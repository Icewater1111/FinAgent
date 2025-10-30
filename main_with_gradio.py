import os
import sys
import time
import asyncio
import threading
from datetime import datetime
from typing import List, Tuple, Optional

from dotenv import load_dotenv
import gradio as gr

import tkinter as tk
from tkinter import ttk


from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferWindowMemory

# 回调 handler 的 import
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

# 本地工具/字典导入
from all_tool import all_tools

import webview

# 检查是否在 PyInstaller 打包环境中运行
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # 如果是打包环境，使用 _MEIPASS 作为基础路径
    base_path = sys._MEIPASS
else:
    # 如果是开发环境，使用当前文件所在的目录作为基础路径
    base_path = os.path.dirname(__file__)

print(base_path)
dotenv_path = os.path.join(base_path, '.env')

# 显式加载 .env 文件
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"成功从内部加载 .env 文件: {dotenv_path}")
else:
    # 如果内部路径找不到，作为备用，尝试从当前工作目录加载
    print(f"内部 .env 文件未找到: {dotenv_path}，尝试从当前工作目录加载...")
    #  尝试默认加载方式
    load_dotenv(override=True)

# 环境变量与配置

now = datetime.now()
DashScope_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# API 工具字典
API_TOOL_dic_path = os.path.join(base_path, "api_dic.json")
API_TOOL_dic = {}

# 定义一个包含不同索引名称的列表
index_list = ["AM", "CN", "HK", "OT"]
# 初始化两个字典，用于存储不同类型的 FAISS 数据库
faiss_databases = {}      # 存储全文检索数据库
faiss_key_databases = {}  # 存储关键词检索数据库

# 封装初始化过程
def initialize_all(update_progress=None):

    # 将仅在此函数中使用的导入移至此处
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import DashScopeEmbeddings
    from all_tool import load_dict_from_json

    global API_TOOL_dic, embeddings, faiss_databases, faiss_key_databases
    API_TOOL_dic = load_dict_from_json(API_TOOL_dic_path)

    # Embeddings 与 FAISS 数据库加载
    try:
        # 尝试初始化 DashScopeEmbeddings 模型
        embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=DashScope_API_KEY,
        )
        if update_progress:
            update_progress(1, 6, "Embeddings 初始化完成")
    except Exception as e:
        # 如果初始化失败，打印错误信息并退出程序
        print(f"Error initializing DashScopeEmbeddings: {e}")
        sys.exit(1)  # 发生了严重错误，退出程序


    index_list = ["AM", "CN", "HK", "OT"]
    faiss_databases = {}
    faiss_key_databases = {}

    # 遍历索引列表，为每个索引加载对应的 FAISS 数据库
    for i, index in enumerate(index_list, start=2):
        # 加载全文数据库
        try:
            # 从本地加载全文 FAISS 数据库
            faiss_index_dir = os.path.join(base_path, f"{index}_APISPECS_faiss_index")
            faiss_databases[index] = FAISS.load_local(
            faiss_index_dir, embeddings, allow_dangerous_deserialization=True
            )
            if update_progress:
                update_progress(i, len(index_list) + 2, f"{index} 全文数据库加载成功")
        except Exception as e:
            # 如果加载失败，打印错误信息并退出程序
            print(f"{index} 全文数据库加载失败: {e}")
            sys.exit(1)

        # 加载关键词数据库
        try:
            # 从本地加载全文 FAISS 数据库
            faiss_key_index_dir = os.path.join(base_path, f"{index}_APISPECS_key_faiss_index")
            faiss_key_databases[index] = FAISS.load_local(
            faiss_key_index_dir, embeddings, allow_dangerous_deserialization=True
            )
            if update_progress:
                update_progress(i, len(index_list) + 2, f"{index} 关键词数据库加载成功")
        except Exception as e:
            # 如果加载失败，打印错误信息并退出程序
            print(f"{index} 关键词数据库加载失败: {e}")
            sys.exit(1)

    update_progress(len(index_list) + 1, len(index_list) + 2, "正在打开gradio界面")
    # 在一个单独的线程中启动 Gradio 服务器
    gradio_thread = threading.Thread(target=run_gradio_server, daemon=True)
    gradio_thread.start()
    # 给予 Gradio 足够的时间启动
    time.sleep(3)

    if update_progress:
        update_progress(len(index_list) + 2, len(index_list) + 2, "初始化完成")
        return embeddings, faiss_databases, faiss_key_databases

# Prompt 模板
# 分类关键词提取 key_rag_prompt
key_rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
    """你是一个API查询助手，精通金融数据API文档的结构和内容，当前金融API文档分为美股市场、中国大陆市场、港股市场和其他四类。

    你的第一个任务是根据用户查询，确定其所需要API的分类。分类标准如下：
    *   **AM (美股市场):** 针对美国上市公司、或查询明确指向美国市场/公司，以及Alpha Vantage提供的非中国公司通用财务数据。
    *   **CN (中国大陆市场):** 针对中国A股上市公司、或查询明确指向中国大陆市场，以及与A+H股相关的中国公司数据。
    *   **HK (港股市场):** 针对香港上市公司、或查询明确指向香港市场，以及与A+H股相关的港股数据或非A股的其他市场。
    *   **OT (其他市场与类型):** 非上述明确分类的查询等。

    请在回答时最先输出对应的分类代码（“AM”、“CN”、“HK”或“OT”），之后你的任务是只返回概括出的关键词列表。
    当前时间：{current_time}
    ---
    示例开始 ---

    用户查询: 我要问腾讯的最近的股价

    关键词提取: HK,腾讯控股（00700）,港股,实时行情数据

    ---

    用户查询: 我想看看阿里巴巴过去一年的股价走势

    关键词提取: HK,阿里巴巴（09988）,港股,过去一年,历史行情数据

    ---

    用户查询: 告诉我小米公司的基本情况

    关键词提取: CN,小米集团（01810）,中国公司,详细公司资料,基本信息

    ---

    用户查询: 想了解苹果公司的财务状况

    关键词提取: AM,苹果公司（AAPL）,美国公司,财务状况

    ---
    用户查询: 腾讯有什么新闻？

    关键词提取: CN,腾讯控股,市场新闻,金融新闻,新闻情绪,新闻追踪

    ---
    用户查询: 我想了解华为在国际上的报道。

    关键词提取: CN,华为公司,国际媒体,英文新闻资讯,国际舆情分析

    ---
    用户查询: 告诉我关于特斯拉的英文新闻。

    关键词提取: AM,特斯拉公司（TSLA）,国际媒体,英文新闻资讯,国际舆情监控

    ---
    用户查询: 比特币最近的行情怎么样？

    关键词提取: OT,比特币,加密货币,实时行情,数字货币

    ---
    用户查询: 查询一下英伟达的年度财报。

    关键词提取: AM,英伟达（NVDA）,年度财报,损益表,资产负债表,现金流量表,财务数据

    ---
    用户查询: 港股哪些股票最近停牌了？

    关键词提取: HK,港股,股票停牌,停复牌信息,交易状态

    ---
    示例结束 ---

    用户查询: {init_query}

    关键词提取:
    """
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}"),
])

# 描述 API 文档查询 rag_prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个API查询助手，精通金融数据API文档的结构和内容，当前金融API文档分为美股市场、中国大陆市场、港股市场和其他四类，你的第一个任务是根据查询，确定其所需要API的分类，不一定完全看公司地区，也要考虑公司在哪里上市，以及跟目前查询最相关的市场
     如果属于美股市场，请在回答时最先输出“AM”；如果属于中国大陆市场，请在回答时最先输出“CN”；如果属于港股市场，请在回答时最先输出“HK”；如果属于其他市场与类型，则在回答时最先输出“OT”，之后你的任务是
     根据用户的自然语言查询，生成一个**目标API的描述文本**。这个扩展后的查询旨在帮助向量数据库更好地匹配到最相关的API文档。
     请将文本扩展为以下形式：

     "功能描述": (根据用户问题,概括出需要使用的API功能),
     "数据类型": (即API查找的是什么类型数据, 比如实时行情数据,公司信息等),
     "数据粒度": (分为实时、日频、周频、年频、全部数据以及非时间序列数据),
     "关键词": (根据用户输入概括出的关键词)

     若用户未提及某个标题的内容,默认为空。



     当前时间：{current_time}

     如果用户输入是一个完整的自然语言问题，请你仿照以下格式进行API描述，以生成更清晰的查询：

         示例开始 ---

         用户查询: 我要问腾讯的最近的股价

         扩展查询: HK,查询腾讯控股（00700）在香港股票市场的实时行情数据，包括最新价、涨跌幅、成交量和成交额。
         API描述: HK,
                 "功能描述": 查询指定股票在香港股票市场的实时行情数据，包括最新价、涨跌幅、成交量和成交额,
                 "数据类型": 实时行情数据,
                 "数据粒度": 实时,
                 "关键词": 港股,实时行情数据.

         ---
         用户查询: 我要问东财转2 (123098)的历史转股溢价率和纯债溢价率

         API描述: CN
                 "功能描述": 获取指定沪深可转债的历史价值分析数据。此接口提供特定可转债在不同日期的收盘价、纯债价值、转股价值、纯债溢价率和转股溢价率等关键指标,
                 "数据类型": 可转债价值分析，历史估值数据，时间序列数据,
                 "数据粒度": 日频，针对单只可转债,
                 "关键词": 可转债, 沪深可转债, 价值分析,纯债溢价率,转股溢价率,历史数据

         ---
         ---
         用户查询: 我要问查询比亚迪公司的最新动态和中文新闻资讯，包括新闻标题、来源、发布日期和链接等，用于公司舆情监控和投资决策。

         API描述: HK,
                 "功能描述": 查询指定公司的最新中文新闻资讯, 包含新闻标题、来源、发布日期和链接。
                 "数据类型": 实时新闻数据,
                 "数据粒度": 实时,
                 "关键词": 最新动态,市场新闻,金融新闻,新闻情绪,新闻追踪,舆情分析

         ---

         用户查询: 我想看看阿里巴巴过去一年的股价走势

         API描述: HK,
                 "功能描述": 查询指定股票在香港股票市场的历史行情数据, 包括开盘价、收盘价、最高价、最低价、成交量、成交额和K线数据。
                 "数据类型": 历史行情数据,
                 "数据粒度": 月频,
                 "关键词": 港股,过去一年,历史行情数据
         ---

         用户查询: 告诉我小米的基本情况

         API描述: CN,
                 "功能描述": 查询指定公司的详细公司资料和基本信息，包括公司名称、注册地、所属行业、董事长和公司介绍。
                 "数据类型": 公司信息,
                 "数据粒度": 非时间序列数据,
                 "关键词": 中国公司,详细公司资料,基本信息


         ---
         用户查询: 腾讯有什么新闻？

         API描述: HK,
                 "功能描述": 查询指定公司的中文新闻资讯，包括新闻标题、来源、发布日期和链接。
                 "数据类型": 新闻资讯,
                 "数据粒度": 实时数据,
                 "关键词": 市场新闻,金融新闻,新闻情绪,新闻追踪

         ---
         用户查询: 我想了解华为在国际上的报道。

         API描述: CN,
                 "功能描述": 查询指定公司在国际媒体上的英文新闻资讯，包括新闻标题、来源、发布日期和链接。
                 "数据类型": 新闻资讯,
                 "数据粒度": 实时数据,
                 "关键词": 国际媒体,英文新闻资讯,国际舆情分析

         ---
         用户查询: 告诉我关于特斯拉的英文新闻。

         API描述: AM,
                 "功能描述": 查询指定公司在国际媒体上的英文新闻资讯，包括新闻标题、来源、发布日期和链接。
                 "数据类型": 新闻资讯,
                 "数据粒度": 实时数据,
                 "关键词": 国际媒体,英文新闻资讯,国际舆情监控

         ---
         示例结束 ---

     请只返回最终需要寻找的API描述，不要添加任何解释或说明文字。
     """),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}"),
])

# 拆分查询
rag_prompt_split = ChatPromptTemplate.from_messages([
    (
        "system",
        '''请判断下面这个查询查询几项内容，并且只将所查询的内容按照‘1.\n 2.\n’的格式列出，尽可能的让每个单独的查询内容都拥有准确的时间空间信息，注意我们只保留对金融部分的查询。
        如果用户输入不是一个完整的自然语言查询，请你结合历史对话内容 chat_history，尝试补全用户的真实意图生成一个完整查询，然后再进行分割查询。
        
        请不要直接进行回答，你的任务只是分割查询。

        示例开始 ---
        用户查询: 我想分析可转债‘113527’过去三个月的价值分析数据并获取同期比特币每日收盘价

        分割查询: 1.分析可转债‘113527’过去三个月的价值分析数据。\n 2.获取过去三个月比特币每日收盘价。\n
        ---
        用户查询: 计算食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的速动比率、资产负债率和每股收益，并按综合评级给出前 5 名。

        分割查询: 1.查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的速动比率。\n 2.查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的资产负债率。\n 3.查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的每股收益。\n
        ---
        用户查询: 获取全市场未来 3 个月（2025-07-22 至 2025-10-22）的财报日历，筛选出同时有 ≥5 次机构调研的公司并排名。

        分割查询: 1.获取全市场未来 3 个月（2025-07-22 至 2025-10-22）的财报日历。\n
        ---
        示例结束 ---

        用户查询: {init_query}
        
        请直接输出分割查询的内容，不要输出其他解释说明
        ''',
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}"),
])

# LLM 初始化

def set_active_model(selected_model: str):
    """根据 selected_model 字符串设置全局 model 和 model_split"""
    # 声明全局变量 model 和 model_split
    global model, model_split
    # 将传入的 selected_model 转换为小写，如果为空则默认为 "qwen-max"
    selected = (selected_model or "qwen-max").lower()
    try:
        # 设置分割模型为 ChatTongyi(qwen-plus)
        model_split = ChatTongyi(model="qwen-turbo", model_kwargs={"seed": 45}, api_key=DashScope_API_KEY)
        # 根据 selected_model 的值设置不同的模型
        if selected in {"qwen-max", "qwen_max", "qwen max", "qwenmax"}:
            # 如果选择的是 qwen-max，则设置主模型为 ChatTongyi(qwen-max)
            model = ChatTongyi(model="qwen-max", api_key=DashScope_API_KEY)
        elif selected in {"qwen-plus", "qwen_plus", "qwen plus"}:
            # 如果选择的是 qwen-plus，则设置主模型为 ChatTongyi(qwen-plus)
            model = ChatTongyi(model="qwen-plus", api_key=DashScope_API_KEY)
        elif selected in {"deepseek", "deepseek-chat", "deepseek_chat"}:
            # 如果选择的是 deepseek 模型
            try:
                # 使用 init_chat_model 初始化 deepseek-chat 模型
                model = init_chat_model("deepseek-chat", model_provider="deepseek", seed=2025)
            except Exception:
                # 如果 init_chat_model 不可用或初始化失败，则使用 ChatTongyi(qwen-max)
                model = ChatTongyi(model="qwen-max", api_key=DashScope_API_KEY)
        else:
            # 如果没有匹配到任何特定模型，则使用 ChatTongyi(qwen-max) 作为主模型
            model = ChatTongyi(model="qwen-max", api_key=DashScope_API_KEY)
        # 打印当前激活的模型名称
        print(f"Active model set to: {selected_model}")
    except Exception as e:
        # 如果设置模型过程中发生错误，则打印错误信息
        print(f"Failed to set model to {selected_model}: {e}")
        # 如果初始化失败，保持之前的模型不变

# 在程序启动时设置默认模型为 "qwen-max"
set_active_model("qwen-max")

# 构建链
# 关键词链
key_rag_llm_chain = key_rag_prompt.partial(current_time=str(datetime.now())) | model | StrOutputParser()
# 完整API描述扩展链
rag_llm_chain = rag_prompt.partial(current_time=str(datetime.now())) | model | StrOutputParser()
# 拆分链
rag_llm_chain_split = rag_prompt_split | model_split | StrOutputParser()


# 工具函数
def split_numbered_items(s: str) -> List[str]:
    """
    将一个包含编号列表的字符串拆分成单个项目的列表。
    例如："1. Item One 2. Item Two" -> ["Item One", "Item Two"]
    """
    # 查找所有数字点空格的起始位置，表示一个新项目的开始
    starts = [i for i in range(len(s) - 2) if s[i].isdigit() and s[i + 1] == '.' and s[i + 2].isspace()]
    # 在列表末尾添加字符串的总长度，以便处理最后一个项目
    starts.append(len(s))
    items: List[str] = []
    # 遍历起始位置列表，提取每个项目的内容
    for idx in range(len(starts) - 1):
        # 截取当前项目对应的字符串片段
        seg = s[starts[idx]:starts[idx + 1]]
        # 找到第一个点号的位置
        dot_pos = seg.find('.')
        # 提取点号之后的内容并去除首尾空格
        content = seg[dot_pos + 1:].strip()
        # 如果内容被双星号包裹，则去除双星号
        if content.startswith("**") and content.endswith("**"):
            content = content[2:-2].strip()
        items.append(content)
    return items


def extract_market_category(llm_response_text: str) -> Optional[str]:
    """
    从LLM的回答中提取前两个字母的市场分类代码。
    预期LLM回答格式为 "XX,扩展查询文本..." 或 "XX扩展查询文本..."。
    例如："HK,腾讯控股" -> "HK"
    """
    # 检查输入是否为有效的非空字符串
    if not isinstance(llm_response_text, str) or not llm_response_text:
        return None
    # 去除字符串首尾空格
    cleaned_response = llm_response_text.strip()
    # 如果字符串长度至少为2，则尝试提取前两个字符作为分类
    if len(cleaned_response) >= 2:
        category = cleaned_response[:2].upper()
        # 检查提取的分类是否在预定义的市场分类集合中
        if category in {"AM", "CN", "HK", "OT"}:
            return category
    return None


# 自定义回调 handler
class ToolLoggingHandler(BaseCallbackHandler):
    """
    一个自定义的 LangChain 回调处理器，用于捕获和记录工具调用的详细过程。
    这些记录可以用于在UI中实时显示或调试。
    """
    def __init__(self):
        # 初始化一个列表来存储所有捕获到的日志条目
        self.entries: List[str] = []
        # 初始化一个线程锁，用于确保对 entries 列表的线程安全访问
        self.lock = threading.Lock()

    def _append(self, text: str):
        """
        线程安全地向日志条目列表追加文本。
        """
        with self.lock:
            self.entries.append(text)

    def pop_entries(self) -> List[str]:
        """
        返回当前所有新的日志条目，并清空内部缓冲区（线程安全）。
        """
        with self.lock:
            # 复制当前条目
            out = self.entries.copy()
            # 清空缓冲区
            self.entries.clear()
        return out

    def _format_output(self, output) -> str:
        """
        将工具的返回值格式化为适合在聊天框或日志中展示的字符串。
        尝试将 DataFrame 转换为 Markdown，将 dict/list 转换为 JSON，否则转换为 repr 或 str。
        """
        try:
            import pandas as pd
        except Exception:
            pd = None
        try:
            import json as _json
        except Exception:
            _json = None

        # 将DataFrame转换为markdown
        try:
            if pd is not None and isinstance(output, pd.DataFrame):
                try:
                    # 尝试转换为 Markdown 表格
                    return output.to_markdown(index=False)
                except Exception:
                    # 失败则转换为 CSV
                    return output.to_csv(index=False)
            # 将list/dict转换为json
            if _json is not None and isinstance(output, (dict, list)):
                try:
                    # 尝试转换为 JSON
                    return _json.dumps(output, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

        # 最后退化为 repr
        try:
            # 尝试使用 repr() 获取对象的表示
            return repr(output)
        except Exception:
            # 最终回退到 str()
            return str(output)

    def _repr_input(self, inp) -> str:
        """
        将工具的输入格式化为适合日志展示的字符串。
        尝试将 dict/list 转换为 JSON，否则转换为 str 或 repr。
        """
        try:
            import json as _json
            if isinstance(inp, (dict, list)):
                # 尝试转换为 JSON
                return _json.dumps(inp, ensure_ascii=False)
        except Exception:
            pass
        try:
            # 尝试转换为字符串
            return str(inp)
        except Exception:
            # 最终回退到 repr()
            return repr(inp)

    def on_tool_start(self, serialized, input_str, run_id=None, **kwargs):
        """
         当工具开始执行时调用。记录工具名称和输入参数。
        """
        try:
            # 尝试从 serialized 对象中获取工具名称
            name = serialized.get("name") if isinstance(serialized, dict) else getattr(serialized, "name", str(serialized))
        except Exception:
            name = str(serialized)
        # 格式化输入
        input_repr = self._repr_input(input_str)
        # 追加日志
        self._append(f"Invoking: {name} with {input_repr}")

    def on_tool_end(self, output, run_id=None, **kwargs):
        """
        当工具执行结束时调用。记录工具的返回值。
        """
        # 格式化输出
        formatted = self._format_output(output)
        # 追加日志
        self._append("Tool returned: " + formatted)

    def on_agent_action(self, action, run_id=None, **kwargs):
        """
        当 Agent 决定执行一个动作（通常是调用工具）时调用。
        """
        tool_name = None
        tool_input = None
        try:
            # 尝试从 action 对象中提取工具名和输入
            if isinstance(action, dict):
                tool_name = action.get("tool") or action.get("name")
                tool_input = action.get("tool_input") or action.get("input") or action.get("args")
            else:
                tool_name = getattr(action, "tool", None) or getattr(action, "name", None)
                tool_input = getattr(action, "tool_input", None) or getattr(action, "input", None)
        except Exception:
            pass
        if tool_name:
            # 格式化输入
            input_repr = self._repr_input(tool_input)
            # 追加日志
            self._append(f"→ Agent 执行动作: 调用 {tool_name}，参数: {input_repr}")
        else:
            try:
                # 如果无法提取工具信息，则直接记录动作的 repr
                self._append(f"→ Agent 执行动作: {repr(action)}")
            except Exception:
                # 最终回退到 str()
                self._append(f"→ Agent 执行动作: {str(action)}")

    def on_agent_finish(self, finish, run_id=None, **kwargs):
        """
        当 Agent 完成其所有动作并给出最终回答时调用。
        """
        # 追加日志
        self._append(f"🏁 Agent 完成: {repr(finish)}")

# Chat 处理函数
# 定义记忆窗口的大小，这里设置为5个回合
memory_k = 5
# 初始化一个对话缓冲区记忆，用于存储聊天历史，以便 Agent 能够记住之前的对话
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=memory_k, return_messages=True)

# 定义聊天提示模板，包含系统指令、聊天历史、用户输入和 Agent 暂存区
prompt = ChatPromptTemplate.from_messages([
   (
       "system",
       f"""你是一个专业的金融助手，当前时间为 {now}。\n你的任务是帮助用户获取与金融相关的数据、解释、分析或回答问题。\n你可以调用一组 API 工具来获取股票、债券、公司财务、宏观经济等数据。\n请按照以下规则操作：\n1. 你应当优先调用 API 工具来获取实时或历史数据，而不是依赖自己的知识编造数据。 只有当问题属于概念解释、定义说明、通用知识时，才直接回答，不使用工具。\n2. 如果用户的问题是自然语言查询，例如： - “什么是市盈率？” - “你是谁？” - “请解释一下ETF的分类。” 这类问题不需要调用 API 工具，请直接用自己的知识回答。\n3. 当用户没有说明查询时间范围时，请默认查找最近的数据；如果最近无数据，可以适当往历史数据中查找。\n你的回答应简洁清晰，使用中文输出。""",
   ),
   ("placeholder", "{chat_history}"),  # 聊天历史的占位符
   ("human", "{input}"),             # 用户输入的占位符
   ("placeholder", "{agent_scratchpad}"),  # Agent 思考过程的占位符
])


def messages_to_dicts(messages):
    """
    将 LangChain 消息对象（或字典）转换为 {'role','content'} 字典列表。
    这是一个兼容性函数，用于确保聊天历史格式一致。
    """
    out: List[dict] = []
    try:
        from langchain_core.messages import HumanMessage as _HumanMessage, AIMessage as _AIMessage
    except Exception:
        _HumanMessage = None
        _AIMessage = None

    if not messages:
        return out

    for m in messages:
        # 如果已经是带有 role/content 的字典，则直接添加
        if isinstance(m, dict) and 'role' in m and 'content' in m:
            out.append(m)
            continue

        # 处理已知的 LangChain 消息类型
        role = None
        if _HumanMessage is not None and isinstance(m, _HumanMessage):
            role = 'user'
        elif _AIMessage is not None and isinstance(m, _AIMessage):
            role = 'assistant'
        else:
            # 尝试从消息对象中获取 role 或 type 属性，默认为 'user'
            role = getattr(m, 'role', None) or getattr(m, 'type', None) or 'user'
        # 获取消息内容
        content = getattr(m, 'content', None)
        if content is None:
            try:
                # 如果内容为空，尝试转换为字符串
                content = str(m)
            except Exception:
                content = ''
        out.append({'role': role, 'content': content})
    return out

async def expand_query(single_split_query: str, memory: ConversationBufferWindowMemory, mode: str) -> str:
    """
    使用 RAG LLM 链扩展单个拆分后的查询。
    它会将聊天历史转换为字典格式，然后传递给 RAG LLM 链。
    """
    # 将 memory.buffer 转换为普通字典
    chat_history = messages_to_dicts(getattr(memory, 'buffer', []) or [])
    if mode == "关键词模式":
        chain = key_rag_llm_chain
    else:
        chain = rag_llm_chain
    # 调用 RAG LLM 链进行查询扩展
    return await chain.ainvoke({
        "init_query": single_split_query,
        "chat_history": chat_history,
    })


async def retrieve(single_last_query: str, mode: str, topk: int):
    """
    根据给定的查询、检索模式和 Top-K 值从 FAISS 数据库中检索相关文档。
    首先从查询中提取市场分类，然后根据模式选择全文或关键词数据库进行检索。
    """
    # 从查询中提取市场分类
    query_index = extract_market_category(single_last_query)
    if not query_index:
        # 如果未能提取市场分类，则返回空列表
        return query_index, []
    if mode == "关键词模式":
        # 如果是关键词模式，从关键词 FAISS 数据库中进行相似性搜索
        docs = await faiss_key_databases[query_index].asimilarity_search(single_last_query, k=topk)
    else:
        # 否则，从全文 FAISS 数据库中进行相似性搜索
        docs = await faiss_databases[query_index].asimilarity_search(single_last_query, k=topk)
    return query_index, docs

# Gradio 界面

def normalize_chat_history(history) -> List[Tuple[str, str]]:
    """
    将不同格式的聊天历史标准化为 [(用户消息, 助手消息), ...] 的列表。
    """
    if not history:
        return []
    normalized: List[Tuple[str, str]] = []
    for item in history:
        # 处理 {'role', 'content'} 字典格式
        if isinstance(item, dict) and 'role' in item and 'content' in item:
            if item['role'] == 'user':
                # 用户消息，助手消息为空
                normalized.append((item['content'], ""))
            elif item['role'] == 'assistant':
                # 如果前一条是用户消息且助手消息为空，则更新前一条的助手消息
                if normalized and normalized[-1][1] == "":
                    user_text = normalized[-1][0]
                    normalized[-1] = (user_text, item['content'])
                else:
                    # 否则，助手消息，用户消息为空
                    normalized.append(("", item['content']))
        # 处理 [(user, assistant), ...] 元组/列表格式
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append((item[0], item[1]))
        else:
            # 其他未知格式，简单转换为字符串作为用户消息
            normalized.append((str(item), ""))
    return normalized


def format_chat_for_gradio(chat_pairs: List[Tuple[str, str]]):
    """
    将内部的 [(user, assistant), ...] 对列表转换为 Gradio Chatbot 组件所需的 messages 格式。
    """
    formatted = []
    for u, a in chat_pairs:
        if u is not None and u != "":
            # 添加用户消息
            formatted.append({"role": "user", "content": u})
        # 添加助手消息，即使为空也添加，因为 Gradio 期望 role/content
        formatted.append({"role": "assistant", "content": a if a is not None else ""})
    return formatted

# 启动界面
class SplashScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("程序启动中")
        self.geometry("500x200+600+300")
        self.resizable(False, False)


        tk.Label(self, text="正在初始化，请稍候...", font=("Arial", 16)).pack(pady=20)
        self.progress = ttk.Progressbar(self, length=400, mode="determinate")
        self.progress.pack(pady=10)
        self.status_label = tk.Label(self, text="准备中...", font=("Arial", 12))
        self.status_label.pack(pady=5)


        threading.Thread(target=self.background_init, daemon=True).start()


    def update_progress(self, step, total, status):
        percent = int(step / total * 100)
        self.progress["value"] = percent
        self.status_label.config(text=f"{status} ({percent}%)")


    def background_init(self):
        # 进行初始化以及文档加载
        initialize_all(self.update_progress)
        self.after(500, self.start_main_app)


    def start_main_app(self):
        self.destroy()
        # 调用 webview 启动
        start_webview_window()


# Gradio 应用界面定义
with gr.Blocks(theme=gr.themes.Soft(), css='''
#title {text-align:center; font-size:28px; font-weight:bold; background: linear-gradient(90deg, #004e92, #000428); -webkit-background-clip: text; color: transparent;}
#subtitle {text-align:center; font-size:16px; color:#555; margin-bottom:20px;}
.chatbox {border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.08);}
.logs {background:#111; color:#0f0; font-family:monospace; font-size:13px;}
''') as demo:
    # 标题和副标题
    gr.Markdown("<div id='title'>📊 金融API调用助手（流式）</div>")
    gr.Markdown("<div id='subtitle'>基于 RAG + LangChain + FAISS + 智能Agent，实时显示工具调用日志</div>")

    # 垂直布局：高级设置 -> 聊天窗口 -> 日志
    with gr.Column():
        # 1) 高级设置
        with gr.Accordion("⚙️ 高级设置", open=False):
            # 检索模式选择：关键词模式或全文档模式
            mode = gr.Radio(["关键词模式", "全文档模式"], label="🔍 检索模式", value="全文档模式")
            # Top-K 检索数滑块
            topk = gr.Slider(1, 10, value=3, step=1, label="📑 Top-K 检索数")
            # 最大记忆回合数滑块
            memory_window = gr.Slider(1, 10, value=5, step=1, label="🧠 最大记忆回合数")
            # 模型切换下拉菜单
            model_selector = gr.Dropdown(choices=["qwen-max", "qwen-plus", "deepseek-chat"], value="qwen-max", label="模型选择")

        # 2) 聊天窗口
        chatbot = gr.Chatbot(label="💬 对话窗口", type="messages", height=420, elem_classes="chatbox")

        # 3) 调试日志
        with gr.Accordion("📝 调试日志", open=False):
            logs_box = gr.Textbox(label="", lines=15, interactive=False, elem_classes="logs")

    # 底部：输入与按钮行
    with gr.Row():
        # 用户输入文本框
        user_input = gr.Textbox(placeholder="请输入您的金融问题...", label="", scale=6)
        with gr.Column(scale=1):
            # 发送按钮
            submit_btn = gr.Button("📤 发送", variant="primary")
            # 清空对话按钮
            clear_btn = gr.Button("🧹 清空对话", variant="secondary")

    def user_submit_stream(message, history, mode, topk, memory_window, selected_model):
        """
        这是一个 generator 函数，用于处理用户提交并以 streaming 模式向 Gradio 返回结果。
        每次 yield 会更新 Gradio 界面。
        输出顺序与 submit_btn.click 的 outputs 一致：[chatbot, user_input, logs_box]
        """
        # 如果用户切换了模型，立即初始化模型实例
        set_active_model(selected_model)

        # 先把 history 规范化为 [(user, assistant), ...]
        chat = normalize_chat_history(history)
        # 将用户消息先 append 到聊天显示
        chat.append((message, ""))
        # assistant 内容暂时为空
        logs_text = ""

        # 将 memory window 如原逻辑处理
        global memory
        if memory.k != memory_window:
            memory = ConversationBufferWindowMemory(memory_key="chat_history", k=memory_window, return_messages=True)
            logs_text += f"🔄 重置记忆窗口大小为 {memory_window}\n"

        # 立刻 yield 出用户消息，这会更新聊天窗口和输入框
        yield format_chat_for_gradio(chat), "", logs_text

        # 拆分 & 扩展 查询
        try:
            # 拆分查询
            chat_history = messages_to_dicts(getattr(memory, 'buffer', []) or [])
            splited_query = rag_llm_chain_split.invoke({"init_query": message, "chat_history": chat_history})
        except Exception as e:
            # 记录错误信息到聊天
            chat[-1] = (message, f"错误：拆分查询失败: {e}")
            # 记录错误信息到日志
            logs_text += f"✖ 拆分查询失败: {e}\n"
            # 更新界面并返回
            yield format_chat_for_gradio(chat), "", logs_text
            return

        # 解析拆分后的查询
        splited_query_list = split_numbered_items(splited_query)
        # 记录日志
        logs_text += f"✂️ 拆分后的查询: {splited_query_list}\n"
        # 更新界面
        yield format_chat_for_gradio(chat), "", logs_text

        async def expand_all():
            """异步函数，用于并发扩展所有拆分后的查询"""
            return await asyncio.gather(*[expand_query(q, memory, mode) for q in splited_query_list])

        try:
            # 运行异步扩展查询
            last_query_list = asyncio.run(expand_all())
        except Exception as e:
            chat[-1] = (message, f"错误：扩展查询失败: {e}")
            logs_text += f"✖ 扩展查询失败: {e}\n"
            yield format_chat_for_gradio(chat), "", logs_text
            return

        logs_text += f"📝 扩展查询结果: {last_query_list}\n"
        yield format_chat_for_gradio(chat), "", logs_text

        # 并发检索
        async def retrieve_all():
            """异步函数，用于并发检索所有扩展后的查询"""
            return await asyncio.gather(*[retrieve(q, mode, topk) for q in last_query_list])

        try:
            # 运行异步检索
            retrieval_results = asyncio.run(retrieve_all())
        except Exception as e:
            chat[-1] = (message, f"错误：检索失败: {e}")
            logs_text += f"✖ 检索失败: {e}\n"
            yield format_chat_for_gradio(chat), "", logs_text
            return

        results, query_tool_names = [], []
        for query_index, docs in retrieval_results:
            if not query_index:
                continue
            logs_text += f"🔍 提取市场分类: {query_index}\n"
            logs_text += f"📖 使用数据库 [{query_index}] 检索 top{topk} 条\n"
            for doc in docs:
                tool_meta_name = doc.metadata.get("name")
                if tool_meta_name and tool_meta_name in API_TOOL_dic:
                    query_tool_names.append(API_TOOL_dic[tool_meta_name])
        # 去重
        query_tool_names = list(set(query_tool_names))
        logs_text += f"🛠️ 检索到的工具: {query_tool_names if query_tool_names else '无'}\n"
        yield format_chat_for_gradio(chat), "", logs_text

        # 如果没有工具，则直接让 LLM 回答，并结束
        query_tools_instances = [tool for tool in all_tools if tool.name in query_tool_names]
        if not query_tools_instances:
            # 直接调用 LLM
            direct_llm_response = model.invoke(message)
            assistant_text = direct_llm_response.content if hasattr(direct_llm_response, "content") else str(direct_llm_response)
            # 更新聊天历史
            chat[-1] = (message, assistant_text)
            logs_text += "⚠️ 未检索到合适工具 → 直接由LLM回答\n"
            # 更新界面并返回
            yield format_chat_for_gradio(chat), "", logs_text
            return

        # 有工具，启动 agent 于独立线程并通过 ToolLoggingHandler 实时抓取日志
        # 创建工具日志处理器
        tool_logger = ToolLoggingHandler()
        # 创建回调管理器
        cb_manager = CallbackManager([tool_logger])
        # 创建 Agent
        agent = create_tool_calling_agent(model, query_tools_instances, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=query_tools_instances, verbose=True, memory=memory, callback_manager=cb_manager)

        # 用于存储 Agent 的最终响应或错误
        response_container = {"response": None, "error": None}

        def _run_agent():
            """在单独线程中运行 Agent 的辅助函数"""
            try:
                try:
                    # 调用 Agent 执行器
                    resp = agent_executor.invoke({"input": message}, callbacks=[tool_logger])
                except TypeError:
                    resp = agent_executor.invoke({"input": message})
                response_container["response"] = resp
            except Exception as e:
                response_container["error"] = e

        # 创建一个守护线程来运行 Agent
        thread = threading.Thread(target=_run_agent, daemon=True)
        # 启动线程
        thread.start()

        # 轮询 handler.entries，持续把中间日志拼成 assistant 的“部分输出”并 yield
        accumulated_tool_trace = ""
        # 当线程还在运行或日志缓冲区有内容时
        while thread.is_alive() or tool_logger.entries:
            # 获取新的日志条目
            new_entries = tool_logger.pop_entries()
            if new_entries:
                for e in new_entries:
                    # 累积工具调用追踪日志
                    accumulated_tool_trace += e + "\n\n"
                # 把当前的工具调用追踪当作 assistant 的临时内容展示
                chat[-1] = (message, accumulated_tool_trace + "···(正在继续调用工具，稍候更新)···")
                # 更新日志文本框
                logs_text += "".join([e + "\n" for e in new_entries])
                # 更新界面
                yield format_chat_for_gradio(chat), "", logs_text
            else:
                # 如果没有新日志，短暂休眠，避免CPU空转
                time.sleep(0.08)

        # 线程已结束，处理最终结果或异常
        if response_container.get("error"):
            err = response_container["error"]
            # 最终文本为错误信息
            final_text = f"工具执行失败: {err}"
            logs_text += f"✖ Agent 执行失败: {err}\n"
        else:
            resp = response_container.get("response")
            if isinstance(resp, dict):
                final_text = resp.get('output') or resp.get('result') or repr(resp)
            else:
                final_text = getattr(resp, 'content', None) or getattr(resp, 'output', None) or str(resp)
            logs_text += "🤖 Agent 调用完成。\n"

        # 把最终回答写入聊天
        chat[-1] = (message, final_text)
        # 最终更新界面
        yield format_chat_for_gradio(chat), "", logs_text

    # 绑定发送按钮的点击事件到 user_submit_stream 函数
    submit_btn.click(
        fn=user_submit_stream,
        inputs=[user_input, chatbot, mode, topk, memory_window, model_selector],
        outputs=[chatbot, user_input, logs_box],
    )

    # 绑定清空按钮的点击事件，清空聊天、输入框和日志
    clear_btn.click(lambda: ([], "", ""), None, [chatbot, user_input, logs_box])

    # 如果作为主程序运行，则启动 Gradio 界面
    if __name__ == "__main__":
        # 定义一个固定的端口

        GRADIO_SERVER_PORT = 7860

        def run_gradio_server():
            """在单独线程中启动 Gradio 服务器"""
            print(f"Starting Gradio server on port {GRADIO_SERVER_PORT}...")
            demo.launch(
                server_name="127.0.0.1",  # 确保只在本地监听
                server_port=GRADIO_SERVER_PORT,
                inbrowser=False,  # 不自动打开浏览器
                share=False,  # 不创建公共共享链接
                # debug=True
            )
            print("Gradio server stopped.")


        def start_webview_window():
            """在主线程中启动 pywebview 窗口"""

            local_url = f"http://127.0.0.1:{GRADIO_SERVER_PORT}"
            print(f"Loading Gradio app in webview from: {local_url}")

            webview.create_window(
                "金融API调用助手",  # 窗口标题
                url=local_url,
                width=1000,
                height=800,
                min_size=(800, 600),  # 最小窗口大小
                resizable=True,
            )
            webview.start()

        # 入口显示 SplashScreen
        splash = SplashScreen()
        splash.mainloop()

        print("Application closed.")

