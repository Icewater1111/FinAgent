import asyncio
import os
import sys
import threading
import time
import io
from contextlib import contextmanager
from datetime import datetime
from dotenv import load_dotenv
import json
import re
from typing import Any, Dict, Tuple, List, Literal, TypedDict, Optional
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.chat_models import ChatTongyi
from langchain.chat_models import init_chat_model
# 导入记忆模块
from langchain.memory import ConversationBufferWindowMemory # 导入滑动窗口记忆
# 回调 handler 的 import
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print("main.py中datetime是什么类型？", type(datetime), datetime)
from pydantic import BaseModel, Field
# 导入 LangGraph
from langgraph.graph import StateGraph, END
# 导入 Gradio 和 webview
import gradio as gr
import webview
import tkinter as tk
from tkinter import ttk
from all_tool import all_tools
now = datetime.now()


print("--- Python 脚本中读取的环境变量 ---")
print("HTTP_PROXY:", os.getenv('HTTP_PROXY'))
print("HTTPS_PROXY:", os.getenv('HTTPS_PROXY'))
print("http_proxy:", os.getenv('http_proxy'))
print("https_proxy:", os.getenv('https_proxy'))
print("-----------------------------------")

# 检查是否在 PyInstaller 打包环境中运行
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # 如果是打包环境，使用 _MEIPASS 作为基础路径
    base_path = sys._MEIPASS
else:
    # 如果是开发环境，使用当前文件所在的目录作为基础路径
    base_path = os.path.dirname(__file__)

print(f"基础路径: {base_path}")
dotenv_path = os.path.join(base_path, '.env')

# 显式加载 .env 文件
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"成功从内部加载 .env 文件: {dotenv_path}")
else:
    # 如果内部路径找不到，作为备用，尝试从当前工作目录加载
    print(f"内部 .env 文件未找到: {dotenv_path}，尝试从当前工作目录加载...")
    load_dotenv(override=True)

# 加载api_tool_dic
API_TOOL_dic_path = os.path.join(base_path, "api_dic.json")
API_TOOL_dic = None

# 加载四种FASSI数据库
DashScope_API_KEY = os.getenv("DASHSCOPE_API_KEY")


class FixedChatTongyi(ChatTongyi):
    """修复了多 tool_calls 流式响应 bug 的 ChatTongyi"""

    def subtract_client_response(self, resp: Any, prev_resp: Any) -> Any:
        """Subtract prev response from curr response.

        Useful when streaming without `incremental_output = True`
        """

        resp_copy = json.loads(json.dumps(resp))
        choice = resp_copy["output"]["choices"][0]
        message = choice["message"]

        prev_resp_copy = json.loads(json.dumps(prev_resp))
        prev_choice = prev_resp_copy["output"]["choices"][0]
        prev_message = prev_choice["message"]

        message["content"] = message["content"].replace(prev_message["content"], "")

        if message.get("tool_calls"):
            for index, tool_call in enumerate(message["tool_calls"]):
                function = tool_call["function"]

                if prev_message.get("tool_calls"):
                    if index < len(prev_message["tool_calls"]):
                        prev_function = prev_message["tool_calls"][index]["function"]

                        if "name" in function:
                            function["name"] = function["name"].replace(
                                prev_function["name"], ""
                            )
                        if "arguments" in function:
                            function["arguments"] = function["arguments"].replace(
                                prev_function["arguments"], ""
                            )

        return resp_copy

# 用于结构化输出的 Pydantic 模型
class SynthesizerOutput(BaseModel):
    """定义了管理者/整合者Chain的最终输出结构。"""
    final_answer: str = Field(description="整合所有中间步骤后，生成给用户的最终答案。")
    reasoning: str = Field(description="对以上决策的简要说明（可选）。")

class IntentOutput(BaseModel):
    """定义了用户意图判断的输出结构。"""
    is_tool_needed: bool = Field(description="判断用户查询是否需要调用金融工具或API。如果是，则为 True；如果是闲聊或概念解释，则为 False。")
    query: str = Field(description="根据对话历史重构后的用户完整查询。")

class SubQueryOutput(BaseModel):
    """定义了复杂查询分解后的子查询列表结构。"""
    sub_queries: List[str] = Field(description="分解后的子查询列表。每个子查询都应是独立的、信息完整的可执行任务。")

class TaskInfo(BaseModel):
    """单个任务的信息"""
    task: str = Field(description="任务描述")
    needs_tool: bool = Field(description="该任务是否需要调用金融工具或API。如果需要获取金融数据（如股价、财务指标、新闻等），则为True；如果只需要基于已有数据进行计算、比较、分析等，则为False。")

class HierarchicalTaskOutput(BaseModel):
    """定义了层级任务结构，支持动态级别和依赖管理。"""
    tasks_by_level: Dict[str, List[TaskInfo]] = Field(description="按级别组织的任务字典。key为级别号字符串（从'1'开始），value为该级别的任务信息列表。每个任务包含任务描述和是否需要工具的判断。")
    max_level: int = Field(description="最大级别数，表示有多少个任务级别。", default=1)
    dependencies: Dict[str, List[int]] = Field(description="任务依赖关系映射。key为任务描述，value为它所依赖的任务索引列表（从1开始）。例如：{'任务2': [1]} 表示任务2依赖任务1。", default_factory=dict)

class ExpandedQueryOutput(BaseModel):
    """定义了查询扩展和市场分类的输出结构。"""
    market_category: Literal["AM", "CN", "HK", "OT"] = Field(description="根据用户查询内容判断出的目标市场分类。")
    expanded_query: str = Field(description="经过详细扩展、用于向量检索的查询文本。")

# Pydantic 解析器实例化
synthesizer_parser = PydanticOutputParser(pydantic_object=SynthesizerOutput)
intent_parser = PydanticOutputParser(pydantic_object=IntentOutput)
sub_query_parser = PydanticOutputParser(pydantic_object=SubQueryOutput)
expanded_query_parser = PydanticOutputParser(pydantic_object=ExpandedQueryOutput)
hierarchical_task_parser = PydanticOutputParser(pydantic_object=HierarchicalTaskOutput)


# 封装初始化过程
def initialize_all(update_progress=None):

    # 将仅在此函数中使用的导入移至此处
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import DashScopeEmbeddings
    from all_tool import load_dict_from_json

    """初始化所有数据库和资源"""
    global API_TOOL_dic, faiss_databases, faiss_key_databases, embeddings

    API_TOOL_dic = load_dict_from_json(API_TOOL_dic_path)
    
    try:
        # 尝试初始化 DashScopeEmbeddings 模型
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=DashScope_API_KEY,
        )
        if update_progress:
            update_progress(1, 6, "Embeddings 初始化完成")
    except Exception as e:
        print(f"Error initializing DashScopeEmbeddings: {e}")
        sys.exit(1)

    # 加载完整数据库
    index_list = ["AM", "CN", "HK", "OT"]
    faiss_databases = {}
    faiss_key_databases = {}
    
    for i, index in enumerate(index_list, start=2):
        # 加载全文数据库
        try:
            database_path = os.path.join(base_path, f"{index}_APISPECS_faiss_index")
            faiss_databases[index] = FAISS.load_local(database_path, embeddings, allow_dangerous_deserialization=True)
            if update_progress:
                update_progress(i, len(index_list) + 2, f"{index} 全文数据库加载成功")
            print(f"{index}_FAISS 数据库已从 {database_path} 成功加载。")
        except Exception as e:
            print(f"加载 FAISS 数据库失败: {e}")
            print("请确保创建database的py文件已运行且数据库文件存在。")
            sys.exit(1)
        
        # 加载关键词数据库
        try:
            key_database_path = os.path.join(base_path, f"{index}_APISPECS_key_faiss_index")
            faiss_key_databases[index] = FAISS.load_local(key_database_path, embeddings, allow_dangerous_deserialization=True)
            if update_progress:
                update_progress(i, len(index_list) + 2, f"{index} 关键词数据库加载成功")
            print(f"{index}_FAISS 关键词数据库已从 {key_database_path} 成功加载。")
        except Exception as e:
            print(f"加载 FAISS 关键词数据库失败: {e}")
            print("请确保创建database的py文件已运行且数据库文件存在。")
            sys.exit(1)
    
    if update_progress:
        update_progress(len(index_list) + 1, len(index_list) + 2, "正在打开gradio界面")
    
    # 在一个单独的线程中启动 Gradio 服务器
    gradio_thread = threading.Thread(target=run_gradio_server, daemon=True)
    gradio_thread.start()
    # 给予 Gradio 足够的时间启动
    time.sleep(3)
    
    if update_progress:
        update_progress(len(index_list) + 2, len(index_list) + 2, "初始化完成")
    
    return embeddings, faiss_databases, faiss_key_databases

# 全局变量，将在初始化时填充
faiss_databases = {}
faiss_key_databases = {}
embeddings = None


# rag_prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个API查询助手，精通金融数据API文档的结构和内容，当前金融API文档分为美股市场、中国大陆市场、港股市场和其他四类。
     你的任务是根据用户的自然语言查询，判断其市场分类，并生成一个**目标API的描述文本**。这个扩展后的查询旨在帮助向量数据库更好地匹配到最相关的API文档。
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
     
     你必须严格按照以下 JSON 格式输出，不得添加任何解释性文字：
     {format_instructions}
     
     注意：在生成 "expanded_query" 字段时，请遵循上述示例中的详细格式要求，包含"功能描述"、"数据类型"等内容。
     """),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}")
]).partial(
    current_time=str(datetime.now()),
    format_instructions=expanded_query_parser.get_format_instructions()
)


# 初始化LLM
model = FixedChatTongyi(
    model="qwen-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
model_reuse = FixedChatTongyi(
    model="qwen-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
model_split = FixedChatTongyi(
    model="qwen-max",
    model_kwargs={"seed": 45},
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
manage_llm = FixedChatTongyi(
    model="qwen-max", 
    model_kwargs={"seed": 123},
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# history_prompt
history_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个金融问题理解助手。
     你的任务是：根据智能体与用户的对话历史以及当前用户的输入，首先准确判断并整理出用户当前的查询需求。
     
     - 如果涉及金融数据或相关金融信息查询(如股票、债券、公司信息、金融新闻舆情等等)或任何需要结合金融信息的分析性问题，请将 `is_tool_needed` 设为 `true`。
     - 如果用户的问题属于概念解释（如“什么是市盈率？”）、打招呼或一般性对话，请将 `is_tool_needed` 设为 `false`。
     
     无论判断结果如何，都请将用户（可能不完整）的输入结合上下文，重构为一个清晰、完整的查询或问题，并填入 `query` 字段。

     你必须严格按照以下 JSON 格式输出，不得添加任何解释性文字：
     {format_instructions}
     """),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}")
]).partial(format_instructions=intent_parser.get_format_instructions())


# 为闲聊创建专用的对话链
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", f"你是一个友好、乐于助人的AI助手,当前时间为 {now}。请根据对话历史和用户当前的问题，提供一个简洁、流畅、有帮助的回答。"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

# rag_prompt_split，支持层级任务和依赖管理
rag_prompt_split = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个专业的任务分解助手,当前时间为 {current_time}。你的任务是将用户的复杂查询分解为层级化的任务列表，识别任务之间的依赖关系，并判断每个任务是否需要调用金融工具。

【任务层级说明】
- 任务可以分成多个级别，级别从1开始编号
- 第一级任务：可以直接运行的原子任务，不依赖其他任务的结果。这些任务可以并行执行。
- 第二级任务：需要第一级任务的结果才能执行的任务。只有当第一级任务完成后，才能执行第二级任务。
- 第三级及以后：需要前一级任务的结果才能执行。可以有多级，根据实际依赖关系确定。

【工具需求判断】
对于每个任务，你需要判断它是否需要调用金融工具或API：
- **需要工具（needs_tool=true）**：任务需要获取新的金融数据，如：
  - 查询股价、行情数据（如"查询特斯拉的股价"、"获取苹果的股价"）
  - 获取财务指标、公司信息（如"查询公司的市盈率"、"获取财务报告"）
  - 查询新闻、公告（如"获取公司新闻"、"查询最新公告"）
  - 获取市场数据、行业数据等（如"查询行业数据"、"获取市场行情"）
  - **重要**：如果任务描述中包含"查询"、"获取"、"搜索"等动词，且目标是金融数据，通常需要工具
  
- **不需要工具（needs_tool=false）**：任务只需要基于已有数据进行处理，如：
  - 比较、计算、分析已有数据（如"比较两个股价"、"计算平均值"）
  - 排序、筛选、统计已有数据（如"按价格排序"、"筛选前5名"）
  - 基于前一级任务的结果进行推理、判断（如"根据股价确定更高的公司"）
  - 生成报告、总结等（如"生成分析报告"、"总结数据"）
  - **重要**：如果任务描述中包含"根据"、"基于"、"比较"、"计算"等动词，且操作对象是已有数据，通常不需要工具

【依赖关系识别】
仔细分析每个任务，判断它是否需要其他任务的结果：
- 如果任务A需要任务B的数据或结果，则任务A依赖于任务B，任务A的级别应该比任务B高
- 如果任务可以独立执行，不依赖任何其他任务，则属于第一级任务
- 如果任务需要等待其他任务完成，则属于相应的后续级别
- 根据依赖关系的深度，可以有多级任务（不限于三级）

【输出要求】
你必须严格按照以下 JSON 格式输出，不得添加任何解释性文字：
{format_instructions}

示例开始 ---

用户查询: 我想分析可转债'113527'过去三个月的价值分析数据并获取同期比特币每日收盘价

输出:
{{
  "tasks_by_level": {{
    "1": [
      {{"task": "分析可转债'113527'过去三个月的价值分析数据", "needs_tool": true}},
      {{"task": "获取过去三个月比特币每日收盘价", "needs_tool": true}}
    ]
  }},
  "max_level": 1,
  "dependencies": {{}}
}}

说明：这两个任务都需要获取金融数据，所以needs_tool都是true。

---
用户查询: 计算食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的速动比率、资产负债率和每股收益，并按综合评级给出前 5 名。

输出:
{{
  "tasks_by_level": {{
    "1": [
      {{"task": "查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的速动比率", "needs_tool": true}},
      {{"task": "查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的资产负债率", "needs_tool": true}},
      {{"task": "查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的每股收益", "needs_tool": true}}
    ],
    "2": [
      {{"task": "根据速动比率、资产负债率和每股收益数据，按综合评级给出前 5 名", "needs_tool": false}}
    ]
  }},
  "max_level": 2,
  "dependencies": {{
    "根据速动比率、资产负债率和每股收益数据，按综合评级给出前 5 名": [1, 2, 3]
  }}
}}

说明：前三个任务需要获取财务数据，needs_tool为true。第四个任务只需要基于已有数据进行分析排序，needs_tool为false。

---
用户查询: 查询特斯拉和苹果股价, 为我介绍股价更高的公司的最近新闻

输出:
{{
  "tasks_by_level": {{
    "1": [
      {{"task": "查询特斯拉的股价", "needs_tool": true}},
      {{"task": "查询苹果的股价", "needs_tool": true}}
    ],
    "2": [
      {{"task": "比较特斯拉和苹果的股价，确定股价更高的公司", "needs_tool": false}}
    ],
    "3": [
      {{"task": "为我介绍股价更高的那家公司的最近新闻", "needs_tool": true}}
    ]
  }},
  "max_level": 3,
  "dependencies": {{
    "比较特斯拉和苹果的股价，确定股价更高的公司": [1, 2],
    "为我介绍股价更高的那家公司的最近新闻": [3]
  }}
}}

说明：第一级任务需要获取股价数据，needs_tool为true。第二级任务只需要比较已有数据，needs_tool为false。第三级任务需要获取新闻数据，needs_tool为true。

---
示例结束 ---

注意：
1. tasks_by_level 的 key 必须是字符串格式的数字（"1", "2", "3"等）
2. 每个任务必须包含 task（任务描述）和 needs_tool（是否需要工具）两个字段
3. 仔细判断每个任务是否需要工具，不要误判
"""),
    ("human", "用户查询: {init_query}\n\n请分析并输出层级任务结构（包含每个任务是否需要工具的判断）：")
]).partial(current_time=str(datetime.now()),
    format_instructions=hierarchical_task_parser.get_format_instructions())

data_reuse_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个结合对话历史尝试金融数据复用的数据助手。输入：一个查询列表
     你的任务是：根据智能体与用户的对话历史中包含的金融数据，分析出当前问题列表中是否有查询所需数据在对话历史中已被得到。如果有这样可以使用对话历史解答的查询，则将它剔除查询列表，并将该查询对应的数据附在列表之后。如果无相关数据，则必须保留原查询列表不变，且禁止无关的任何输入。
     **严格禁止事项：**
     *   **严禁**生成、编造、推断任何新的数据或信息。
     *   **严禁**对查询进行回答、分析或总结。你的角色仅限于数据识别和复用。
     *   **严禁**添加任何额外的解释性文字、前缀、后缀或无关内容。输出必须严格遵循示例格式。

     示例开始 ---

        **查询列表: **"1.分析可转债‘113527’过去三个月的价值分析数据。\n
                2.获取过去三个月比特币每日收盘价。\n"

        **上下文中存在：**"以下是过去三个月比特币（BTC）的每日收盘价数据：

                    - 2025-11-16: 95374.01 USD
                    - 2025-11-15: 95544.94 USD
                    - 2025-11-14: 94503.96 USD
                    - 2025-11-13: 99614.54 USD
                    - 2025-11-12: 101639.01 USD" 

        **你的输出：**"1.分析可转债‘113527’过去三个月的价值分析数据。\n
                 以下是过去三个月比特币（BTC）的每日收盘价数据：

                    - 2025-11-16: 95374.01 USD
                    - 2025-11-15: 95544.94 USD
                    - 2025-11-14: 94503.96 USD
                    - 2025-11-13: 99614.54 USD
                    - 2025-11-12: 101639.01 USD \n"

        ---
        ---
        **查询列表: **"1.查询公司A的最新股价。\n
                2.查询公司B的最新股价。\n
                3.比较公司A和公司B的最新股价，确定股价更高的公司。\n
                4.输出股价更高公司的简介。\n"

        **上下文中存在：**"以下是公司A的最新股价：404.35 美元。公司B的最新股价为 272.41 美元。" 

        **你的输出：**"3.比较公司A和公司B的最新股价，确定股价更高的公司。\n
                4.输出股价更高公司的简介。\n
                以下是公司A的最新股价：404.35 美元。公司B的最新股价为 272.41 美元。"

        ---
        ---
        **查询列表: **"1.查询公司A的最新股价。\n"

        **上下文中存在：**"以下是公司A的最新股价：404.35 美元。公司B的最新股价为 272.41 美元。" 

        **你的输出：**"以下是公司A的最新股价：404.35 美元。公司B的最新股价为 272.41 美元。"
        ---
        ---
        **查询列表: **"1.分析可转债‘113527’过去三个月的价值分析数据。\n
                2.获取过去三个月比特币每日收盘价。\n"

        **上下文中无相关数据。**

        **你的输出：**"1.分析可转债‘113527’过去三个月的价值分析数据。\n
                2.获取过去三个月比特币每日收盘价。\n"

        示例结束 ---
        
        查询列表：
        
        你的输出：
     """),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}")
])

data_reuse_chain = data_reuse_prompt | model_reuse | StrOutputParser()

# 简单架构的查询拆分prompt
simple_split_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        '''请判断下面这个查询查询几项内容，并且只将所查询的内容按照'1. XXXX\n 2. XXXX\n'的格式列出，尽可能的让每个单独的查询内容都拥有准确的时间空间信息，注意我们只保留对金融部分的查询。
        如果用户输入不是一个完整的自然语言查询，请你结合历史对话内容 chat_history，尝试补全用户的真实意图生成一个完整查询，然后再进行分割查询。
        
        请不要直接进行回答，你的任务只是分割查询。

        示例开始 ---
        用户查询: 我想分析可转债'113527'过去三个月的价值分析数据并获取同期比特币每日收盘价

        分割查询: 1. 分析可转债'113527'过去三个月的价值分析数据。\n 2. 获取过去三个月比特币每日收盘价。\n
        ---
        用户查询: 计算食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的速动比率、资产负债率和每股收益，并按综合评级给出前 5 名。

        分割查询: 1. 查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的速动比率。\n 2. 查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的资产负债率。\n 3. 查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的每股收益。\n
        ---
        用户查询: 获取全市场未来 3 个月（2025-07-22 至 2025-10-22）的财报日历，筛选出同时有 ≥5 次机构调研的公司并排名。

        分割查询: 1. 获取全市场未来 3 个月（2025-07-22 至 2025-10-22）的财报日历。\n
        ---
        示例结束 ---

        用户查询: {init_query}
        
        请直接输出分割查询的内容，不要输出其他解释说明
        ''',
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}"),
])

simple_split_chain = simple_split_prompt | model_split | StrOutputParser()

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
            # 从消息对象中获取 role 或 type 属性，默认为 'user'
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

# 简单架构的prompt
simple_agent_prompt = ChatPromptTemplate.from_messages([
   (
       "system",
       f"""你是一个专业的金融助手，当前时间为 {now}。\n你的任务是帮助用户获取与金融相关的数据、解释、分析或回答问题。\n你可以调用一组 API 工具来获取股票、债券、公司财务、宏观经济等数据。\n请按照以下规则操作：\n
       1. 你应当优先调用 API 工具来获取实时或历史数据，而不是依赖自己的知识编造数据。 只有当问题属于概念解释、定义说明、通用知识时，才直接回答，不使用工具。\n
       2. 如果用户的问题是自然语言查询，例如： - "什么是市盈率？" - "你是谁？" - "请解释一下ETF的分类。" 这类问题不需要调用 API 工具，请直接用自己的知识回答。\n
       3. 当用户没有说明查询时间范围时，请默认查找最近的数据；如果最近无数据，可以适当往历史数据中查找。\n你的回答应简洁清晰，使用中文输出。
        【重要规则】：答案内容必须保持客观中立，绝对禁止包含任何形式的投资建议。
        【强制行动】：你【必须】在答案末尾另起一行，追加免责声明。
       """,
   ),
   ("placeholder", "{chat_history}"),  # 聊天历史的占位符
   ("human", "{input}"),             # 用户输入的占位符
   ("placeholder", "{agent_scratchpad}"),  # Agent 思考过程的占位符
])

async def simple_expand_query(single_split_query: str, memory) -> ExpandedQueryOutput:
    """简单架构的查询扩展（异步）"""
    return await rag_llm_chain.ainvoke({
        "init_query": single_split_query, 
        "chat_history": memory.buffer
    })

async def simple_retrieve(expanded_output: ExpandedQueryOutput, mode: str, topk: int):
    """简单架构的检索（异步）"""
    query_index = expanded_output.market_category
    search_query = expanded_output.expanded_query
    
    if mode == "关键词模式":
        docs = await faiss_key_databases[query_index].asimilarity_search(search_query, k=topk)
    else:
        docs = await faiss_databases[query_index].asimilarity_search(search_query, k=topk)
    return query_index, docs

def split_numbered_items_with_data(s: str) -> Tuple[List[str], str]:
    """
    解析带编号的查询列表并提取尾部数据块。

    参数:
        s: 原始字符串，可能含有多行，前面是编号项（例如 "1. ..." / "2. ..."），编号项后可能跟一个或多个未编号的“附加数据块”。

    返回:
        (items, tail)
        - items: List[str]，按编号顺序提取的每项文本（去除编号前缀与前后空白，若项两端被 ** 包裹则去掉 **）。
        - tail: str，编号列表之后剩余的文本（去两端空白）。如果没有尾部则返回空字符串。

    行为说明（启发式）:
        - 识别以行首出现的 "数字. " 模式作为编号开始（支持多位数字）。
        - 对最后一项，若其内容内部出现两个连续换行或出现以 "- "、"•" 等列举符或类似 "YYYY-MM-DD:" 的日期行，则把此处之后的文本视为 tail。
        - 若没有识别到编号项，则返回 ([], s.strip())。
    """
    if not s or not s.strip():
        return [], ""

    # 正则匹配行首的编号 "数字. "
    pattern = re.compile(r'(?m)^\s*\d+\.\s*')
    matches = list(pattern.finditer(s))

    if not matches:
        return [], s.strip()

    items: List[str] = []
    tail: str = ""

    for idx, m in enumerate(matches):
        content_start = m.end()
        # 如果后面还有编号项，则该编号项内容到下一个编号开始前结束
        if idx + 1 < len(matches):
            next_start = matches[idx + 1].start()
            content = s[content_start:next_start].strip()
            # 去除包裹的 ** ... **
            if content.startswith("**") and content.endswith("**"):
                content = content[2:-2].strip()
            items.append(content)
        else:
            # 最后一项：需要将可能的尾部分离出来
            last_seg = s[content_start:].rstrip()

            # 找到第一个明显表示“尾部开始”的位置：
            # - 两个及以上连续换行被视为分段标记
            # - 或者换行后紧接着以列表符号开头（如 "-"、"•"、"*"等）
            # - 或者换行后紧接着形如 "YYYY-MM-DD:" 的日期行
            tail_split = re.search(r'(\r?\n\s*\r?\n)|(\r?\n\s*[-•\u2022\*]\s+)|(\r?\n\s*\d{4}-\d{2}-\d{2}\s*:)', last_seg)

            if tail_split:
                split_idx = tail_split.start()
                item_part = last_seg[:split_idx].strip()
                tail = last_seg[split_idx:].strip()
            else:
                # 无明显尾部，全部作为最后一项
                item_part = last_seg.strip()
                tail = ""

            if item_part.startswith("**") and item_part.endswith("**"):
                item_part = item_part[2:-2].strip()
            items.append(item_part)

    return items, tail

def format_intermediate_steps(intermediate_steps: list, thread_info: dict = None) -> str:
    """
    将 AgentExecutor 返回的原始 intermediate_steps 格式化为
    一个对 LLM 更友好的、简洁的字符串。
    """
    if not intermediate_steps:
        return "执行者 Agent 未调用任何工具。"

    log_parts = []
    for i, (_, observation) in enumerate(intermediate_steps):
        log_parts.append(
            f"步骤 {i+1}: \n"
            f"```\n{observation}\n```"
        )
        
    return "\n---\n".join(log_parts)

# 结果有效性检查
class ResultValidityOutput(BaseModel):
    """定义了结果有效性检查的输出结构"""
    is_valid: bool = Field(description="判断工具返回的结果是否有效且值得处理。如果结果包含有意义的金融数据或信息，则为True；如果结果为空、错误、无意义或重复，则为False。")
    reason: str = Field(description="简要说明判断结果是否有效的原因。")
    confidence: float = Field(description="判断的置信度，范围0-1，1表示非常确定。")

result_validity_parser = PydanticOutputParser(pydantic_object=ResultValidityOutput)

# 结果有效性检查prompt
result_validity_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个工具结果质量评估专家。你的任务是评估工具返回的结果是否有效且值得进一步处理。

【评估标准】
**有效结果（is_valid=true）的特征：**
- 包含有意义的金融数据（股价、财务指标、公司信息等）
- 返回了具体的数值、日期、名称等信息
- 结果与用户查询相关且有用
- 虽然有部分错误但整体包含有价值信息

**无效结果（is_valid=false）的特征：**
- 完全为空或null
- 明确的错误信息（如"API调用失败"、"数据不存在"等）
- 无意义的重复字符或乱码
- 与查询完全不相关的信息
- 只有格式化信息没有实际数据

【重要说明】
- 即使结果不完美，只要包含有用信息就应标记为有效
- 对于边界情况（部分有效），优先标记为有效，让下游处理
- confidence应该反映你的判断确定性

你必须严格按照以下 JSON 格式输出，不得添加任何解释性文字：
{format_instructions}
"""),
    ("human", """【用户查询】{query}

【工具名称】{tool_name}

【工具返回结果】
{tool_result}

请评估这个工具返回结果的有效性：""")
]).partial(format_instructions=result_validity_parser.get_format_instructions())

# 创建专门用于结果有效性检查的小LLM
validity_check_llm = FixedChatTongyi(
    model="qwen-turbo",  # 使用更轻量级的模型进行快速检查
    model_kwargs={"seed": 999},
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

result_validity_chain = result_validity_prompt | validity_check_llm | result_validity_parser

async def check_result_validity(query: str, tool_name: str, tool_result: str) -> ResultValidityOutput:
    """检查工具结果的有效性（异步版本）"""
    try:
        result = await result_validity_chain.ainvoke({
            "query": query,
            "tool_name": tool_name,
            "tool_result": tool_result
        })
        return result
    except Exception as e:
        # 如果检查失败，默认认为结果有效
        print(f"⚠️ 结果有效性检查失败，默认标记为有效: {e}")
        return ResultValidityOutput(
            is_valid=True,
            reason=f"有效性检查失败，默认保留: {str(e)[:100]}",
            confidence=0.5
        )

def check_result_validity_sync(query: str, tool_name: str, tool_result: str) -> ResultValidityOutput:
    """检查工具结果的有效性（同步版本，用于多线程环境）"""
    try:
        # 在同步环境中使用 invoke 而不是 ainvoke
        result = result_validity_chain.invoke({
            "query": query,
            "tool_name": tool_name,
            "tool_result": tool_result
        })
        return result
    except Exception as e:
        # 如果检查失败，默认认为结果有效
        print(f"⚠️ 结果有效性检查失败，默认标记为有效: {e}")
        return ResultValidityOutput(
            is_valid=True,
            reason=f"有效性检查失败，默认保留: {str(e)[:100]}",
            confidence=0.5
        )

# 构建加工链
rag_llm_chain = None
chat_chain = None
history_query_chain = None
rag_llm_chain_split = None
synthesizer_chain = None

def rebuild_chains():
    """重新构建所有链（当模型切换时调用）"""
    global rag_llm_chain, chat_chain, history_query_chain, rag_llm_chain_split, synthesizer_chain
    rag_llm_chain = rag_prompt | model | expanded_query_parser
    chat_chain = chat_prompt | model | StrOutputParser()
    history_query_chain = history_prompt | model | intent_parser
    rag_llm_chain_split = rag_prompt_split | model_split | hierarchical_task_parser
    synthesizer_chain = manager_prompt | manage_llm | synthesizer_parser

# 实例化记忆模块
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

worker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一个严谨、高效且绝对服从指令的金融数据采集员，严格遵循程序指令。

**当前时间为 {current_time}。**

【重要说明：单任务执行模式】
你当前正在单任务执行架构中工作。系统会为每个子任务创建一个独立的执行者Agent（你），你可以使用分配给该任务的所有工具。你绝不能直接回答用户，也不能进行任何计算或推理，只能调用工具获取原始数据。

你的唯一任务是根据用户查询（子任务），使用可用的工具采集相关的"原始数据点"。你可以根据需要使用一个或多个工具来完成该任务。如果所有工具都不适用，则直接结束，不调用工具。
你的行动【必须】遵循以下流程：
============================================================
【执行流程】
---
**【步骤 1：任务分析和工具选择】**
*   **任务**: 仔细分析用户查询（子任务）的需求，评估可用的工具，选择最适合的工具来完成该任务。
*   **判断标准**:
    *   工具的功能是否与查询需求相关？
    *   工具能否提供查询所需的数据或信息？
    *   工具的参数要求是否与查询内容匹配？
    *   是否需要使用多个工具来完整回答查询？
*   **决策**:
    *   **如果有适用的工具**: 进入【步骤 2】
    *   **如果所有工具都不适用**: 直接进入【步骤 3】，不调用工具，返回空结果

---
**【步骤 2：调用工具】**
*   **前提条件**: 已在【步骤 1】中确认有适用的工具
*   **行动**: 
    *   根据查询需求，选择合适的工具（可以是一个或多个）进行调用
    *   如果单个工具无法完整回答查询，可以依次调用多个工具
    *   每次调用工具后，审视结果，判断是否需要继续调用其他工具
*   **审视结果**:
    *   如果已获取足够的数据来回答查询，停止调用工具，将工具调用结果返回
    *   如果工具调用失败或返回错误，分析错误原因，可以尝试其他工具或返回错误信息
    *   如果还需要更多数据，继续调用其他适用的工具

---
**【步骤 3：完成】**
*   **条件**: 
    *   工具调用完成后（无论成功或失败），或
    *   在【步骤 1】中判断所有工具都不适用
*   **行动**: 立即停止，将结果返回。系统会将你的结果与其他并行Agent的结果合并。
*   **注意**: 如果所有工具都不适用，你应该直接结束，不调用工具，返回空的中间步骤。

============================================================
【思考格式】
这是你向系统展示你工作状态的唯一方式。你的每一次思考（Thought）都【必须】遵循此格式。
子任务: [你当前处理的子任务内容]
可用工具: [列出所有可用的工具名称]
任务分析: [详细分析该子任务需要什么数据，哪些工具可能适用]
工具选择: [说明你选择使用哪些工具，以及选择的理由]
决策: [明确声明下一步的具体行动：
      - 如果有适用工具：'将调用工具XXX，参数为...'
      - 如果所有工具都不适用：'所有工具都不适用，直接结束，不调用工具']
"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(current_time=str(datetime.now()))

manager_prompt = ChatPromptTemplate.from_messages([
    ("system",
          """你是一位顶级的金融分析师和 AI 数据整合专家, 当前时间为 {current_time}。你的职责是接收用户的【原始问题】和多个并行执行者Agent采集的【原始数据】以及前文的【可复用数据】，完成最终的分析、整合与输出。
============================================================
【输入数据说明】
============================================================
【单任务并行执行模式说明】
系统会为每个子任务创建一个独立的执行者Agent，每个Agent负责完成一个子任务，可以使用一个或多个工具。你接收到的中间数据是多个Agent并行执行后合并的结果，每个Agent完成了一个子任务（可能调用了多个工具）。

formatted_intermediate_steps格式示例：
步骤 1: 

（工具返回的原始字符串）

步骤 2: 

（工具返回的原始字符串）

注意：每个步骤来自不同的Agent，每个Agent完成了一个子任务（可能调用了多个工具）。

============================================================

你必须严格按照以下流程工作：

============================================================
【第一部分：分析与计算】
1.  **审视原料**: 检查所有执行者Agent提供的【原始数据】。注意这些数据来自不同的工具Agent，每个Agent只负责一个工具。
2.  **数据整合**: 将来自不同工具的数据进行整合，去除重复，识别关联关系，理解数据之间的逻辑联系。
3.  **执行计算**: 如果问题需要，基于原始数据进行计算。
    *   例如：若需计算市盈率，公式为 市盈率 = 股价 / 每股收益。
4.  **生成核心答案**: 综合所有工具返回的信息，生成一个专业、清晰、完整、直接回应用户【原始问题】的核心答案，尽可能的保留向你提供的数据，使该回答内容丰富，允许适当包装，但禁止编造虚假数据。

   **答案质量要求：**
   *   **高质量内容**：确保答案准确、专业、有深度，充分体现金融数据的价值。
   *   **清晰表达**：使用简洁明了的语言，避免冗长和模糊的表述，确保用户能够快速理解。
   *   **结构化呈现**：优先采用**分点作答**的方式，使用清晰的层次结构（如使用数字编号、项目符号等）。
   *   **美观格式**：
     - 合理使用换行、分段，避免大段文字堆砌
     - 重要数据、关键信息可以使用适当的分隔或强调
     - 如果涉及多个维度或类别，使用清晰的分组和标题
     - 确保答案整体布局美观、易读
   *   **完整性**：确保答案全面回答用户的问题，不遗漏关键信息。

============================================================
【第二部分：合规性审查 (最终步骤)】
这是你输出前的最后一道关卡。
1.  审查你在【第一部分】生成的核心答案。
2.  **【强制规则】**: 答案内容必须保持客观中立，绝对禁止包含任何形式的投资建议。
3.  **【强制行动】**: **重要**你【必须】在答案末尾另起一行，追加免责声明！
4.  将经过审查和追加了免责声明的答案，作为最终的 `final_answer`。

============================================================
【输出格式】
**重要：你只能输出 JSON 格式的内容，不能输出任何其他文字、解释、中间步骤或格式化内容。**

你必须严格按照以下 JSON 格式输出，不得添加任何解释性文字、代码块标记或其他内容：
{format_instructions}

**输出示例：**
{{
  "final_answer": "根据查询到的数据，...",
  "reasoning": "通过比较...得出结论。"
}}


**重要提醒：**
- 只输出 JSON 对象本身，不要包含代码块标记（```json 或 ```）
- 必须追加免责声明！！！
"""),
    ("human", "【原问题】:\n{query}\n\n【中间数据（来自多个单工具Agent）】:\n{formatted_intermediate_steps}\n【已有可用数据】:\n{data}")
]).partial(current_time=str(datetime.now()),format_instructions=synthesizer_parser.get_format_instructions())

# 初始化链
rebuild_chains()

# LangGraph 状态定义
class GraphState(TypedDict, total=False):
    """LangGraph 状态定义"""
    query: str  # 原始查询
    processed_query: str  # 处理后的查询
    is_tool_needed: bool  # 是否需要工具
    hierarchical_tasks: HierarchicalTaskOutput  # 层级任务结构
    task_results: Dict[str, Any]  # 任务执行结果，key为任务描述，value为结果
    all_intermediate_steps: List[Tuple]  # 所有中间步骤
    formatted_steps: str  # 格式化后的中间步骤
    final_answer: str  # 最终答案
    data: str  # 从历史中复用的数据
    current_level: int  # 当前执行的任务级别
    completed_levels: List[int]  # 已完成的任务级别
    _task_tools_map: Dict[str, List[str]]  # 临时字段：任务到工具的映射
    _tasks_without_tools: List[str]  # 临时字段：不需要工具的任务列表
    _has_next_level: bool  # 临时字段：是否还有下一级别需要执行
    task_reused_data_map: Dict[str, str]  # 任务到复用数据的映射
    validity_check_stats: Dict[str, int]  # 有效性检查统计信息
    retrieval_mode: str  # 检索模式："关键词模式" 或 "全文档模式"
    topk: int  # Top-K 检索数

async def expand_query(single_split_query: str, memory) -> ExpandedQueryOutput:
    """扩展查询并返回一个结构化对象"""
    return await rag_llm_chain.ainvoke({
        "init_query": single_split_query, 
        "chat_history": memory.buffer
    })

async def retrieve(expanded_output: ExpandedQueryOutput, mode: str, topk: int):
    """根据结构化的扩展查询结果进行检索"""
    query_index = expanded_output.market_category
    search_query = expanded_output.expanded_query # 用于检索的文本
    
    if mode == "关键词模式":
        docs = await faiss_key_databases[query_index].asimilarity_search(search_query, k=topk)
    else:
        docs = await faiss_databases[query_index].asimilarity_search(search_query, k=topk)
    return query_index, docs

# LangGraph 节点函数
def intent_judge_node(state: GraphState) -> GraphState:
    """意图判断节点：判断是否需要工具"""
    intent_result = history_query_chain.invoke({
        "init_query": state["query"], 
        "chat_history": memory.buffer
    })
    state["is_tool_needed"] = intent_result.is_tool_needed
    state["processed_query"] = intent_result.query
    print(f"结合上下文推理后的查询：{state['processed_query']}")
    print(f"是否需要调用工具：{state['is_tool_needed']}")
    return state

def chat_node(state: GraphState) -> GraphState:
    """闲聊节点：直接回答，不需要工具"""
    direct_llm_response = chat_chain.invoke({
        "input": state["processed_query"],
        "chat_history": memory.buffer_as_messages 
    })
    state["final_answer"] = direct_llm_response
    print("\n[直接回答]:", direct_llm_response)
    memory.save_context({"input": state["query"]}, {"output": direct_llm_response})
    return state

def task_split_node(state: GraphState) -> GraphState:
    """任务分割节点：将查询分解为层级任务，并进行数据复用检查"""
    hierarchical_tasks = rag_llm_chain_split.invoke({"init_query": state["processed_query"]})
    
    # 先打印LLM的原始判断结果（数据复用检查之前）
    print(f"\n📋 LLM原始任务层级结构 (共 {hierarchical_tasks.max_level} 级):")
    for level in range(1, hierarchical_tasks.max_level + 1):
        level_key = str(level)
        if level_key in hierarchical_tasks.tasks_by_level:
            task_infos = hierarchical_tasks.tasks_by_level[level_key]
            task_descriptions = [f"{task_info.task} (LLM判断需要工具: {task_info.needs_tool})" for task_info in task_infos]
            print(f"  第{level}级任务 ({len(task_infos)}个):")
            for task_desc in task_descriptions:
                print(f"    - {task_desc}")
    
    # 数据复用检查 - 收集所有级别的任务（提取任务描述）
    all_tasks = []
    task_to_level_map = {}  # 记录任务到级别的映射，用于后续更新
    for level in range(1, hierarchical_tasks.max_level + 1):
        level_key = str(level)
        if level_key in hierarchical_tasks.tasks_by_level:
            # 提取任务描述
            task_infos = hierarchical_tasks.tasks_by_level[level_key]
            for task_info in task_infos:
                all_tasks.append(task_info.task)
                task_to_level_map[task_info.task] = level_key
    
    # 存储任务到复用数据的映射
    task_reused_data_map = {}  # {task: reused_data}
    
    if all_tasks:
        tasks_text = "\n".join([f"{i+1}.{task}" for i, task in enumerate(all_tasks)])

        print("处理前的任务：", tasks_text)
        # print("历史：", memory.buffer)

        dealed_tasks = data_reuse_chain.invoke({
            "init_query": tasks_text,
            "chat_history": memory.buffer
        })

        print("处理后的任务：", dealed_tasks)
        
        # 解析处理后的任务，提取复用的数据
        remaining_tasks, reused_data = split_numbered_items_with_data(dealed_tasks)

        # print("剩余任务：", remaining_tasks)
        # print("复用数据：", reused_data)

        
        # 识别哪些任务被移除了（可以复用数据）
        original_tasks_set = set(all_tasks)
        remaining_tasks_set = set(remaining_tasks)
        reused_tasks = original_tasks_set - remaining_tasks_set
        
        # 打印数据复用检查结果
        print(f"\n♻️ 数据复用检查:")
        print(f"  - 原始任务数: {len(all_tasks)}")
        print(f"  - 剩余任务数: {len(remaining_tasks)}")
        print(f"  - 可复用任务数: {len(reused_tasks)}")
        
        if reused_tasks:
            print(f"  可复用数据的任务:")
            for task in reused_tasks:
                print(f"    - {task[:60]}...")
        else:
            print(f"  没有任务可以从历史中复用数据")
        
        # 验证：只有当有复用数据且确实有任务被移除时，才进行数据关联
        # 如果所有任务都被移除但没有数据，可能是 data_reuse_chain 输出格式有问题
        if len(reused_tasks) == len(all_tasks) and not reused_data:
            print(f"  ⚠️ 警告: 所有任务都被标记为可复用，但没有提供复用数据，可能是 data_reuse_chain 输出格式异常")
            # 这种情况下，不进行数据复用，恢复所有任务
            reused_tasks = set()
        
        # 如果有复用的数据，需要将数据与对应的任务关联
        if reused_data and reused_tasks:
            # 根据 data_reuse_chain 的输出格式，复用的数据通常对应最后一个被移除的任务
            # 或者如果有多个任务可以复用，数据可能合并在一起
            # 这里简化处理：将复用数据与所有复用任务关联
            # 在实际使用中，可以根据数据内容进行更精确的匹配
            for task in reused_tasks:
                task_reused_data_map[task] = reused_data
            
            print(f"📦 复用的数据（共 {len(reused_tasks)} 个任务）: {reused_data[:200]}..." if len(reused_data) > 200 else f"📦 复用的数据: {reused_data}")
        
        # 更新层级任务结构：将可以复用数据的任务标记为不需要工具
        # 只更新那些原本需要工具但现在可以复用数据的任务
        for level in range(1, hierarchical_tasks.max_level + 1):
            level_key = str(level)
            if level_key in hierarchical_tasks.tasks_by_level:
                task_infos = hierarchical_tasks.tasks_by_level[level_key]
                for task_info in task_infos:
                    if task_info.task in reused_tasks:
                        # 任务可以复用数据，标记为不需要工具
                        original_needs_tool = task_info.needs_tool
                        task_info.needs_tool = False
                        if original_needs_tool:
                            print(f"  ✅ 任务 '{task_info.task[:50]}...' 已从需要工具改为不需要工具（数据可复用）")
                        else:
                            print(f"  ℹ️ 任务 '{task_info.task[:50]}...' 原本就不需要工具，已关联复用数据")
        
        # 存储全局复用数据
        state["data"] = reused_data
    else:
        state["data"] = ""
    
    # 存储任务到复用数据的映射
    state["task_reused_data_map"] = task_reused_data_map
    
    # 初始化状态
    state["hierarchical_tasks"] = hierarchical_tasks
    state["task_results"] = {}
    state["current_level"] = 1
    state["completed_levels"] = []
    state["all_intermediate_steps"] = []
    state["validity_check_stats"] = {"total_checks": 0, "valid_results": 0, "filtered_results": 0}
    
    # 打印最终的任务层级结构
    print(f"\n📋 最终任务层级结构 (共 {hierarchical_tasks.max_level} 级，已考虑数据复用):")
    for level in range(1, hierarchical_tasks.max_level + 1):
        level_key = str(level)
        if level_key in hierarchical_tasks.tasks_by_level:
            task_infos = hierarchical_tasks.tasks_by_level[level_key]
            task_descriptions = []
            for task_info in task_infos:
                has_reused_data = task_info.task in task_reused_data_map
                if has_reused_data:
                    task_descriptions.append(f"{task_info.task} (需要工具: {task_info.needs_tool}, 有复用数据: 是)")
                else:
                    task_descriptions.append(f"{task_info.task} (需要工具: {task_info.needs_tool})")
            print(f"  第{level}级任务 ({len(task_infos)}个):")
            for task_desc in task_descriptions:
                print(f"    - {task_desc}")
    
    return state

def expand_and_retrieve_node(state: GraphState) -> GraphState:
    """扩展查询和RAG检索节点：为当前级别需要工具的任务进行扩展和检索"""
    hierarchical_tasks = state["hierarchical_tasks"]
    current_level = state["current_level"]
    
    # 获取当前级别的任务信息
    level_key = str(current_level)
    task_infos = hierarchical_tasks.tasks_by_level.get(level_key, [])
    
    if not task_infos:
        print(f"⚠️ [级别 {current_level}] 没有任务需要处理")
        state["_task_tools_map"] = {}
        state["_tasks_without_tools"] = []
        return state
    
    # 根据LLM的判断，分离需要工具和不需要工具的任务
    tasks_need_tools = [task_info.task for task_info in task_infos if task_info.needs_tool]
    tasks_without_tools = [task_info.task for task_info in task_infos if not task_info.needs_tool]
    
    # 打印LLM的判断结果
    print(f"\n🤖 [级别 {current_level}] LLM任务工具需求判断:")
    for idx, task_info in enumerate(task_infos, 1):
        status = "✅ 需要工具" if task_info.needs_tool else "ℹ️ 不需要工具"
        print(f"  任务 {idx}: {task_info.task} - {status}")
    
    # 对于不需要工具的任务，直接标记，不进行RAG检索
    task_tools_map = {}  # {task: [tool_names]}
    for task in tasks_without_tools:
        task_tools_map[task] = []
    
    # 只对需要工具的任务进行RAG检索
    if not tasks_need_tools:
        print(f"ℹ️ [级别 {current_level}] 所有任务都不需要工具，跳过RAG检索")
        state["_task_tools_map"] = task_tools_map
        state["_tasks_without_tools"] = tasks_without_tools
        return state
    
    # 从状态中获取 topk 和检索模式，如果没有则使用默认值
    topk = state.get("topk", 5)
    mode = state.get("retrieval_mode", "全文档模式")
    
    # 使用 asyncio.run 执行异步操作
    async def async_expand_and_retrieve():
        # 并发扩展查询
        expanded_outputs = await asyncio.gather(*[
            expand_query(task, memory) for task in tasks_need_tools
        ])
        print(f"📝 [级别 {current_level}] 扩展查询完成，共 {len(expanded_outputs)} 个需要工具的任务")
        
        # 并发检索
        retrieval_results = await asyncio.gather(*[
            retrieve(eo, mode, topk) for eo in expanded_outputs
        ])
        
        # 为每个需要工具的任务提取工具
        for idx, (task, (query_index, docs)) in enumerate(zip(tasks_need_tools, retrieval_results)):
            print(f"🔍 [级别 {current_level}, 任务 {idx+1}] 市场分类: {query_index}")
            
            tool_names = []
            for doc in docs:
                tool_meta_name = doc.metadata.get("name")
                if tool_meta_name and tool_meta_name in API_TOOL_dic:
                    tool_name = API_TOOL_dic[tool_meta_name]
                    tool_names.append(tool_name)
            
            tool_names = list(set(tool_names))  # 去重
            task_tools_map[task] = tool_names
            
            if tool_names:
                print(f"✅ [级别 {current_level}, 任务 {idx+1}] 检索到的工具: {tool_names}")
            else:
                print(f"⚠️ [级别 {current_level}, 任务 {idx+1}] 未检索到工具，将由LLM直接处理")
                # 如果检索不到工具，也加入不需要工具列表
                tasks_without_tools.append(task)
                task_tools_map[task] = []
        
        return task_tools_map, tasks_without_tools
    
    # 执行异步操作
    task_tools_map, tasks_without_tools = asyncio.run(async_expand_and_retrieve())
    
    # 将工具映射和不需要工具的任务列表存储到状态中
    state["_task_tools_map"] = task_tools_map
    state["_tasks_without_tools"] = tasks_without_tools
    
    # 统计信息
    tasks_with_tools = [task for task in tasks_need_tools if task in task_tools_map and task_tools_map[task]]
    print(f"\n📊 [级别 {current_level}] 任务工具需求统计:")
    print(f"  - 需要工具的任务: {len(tasks_with_tools)} 个")
    print(f"  - 不需要工具的任务: {len(tasks_without_tools)} 个")
    
    return state

def execute_tasks_node(state: GraphState) -> GraphState:
    """执行任务节点：并行执行当前级别需要工具的任务，对不需要工具的任务用LLM处理"""
    hierarchical_tasks = state["hierarchical_tasks"]
    current_level = state["current_level"]
    task_tools_map = state.get("_task_tools_map", {})
    tasks_without_tools = state.get("_tasks_without_tools", [])
    
    # 获取当前级别的任务
    level_key = str(current_level)
    task_infos = hierarchical_tasks.tasks_by_level.get(level_key, [])
    tasks = [task_info.task for task_info in task_infos]  # 提取任务描述
    
    if not tasks:
        state["completed_levels"].append(current_level)
        return state
    
    # 处理不需要工具的任务- 并行处理
    if tasks_without_tools:
        print(f"\n💬 [级别 {current_level}] 开始并行处理 {len(tasks_without_tools)} 个不需要工具的任务...")
        # 创建任务到索引的映射
        task_to_idx = {task: idx for idx, task in enumerate(tasks)}
        task_reused_data_map = state.get("task_reused_data_map", {})
        
        # 提取之前级别的结果
        previous_results = []
        for item in state["all_intermediate_steps"]:
            if len(item) == 6:
                _, _, _, level, _, observation = item
                if level < current_level:
                    previous_results.append(str(observation))
        
        previous_text = "\n\n【之前任务的结果】\n" + "\n\n".join(previous_results) if previous_results else ""
        if previous_results:
            print(f"  📋 [级别 {current_level}] 已提取之前级别的结果，将作为上下文传递给所有任务")
        
        def process_single_task_without_tool(task: str) -> tuple:
            """处理单个不需要工具的任务"""
            try:
                # 检查任务是否有复用数据
                if task in task_reused_data_map:
                    # 任务有复用数据，直接使用数据
                    reused_data = task_reused_data_map[task]
                    print(f"  ♻️ [级别 {current_level}] 任务 '{task[:50]}...' 使用复用数据")
                    
                    # 将复用数据作为观察结果存储
                    class VirtualAction:
                        def __init__(self, tool_name, tool_input):
                            self.tool = tool_name
                            self.tool_input = tool_input
                    
                    virtual_action = VirtualAction('data_reuse', {'task': task})
                    virtual_observation = reused_data
                    task_idx = task_to_idx.get(task, -1)
                    return ('data_reuse', task_idx, virtual_action, virtual_observation, None)
                else:
                    # 没有复用数据，使用LLM直接处理
                    task_input = task + previous_text if previous_text else task
                    
                    llm_response = chat_chain.invoke({
                        "input": task_input,
                        "chat_history": memory.buffer_as_messages
                    })
                    
                    # 将LLM处理的结果作为观察结果存储
                    class VirtualAction:
                        def __init__(self, tool_name, tool_input):
                            self.tool = tool_name
                            self.tool_input = tool_input
                    
                    virtual_action = VirtualAction('llm_direct_response', {'task': task})
                    virtual_observation = llm_response
                    task_idx = task_to_idx.get(task, -1)
                    print(f"  ✅ [级别 {current_level}] 任务 '{task[:50]}...' 已由LLM处理完成")
                    return ('llm_direct', task_idx, virtual_action, virtual_observation, None)
            except Exception as e:
                print(f"  ❌ [级别 {current_level}] 任务 '{task[:50]}...' 处理失败: {e}")
                return (None, task_to_idx.get(task, -1), None, None, str(e))
        
        # 并行处理所有不需要工具的任务
        from concurrent.futures import ThreadPoolExecutor, as_completed
        begin_time_no_tool = time.time()
        
        with ThreadPoolExecutor(max_workers=min(len(tasks_without_tools), 10)) as executor:
            future_to_task = {
                executor.submit(process_single_task_without_tool, task): task 
                for task in tasks_without_tools
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result_type, task_idx, virtual_action, virtual_observation, error = future.result()
                    if result_type and virtual_action and virtual_observation:
                        # 根据结果类型选择线程ID
                        thread_id = -1 if result_type == 'data_reuse' else 0
                        state["all_intermediate_steps"].append((thread_id, result_type, task_idx, current_level, virtual_action, virtual_observation))
                except Exception as e:
                    print(f"  ❌ [级别 {current_level}] 任务 '{task[:50]}...' 并行处理异常: {e}")
        
        end_time_no_tool = time.time()
        print(f"  ✅ [级别 {current_level}] 所有不需要工具的任务处理完成，总耗时: {end_time_no_tool - begin_time_no_tool:.2f} 秒")
    
    # 为需要工具的任务创建Agent任务
    # 策略：如果总工具数 <= 10，每个工具一个线程；如果 > 10，合并工具但保证线程数 <= 10
    # 每个线程只处理一个子任务
    
    # 首先收集所有需要工具的任务及其工具列表
    task_tool_pairs = []  # [(task_idx, task, tool_names), ...]
    for task_idx, task in enumerate(tasks):
        # 跳过不需要工具的任务
        if task in tasks_without_tools:
            continue
            
        tool_names = task_tools_map.get(task, [])
        if not tool_names:
            # 如果任务在工具映射中但没有工具，也跳过
            print(f"  ⚠️ [级别 {current_level}, 任务 {task_idx+1}] 任务 '{task[:50]}...' 没有可用工具，跳过")
            continue
        
        task_tool_pairs.append((task_idx, task, tool_names))
    
    # 计算总工具数
    total_tools = sum(len(tool_names) for _, _, tool_names in task_tool_pairs)
    max_threads = 10
    
    # 提取之前级别的结果
    previous_results = []
    for item in state["all_intermediate_steps"]:
        if len(item) == 6:
            _, _, _, level, _, observation = item
            if level < current_level:
                previous_results.append(str(observation))
    
    previous_text = "\n\n【之前任务的结果】\n" + "\n\n".join(previous_results) if previous_results else ""
    
    # 根据总工具数决定线程分配策略
    all_agent_tasks = []
    thread_counter = 1
    
    if total_tools <= max_threads:
        # 策略1：总工具数 <= 10，每个工具一个线程
        print(f"📊 [级别 {current_level}] 总工具数 {total_tools} <= {max_threads}，采用策略：每个工具一个线程")
        for task_idx, task, tool_names in task_tool_pairs:
            for tool_name in tool_names:
                all_agent_tasks.append((thread_counter, [tool_name], task, task_idx, current_level, previous_text))
                thread_counter += 1
    else:
        # 策略2：总工具数 > 10，需要合并工具，但保证线程数 <= 10，且每个线程只处理一个子任务
        print(f"📊 [级别 {current_level}] 总工具数 {total_tools} > {max_threads}，采用策略：合并工具，线程数限制为 {max_threads}")
        
        # 计算每个任务应该分配多少线程
        task_thread_allocations = {}  # {task_idx: allocated_threads}
        remaining_threads = max_threads
        
        # 第一轮：按比例分配（向下取整）
        for task_idx, task, tool_names in task_tool_pairs:
            num_tools = len(tool_names)
            allocated = max(1, int(num_tools * max_threads / total_tools))  # 至少1个线程
            task_thread_allocations[task_idx] = allocated
            remaining_threads -= allocated
        
        # 第二轮：将剩余的线程分配给工具数最多的任务
        if remaining_threads > 0:
            # 按工具数排序，优先分配给工具数多的任务
            sorted_tasks = sorted(task_tool_pairs, key=lambda x: len(x[2]), reverse=True)
            for task_idx, task, tool_names in sorted_tasks:
                if remaining_threads <= 0:
                    break
                task_thread_allocations[task_idx] += 1
                remaining_threads -= 1
        
        # 为每个任务分配线程
        for task_idx, task, tool_names in task_tool_pairs:
            num_tools = len(tool_names)
            allocated_threads = task_thread_allocations.get(task_idx, 1)
            
            if num_tools <= allocated_threads:
                # 如果该任务的工具数 <= 分配的线程数，每个工具一个线程
                for tool_idx, tool_name in enumerate(tool_names):
                    if thread_counter > max_threads:
                        # 如果超过限制，将剩余工具合并到该任务的最后一个线程
                        remaining_tools = tool_names[tool_idx:]
                        if remaining_tools:
                            # 找到该任务的最后一个线程
                            for j in range(len(all_agent_tasks) - 1, -1, -1):
                                if all_agent_tasks[j][2] == task:
                                    existing_tools = all_agent_tasks[j][1]
                                    all_agent_tasks[j] = (all_agent_tasks[j][0], existing_tools + remaining_tools, task, task_idx, current_level, previous_text)
                                    break
                        break
                    all_agent_tasks.append((thread_counter, [tool_name], task, task_idx, current_level, previous_text))
                    thread_counter += 1
            else:
                # 如果该任务的工具数 > 分配的线程数，需要将工具分组
                tools_per_thread = (num_tools + allocated_threads - 1) // allocated_threads  # 向上取整
                
                for i in range(0, num_tools, tools_per_thread):
                    if thread_counter > max_threads:
                        # 如果超过限制，将剩余工具合并到该任务的最后一个线程
                        remaining_tools = tool_names[i:]
                        if remaining_tools:
                            # 找到该任务的最后一个线程
                            for j in range(len(all_agent_tasks) - 1, -1, -1):
                                if all_agent_tasks[j][2] == task:
                                    existing_tools = all_agent_tasks[j][1]
                                    all_agent_tasks[j] = (all_agent_tasks[j][0], existing_tools + remaining_tools, task, task_idx, current_level, previous_text)
                                    break
                        break
                    
                    tool_group = tool_names[i:i + tools_per_thread]
                    all_agent_tasks.append((thread_counter, tool_group, task, task_idx, current_level, previous_text))
                    thread_counter += 1
        
        # 最终检查：确保线程数不超过限制
        if len(all_agent_tasks) > max_threads:
            print(f"  ⚠️ [级别 {current_level}] 警告：线程数 {len(all_agent_tasks)} 超过限制 {max_threads}，将进行最终合并")
            # 将最后几个线程的工具合并到前面的线程中（优先合并到同任务的线程）
            excess = len(all_agent_tasks) - max_threads
            for i in range(excess):
                if not all_agent_tasks:
                    break
                last_task_info = all_agent_tasks.pop()
                last_thread_id, last_tools, last_task, last_task_idx, last_level, last_prev = last_task_info
                # 优先找到同一个任务的最后一个线程，合并工具
                merged = False
                for j in range(len(all_agent_tasks) - 1, -1, -1):
                    if all_agent_tasks[j][2] == last_task:  # 同一个任务
                        existing_tools = all_agent_tasks[j][1]
                        all_agent_tasks[j] = (all_agent_tasks[j][0], existing_tools + last_tools, last_task, last_task_idx, last_level, last_prev)
                        merged = True
                        break
                if not merged:
                    # 如果找不到同任务的线程，合并到最后一个线程
                    if all_agent_tasks:
                        last_existing = all_agent_tasks[-1]
                        all_agent_tasks[-1] = (last_existing[0], last_existing[1] + last_tools, last_existing[2], last_existing[3], last_existing[4], last_existing[5])
    
    print(f"  📋 [级别 {current_level}] 最终分配：{len(all_agent_tasks)} 个线程，处理 {len(task_tool_pairs)} 个子任务")
    
    # 执行需要工具的任务
    if not all_agent_tasks:
        print(f"ℹ️ [级别 {current_level}] 没有需要工具执行的任务")
        state["completed_levels"].append(current_level)
        # 清理临时字段
        if "_task_tools_map" in state:
            del state["_task_tools_map"]
        if "_tasks_without_tools" in state:
            del state["_tasks_without_tools"]
        return state
    
    print(f"\n🚀 [级别 {current_level}] 开始并行执行 {len(all_agent_tasks)} 个执行者 Agent（智能线程分配）...")
    begin_time_parallel = time.time()
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def execute_single_task_agent(thread_id: int, tool_names: List[str], task: str, task_idx: int, level: int, previous_text: str = "") -> tuple:
        """执行单个任务Agent，该任务可以使用多个工具，并进行结果有效性检查"""
        # 获取所有工具实例
        tool_instances = [t for t in all_tools if t.name in tool_names]
        
        if not tool_instances:
            print(f"⚠️ [线程 {thread_id}] 任务 '{task[:50]}...' 的工具未找到，跳过")
            return thread_id, task_idx, level, [], 0, 0, 0  # 返回统计信息
        
        print(f"✨ [线程 {thread_id}] 级别 {level} - 任务 {task_idx+1} - 可用工具: {tool_names}")
        
        worker_agent = create_tool_calling_agent(model, tool_instances, worker_prompt)
        worker_executor = AgentExecutor(
            agent=worker_agent,
            tools=tool_instances,
            verbose=False,
            return_intermediate_steps=True,
        )
        
        try:
            task_input = task + previous_text if previous_text else task
            if previous_text:
                print(f"  📋 [线程 {thread_id}] 已添加之前级别的结果作为上下文")
            
            response = worker_executor.invoke({
                "input": task_input, 
                "chat_history": memory.buffer_as_messages
            })
            
            raw_intermediate_steps = response.get("intermediate_steps", [])
            print(f"  📊 [线程 {thread_id}] 原始intermediate_steps数量: {len(raw_intermediate_steps)}")
            
            # 如果有工具调用结果，进行有效性检查
            if raw_intermediate_steps:
                valid_intermediate_steps = []
                total_checks = 0
                valid_count = 0
                filtered_count = 0
                
                for action, observation in raw_intermediate_steps:
                    total_checks += 1
                    # 从action中获取工具名称
                    tool_name = getattr(action, 'tool', 'unknown')
                    print(f"  🔍 [线程 {thread_id}] 开始检查第 {total_checks} 个工具调用结果（工具: {tool_name}）...")
                    # 使用同步方式检查结果有效性，避免在线程中使用异步
                    try:
                        validity_result = check_result_validity_sync(task, tool_name, str(observation))
                        
                        if validity_result.is_valid:
                            valid_intermediate_steps.append((action, observation))
                            valid_count += 1
                            print(f"  ✅ [线程 {thread_id}] 工具结果有效性检查: 通过 (置信度: {validity_result.confidence:.2f})")
                            print(f"      原因: {validity_result.reason}")
                        else:
                            filtered_count += 1
                            print(f"  ❌ [线程 {thread_id}] 工具结果有效性检查: 未通过 (置信度: {validity_result.confidence:.2f})")
                            print(f"      原因: {validity_result.reason}")
                            print(f"      已过滤此结果，避免无效数据传入解析LLM")
                    except Exception as validity_error:
                        print(f"  ⚠️ [线程 {thread_id}] 结果有效性检查失败，默认保留: {validity_error}")
                        valid_intermediate_steps.append((action, observation))
                        valid_count += 1
                
                print(f"  📋 [线程 {thread_id}] 检查完成: 总计{total_checks}个，有效{valid_count}个，过滤{filtered_count}个")
                return thread_id, task_idx, level, valid_intermediate_steps, total_checks, valid_count, filtered_count
            else:
                print(f"  ⚠️ [线程 {thread_id}] 没有工具调用结果")
                return thread_id, task_idx, level, [], 0, 0, 0  # 没有工具调用
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ [线程 {thread_id}] Agent 执行出错: {error_msg[:200]}...")
            return thread_id, task_idx, level, [], 0, 0, 0  # 执行失败
    
    # 线程池大小不超过10
    with ThreadPoolExecutor(max_workers=min(len(all_agent_tasks), 10)) as executor:
        future_to_info = {
            executor.submit(execute_single_task_agent, *task): task 
            for task in all_agent_tasks
        }
        
        results_by_thread = {}
        for future in as_completed(future_to_info):
            thread_id, task_idx, level, intermediate_steps, total_checks, valid_count, filtered_count = future.result()
            results_by_thread[thread_id] = (task_idx, level, intermediate_steps, total_checks, valid_count, filtered_count)
    
    end_time_parallel = time.time()
    print(f"\n✅ [级别 {current_level}] 所有 Agent 执行完成，总耗时: {end_time_parallel - begin_time_parallel:.2f} 秒")
    
    # 收集中间步骤并统计有效性检查结果
    intermediate_steps_with_info = []
    validity_stats = state.get("validity_check_stats", {"total_checks": 0, "valid_results": 0, "filtered_results": 0})
    
    for thread_id in sorted(results_by_thread.keys()):
        task_idx, level, thread_steps, total_checks, valid_count, filtered_count = results_by_thread[thread_id]
        for action, observation in thread_steps:
            # 从action中获取工具名称
            tool_name = getattr(action, 'tool', 'unknown')
            intermediate_steps_with_info.append((thread_id, tool_name, task_idx, level, action, observation))
        
        # 累加每个线程的统计信息
        validity_stats["total_checks"] += total_checks
        validity_stats["valid_results"] += valid_count
        validity_stats["filtered_results"] += filtered_count
    
    # 计算过滤结果数
    validity_stats["filtered_results"] = validity_stats["total_checks"] - validity_stats["valid_results"]
    
    # 更新状态
    state["all_intermediate_steps"].extend(intermediate_steps_with_info)
    state["completed_levels"].append(current_level)
    state["validity_check_stats"] = validity_stats
    
    # 清理临时字段
    if "_task_tools_map" in state:
        del state["_task_tools_map"]
    if "_tasks_without_tools" in state:
        del state["_tasks_without_tools"]
    
    return state

def check_next_level_node(state: GraphState) -> GraphState:
    """检查下一级别节点：判断是否还有下一级别的任务需要执行（支持动态级别）"""
    hierarchical_tasks = state["hierarchical_tasks"]
    current_level = state["current_level"]
    
    # 检查是否有下一级别的任务
    next_level = current_level + 1
    if next_level <= hierarchical_tasks.max_level:
        next_level_key = str(next_level)
        if next_level_key in hierarchical_tasks.tasks_by_level and hierarchical_tasks.tasks_by_level[next_level_key]:
            state["current_level"] = next_level
            state["_has_next_level"] = True  # 标记还有下一级别需要执行
            print(f"\n➡️ 进入第 {next_level} 级任务处理")
            return state
    
    # 没有下一级别，标记所有级别已完成
    state["_has_next_level"] = False
    print(f"\n✅ 所有级别任务已完成，进入结果汇总阶段")
    return state

def synthesize_node(state: GraphState) -> GraphState:
    """结果汇总节点：整合所有任务结果并进行合规性过滤"""
    # 格式化中间步骤
    intermediate_steps_result = []
    thread_info_map = {}
    for i, item in enumerate(state["all_intermediate_steps"]):
        if len(item) == 6:  # (thread_id, tool_name, task_idx, level, action, observation)
            thread_id, tool_name, _, _, action, observation = item
            # 过滤掉可复用的数据（data_reuse）和LLM直接处理的结果
            # 可复用的数据已经在state["data"]中，不需要在formatted_steps中重复
            if tool_name != 'data_reuse':
                intermediate_steps_result.append((action, observation))
                thread_info_map[len(intermediate_steps_result) - 1] = thread_id
    
    if not intermediate_steps_result:
        formatted_steps = "执行者 Agent 未调用任何工具。"
        print("⚠️ 未检索到相关API工具或工具执行失败，将尝试由语言模型直接回答...")
    else:
        formatted_steps = format_intermediate_steps(intermediate_steps_result, thread_info=thread_info_map)
    
    state["formatted_steps"] = formatted_steps

    # print(f"  [管理者 Chain] 输入的query: {state['processed_query']}")
    # print(f"  [管理者 Chain] 输入的formatted_steps: {formatted_steps}")
    # print(f"  [管理者 Chain] 输入的data: {state.get('data', '')}")
    
    print("\n🔬 开始执行 [管理者 Chain] 进行数据整合和合规性过滤...")
    
    # 重试机制：最多重试2次
    max_retries = 2
    synthesis_result = None
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"  🔄 第 {attempt + 1} 次尝试解析（共 {max_retries + 1} 次）...")
            
            synthesis_result = synthesizer_chain.invoke({
                "query": state["processed_query"],
                "formatted_intermediate_steps": formatted_steps,
                "data": state.get("data", "")
            })
            
            # 如果成功解析，跳出循环
            state["final_answer"] = synthesis_result.final_answer
            print("\n[最终答案]:")
            print(synthesis_result.final_answer)
            break
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️ 管理者 Chain 执行步骤出错（尝试 {attempt + 1}/{max_retries + 1}）: {error_msg}")
            
            # 如果是最后一次尝试，使用备用方案
            if attempt == max_retries:
                print("  🔧 尝试使用备用方案：直接从 LLM 输出中提取答案...")
                answer_extracted = False  # 标志变量，表示是否成功提取答案
                try:
                    # 备用方案：直接调用 LLM，不使用结构化解析器
                    raw_response = manage_llm.invoke(
                        manager_prompt.format(
                            query=state["processed_query"],
                            formatted_intermediate_steps=formatted_steps,
                            data=state.get("data", ""),
                            format_instructions=synthesizer_parser.get_format_instructions()
                        )
                    )
                    
                    # 尝试从原始输出中提取 JSON
                    raw_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
                    
                    # 方法1：尝试提取完整的 JSON 对象
                    # 查找从第一个 { 开始到匹配的 } 结束的 JSON 对象
                    brace_count = 0
                    json_start = raw_content.find('{')
                    if json_start != -1:
                        json_str = ""
                        for i in range(json_start, len(raw_content)):
                            char = raw_content[i]
                            json_str += char
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # 找到了完整的 JSON 对象
                                    try:
                                        parsed_json = json.loads(json_str)
                                        if "final_answer" in parsed_json:
                                            state["final_answer"] = parsed_json["final_answer"]
                                            print("\n[最终答案]（从备用方案提取）:")
                                            print(state["final_answer"])
                                            answer_extracted = True
                                            break
                                    except json.JSONDecodeError:
                                        pass
                                    break
                    
                    # 方法2：如果方法1失败，尝试直接查找 final_answer 字段的值
                    if not answer_extracted:
                        # 匹配 "final_answer": "..." 或 "final_answer": """..."""
                        final_answer_patterns = [
                            r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"',  # 单行字符串
                            r'"final_answer"\s*:\s*"""((?:[^"]|"(?!""))*?)"""',  # 多行字符串
                        ]
                        
                        for pattern in final_answer_patterns:
                            final_answer_match = re.search(pattern, raw_content, re.DOTALL)
                            if final_answer_match:
                                state["final_answer"] = final_answer_match.group(1)
                                print("\n[最终答案]（从备用方案提取）:")
                                print(state["final_answer"])
                                answer_extracted = True
                                break
                    
                    # 方法3：如果还是无法提取，使用原始内容的前500个字符作为答案
                    if not answer_extracted:
                        state["final_answer"] = raw_content[:500] if len(raw_content) > 500 else raw_content
                        print("\n[最终答案]（使用原始输出）:")
                        print(state["final_answer"])
                        answer_extracted = True
                    
                    # 如果成功提取答案，跳出重试循环
                    if answer_extracted:
                        break
                    
                except Exception as fallback_error:
                    print(f"  ❌ 备用方案也失败: {fallback_error}")
                    state["final_answer"] = "处理过程中发生错误，无法生成最终答案。请重试或检查输入数据。"
            else:
                # 如果不是最后一次，等待一下再重试
                import time
                time.sleep(0.5)
    
    # 保存到记忆
    if state.get("final_answer"):
        memory.save_context({"input": state["query"]}, {"output": state["final_answer"]})
        print("✅ 本轮有效问答已存入记忆。")
    
    # 显示有效性检查统计信息
    validity_stats = state.get("validity_check_stats", {"total_checks": 0, "valid_results": 0, "filtered_results": 0})
    if validity_stats["total_checks"] > 0:
        print(f"\n📊 [结果有效性检查统计]")
        print(f"  - 总检查次数: {validity_stats['total_checks']}")
        print(f"  - 有效结果: {validity_stats['valid_results']}")
        print(f"  - 过滤结果: {validity_stats['filtered_results']}")
        print(f"  - 过滤比例: {validity_stats['filtered_results']/validity_stats['total_checks']*100:.1f}%")
        print(f"  - 有效比例: {validity_stats['valid_results']/validity_stats['total_checks']*100:.1f}%")
        print(f"💡 通过有效性检查，有效避免了 {validity_stats['filtered_results']} 个无效结果传入解析LLM，节省了上下文空间")
    else:
        print(f"\n📊 [结果有效性检查统计] 本次查询未进行工具结果有效性检查")
    
    return state

def should_use_tool(state: GraphState) -> str:
    """路由函数：判断是否需要使用工具"""
    if state["is_tool_needed"]:
        return "task_split"
    else:
        return "chat"

def should_continue_level(state: GraphState) -> str:
    """路由函数：判断是否继续执行下一级别（支持动态级别）"""
    # 检查 check_next_level_node 设置的标志
    has_next_level = state.get("_has_next_level", False)
    
    if has_next_level:
        # 还有下一级别需要执行，继续执行当前级别
        current_level = state["current_level"]
        print(f"🔄 路由决策: 继续执行第 {current_level} 级任务")
        # 清除标志，避免影响后续判断
        if "_has_next_level" in state:
            del state["_has_next_level"]
        return "expand_and_retrieve"
    else:
        # 没有下一级别，进入汇总
        current_level = state["current_level"]
        hierarchical_tasks = state["hierarchical_tasks"]
        print(f"🔄 路由决策: 所有级别已完成（当前级别: {current_level}, 最大级别: {hierarchical_tasks.max_level}），进入汇总阶段")
        # 清除标志
        if "_has_next_level" in state:
            del state["_has_next_level"]
        return "synthesize"

# 构建 LangGraph
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("intent_judge", intent_judge_node)
workflow.add_node("chat", chat_node)
workflow.add_node("task_split", task_split_node)
workflow.add_node("expand_and_retrieve", expand_and_retrieve_node)
workflow.add_node("execute_tasks", execute_tasks_node)
workflow.add_node("check_next_level", check_next_level_node)
workflow.add_node("synthesize", synthesize_node)

# 设置入口点
workflow.set_entry_point("intent_judge")

# 添加条件边
workflow.add_conditional_edges(
    "intent_judge",
    should_use_tool,
    {
        "task_split": "task_split",
        "chat": "chat"
    }
)

# 任务分割后进入扩展和检索
workflow.add_edge("task_split", "expand_and_retrieve")

# 扩展检索后执行任务
workflow.add_edge("expand_and_retrieve", "execute_tasks")

# 执行任务后检查下一级别
workflow.add_edge("execute_tasks", "check_next_level")

# 检查下一级别后决定是继续还是汇总
workflow.add_conditional_edges(
    "check_next_level",
    should_continue_level,
    {
        "expand_and_retrieve": "expand_and_retrieve",
        "synthesize": "synthesize"
    }
)

# 汇总和聊天节点结束
workflow.add_edge("synthesize", END)
workflow.add_edge("chat", END)

# 编译图
app = workflow.compile()

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

# 日志收集器
class LogCollector:
    """收集 print 输出的日志收集器，支持实时流式输出"""
    def __init__(self):
        self.logs = []
        self.buffer = io.StringIO()
        self.lock = threading.Lock()  # 线程锁，确保线程安全
    
    def write(self, text):
        """重定向 stdout 的 write 方法"""
        # 记录所有内容
        if text:
            with self.lock:
                self.logs.append(text)
                self.buffer.write(text)
        return len(text) if text else 0
    
    def flush(self):
        """重定向 stdout 的 flush 方法"""
        self.buffer.flush()
    
    def get_logs(self):
        """获取收集的日志"""
        with self.lock:
            return ''.join(self.logs)
    
    def get_new_logs(self, last_count=0):
        """获取新增的日志（从 last_count 位置开始）"""
        with self.lock:
            if len(self.logs) > last_count:
                new_logs = self.logs[last_count:]
                return ''.join(new_logs), len(self.logs)
            return '', len(self.logs)
    
    def clear(self):
        """清空日志"""
        with self.lock:
            self.logs.clear()
            self.buffer = io.StringIO()

@contextmanager
def capture_logs(log_collector):
    """上下文管理器：临时重定向 stdout 以捕获日志"""
    original_stdout = sys.stdout
    try:
        sys.stdout = log_collector
        yield log_collector
    finally:
        sys.stdout = original_stdout

# Gradio 界面辅助函数
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
    gr.Markdown("<div id='title'>FinRAGent - 金融API调用助手</div>")
    gr.Markdown("<div id='subtitle'>基于 LangGraph + RAG + FAISS + 层级任务处理，实时显示执行过程</div>")

    # 垂直布局：高级设置 -> 聊天窗口 -> 日志
    with gr.Column():
        # 1) 高级设置
        with gr.Accordion("⚙️ 高级设置", open=False):
            # 是否使用分层结构
            use_hierarchical = gr.Radio(["是", "否"], value="否", label="🏗️ 是否使用分层结构", info="选择'是'使用LangGraph层级任务处理，选择'否'使用简单单Agent架构")
            # 检索模式选择
            retrieval_mode = gr.Radio(["全文档模式", "关键词模式"], value="全文档模式", label="🔍 检索模式", info="选择'全文档模式'使用全文数据库检索，选择'关键词模式'使用关键词数据库检索")
            # Top-K 检索数滑块
            topk = gr.Slider(1, 10, value=5, step=1, label="📑 Top-K 检索数", info="每次检索返回的相关文档数量")
            # 最大记忆回合数滑块
            memory_window = gr.Slider(1, 10, value=5, step=1, label="🧠 最大记忆回合数")
            # 模型切换下拉菜单
            model_selector = gr.Dropdown(choices=["qwen-max", "qwen-plus", "deepseek-chat"], value="qwen-max", label="模型选择")

        # 2) 聊天窗口
        chatbot = gr.Chatbot(label="💬 对话窗口", type="messages", height=420, elem_classes="chatbox")

        # 3) 调试日志
        with gr.Accordion("📝 执行日志", open=False):
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

    def user_submit_stream(message, history, use_hierarchical, retrieval_mode, topk, memory_window, selected_model):
        """
        这是一个 generator 函数，用于处理用户提交并以 streaming 模式向 Gradio 返回结果。
        每次 yield 会更新 Gradio 界面。
        输出顺序与 submit_btn.click 的 outputs 一致：[chatbot, user_input, logs_box]
        """
        # 如果用户切换了模型，立即初始化模型实例
        if selected_model:
            global model, model_split, manage_llm
            if selected_model.lower() in {"qwen-max", "qwen_max", "qwen max", "qwenmax"}:
                model = FixedChatTongyi(model="qwen-max", api_key=DashScope_API_KEY)
                model_split = FixedChatTongyi(model="qwen-max", model_kwargs={"seed": 45}, api_key=DashScope_API_KEY)
                manage_llm = FixedChatTongyi(model="qwen-max", model_kwargs={"seed": 123}, api_key=DashScope_API_KEY)
            elif selected_model.lower() in {"qwen-plus", "qwen_plus", "qwen plus"}:
                model = FixedChatTongyi(model="qwen-plus", api_key=DashScope_API_KEY)
                model_split = FixedChatTongyi(model="qwen-plus", model_kwargs={"seed": 45}, api_key=DashScope_API_KEY)
                manage_llm = FixedChatTongyi(model="qwen-plus", model_kwargs={"seed": 123}, api_key=DashScope_API_KEY)
            elif selected_model.lower() in {"deepseek", "deepseek-chat", "deepseek_chat"}:
                try:
                    model = init_chat_model("deepseek-chat", model_provider="deepseek", seed=2025)
                    model_split = init_chat_model("deepseek-chat", model_provider="deepseek", seed=45)
                    manage_llm = init_chat_model("deepseek-chat", model_provider="deepseek", seed=123)
                except Exception:
                    model = FixedChatTongyi(model="qwen-max", api_key=DashScope_API_KEY)
                    model_split = FixedChatTongyi(model="qwen-max", model_kwargs={"seed": 45}, api_key=DashScope_API_KEY)
                    manage_llm = FixedChatTongyi(model="qwen-max", model_kwargs={"seed": 123}, api_key=DashScope_API_KEY)
            # 重新构建链
            rebuild_chains()

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

        # 根据用户选择决定使用哪种架构
        if use_hierarchical == "否":
            # 简单架构模式
            # 显示处理中状态
            chat[-1] = (message, "⏳ 正在处理中，请稍候...")
            logs_text += "🚀 开始执行简单架构（单Agent模式）...\n"
            yield format_chat_for_gradio(chat), "", logs_text
            
            # 拆分查询
            try:
                chat_history = messages_to_dicts(getattr(memory, 'buffer', []) or [])
                splited_query = simple_split_chain.invoke({"init_query": message, "chat_history": chat_history})
                print("splited_query", splited_query)
            except Exception as e:
                chat[-1] = (message, f"错误：拆分查询失败: {e}")
                logs_text += f"✖ 拆分查询失败: {e}\n"
                yield format_chat_for_gradio(chat), "", logs_text
                return
            
            # 解析拆分后的查询
            splited_query_list = split_numbered_items(splited_query)
            logs_text += f"✂️ 拆分后的查询: {splited_query_list}\n"
            yield format_chat_for_gradio(chat), "", logs_text
            
            # 并发扩展查询
            async def expand_all():
                return await asyncio.gather(*[simple_expand_query(q, memory) for q in splited_query_list])
            
            try:
                expanded_outputs = asyncio.run(expand_all())
            except Exception as e:
                chat[-1] = (message, f"错误：扩展查询失败: {e}")
                logs_text += f"✖ 扩展查询失败: {e}\n"
                yield format_chat_for_gradio(chat), "", logs_text
                return
            
            # 并发检索
            async def retrieve_all():
                return await asyncio.gather(*[simple_retrieve(eo, retrieval_mode, int(topk)) for eo in expanded_outputs])
            
            try:
                retrieval_results = asyncio.run(retrieve_all())
            except Exception as e:
                chat[-1] = (message, f"错误：检索失败: {e}")
                logs_text += f"✖ 检索失败: {e}\n"
                yield format_chat_for_gradio(chat), "", logs_text
                return
            
            # 提取工具
            query_tool_names = []
            for query_index, docs in retrieval_results:
                if not query_index:
                    continue
                logs_text += f"🔍 提取市场分类: {query_index}\n"
                logs_text += f"📖 使用数据库 [{query_index}] 检索 top{topk} 条\n"
                for doc in docs:
                    tool_meta_name = doc.metadata.get("name")
                    if tool_meta_name and tool_meta_name in API_TOOL_dic:
                        query_tool_names.append(API_TOOL_dic[tool_meta_name])
            
            query_tool_names = list(set(query_tool_names))
            query_tools_instances = [tool for tool in all_tools if tool.name in query_tool_names]
            logs_text += f"🛠️ 检索到的工具: {query_tool_names if query_tool_names else '无'}\n"
            yield format_chat_for_gradio(chat), "", logs_text
            
            # 如果没有工具，直接让LLM回答
            if not query_tools_instances:
                direct_llm_response = model.invoke(message)
                assistant_text = direct_llm_response.content if hasattr(direct_llm_response, "content") else str(direct_llm_response)
                chat[-1] = (message, assistant_text)
                logs_text += "⚠️ 未检索到合适工具 → 直接由LLM回答\n"
                yield format_chat_for_gradio(chat), "", logs_text
                memory.save_context({"input": message}, {"output": assistant_text})
                return
            
            # 有工具，启动 agent 于独立线程并通过 ToolLoggingHandler 实时抓取日志
            # 创建工具日志处理器
            tool_logger = ToolLoggingHandler()
            # 创建回调管理器
            cb_manager = CallbackManager([tool_logger])
            # 创建 Agent
            agent = create_tool_calling_agent(model, query_tools_instances, simple_agent_prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=query_tools_instances, 
                verbose=True, 
                memory=memory,
                callback_manager=cb_manager
            )
            
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
            
            # 轮询 handler.entries，持续把中间日志拼成 assistant 的"部分输出"并 yield
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
            # 保存到记忆
            memory.save_context({"input": message}, {"output": final_text})
            # 最终更新界面
            yield format_chat_for_gradio(chat), "", logs_text
            
            return
        
        # 分层架构模式（LangGraph）
        # 初始化状态
        initial_state = {
            "query": message,
            "processed_query": "",
            "is_tool_needed": False,
            "hierarchical_tasks": None,
            "task_results": {},
            "all_intermediate_steps": [],
            "formatted_steps": "",
            "final_answer": "",
            "data": "",
            "current_level": 1,
            "completed_levels": [],
            "retrieval_mode": retrieval_mode,  # 添加检索模式到状态中
            "topk": int(topk)  # 添加 topk 到状态中
        }

        # 显示处理中状态
        chat[-1] = (message, "⏳ 正在处理中，请稍候...")
        logs_text += "🚀 开始执行 LangGraph 工作流（分层架构）...\n"
        yield format_chat_for_gradio(chat), "", logs_text
        
        # 创建日志收集器（用于捕获所有 print 输出）
        log_collector = LogCollector()
        
        # 用于存储执行结果
        execution_result = {"final_state": None, "error": None, "completed": False}
        
        def run_langgraph():
            """在单独线程中运行 LangGraph"""
            try:
                with capture_logs(log_collector):
                    # 执行 LangGraph
                    execution_result["final_state"] = app.invoke(initial_state)
                execution_result["completed"] = True
            except Exception as e:
                # 确保 stdout 已恢复
                if sys.stdout is log_collector:
                    sys.stdout = sys.__stdout__
                execution_result["error"] = e
                execution_result["completed"] = True
        
        # 在单独线程中启动 LangGraph 执行
        begin_time = time.time()
        execution_thread = threading.Thread(target=run_langgraph, daemon=True)
        execution_thread.start()
        
        # 实时监控日志并更新界面
        last_log_count = 0
        try:
            while execution_thread.is_alive() or not execution_result["completed"]:
                # 检查是否有新的日志
                new_logs, current_count = log_collector.get_new_logs(last_log_count)
                if new_logs:
                    logs_text += new_logs
                    last_log_count = current_count
                    # 实时更新日志显示
                    yield format_chat_for_gradio(chat), "", logs_text
                
                # 短暂休眠，避免 CPU 占用过高
                time.sleep(0.1)
            
            # 获取最后剩余的日志
            new_logs, _ = log_collector.get_new_logs(last_log_count)
            if new_logs:
                logs_text += new_logs
                yield format_chat_for_gradio(chat), "", logs_text
            
            end_time = time.time()
            
            # 检查是否有错误
            if execution_result["error"]:
                raise execution_result["error"]
            
            final_state = execution_result["final_state"]
            
            # 获取最终答案
            final_answer = final_state.get("final_answer", "")
            if final_answer:
                chat[-1] = (message, final_answer)
            else:
                chat[-1] = (message, "处理完成，但未生成最终答案。")
            
            logs_text += f"\n✅ 执行完成\n"
            logs_text += f"⏱️ 总执行时间: {end_time - begin_time:.2f} 秒\n"
            
            # 显示执行统计信息
            if "validity_check_stats" in final_state:
                stats = final_state["validity_check_stats"]
                if stats.get("total_checks", 0) > 0:
                    logs_text += f"\n📊 结果有效性检查统计:\n"
                    logs_text += f"  - 总检查次数: {stats['total_checks']}\n"
                    logs_text += f"  - 有效结果: {stats['valid_results']}\n"
                    logs_text += f"  - 过滤结果: {stats['filtered_results']}\n"
            
            # 最终更新界面
            yield format_chat_for_gradio(chat), "", logs_text
            
        except Exception as e:
            # 确保 stdout 已恢复
            if sys.stdout is log_collector:
                sys.stdout = sys.__stdout__
            
            # 获取最后剩余的日志
            new_logs, _ = log_collector.get_new_logs(last_log_count)
            if new_logs:
                logs_text += new_logs
                yield format_chat_for_gradio(chat), "", logs_text
            
            error_msg = f"执行过程中发生错误: {e}"
            chat[-1] = (message, f"❌ {error_msg}")
            logs_text += f"\n❌ {error_msg}\n"
            import traceback
            logs_text += traceback.format_exc()
            yield format_chat_for_gradio(chat), "", logs_text

    # 绑定发送按钮的点击事件到 user_submit_stream 函数
    submit_btn.click(
        fn=user_submit_stream,
        inputs=[user_input, chatbot, use_hierarchical, retrieval_mode, topk, memory_window, model_selector],
        outputs=[chatbot, user_input, logs_box],
    )

    def clear_all():
        """清空聊天界面和对话记忆"""
        global memory
        memory.clear()  # 清除对话记忆
        return [], "", ""  # 返回空列表和空字符串，清空聊天、输入框和日志

    clear_btn.click(clear_all, None, [chatbot, user_input, logs_box])

# 启动函数
def run_gradio_server():
    """在单独线程中启动 Gradio 服务器"""
    GRADIO_SERVER_PORT = 7860
    print(f"Starting Gradio server on port {GRADIO_SERVER_PORT}...")
    demo.launch(
        server_name="127.0.0.1",  # 确保只在本地监听
        server_port=GRADIO_SERVER_PORT,
        inbrowser=False,  # 不自动打开浏览器
        share=False,  # 不创建公共共享链接
    )
    print("Gradio server stopped.")

def start_webview_window():
    """在主线程中启动 pywebview 窗口"""
    GRADIO_SERVER_PORT = 7860
    local_url = f"http://127.0.0.1:{GRADIO_SERVER_PORT}"
    print(f"Loading Gradio app in webview from: {local_url}")
    
    webview.create_window(
        "FinRAGent - 金融API调用助手",  # 窗口标题
        url=local_url,
        width=1000,
        height=800,
        min_size=(800, 600),  # 最小窗口大小
        resizable=True,
    )
    webview.start()

if __name__ == "__main__":
    # 入口显示 SplashScreen
    splash = SplashScreen()
    splash.mainloop()
    
    print("Application closed.")
