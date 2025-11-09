import asyncio
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import json
import re
from typing import Any, Dict, Tuple
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain.chat_models import init_chat_model
# 导入记忆模块
from langchain.memory import ConversationBufferWindowMemory # 导入滑动窗口记忆
from langchain_core.messages import HumanMessage, AIMessage # 用于手动构建消息，如果需要的话
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from all_tool import all_tools, load_dict_from_json # 确保这些是正确的导入路径
print("main.py中datetime是什么类型？", type(datetime), datetime)
import pandas as pd  # 新增导入 pandas
from langchain.tools import tool # 新增导入 Tool
from langchain.tools import Tool
from langchain_core.documents import Document
from pydantic import BaseModel, Field
now = datetime.now()


print("--- Python 脚本中读取的环境变量 ---")
print("HTTP_PROXY:", os.getenv('HTTP_PROXY'))
print("HTTPS_PROXY:", os.getenv('HTTPS_PROXY'))
print("http_proxy:", os.getenv('http_proxy'))
print("https_proxy:", os.getenv('https_proxy'))
print("-----------------------------------")
# 加载api_tool_dic
API_TOOL_dic_path = "./api_dic.json"
API_TOOL_dic = load_dict_from_json(API_TOOL_dic_path)
load_dotenv(override=True)

# 加载四种FASSI数据库
DashScope_API_KEY = os.getenv("DASHSCOPE_API_KEY")

try:
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=DashScope_API_KEY
    )
    print("DashScopeEmbeddings model initialized successfully.")
except Exception as e:
    print(f"Error initializing DashScopeEmbeddings: {e}")
    exit()


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
                    # print(f"message: {message["tool_calls"]}")
                    # print(f"prev_function: {prev_message["tool_calls"]}")
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
class SynthesizerOutput(BaseModel):
    """定义了管理者/整合者Chain的最终输出结构。"""
    final_answer: str = Field(description="整合所有中间步骤后，生成给用户的最终答案。")
    data_to_scratchpad: list[Dict[str, Any]] = Field(description="一个字典列表，包含需要存入数据暂存区的数据。每个字典应有 'key' 和 'value' 两个键。例如：[{'key': '特斯拉_股价', 'value': 200}]。")
    should_cache: bool = Field(description="综合判断，这个回答是否质量高到值得存入长期缓存。")
    reasoning: str = Field(description="对以上决策（特别是缓存决策）的简要说明。")
# ==================== 新增：缓存管理器 ====================
class CacheManager:
    def __init__(self, embeddings_model):
        print("正在初始化缓存管理器...")
        # 1. 初始化一个空的DataFrame作为“表格”
        self.df = pd.DataFrame(columns=['Question', 'Answer'])
        
        # 2. 初始化一个内存中的FAISS作为“语义索引”
        self.embeddings = embeddings_model
        initial_doc = Document(page_content="placeholder_for_init", metadata={"answer": "init"})
        self.vectorstore = FAISS.from_documents([initial_doc], self.embeddings)
        
        # 3. 设置相似度阈值
        self.threshold = 0.25  # 使用我们之前讨论过的更合理的阈值

    def add(self, question: str, answer: str):
        """向缓存中添加一个新的问答对"""
        # 更新DataFrame
        new_row = pd.DataFrame([{'Question': question, 'Answer': answer}])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # 更新FAISS语义索引
        new_doc = Document(page_content=question, metadata={'answer': answer})
        self.vectorstore.add_documents([new_doc])
        
        print("💾 缓存已更新。当前缓存数量:", len(self.df))
        print("当前缓存表格:\n", self.df) # 可取消注释以查看表格

    def search(self, query: str) -> str:
        """根据语义相似度在缓存中搜索答案"""
        print("🔍 正在缓存中搜索相似问答...")
        results = self.vectorstore.similarity_search_with_score(query, k=1)
        
        if results and results[0][1] < self.threshold:
            doc, score = results[0]
            cached_question = doc.page_content
            cached_answer = doc.metadata['answer']
            
            response = (
                f"✅ 缓存命中 (相似度得分: {score:.3f})！\n"
                f"相似的历史问题是: '{cached_question}'\n"
                f"对应的答案是: {cached_answer}"
            )
            return response
        
        return "❌ 缓存中未找到足够相似的答案，请继续使用其他工具。"

# 实例化缓存管理器
cache_manager = CacheManager(embeddings)
# ==================== 新增：创建缓存搜索工具 ====================
cache_search_tool = Tool(
    name="ConversationCacheSearch",
    func=cache_manager.search,
    description="【最高优先级，任务起点】在开始任何新任务前，必须首先调用此工具。它用于检查是否存在一个与用户当前问题高度相似的历史问答，并返回其完整的最终答案。如果此工具返回了一个具体的答案，应直接采纳该答案并结束任务。只有当此工具明确返回'未找到'时，才应继续执行其他步骤来从头解决问题。"
)
# ==================== 修改：实现方案三的“数据暂存区” ====================
class DataScratchpad:
    """
    一个具有模糊键名匹配功能的键值存储，用作Agent的短期工作记忆。
    """
    def __init__(self, embeddings_model):
        self.data: Dict[str, Any] = {}
        self.embeddings = embeddings_model
        self.key_vectorstore: FAISS | None = None
        self.key_similarity_threshold = 0.25
        print("📝 具备模糊匹配功能的数据暂存区 (Scratchpad) 已初始化。")

    def _save_data(self, key: str, value: Any):
        """内部保存逻辑"""
        self.data[key] = value
        new_key_doc = Document(page_content=key)
        if self.key_vectorstore is None:
            self.key_vectorstore = FAISS.from_documents([new_key_doc], self.embeddings)
        else:
            self.key_vectorstore.add_documents([new_key_doc])

    def _retrieve_data(self, key: str) -> Any:
        """内部检索逻辑"""
        if self.key_vectorstore is None:
            return "暂存区为空，无法检索。"
        
        results_with_scores = self.key_vectorstore.similarity_search_with_score(key, k=1)
        
        if results_with_scores and results_with_scores[0][1] < self.key_similarity_threshold:
            most_similar_key_doc, score = results_with_scores[0]
            most_similar_key = most_similar_key_doc.page_content
            retrieved_value = self.data.get(most_similar_key, "内部错误：在字典中找不到已匹配的键。")
            print(f"暂存区模糊检索成功: 传入key='{key}', 匹配到最相似key='{most_similar_key}' (得分: {score:.3f})")
            return retrieved_value
        
        print(f"暂存区模糊检索失败: 传入key='{key}'，未找到相似的已存键名。")
        return "未在暂存区中找到与该键名相关的数据。"

# 【修改】实例化暂存区时传入 embeddings
scratchpad = DataScratchpad(embeddings)

# 【修改】优化工具描述，引导Agent生成规范的键名（仍然是好习惯）
class SaveArgs(BaseModel):
    key: str = Field(description="用于存储和检索数据的唯一标识符。强烈建议遵循【实体_属性】的命名规范，例如 '特斯拉_股价'。")
    value: Any = Field(description="要存储的原始数据值，可以是数字、字符串或字典。")

class RetrieveArgs(BaseModel):
    key: str = Field(description="要检索的数据的唯一标识符。即使不完全确定键名，系统也会尝试模糊匹配。提供一个规范的键名，会得到最准确的结果。")


# 创建暂存区工具 (保持不变)
@tool(args_schema=SaveArgs)
def save_to_scratchpad(key: str, value: Any) -> str:
    """【暂存中间结果】当你通过外部API工具成功获取到一个可复用的原始数据点（例如一个具体的股价、财报数字、指标）后，应立即使用此工具将其存入数据暂存区，以便在当前任务的后续步骤中直接使用。键名(key)应严格遵循'实体_属性'格式，例如 key='特斯拉_股价', value=200。"""
    scratchpad._save_data(key, value)
    print(f"暂存区保存: key='{key}', value={value}")
    return f"数据点 '{key}' 已成功保存到暂存区。"

@tool(args_schema=RetrieveArgs)
def retrieve_from_scratchpad(key: str) -> Any:
    """【API调用前检查】在调用任何外部API（如查询股价、财报的工具）来获取原始数据之前，必须先使用此工具，通过一个描述性的键名（key）来检查工作暂存区中是否已存在所需的数据点。这可以避免不必要的API调用。例如，在需要特斯拉股价时，先用 key='特斯拉_股价' 在此检索。"""
    return scratchpad._retrieve_data(key)
# ==================== 修改结束 ====================
# 加载***完整数据库***
index_list = ["AM", "CN", "HK", "OT"]
faiss_databases = {}
for index in index_list:

    database_path = f"./{index}_APISPECS_faiss_index"
    try:
        faiss_databases[index] = FAISS.load_local(database_path, embeddings, allow_dangerous_deserialization=True)
        print(f"{index}_FAISS 数据库已从 {database_path} 成功加载。")
    except Exception as e:
        print(f"加载 FAISS 数据库失败: {e}")
        print("请确保创建database的py文件已运行且数据库文件存在。")
        exit()

faiss_key_databases = {}
for index in index_list:

    key_database_path = f"./{index}_APISPECS_key_faiss_index"
    try:
        faiss_key_databases[index] = FAISS.load_local(key_database_path, embeddings, allow_dangerous_deserialization=True)
        print(f"{index}_FAISS 数据库已从 {key_database_path} 成功加载。")
    except Exception as e:
        print(f"加载 FAISS 数据库失败: {e}")
        print("请确保创建database的py文件已运行且数据库文件存在。")
        exit()

key_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", 
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

    如果用户输入不是一个完整的自然语言查询（如“是的”、“再查一次”、“用01810试试看”），请你结合下面的历史对话内容 chat_history，尽力补全用户的真实意图，然后再按照自然语言扩展规则进行扩展查询。

    请只返回最终扩展后的查询内容，不要添加任何解释或说明文字。
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}")
])

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
    ("human", "{init_query}")
])
# RAG 扩展查询 Prompt
'''
rag_prompt = PromptTemplate(
    template = 
    """你是一个API查询助手，精通金融数据API文档的结构和内容。
        你的任务是根据用户的自然语言查询，根据需要将其扩展成一个更详细、更精确、包含更多API文档中常见词汇（如功能描述、支持市场、数据类型、数据粒度、数据特点、应用场景、限制与注意事项、以及API的关键词）的查询文本。这个扩展后的查询旨在帮助向量数据库更好地匹配到最相关的API文档。
        只返回扩展后的查询文本，不要添加任何解释或额外文字。如果原始查询信息足够丰富则不必过多扩展。

        ---
        示例开始 ---

        用户查询: 我要问腾讯的最近的股价

        扩展查询: 查询腾讯控股（00700）在香港股票市场的实时行情数据，包括最新价、涨跌幅、成交量、成交额等。需要获取港股盘中数据。

        ---
        用户查询: 我要问查询比亚迪公司的最新动态和中文新闻资讯，包括新闻标题、来源、发布日期和链接等，用于公司舆情监控和投资决策。

        扩展查询: 查询比亚迪公司的最新动态和中文新闻资讯，包括新闻标题、来源、发布日期和链接等，用于公司舆情监控和投资决策，可以使用NewsAPI或者东方财富网提供的新闻资讯以及NEWS_SENTIMENT。

        ---

        用户查询: 我想看看阿里巴巴过去一年的股价走势

        扩展查询: 查询阿里巴巴（09988）在香港股票市场过去一年的历史行情数据，包括开盘价、收盘价、最高价、最低价、成交量、成交额等K线数据。可能需要进行复权处理，用于技术分析和量化回测。

        ---

        用户查询: 告诉我小米的基本情况

        扩展查询: 查询小米集团的详细公司资料和基本信息，包括公司名称、注册地、所属行业、董事长、公司介绍等上市公司基本资料，用于公司基本面研究和尽职调查。

        ---

        用户查询: 想了解苹果的财务状况

        扩展查询: 查询苹果公司（AAPL）作为美国公司的综合信息，包括公司基本资料、财务比率、估值指标和关键业务指标等财务状况，通常在财报发布当天更新，用于基本面分析和投资决策。

        ---
        用户查询: 腾讯有什么新闻？

        扩展查询: 查询腾讯控股的中文新闻资讯，包括新闻标题、来源、发布日期和链接等，用于公司舆情监控，可以使用NewsAPI或者东方财富网提供的新闻资讯以及NEWS_SENTIMENT。

        ---
        用户查询: 我想了解华为在国际上的报道。

        扩展查询: 查询华为公司在国际媒体上的英文新闻资讯，包括新闻标题、来源、发布日期和链接等，用于国际舆情分析，可以使用NewsAPI或者东方财富网提供的新闻资讯以及NEWS_SENTIMENT。

        ---
        用户查询: 告诉我关于特斯拉的英文新闻。

        扩展查询: 查询特斯拉公司（TSLA）在国际媒体上的英文新闻资讯，包括新闻标题、来源、发布日期和链接等，用于国际公司舆情监控和投资决策，可以使用NewsAPI或者东方财富网提供的新闻资讯以及NEWS_SENTIMENT。

        ---
        示例结束 ---

        用户查询: {init_query}

        扩展查询:
    """
)
'''
# 初始化LLM
# model = init_chat_model("deepseek-chat", model_provider="deepseek", seed=2025)

model = FixedChatTongyi(
    model="qwen-max",  # 或 qwen-max / qwen-turbo 等
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 配合语义分割以适应多API问题
model_split = FixedChatTongyi(
    model="qwen-turbo",
    model_kwargs={"seed": 45},
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
# 新增一个专门用于质检的LLM实例
judge_llm = FixedChatTongyi(
    model="qwen-turbo", # 使用一个速度快、成本效益高的模型
    model_kwargs={"seed": 123},
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
# 1. 创建 Pydantic 解析器
synthesizer_parser = PydanticOutputParser(pydantic_object=SynthesizerOutput)

history_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个金融问题理解助手。
     你的任务是：根据智能体与用户的对话历史以及当前用户的输入，首先准确判断并整理出用户当前的查询需求及相关分析需求，
     如果涉及金融数据或相关金融信息查询(如股票、债券、公司信息、金融新闻舆情等等)或任何需要结合金融信息的分析性问题，则以如下格式输出重新组织用户需求后的完整查询内容："True,[重新组织后的查询]"
     注意：禁止对用户的查询或提问进行任何形式的解释、分析、推理或回答。禁止直接回应用户的需求。输出必须是一个问题或查询。
     输出中不得包含任何额外内容（例如提示语、解释性文字或结论）
     如果用户的问题属于概念解释（如“什么是市盈率？”）、打招呼或一般性对话，则输出："False,[重新组织后的查询]"
     """),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}")
])
history_query_chain = history_prompt | model | StrOutputParser()

# ==================== 新增：为闲聊创建专用的对话链 ====================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好、乐于助人的AI助手。请根据对话历史和用户当前的问题，提供一个流畅、有帮助的回答。"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

def _strip_matching_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1].strip()
    return s


def parse_history_output(text: str) -> Tuple[bool, str]:
    """
    解析 history_prompt 的输出格式，返回 (flag, content)：
      - flag: bool，表示 True/False（对大小写不敏感）
      - content: str，方括号中的文本或逗号后面的原始文本（去掉外层引号与首尾空白）

    支持的输入形式举例：
      "True,[重新组织后的查询]"
      "False,[用户原始输入]"
      "true, 这是没有方括号的内容"
      "False, '原始输入在引号内'"

    在无法解析时抛出 ValueError。
    """
    if text is None:
        raise ValueError("输入为空")

    txt = text.strip()

    # 1) 尝试匹配带方括号的形式： True,[...]
    m = re.match(r'^\s*(True|False)\s*,\s*\[\s*(.*)\s*\]\s*$', txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        flag_str = m.group(1).lower()
        content = m.group(2)
        content = _strip_matching_quotes(content)
        return (flag_str == 'true', content)

    # 2) 回退：匹配不带方括号的形式： True,内容
    m2 = re.match(r'^\s*(True|False)\s*,\s*(.*)$', txt, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        flag_str = m2.group(1).lower()
        content = m2.group(2).strip()
        content = _strip_matching_quotes(content)
        return (flag_str == 'true', content)

    # 3) 都不匹配 -> 报错
    raise ValueError(f"无法解析输入：{text!r}")

rag_prompt_split = PromptTemplate(
    template =
    """请判断下面这个查询查询几项内容，并且只将所查询的内容按照‘1.\n 2.\n’的格式列出，尽可能的让每个单独的查询内容都拥有准确的时间空间信息，注意我们只保留对金融部分的查询。

        示例开始 ---

        用户查询: 我想分析可转债‘113527’过去三个月的价值分析数据并获取同期比特币每日收盘价

        分割查询: 1.分析可转债‘113527’过去三个月的价值分析数据。\n
                2.获取过去三个月比特币每日收盘价。\n

        ---
        用户查询: 计算食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的速动比率、资产负债率和每股收益，并按综合评级给出前 5 名。

        分割查询: 1.查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的速动比率。\n
                2.查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的资产负债率。\n
                3.查询食品饮料行业（贵州茅台 600519.SH、五粮液 000858.SZ、泸州老窖 000568.SZ）最新报告期的每股收益。\n

        ---
        用户查询: 获取全市场未来 3 个月（2025-07-22 至 2025-10-22）的财报日历，筛选出同时有 ≥5 次机构调研的公司并排名。

        分割查询: 1.获取全市场未来 3 个月（2025-07-22 至 2025-10-22）的财报日历。\n

        ---
        示例结束 ---

        用户查询: {init_query}

        分割查询:
    """
)
rag_llm_chain_split = rag_prompt_split | model_split | StrOutputParser()

def split_numbered_items(s: str) -> list[str]:
    # 1. 找到每个“数字. ”的起始位置（点后必须跟空格）
    starts = []
    for i in range(len(s) - 2):
        if s[i].isdigit() and s[i + 1] == '.' and s[i + 2].isspace():
            starts.append(i)
    # 最后一项的结束位置
    starts.append(len(s))

    items = []
    # 2. 按各起始位置区间切片
    for idx in range(len(starts) - 1):
        seg = s[starts[idx]:starts[idx + 1]]
        # 3. 去掉“数字. ”前缀
        dot_pos = seg.find('.')
        content = seg[dot_pos + 1:].strip()
        # 4. 去除两端的 **（若存在）
        if content.startswith("**") and content.endswith("**"):
            content = content[2:-2].strip()
        items.append(content)
    return items

def extract_market_category(llm_response_text: str) -> str | None:
    """
    从LLM的回答中提取前两个字母的市场分类代码。
    预期LLM回答格式为 "XX,扩展查询文本..." 或 "XX扩展查询文本..."

    Args:
        llm_response_text (str): LLM的原始回答字符串。

    Returns:
        str | None: 提取到的两个字母分类代码（大写），如果无法提取则返回 None。
    """
    if not isinstance(llm_response_text, str) or not llm_response_text:
        return None

    # 移除回答前后的空白字符
    cleaned_response = llm_response_text.strip()

    # 确保字符串长度至少为2，才能提取前两个字符
    if len(cleaned_response) >= 2:
        category = cleaned_response[:2].upper() # 提取前两个字符并转换为大写

        # 可以在这里添加一个可选的验证步骤，确保提取的分类是预期的
        expected_categories = {"AM", "CN", "HK", "OT"}
        if category in expected_categories:
            return category
        else:
            return None # 如果提取的分类不在预期列表中，则视为无效
    else:
        return None # 如果字符串太短，无法提取分类

def format_intermediate_steps(intermediate_steps: list) -> str:
    """
    将 AgentExecutor 返回的原始 intermediate_steps 格式化为
    一个对 LLM 更友好的、简洁的字符串。
    """
    if not intermediate_steps:
        return "执行者 Agent 未调用任何工具。"

    log_parts = []
    for i, (action, observation) in enumerate(intermediate_steps):
        # action 是 ToolAgentAction 对象
        # observation 是工具返回的字符串结果
        
        tool_name = action.tool
        tool_input = action.tool_input
        
        # 将工具输入（通常是字典）转换为易读的JSON字符串
        # ensure_ascii=False 保证中文字符正常显示
        tool_input_str = json.dumps(tool_input, ensure_ascii=False, indent=2)

        log_parts.append(
            f"步骤 {i+1}: 工具调用\n"
            f"---工具名称: `{tool_name}`\n"
            f"---工具输入:\n```json\n{tool_input_str}\n```\n"
            f"---工具输出:\n```\n{observation}\n```"
        )
        
    return "\n---\n".join(log_parts)

# 构建加工链
key_rag_llm_chain = key_rag_prompt.partial(current_time=str(datetime.now())) | model | StrOutputParser()
rag_llm_chain = rag_prompt.partial(current_time=str(datetime.now())) | model | StrOutputParser()
chat_chain = chat_prompt | model | StrOutputParser()

# 1. 实例化记忆模块
# memory_key 必须与 ChatPromptTemplate 中用于历史对话的 placeholder 名称一致
# k=5 表示记住最近的5轮对话（用户输入+AI输出算一轮）
# return_messages=True 表示返回消息对象列表，这与ChatPromptTemplate的期望一致
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

# 2. 修改Agent的Prompt，添加 {chat_history} 占位符
# 这个占位符将由 AgentExecutor 自动填充，包含记忆中的历史对话

#对于不需要API调用的提问 处理方式二：
#经过测试 不正确 openai规范中传入agent的tool不能为空列表
worker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""你是一个高效、智能、严谨的顶级金融数据收集助手，当前时间为 {now}。
你的唯一任务是根据用户的请求，规划并执行一系列工具调用来获取所有必要的原始数据。

你必须严格遵守以下【工作流程】：

**第一步：检查长期记忆（对话缓存）**
   - 永远首先调用 `ConversationCacheSearch` 工具，检查是否已经回答过完全相同或非常相似的问题。
   - 如果找到答案，直接返回该答案，任务结束。

**第二步：规划并执行数据获取**
   - 如果缓存未命中，你需要分析用户的问题，规划需要获取哪些数据。
   - 在调用任何外部API工具（例如 `get_stock_price`）之前，必须先调用 `retrieve_from_scratchpad` 工具检查所需数据是否已存在。

   ---
   **【调用 `retrieve_from_scratchpad` 的黄金法则】**
   你必须遵循以下三步法来构建最精确的键名(key)：

   1.  **第一步：识别实体** - 从用户问题中找出核心的金融实体，例如：“特斯拉”、“苹果公司”、“贵州茅台”。
   2.  **第二步：识别指标** - 从用户问题中找出需要查询的**最具体**的指标或属性，例如：“跌幅”、“市盈率”、“最新报告期的速动比率”、“前一交易日收盘价”。
   3.  **第三步：精确组合** - 严格按照 **“实体_指标”** 的格式组合键名。

   **示例:**
   - **正例 (必须这样做):**
     - 用户提问：“特斯拉的跌幅是多少？” -> 你构建的键名必须是 `key='特斯拉_跌幅'`。
     - 用户提问：“苹果公司的市盈率” -> 你构建的键名必须是 `key='苹果公司_市盈率'`。
     - 用户提问：“茅台的最新股价” -> 你构建的键名必须是 `key='贵州茅台_最新股价'`。
     - 用户提问：“特斯拉前一天的收盘价” -> 你构建的键名必须是 `key='特斯拉_前一交易日收盘价'`。

   - **反例 (绝对禁止):**
     - 用户提问：“特斯拉的跌幅是多少？” -> 禁止使用 `key='特斯拉_股价'`。（错误：过于宽泛，没有精确到“跌幅”）
     - 用户提问：“苹果公司的市盈率” -> 禁止使用 `key='苹果公司_财务数据'`。（错误：过于笼统，“财务数据”是一个类别，而不是具体指标）

   **核心要求：** 禁止使用任何模糊、笼统或上位的词语（如‘股价’、‘数据’、‘信息’）作为指标，除非用户的问题本身就如此宽泛。你的目标是实现1对1的精确匹配。
   ---

   - 如果通过精确的键名在暂存区找到了所有需要的数据，就**必须停止**并结束任务，不要再调用外部API。
   - 如果暂存区没有所需数据，才去调用相应的外部API工具获取。

**第三步：完成任务**
   - 当你认为已经收集到了回答问题所需的全部原始数据后，停止工具调用。
   - 你不需要生成最终的、格式优美的答案，只需确保你调用的工具结果已经包含了足够的信息即可。
"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# 2. 创建管理者 Prompt
manager_prompt = ChatPromptTemplate.from_messages([
    ("system",
          """你是一位资深的金融分析师和AI数据管家。
你的任务是接收用户的【原问题】和AI执行者收集到的【中间数据摘要】，然后完成以下四项工作：

1.  **整合答案 (final_answer)**: 根据【中间数据摘要】，首先合成一个直接、准确的答案来回应用户的【原问题】。

2.  **数据暂存 (data_to_scratchpad)**: 这是你的核心职责。你必须将【中间数据摘要】中的每一个工具输出都视为一个等待解析的“数据包”。你的任务是：
    - **全面解析**：仔细阅读工具输出的完整内容。
    - **最大化提取**: 从中提取出 **所有** 独立的、具有复用价值的原始数据点。一个工具的输出往往包含多个有用的信息，你必须全部识别出来。
    - **逐条构建**: 为每一个提取出的数据点，都创建一个独立的 key-value 字典，并遵循“实体_指标”的命名规范。
    - **汇总列表**: 将所有构建好的字典汇总成一个列表。

3.  **缓存决策 (should_cache)**: 评估你生成的 `final_answer` 的质量。只有当答案有效、信息详实且无误时，才将 `should_cache` 设为 `True`。

4.  **合规性审查**: 确保你的 `final_answer` 中立客观，不包含任何投资建议。如果内容涉及预测或建议，请在末尾追加：\n【免责声明：以上内容仅供参考，不构成任何投资建议。】

**【中间数据摘要】格式示例:**
步骤 1: 工具调用
---工具名称: tool_name_A
---工具输入:{{"parameter_key_1": "parameter_value_1","parameter_key_2": "parameter_value_2"}}
---工具输出:这是工具A返回的原始结果字符串，可能是JSON、纯文本或其他格式。
步骤 2: 工具调用
---工具名称: tool_name_B
---工具输入:{{"another_key": "another_value"}}
---工具输出:这是工具B返回的原始结果字符串，可能是JSON、纯文本或其他格式。
**输出格式:**
你在综合信息输出时，请你输出完整，比如在输出新闻相关内容时，不仅仅输出标题，还要输出来源、时间、链接等信息，其他信息时同理。
你必须严格按照以下JSON格式进行输出，不要添加任何其他解释性文字。
{format_instructions}
"""),
    ("human", "【原问题】:\n{query}\n\n【中间数据】:\n{formatted_intermediate_steps}")
]).partial(format_instructions=synthesizer_parser.get_format_instructions())
# 3. 组合成完整的管理者/整合者链
synthesizer_chain = manager_prompt | judge_llm | synthesizer_parser

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", f"你是金融助手，请根据用户的问题，给出相应的金融信息，当前时间为{now}。当用户未指明查找时间时，默认查按照近期数据，且如果查找没有结果，可以适当向前查找历史数据。"),
#         # 插入历史对话的占位符
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )

# 3. 创建Agent
# 由于工具是动态检索的，Agent的创建和执行需要放在循环内部，或者在每次循环中更新工具列表。
# 为了简化，我们先将AgentExecutor的创建放在循环外部，但要注意，如果工具集每次都根据RAG结果变化，
# 那么AgentExecutor需要重新创建或动态更新其工具列表（这在LangChain中不直接支持，通常意味着AgentExecutor需要重新初始化）。
# 对于记忆功能，AgentExecutor的初始化只需要一次。工具的动态选择通常在RAG步骤中完成，然后传递给Agent。

async def expand_query(single_split_query, memory):
    return await rag_llm_chain.ainvoke({"init_query": single_split_query, "chat_history": memory.buffer})

async def retrieve(single_last_query, mode, topk):
    query_index = extract_market_category(single_last_query)
    if not query_index:
        return query_index, []
    if mode == "关键词模式":
        docs = await faiss_key_databases[query_index].asimilarity_search(single_last_query, k=topk)
    else:
        docs = await faiss_databases[query_index].asimilarity_search(single_last_query, k=topk)
    return query_index, docs

# 循环开始，允许用户进行多轮对话
print("欢迎使用金融API调用助手！输入 '退出' 结束对话。")
while True:
    query = input("\n请输入想要询问的问题：")
    if query.lower() == '退出':
        print("对话结束。")
        break

    query = history_query_chain.invoke({"init_query": query, "chat_history": memory.buffer})
    rag_use, query = parse_history_output(query)
    print(f"结合上下文推理后的查询：{query}")
    if not rag_use:
        print("无需调用工具，由语言模型直接回答...")
        direct_llm_response = chat_chain.invoke({
            "input": query,
            "chat_history": memory.buffer_as_messages # 传入完整的历史消息
        })
        
        print("\n[直接回答]:", direct_llm_response)
         # 【新增】手动将闲聊内容存入记忆
        memory.save_context({"input": query}, {"output": direct_llm_response})
        continue
    # RAG 和工具检索部分 (这部分仍然是每次查询都执行，因为工具的选择可能依赖于当前查询)
    begin_time_rag_process = time.time()
    splited_query = rag_llm_chain_split.invoke({"init_query": query})
    splited_query_list = split_numbered_items(splited_query)
    mode = 1
    topk = 3
    async def main():
        # 1️⃣ 并发扩展查询
        last_query_list = await asyncio.gather(*[
            expand_query(q, memory) for q in splited_query_list
        ])
        print(f"📝 扩展查询结果: {last_query_list}")

        # 2️⃣ 并发检索
        retrieval_results = await asyncio.gather(*[
            retrieve(q, mode, topk) for q in last_query_list
        ])

        results, query_tool_names = [], []
        for query_index, docs in retrieval_results:
            if not query_index:
                continue
            print(f"🔍 提取市场分类: {query_index}")
            print(f"📖 使用数据库 [{query_index}] 检索 top{topk} 条")

            results.append(docs)
            for doc in docs:
                tool_meta_name = doc.metadata.get("name")
                if tool_meta_name and tool_meta_name in API_TOOL_dic:
                    query_tool_names.append(API_TOOL_dic[tool_meta_name])

        query_tool_names = list(set(query_tool_names))
        print(f"🛠️ 检索到的工具: {query_tool_names if query_tool_names else '无'}")
        return results, query_tool_names, last_query_list


    results, query_tool_names, last_query_list = asyncio.run(main())
    end_time_rag_process = time.time()
    print(f"扩展和RAG检索执行时间: {end_time_rag_process - begin_time_rag_process} 秒")

    # 筛选出实际检索到的工具实例
    query_tools_instances = [tool for tool in all_tools if tool.name in query_tool_names]
    # 3. 【修改】将RAG检索到的API工具和缓存工具\数据暂存区合并
    worker_tools = [cache_search_tool, retrieve_from_scratchpad] + query_tools_instances
    print(f"✨ 执行者 Agent执行者 Agent本轮可用总工具: {[t.name for t in worker_tools]}")

    # 4. 创建并执行 Agent
    # 如果一个工具都没有（既没有API工具，也没有缓存工具），则直接回答
    if not query_tools_instances:
        print("未检索到相关API工具，将尝试由语言模型直接回答...")
        # 这种情况可以简化处理，直接让 '管理者' 基于空数据进行回答
        intermediate_steps_result = []
    else:
        # 步骤 4: 创建并执行“执行者 Agent”
        worker_agent = create_tool_calling_agent(model, worker_tools, worker_prompt)
        worker_executor = AgentExecutor(
            agent=worker_agent,
            tools=worker_tools,
            verbose=False,
            return_intermediate_steps=True, # 【核心修改】确保返回中间步骤
            #memory=memory
        )
        
        print("\n🚀 开始执行 [执行者 Agent] 以收集数据...")
        begin_time_llm_response = time.time()
        response = worker_executor.invoke({"input": query, "chat_history": memory.buffer_as_messages})
        end_time_llm_response = time.time()
        print(f"执行者 Agent 运行时间: {end_time_llm_response - begin_time_llm_response:.2f} 秒")

        intermediate_steps_result = response.get("intermediate_steps", [])
        print("\n[原始数据] 执行者收集到的中间步骤:")
        print(intermediate_steps_result)
        formatted_steps = format_intermediate_steps(intermediate_steps_result)
        print("\n[格式化数据] 整理后准备传给管理者的数据:")
        print(formatted_steps)
        # 步骤 5: 执行“管理者/整合者 Chain”
        print("\n🔬 开始执行 [管理者 Chain] 进行数据整合、存储和质检...")
        try:
            # 调用整合链，传入原始问题和中间步骤
            synthesis_result = synthesizer_chain.invoke({
                "query": query,
                "formatted_intermediate_steps": formatted_steps
            })
            # 步骤 6: 处理整合结果
            # 6.1. 将数据存入暂存区
            if synthesis_result.data_to_scratchpad:
                print("\n💾 正在将中间结果存入数据暂存区...")
                for item in synthesis_result.data_to_scratchpad:
                    if 'key' in item and 'value' in item:
                        save_to_scratchpad.invoke(item) # 直接调用工具函数
                    else:
                        print(f"  - 格式错误，跳过存储: {item}")
            
            # 6.2. 将最终答案存入长期缓存（如果需要）
            if synthesis_result.should_cache:
                cache_manager.add(query, synthesis_result.final_answer)
                print("\n✅ 评估通过，高质量回答已存入长期缓存。")
            else:
                print(f"\n❌ 评估未通过，原因: {synthesis_result.reasoning}。此回答将不会被缓存。")

            # 6.3. 向用户展示最终答案
            print("\n[最终答案]:")
            print(synthesis_result.final_answer)
            final_answer_to_display = synthesis_result.final_answer
        except Exception as e:
            print(f"\n⚠️ 管理者 Chain 执行步骤出错: {e}")
            print("将以降级模式处理：直接展示执行者的原始输出（如果存在）。")
            final_answer_to_display = response.get("output", "处理过程中发生错误，无法生成最终答案。") if 'response' in locals() else "处理过程中发生错误，无法生成最终答案。"
            print("\n[原始输出]:")
            print(final_answer_to_display)

        # 【新增】无论成功或失败，都在最后手动保存最终的问答对到记忆中
        if final_answer_to_display:
            memory.save_context({"input": query}, {"output": final_answer_to_display})
            print("✅ 本轮有效问答已存入记忆。")


