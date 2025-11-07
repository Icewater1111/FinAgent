import asyncio
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import json
from typing import Any, Dict
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
# 新增质检员的相关内容 为结构化输出定义 Pydantic 模型
class AnswerEvaluation(BaseModel):
    """用于评估和修正Agent回答的数据模型。"""
    is_valid_answer: bool = Field(description="回答是否是一个有效的、信息丰富的答案，而不是错误报告或空洞的回复。")
    should_cache: bool = Field(description="综合判断，这个回答是否质量高到值得存入缓存，供未来直接使用。")
    revised_answer: str = Field(description="经过合规性审查和修正后的最终回答文本，将直接展示给用户。")
    reasoning: str = Field(description="对以上判断的简要说明，便于调试。")
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
        self.threshold = 0.2  # 使用我们之前讨论过的更合理的阈值

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
        self.key_similarity_threshold = 0.2
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
    key: str = Field(description="要检索的数据的唯一标识符。即使不完全确定键名，系统也会尝试模糊匹配。提供一个规范的键名（如'特斯拉_股价'）会得到最准确的结果。")


# 创建暂存区工具 (保持不变)
@tool(args_schema=SaveArgs)
def save_to_scratchpad(key: str, value: Any) -> str:
    """【暂存中间结果】当你通过外部API工具成功获取到一个可复用的原始数据点（例如一个具体的股价、财报数字、指标）后，应立即使用此工具将其存入工作暂存区，以便在当前任务的后续步骤中直接使用。键名(key)应严格遵循'实体_属性'格式，例如 key='特斯拉_股价', value=200。"""
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
pydantic_parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)

# 2. 创建质检 Prompt
evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个资深的AI回答质检员，负责审查金融领域AI助手的回答。
你的任务是根据用户的【原问题】和AI助手的【待评估的回答】，从【有效性】和【合规性】两个维度进行评估，并输出一个JSON对象。

**评估标准:**

1.  **有效性判断 (is_valid_answer):**
    - **有效回答 (True):** 提供了具体数据、分析、计算结果或相关信息，成功地回应了用户问题。
    - **无效回答 (False):** 包含明确的错误信息（如“网络连接失败”、“API调用出错”、“超时”），或实质为空的回答（如“查询无结果”、“找不到相关信息”、“我无法回答这个问题”）。

2.  **合规性审查与修正 (revised_answer):**
    - **禁止投资建议:** 任何暗示买卖、推荐具体股票或带有强烈主观倾向的判断，都必须改写为中立客观的陈述。如果无法避免，必须在回答末尾添加免责声明。
    - **添加免责声明:** 如果回答涉及预测、估值或看起来像投资建议，请在末尾统一追加：\n\n【免责声明：以上内容仅供参考，不构成任何投资建议。】
    - **保持中立:** 移除任何带有感情色彩或夸张的词汇。
    - **内容安全:** 确保回答不包含任何违法、不道德或攻击性的内容。

3.  **缓存决策 (should_cache):**
    - 只有当 `is_valid_answer` 为 `True` 且回答内容详实、合理时，`should_cache` 才应为 `True`。
    - 对于无效回答或质量较低的回答，`should_cache` 必须为 `False`。

**输出格式:**
你必须严格按照以下JSON格式进行输出，不要添加任何其他解释性文字。
{format_instructions}
"""),
    ("human", "【原问题】:\n{query}\n\n【待评估的回答】:\n{answer}")
]).partial(format_instructions=pydantic_parser.get_format_instructions())

# 3. 组合成完整的评估链
evaluation_chain = evaluation_prompt | judge_llm | pydantic_parser
history_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """你是一个金融问题理解助手。你的任务是：根据智能体与用户的对话历史以及当前用户的输入，准确判断并整理出用户当前的查询需求及相关分析需求，输出重新组织用户需求后的完整查询内容。
     注意：禁止对用户的查询或提问进行任何形式的解释、分析、推理或回答。禁止直接回应用户的需求。输出必须是一个问题或查询。
     输出中不得包含任何额外内容（例如提示语、解释性文字或结论）"""),
    ("placeholder", "{chat_history}"),
    ("human", "{init_query}")
])
history_query_chain = history_prompt | model | StrOutputParser()

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
    
# 构建加工链
key_rag_llm_chain = key_rag_prompt.partial(current_time=str(datetime.now())) | model | StrOutputParser()
rag_llm_chain = rag_prompt.partial(current_time=str(datetime.now())) | model | StrOutputParser()


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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""你是一个高效、智能的顶级金融分析助手，当前时间为 {now}。
你的核心任务是利用可用工具，准确、快速地回答用户的金融问题。

为了最高效地完成任务，你必须理解并区分两种可用的记忆辅助工具：
- **对话缓存 (ConversationCacheSearch)**: 这是你的长期记忆。它存储了过去已经完整回答过且回答良好的问题和答案。
- **工作暂存区 (retrieve/save_from_scratchpad)**: 这是你的短期草稿纸。它存储了在解决当前问题过程中获取到的原始数据点（如股价、市盈率等）。

你必须严格遵守以下【四步工作流程】：

**第一步：检查长期记忆（对话缓存）**
   - 永远首先调用 `ConversationCacheSearch` 工具，检查是否已经回答过完全相同或非常相似的问题。
   - 如果找到答案，直接返回该答案，任务结束。

**第二步：规划并执行任务**
   - 如果缓存未命中，你需要分析用户的问题，规划如何获取数据来构建答案。
   - 在调用任何外部API工具（例如，`get_stock_price`, `get_financial_reports`等）获取一个具体的数据点之前，必须先调用 `retrieve_from_scratchpad` 工具，检查你的“草稿纸”上是否已经有这个数据。
   - 如果暂存区没有，才去调用相应的外部API工具获取数据。

**第三步：记录到草稿纸（工作暂存区）**
   - 每当你通过外部API工具成功获得一个原始数据点（数字、文本等），必须立即调用 `save_to_scratchpad` 工具，将其记录到你的“草稿纸”上。这对于需要多个数据进行计算的复杂问题至关重要。

**第四步：综合并给出最终答案**
   - 当你收集齐所有需要的数据后，进行最终的计算、分析或整理，并向用户提供清晰、完整的答案。

---
**其他重要规则:**
- **直接回答**: 如果用户的问题不涉及数据查询，而是概念解释（如“什么是市盈率？”）、打招呼或一般性对话，请直接用你的知识回答，不要使用工具。
- **多步任务**: 如果一个问题需要多个数据点（例如“比较苹果和微软的市盈率”），请严格遵循上述流程，分别为苹果和微软执行“检查暂存区 -> 调用API -> 保存到暂存区”的步骤。
- **默认时间**: 当用户没有说明查询时间范围时，请默认查找最近的数据。
- **响应完整性**: 确保你的最终回答响应了用户问题的所有部分。
"""),

        ("placeholder", "{chat_history}"),   # 保留记忆
        ("human", "{input}"),                # 用户输入
        ("placeholder", "{agent_scratchpad}"),
    ]
)

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
    print(f"结合上下文推理后的查询：{query}")
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
    final_tools = [cache_search_tool, save_to_scratchpad, retrieve_from_scratchpad] + query_tools_instances
    print(f"✨ Agent本轮可用总工具: {[t.name for t in final_tools]}")

    # 4. 创建并执行 Agent
    # 如果一个工具都没有（既没有API工具，也没有缓存工具），则直接回答
    if not final_tools:
        print("无可用工具，正在尝试由语言模型直接回答...")
        direct_llm_response = model.invoke(query)
        raw_final_answer = direct_llm_response.content
        print("\n[直接回答]:", raw_final_answer)
    else:
        agent = create_tool_calling_agent(model, final_tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=final_tools, # 传入本轮所有可用工具
            verbose=True,
            memory=memory # 传入记忆对象
        )
        begin_time_llm_response = time.time()
        response = agent_executor.invoke({"input": query})
        end_time_llm_response = time.time()
        print(f"LLM回答时间: {end_time_llm_response - begin_time_llm_response:.2f} 秒")
        raw_final_answer = response["output"]
        print("\n[待审查]Agent的最终回答:")
        print(raw_final_answer)

        # ==================== 新增：回答评估与修正模块 ====================
        print("\n🔬 正在对回答进行质检和合规性审查...")
        try:
            evaluation_result = evaluation_chain.invoke({
                "query": query,
                "answer": raw_final_answer
            })

            # 使用经过修正的最终答案
            final_answer_to_display = evaluation_result.revised_answer

            print("\n[最终] 修正后的回答:")
            print(final_answer_to_display)

            # 根据质检结果，有条件地存入缓存
            if evaluation_result.should_cache:
                # 注意：我们应该将修正后的、高质量的答案存入缓存
                cache_manager.add(query, final_answer_to_display)
                print("\n✅ 评估通过，高质量回答已存入缓存。")
            else:
                print(f"\n❌ 评估未通过，原因: {evaluation_result.reasoning}。此回答将不会被缓存。")
        
        except Exception as e:
            print(f"\n⚠️ 回答评估步骤出错: {e}")
            print("将以降级模式处理：直接展示原始回答，且不进行缓存。")
            final_answer_to_display = raw_final_answer
            print("\n[原始] Agent的回答:")
            print(final_answer_to_display)
        # ========================== 模块结束 ============================

