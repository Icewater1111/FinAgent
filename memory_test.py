import asyncio
import os
import threading
import time
from datetime import datetime, timedelta
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
    data_to_scratchpad: list[Dict[str, Any]] = Field(
        description=
        "需要写入数据暂存区的原始数据点列表。每个数据点是一个字典，必须包含："
        "\n- 'key': 唯一键名，格式为“实体_属性”。"
        "\n       • 实体使用最能唯一标识该对象的名称，可以是股票代码（AAPL、600519、00700）或常用名称（特斯拉、贵州茅台、腾讯控股）。"
        "\n       • 属性使用简洁的中文短语（如 '股价'、'每股收益'、'营收'）。"
        "\n- 'value': 原始数据点的值（数字、字符串、字典等）。"
        "\n\n"
        "【规则】"
        "\n1. 只保存外部工具返回的“原始数据”，不保存计算结果。"
        "\n2. 一次工具返回中如包含多项可复用数据，应拆成多条 key/value。"
        "\n3. key 必须简洁一致，不包含空格或特殊符号。"
        "\n\n"
        "示例："
        "\n["
        "\n  {'key': 'AAPL_股价', 'value': 172.55},"
        "\n  {'key': '600519_营收', 'value': 1350.2},"
        "\n  {'key': '腾讯控股_净利润', 'value': 159.3}"
        "\n]")
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
        self.threshold = 0.3  # 使用我们之前讨论过的更合理的阈值

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

# ==================== 修改：实现方案三的"数据暂存区" ====================
class DataScratchpad:
    """
    一个具有模糊键名匹配功能的键值存储，用作Agent的短期工作记忆。
    支持实时数据的自动过期清理（每10分钟）。
    """
    def __init__(self, embeddings_model, realtime_ttl_minutes=10):
        self.data: Dict[str, Any] = {}
        self.data_timestamps: Dict[str, datetime] = {}  # 记录每个数据的时间戳
        self.data_types: Dict[str, str] = {}  # 记录数据类型：'realtime' 或 'historical'
        self.embeddings = embeddings_model
        self.key_vectorstore: FAISS | None = None
        self.key_similarity_threshold = 0.45
        self.realtime_ttl_minutes = realtime_ttl_minutes  # 实时数据过期时间（分钟）
        self._cleanup_running = True  # 控制清理线程的运行状态
        
        # 启动后台清理线程
        self._start_cleanup_thread()
        
        print(f"📝 具备模糊匹配功能的数据暂存区 (Scratchpad) 已初始化。")
        print(f"⏰ 实时数据自动清理间隔: {realtime_ttl_minutes} 分钟")
    
    def _normalize_key_for_similarity(self, key: str) -> str:
        """
        用于相似度检索时的 key 归一化：
        - 保留实体部分（第一个下划线前）
        - 属性部分尽量去掉时间段等修饰，只保留指标语义
          例如: META_2025Q3_营收 -> META_营收
        """
        parts = key.split("_")
        if len(parts) <= 1:
            return key  # 没有下划线就直接用原始 key

        entity = parts[0]
        # 剩下的是属性 + 可能包含时间信息
        attr_parts = parts[1:]

        normalized_attr_parts = []
        for p in attr_parts:
            # 过滤掉类似 2025Q3 / 2024Q1 / 2025H1 这种时间标记
            if re.match(r"^\d{4}Q[1-4]$", p):  # 2025Q3 这类
                continue
            if re.match(r"^\d{4}H[1-2]$", p):  # 2025H1 这类
                continue
            if re.match(r"^\d{4}$", p):       # 单独年份
                continue
            normalized_attr_parts.append(p)

        # 如果全部被过滤掉了，就退回原始 key
        if not normalized_attr_parts:
            return key

        normalized_attr = "_".join(normalized_attr_parts)
        return f"{entity}_{normalized_attr}"


    def _is_realtime_by_granularity(self, granularity: str) -> bool:
        """根据数据粒度判断是否为实时数据"""
        if not granularity:
            return False
        granularity_lower = granularity.lower().strip()
        # 实时数据粒度：实时、实时数据
        realtime_granularities = ['实时', '实时数据', 'realtime', 'real-time']
        return granularity_lower in realtime_granularities

    def _save_data(self, key: str, value: Any, data_type: str = None, granularity: str = None):
        """内部保存逻辑，带时间戳和数据类型
        
        Args:
            key: 数据键名
            value: 数据值
            data_type: 数据类型（'realtime' 或 'historical'），如果为 None 则根据 granularity 判断
            granularity: 数据粒度（从扩展查询中提取），用于判断数据类型
        """
        self.data[key] = value
        self.data_timestamps[key] = datetime.now()
        
        # 根据数据粒度判断数据类型，如果未指定
        if data_type is None:
            if granularity:
                data_type = 'realtime' if self._is_realtime_by_granularity(granularity) else 'historical'
            else:
                # 如果没有提供粒度信息，默认使用历史数据（更安全）
                data_type = 'historical'
        self.data_types[key] = data_type
        
        # 更新向量索引
        new_key_doc = Document(page_content=key)
        if self.key_vectorstore is None:
            self.key_vectorstore = FAISS.from_documents([new_key_doc], self.embeddings)
        else:
            self.key_vectorstore.add_documents([new_key_doc])

    def _retrieve_data(self, key: str) -> Any:
        """
        内部检索逻辑，检查数据是否过期（带 key 归一化 + 模糊匹配）。
        命中时返回包含上下文的JSON字符串，未命中则返回提示字符串。
        """
        if self.key_vectorstore is None:
            return "暂存区为空，无法检索。"

        norm_key = self._normalize_key_for_similarity(key)
        results_with_scores = self.key_vectorstore.similarity_search_with_score(norm_key, k=1)

        if results_with_scores and results_with_scores[0][1] < self.key_similarity_threshold:
            most_similar_key_doc, score = results_with_scores[0]
            most_similar_key = most_similar_key_doc.page_content

            if most_similar_key not in self.data:
                return "内部错误：在字典中找不到已匹配的键。"

            if self.data_types.get(most_similar_key) == 'realtime':
                timestamp = self.data_timestamps.get(most_similar_key)
                if timestamp:
                    age = datetime.now() - timestamp
                    if age > timedelta(minutes=self.realtime_ttl_minutes):
                        self._remove_data(most_similar_key)
                        return f"数据已过期: 暂存区中的 '{most_similar_key}' (存在时间超过 {self.realtime_ttl_minutes} 分钟) 已被清理。"

            retrieved_value = self.data[most_similar_key]
            age = datetime.now() - self.data_timestamps.get(most_similar_key, datetime.now())
            age_minutes = age.total_seconds() / 60
            data_type = self.data_types.get(most_similar_key, 'unknown')
            
            # 【核心修改】返回一个包含完整上下文的JSON字符串
            result_payload = {
                "status": "hit",
                "query_key": key,
                "matched_key": most_similar_key,
                "value": retrieved_value,
                "score": float(f"{score:.3f}"),
                "data_type": data_type,
                "age_minutes": float(f"{age_minutes:.1f}")
            }
            print(f"✅ 暂存区模糊检索成功: 命中详情 -> {json.dumps(result_payload, ensure_ascii=False)}")
            # ensure_ascii=False 保证中文字符正常显示
            return json.dumps(result_payload, ensure_ascii=False)

        print(f"❌ 暂存区模糊检索失败: 传入key='{key}'，归一化key='{norm_key}'，未找到相似的已存键名。")
        return "未在暂存区中找到与该键名相关的数据。"


    def _remove_data(self, key: str):
        """删除指定键的数据"""
        if key in self.data:
            del self.data[key]
        if key in self.data_timestamps:
            del self.data_timestamps[key]
        if key in self.data_types:
            del self.data_types[key]
        print(f"🗑️ 已删除过期数据: key='{key}'")

    def _cleanup_expired_realtime_data(self):
        """清理过期的实时数据"""
        now = datetime.now()
        expired_keys = []
        
        for key, data_type in self.data_types.items():
            if data_type == 'realtime':
                timestamp = self.data_timestamps.get(key)
                if timestamp:
                    age = now - timestamp
                    if age > timedelta(minutes=self.realtime_ttl_minutes):
                        expired_keys.append(key)
        
        if expired_keys:
            print(f"\n🧹 开始清理 {len(expired_keys)} 条过期的实时数据...")
            for key in expired_keys:
                self._remove_data(key)
            print(f"✅ 清理完成，已删除 {len(expired_keys)} 条过期数据。当前剩余数据: {len(self.data)} 条")
        else:
            print(f"✅ 暂存区检查完成，无过期数据。当前数据量: {len(self.data)}")

    def _start_cleanup_thread(self):
        """启动后台清理线程，每10分钟执行一次"""
        def cleanup_loop():
            while self._cleanup_running:
                time.sleep(self.realtime_ttl_minutes * 60)  # 等待指定分钟数
                if self._cleanup_running:  # 再次检查状态
                    self._cleanup_expired_realtime_data()
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        print(f"🔄 后台清理线程已启动，每 {self.realtime_ttl_minutes} 分钟清理一次过期数据。")

    def manual_cleanup(self) -> str:
        """手动清理过期的实时数据"""
        before_count = len(self.data)
        self._cleanup_expired_realtime_data()
        after_count = len(self.data)
        removed = before_count - after_count
        return f"手动清理完成，已删除 {removed} 条过期数据。当前剩余数据: {after_count} 条。"

    def get_stats(self) -> dict:
        """获取暂存区统计信息"""
        realtime_count = sum(1 for dt in self.data_types.values() if dt == 'realtime')
        historical_count = sum(1 for dt in self.data_types.values() if dt == 'historical')
        
        return {
            'total': len(self.data),
            'realtime': realtime_count,
            'historical': historical_count,
            'realtime_ttl_minutes': self.realtime_ttl_minutes
        }
# 【修改】实例化暂存区时传入 embeddings
scratchpad = DataScratchpad(embeddings)
class SaveArgs(BaseModel):
    key: str = Field(description=
        "用于存储和检索数据的唯一标识符。必须遵循【实体_属性】格式，例如 'AAPL_股价'、'TSLA_每股收益'。"
        "实体建议优先使用股票代码或统一规范名称。属性应为描述性中文，如 '股价'、'营收'、'总负债'。")
    value: Any = Field(description="要存储的原始数据点，可以是数字、字符串、字典或时间序列。")

class RetrieveArgs(BaseModel):
    key: str = Field(description=
        "要检索的数据标识符。即使不完全精确，系统会尝试模糊匹配，但提供规范化的 key（如 'AAPL_股价'）会得到最准确结果。"
        "你可以先尝试复合指标（如 'AAPL_市盈率'），命中则无需拆解；未命中再尝试基础原料。")

@tool(args_schema=SaveArgs)
def save_to_scratchpad(key: str, value: Any) -> str:
    """【暂存中间原始数据】当你成功通过任意外部工具获取到一个可复用的原始数据点时，应立即保存。
    键名必须遵循 '实体_属性' 格式，如 key='AAPL_股价'。
    暂存区自动支持模糊匹配和即时复用。"""
    scratchpad._save_data(key, value)
    print(f"暂存区保存: key='{key}', value={value}")
    return f"数据点 '{key}' 已成功保存到暂存区。"

@tool(args_schema=RetrieveArgs)
def retrieve_from_scratchpad(key: str) -> Any:
    """
    【API调用前必须执行】在调用任何外部API之前，必须先用此工具从暂存区检索数据。
    
    【重要】此工具的返回格式如下：
    1.  **命中时**: 返回一个包含详细上下文的 JSON 字符串，格式为：
        `{"status": "hit", "query_key": "你的查询键", "matched_key": "实际命中的键", "value": "数据值", ...}`
    2.  **未命中时**: 返回一个提示字符串，如 "未在暂存区中找到..."。

    【你的责任】收到命中结果后，你【必须】检查返回的 `matched_key` 是否与你的 `query_key` 在逻辑上完全等价。
    - **例如**: 你查 `CNY_USA_汇率`，但 `matched_key` 是 `USA_CNY_汇率`。这在语义上相似但逻辑上是倒数关系。你必须识别出这一点，并决定此 `value` 是否可用（可能需要后续计算）。如果不可用，则必须放弃此结果，继续调用外部API。
    - **只有当 `matched_key` 确认无误后**，你才能使用这个 `value` 并终止后续的API调用。
    """
    return scratchpad._retrieve_data(key)

@tool
def clear_expired_scratchpad_data() -> str:
    """【清理实时数据】立即清理暂存区中过期的实时数据，需要获取最新实时数据时可使用。"""
    return scratchpad.manual_cleanup()

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

manage_llm = FixedChatTongyi(
    model="qwen-max", 
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
def extract_granularity_from_expanded_query(expanded_query: str) -> str:
    """
    从扩展查询结果中提取数据粒度信息。
    
    Args:
        expanded_query: 扩展查询的结果字符串，格式如：
            'HK,\n"功能描述": ...,\n"数据类型": ...,\n"数据粒度": 实时,\n"关键词": ...'
    
    Returns:
        str: 提取到的数据粒度（如：'实时'、'日频'、'非时间序列数据'等），如果未找到则返回空字符串
    """
    if not expanded_query:
        return ""
    
    # 尝试匹配 "数据粒度": 值 的模式
    # 支持多种格式：带引号、不带引号、带逗号等
    patterns = [
        r'"数据粒度"\s*:\s*"([^"]+)"',  # "数据粒度": "实时"
        r'"数据粒度"\s*:\s*([^,\n]+)',  # "数据粒度": 实时
        r'数据粒度["\']?\s*[:：]\s*["\']?([^",\n]+)',  # 数据粒度: 实时
    ]
    
    for pattern in patterns:
        match = re.search(pattern, expanded_query, re.IGNORECASE)
        if match:
            granularity = match.group(1).strip()
            # 清理可能的引号和空白
            granularity = granularity.strip('"\'')
            return granularity
    
    return ""
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

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

worker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""当前时间为{now},你是一个严谨的金融数据采集员。你的唯一任务是根据用户问题，规划并获取所有必需的“原始数据点”。你绝对不能进行任何计算、推理或直接回答用户。

你必须严格遵循以下【行动算法】：

============================================================
【第一步：规划】
分析用户问题，拆解出所有需要查询的“数据点”。
例如：查询“苹果公司的市盈率”，你需要规划采集：
1. `AAPL_市盈率` (首选目标)
2. `AAPL_股价` (备用原料)
3. `AAPL_每股收益` (备用原料)

============================================================
【第二步：执行 - 对每个数据点循环执行】
对于规划中的【每一个】数据点，你【必须】严格遵循“先查缓存 -> 再查暂存 -> 最后调API”的顺序。

--- 阶段 A: 查长期缓存 (最高优先级) ---
1.  调用 `ConversationCacheSearch` 工具。
2.  如果命中，获取到完整答案 -> 你的任务【立即结束】，**必须**停止所有后续操作，直接返回已知信息。

--- 阶段 B: 查暂存区 (数据复用) ---
1.  使用【标准键名】调用 `retrieve_from_scratchpad` 工具。
    *   【标准键名规则】: 实体必须是官方代码（如 `AAPL`, `600519`），属性是中文标准名（如 `股价`, `营收`）。
2.  **【强制验证返回结果】**:
    *   **未命中 (返回字符串)**: 直接进入【阶段 C】。
    *   **命中 (返回 JSON 对象)**: 你必须对比你查询的 `query_key` 和它返回的 `matched_key`，然后决策：
        *   **情况1：逻辑等价** (如查 `AAPL_股价` 命中 `Apple_股价`)
            -> **决策**: 数据有效，直接使用。此数据点采集完成,【绝对禁止】再为此数据点调用任何其他工具。
        *   **情况2：逻辑相关但需转换** (如查 `CNY_USD_汇率` 命中 `USD_CNY_汇率`)
            -> **决策**: 数据有效，但需在思考中记录“需进行倒数计算”。此数据点采集完成，【绝对禁止】再为此数据点调用任何其他工具。
        *   **情况3：逻辑无关 (错误匹配)** (如查 `TSLA_股价` 命中 `TSLA_每股收益`)
            -> **决策**: 数据【无效】，必须忽略。将此情况视为【未命中】，立即进入【阶段 C】。

--- 阶段 C: 调用外部 _tool (最后手段) ---
仅在以下情况可调用别的可用的tool：
1.  【阶段 B】中暂存区未命中。
2.  【阶段 B】中暂存区命中但验证为【无效】。

============================================================
【第三步：完成】
当你规划的所有数据点都已通过上述流程成功采集后，你的任务就完成了。立即停止，并将所有采集到的中间步骤信息返回。

"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# 2. 创建管理者 Prompt
manager_prompt = ChatPromptTemplate.from_messages([
    ("system",
          """你是一位顶级的金融分析师和 AI 数据管家。你的职责是接收用户的【原始问题】和执行者采集的【原始数据】，完成最终的分析、整合与数据治理。

============================================================
【输入数据说明】
============================================================
中间数据摘要格式示例：
步骤 1: 工具调用
---工具名称: tool_name
---工具输入:{{...}}
---工具输出:（工具返回的原始字符串）
============================================================

你必须严格按照以下四大流程工作：

============================================================
【第一部分：分析与计算】
1.  **审视原料**: 检查执行者提供的所有【原始数据】。
2.  **执行计算**: 如果问题需要，基于原始数据进行计算。
    *   例如：若需计算市盈率，公式为 市盈率 = 股价 / 每股收益。
3.  **生成核心答案**: 综合所有信息，生成一个专业、清晰、直接回应用户【原始问题】的核心答案。

============================================================
【第二部分：数据治理与暂存 (`data_to_scratchpad`)】
这是你的核心职责。你必须将本次收到的、有价值的【原始数据】整理后存入暂存区。你必须遵循【数据暂存黄金法则】：
*   **法则一：【强制命名标准化】**: 所有存入的 `key` 都必须遵循【实体_属性】格式，实体必须是官方股票代码 (如 `AAPL`, `600519`)。
*   **法则二：【只存原始数据，不存计算结果】**: 绝对禁止保存你自己计算出的复合指标。
*   **法则三：【数据原子化】**: 如遇复杂对象，必须拆分为多条独立的 key-value 对。

============================================================
【第三部分：缓存与决策 (`should_cache`, `reasoning`)】
1.  **缓存决策**: 评估你生成的答案质量，如果准确、完整、高质量，将 `should_cache`设为 `True`。
2.  **生成原因**: 简要说明你的决策。

============================================================
【第四部分：合规性审查 (最终步骤)】
这是你输出前的最后一道关卡。
1.  审查你在【第一部分】生成的核心答案。
2.  **【强制规则】**: 答案内容必须保持客观中立，绝对禁止包含任何形式的投资建议。
3.  **【强制行动】**: 如果答案涉及任何金融投资建议等，你【必须】在答案末尾另起一行，追加标准免责声明："免责声明：以上信息仅供参考，不构成任何投资建议。"
4.  将经过审查和可能追加了免责声明的答案，作为最终的 `final_answer`。

============================================================
【输出格式】
你必须严格按照以下 JSON 格式输出，不得添加任何解释性文字：
{format_instructions}
============================================================
"""),
    ("human", "【原问题】:\n{query}\n\n【中间数据】:\n{formatted_intermediate_steps}")
]).partial(format_instructions=synthesizer_parser.get_format_instructions())
# 3. 组合成完整的管理者/整合者链
synthesizer_chain = manager_prompt | manage_llm | synthesizer_parser

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

def create_memory_instance():
    """创建独立的记忆实例"""
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True
    )

def create_cache_manager():
    """创建独立的缓存管理器实例"""
    return CacheManager(embeddings)

def create_scratchpad():
    """创建独立的数据暂存区实例"""
    return DataScratchpad(embeddings)

def create_sample_tools(cache_manager, scratchpad):
    """为每个样例创建独立的工具集"""
    cache_search_tool = Tool(
        name="ConversationCacheSearch",
        func=cache_manager.search,
        description="【最高优先级，任务起点】在开始任何新任务前，必须首先调用此工具。它用于检查是否存在一个与用户当前问题高度相似的历史问答，并返回其完整的最终答案。如果此工具返回了一个具体的答案，应直接采纳该答案并结束任务。只有当此工具明确返回'未找到'时，才应继续执行其他步骤来从头解决问题。"
    )
    
    # 创建暂存区工具的闭包，绑定到特定的scratchpad实例
    @tool(args_schema=SaveArgs)
    def save_to_scratchpad(key: str, value: Any) -> str:
        """【暂存中间结果】当你通过外部API工具成功获取到一个可复用的原始数据点（例如一个具体的股价、财报数字、指标）后，应立即使用此工具将其存入数据暂存区，以便在当前任务的后续步骤中直接使用。键名(key)应严格遵循'实体_属性'格式，例如 key='特斯拉_股价', value=200。"""
        scratchpad._save_data(key, value)
        print(f"暂存区保存: key='{key}', value={value}")
        return f"数据点 '{key}' 已成功保存到暂存区。"
    
    @tool
    def clear_expired_scratchpad_data() -> str:
        """【手动清理】立即清理暂存区中所有过期的实时数据。当需要确保获取最新数据时，可以使用此工具手动清理过期数据。"""
        return scratchpad.manual_cleanup()
    
    @tool(args_schema=RetrieveArgs)
    def retrieve_from_scratchpad(key: str) -> Any:
        """【API调用前检查】在调用任何外部API（如查询股价、财报的工具）来获取原始数据之前，必须先使用此工具，通过一个描述性的键名（key）来检查工作暂存区中是否已存在所需的数据点。这可以避免不必要的API调用。例如，在需要特斯拉股价时，先用 key='特斯拉_股价' 在此检索。"""
        return scratchpad._retrieve_data(key)
    
    return cache_search_tool, retrieve_from_scratchpad, save_to_scratchpad, clear_expired_scratchpad_data

async def process_single_query(query: str, memory: ConversationBufferWindowMemory, sample_num: int, round_num: int, cache_manager, scratchpad, cache_search_tool, retrieve_from_scratchpad, save_to_scratchpad, clear_expired_scratchpad_data):
    """处理单个查询的完整流程"""
    print(f"\n{'='*50}")
    print(f"样例 {sample_num} - 第{round_num}轮查询: {query}")
    print(f"{'='*50}")
    
    # 记录到输出文件
    with open(memory_test_output_path, 'a', encoding='utf-8') as out_f:
        out_f.write(f"\n{'='*50}\n")
        out_f.write(f"样例 {sample_num} - 第{round_num}轮查询: {query}\n")
        out_f.write(f"{'='*50}\n")
    
    # 查询历史处理
    processed_query = history_query_chain.invoke({"init_query": query, "chat_history": memory.buffer})
    rag_use, query = parse_history_output(processed_query)
    print(f"结合上下文推理后的查询：{query}")
    
    with open(memory_test_output_path, 'a', encoding='utf-8') as out_f:
        out_f.write(f"结合上下文推理后的查询：{query}\n")
    
    if not rag_use:
        print("无需调用工具，由语言模型直接回答...")
        direct_llm_response = chat_chain.invoke({
            "input": query,
            "chat_history": memory.buffer_as_messages
        })
        
        print("\n[直接回答]:", direct_llm_response)
        memory.save_context({"input": query}, {"output": direct_llm_response})
        
        with open(memory_test_output_path, 'a', encoding='utf-8') as out_f:
            out_f.write(f"[直接回答]: {direct_llm_response}\n")
        return
    
    # RAG 和工具检索部分
    begin_time_rag_process = time.time()
    splited_query = rag_llm_chain_split.invoke({"init_query": query})
    splited_query_list = split_numbered_items(splited_query)
    mode = 1
    topk = 3
    
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
    results, query_tool_names, last_query_list = results, query_tool_names, last_query_list
    end_time_rag_process = time.time()
    print(f"扩展和RAG检索执行时间: {end_time_rag_process - begin_time_rag_process} 秒")
    
    # 保存扩展查询结果，供后续使用
    expanded_queries = last_query_list
    
    # 筛选出实际检索到的工具实例
    query_tools_instances = [tool for tool in all_tools if tool.name in query_tool_names]
    worker_tools = [cache_search_tool, retrieve_from_scratchpad] + query_tools_instances
    print(f"✨ 执行者 Agent本轮可用总工具: {[t.name for t in worker_tools]}")
    
    with open(memory_test_output_path, 'a', encoding='utf-8') as out_f:
        out_f.write(f"检索到的工具: {[t.name for t in worker_tools]}\n")

    # 创建并执行 Agent
    if not query_tools_instances:
        print("未检索到相关API工具，将尝试由语言模型直接回答...")
        intermediate_steps_result = []
        final_answer_to_display = "未找到相关工具来回答此问题。"
    else:
        # 创建执行者 Agent
        worker_agent = create_tool_calling_agent(model, worker_tools, worker_prompt)
        worker_executor = AgentExecutor(
            agent=worker_agent,
            tools=worker_tools,
            verbose=False,
            return_intermediate_steps=True,
        )
        
        print("\n🚀 开始执行 [执行者 Agent] 以收集数据...")
        begin_time_llm_response = time.time()
        response = worker_executor.invoke({"input": query, "chat_history": memory.buffer_as_messages})
        end_time_llm_response = time.time()
        print(f"执行者 Agent 运行时间: {end_time_llm_response - begin_time_llm_response:.2f} 秒")

        intermediate_steps_result = response.get("intermediate_steps", [])
        formatted_steps = format_intermediate_steps(intermediate_steps_result)
        
        # 记录worker_agent的思考过程
        with open(memory_test_output_path, 'a', encoding='utf-8') as out_f:
            out_f.write(f"\n[Worker Agent 思考过程]:\n{formatted_steps}\n")
        
        print("\n[格式化数据] 整理后准备传给管理者的数据:")
        print(formatted_steps)
        
        # 执行管理者/整合者 Chain
        print("\n🔬 开始执行 [管理者 Chain] 进行数据整合、存储和质检...")
        try:
            synthesis_result = synthesizer_chain.invoke({
                "query": query,
                "formatted_intermediate_steps": formatted_steps
            })
            
            # 处理整合结果
            if synthesis_result.data_to_scratchpad:
                print("\n💾 正在将中间结果存入数据暂存区...")
                expanded_granularity = None
                if expanded_queries:
                    for expanded_query in expanded_queries:
                        extracted_granularity = extract_granularity_from_expanded_query(expanded_query)
                        if extracted_granularity:
                            expanded_granularity = extracted_granularity
                            break
                
                is_realtime_api = False
                if expanded_granularity:
                    expanded_granularity_lower = expanded_granularity.lower().strip()
                    is_realtime_api = expanded_granularity_lower in ['实时', '实时数据', 'realtime', 'real-time']
                
                for item in synthesis_result.data_to_scratchpad:
                    if 'key' in item and 'value' in item:
                        granularity = item.get('granularity', '')
                        if is_realtime_api:
                            granularity = '实时'
                            print(f"  🔄 检测到实时API，强制标记为实时: {item['key']}")
                        elif not granularity and expanded_granularity:
                            granularity = expanded_granularity
                        
                        scratchpad._save_data(item['key'], item['value'], granularity=granularity)
                        data_type = scratchpad.data_types.get(item['key'], 'unknown')
                        print(f"  ✅ 已保存: key='{item['key']}', type={data_type}, granularity={granularity if granularity else '未指定'}")
            
            # 将最终答案存入长期缓存（如果需要）
            if synthesis_result.should_cache:
                cache_manager.add(query, synthesis_result.final_answer)
                print("\n✅ 评估通过，高质量回答已存入长期缓存。")
            else:
                print(f"\n❌ 评估未通过，原因: {synthesis_result.reasoning}。此回答将不会被缓存。")

            # 展示最终答案
            print("\n[最终答案]:")
            print(synthesis_result.final_answer)
            final_answer_to_display = synthesis_result.final_answer
            
            # 记录最终答案
            with open(memory_test_output_path, 'a', encoding='utf-8') as out_f:
                out_f.write(f"\n[最终答案]:\n{synthesis_result.final_answer}\n")
                
        except Exception as e:
            print(f"\n⚠️ 管理者 Chain 执行步骤出错: {e}")
            final_answer_to_display = response.get("output", "处理过程中发生错误，无法生成最终答案。")
            
            with open(memory_test_output_path, 'a', encoding='utf-8') as out_f:
                out_f.write(f"\n[执行出错]: {e}\n")

    # 保存问答对到记忆中
    if final_answer_to_display:
        memory.save_context({"input": query}, {"output": final_answer_to_display})
        print("✅ 本轮有效问答已存入记忆。")

#输入路径：
memory_test_file_path = "./test/memory_test.txt"
memory_test_output_path = "./test/memory_test_output.txt"

async def main():
    """主异步函数"""
    # 清空输出文件
    with open(memory_test_output_path, 'w', encoding='utf-8') as f:
        f.write("Memory Test Results\n")
        f.write(f"开始时间: {datetime.now()}\n")
    with open(memory_test_file_path, 'r', encoding='utf-8') as f:
        print(f"--- 开始读取文件: {memory_test_file_path} 并执行memory测试 ---")
        #每次循环都先空读一行，也就是每三行作为一个测试单元，第一行为空，第二行是用户第一轮输入，第三行是用户第二轮输入
        #每个样例应该是单独的memory和agent执行环境
        lines = f.readlines()
        
        total_samples = len(lines) // 3
        print(f"总共发现 {total_samples} 个样例")
        
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
                
            sample_num = i // 3 + 1
            
            # 获取三行内容：空行、第一轮输入、第二轮输入
            empty_line = lines[i].strip()
            first_query = lines[i + 1].strip()
            second_query = lines[i + 2].strip()
            
            print(f"\n{'='*60}")
            print(f"开始处理样例 {sample_num}/{total_samples}")
            print(f"第一轮问题: {first_query}")
            print(f"第二轮问题: {second_query}")
            print(f"{'='*60}")
            
            # 为每个样例创建独立的记忆实例、缓存管理器和暂存区
            sample_memory = create_memory_instance()
            sample_cache_manager = create_cache_manager()
            sample_scratchpad = create_scratchpad()
            cache_search_tool, retrieve_from_scratchpad, save_to_scratchpad, clear_expired_scratchpad_data = create_sample_tools(sample_cache_manager, sample_scratchpad)
            
            print(f"🔧 样例 {sample_num} 已创建独立的执行环境:")
            print(f"  - 记忆实例: ConversationBufferWindowMemory")
            print(f"  - 缓存管理器: CacheManager")
            print(f"  - 数据暂存区: DataScratchpad")
            print(f"  - 工具集: {[cache_search_tool.name, retrieve_from_scratchpad.name, save_to_scratchpad.name, clear_expired_scratchpad_data.name]}")
            
            try:
                # 处理第一轮查询
                await process_single_query(first_query, sample_memory, sample_num, 1, sample_cache_manager, sample_scratchpad, cache_search_tool, retrieve_from_scratchpad, save_to_scratchpad, clear_expired_scratchpad_data)
                
                # 等待一下，避免API调用过快
                await asyncio.sleep(2)
                
                # 处理第二轮查询
                await process_single_query(second_query, sample_memory, sample_num, 2, sample_cache_manager, sample_scratchpad, cache_search_tool, retrieve_from_scratchpad, save_to_scratchpad, clear_expired_scratchpad_data)
                
                # 等待一下，避免API调用过快
                await asyncio.sleep(2)
                
                # 显示样例完成后的统计信息
                cache_stats = sample_cache_manager.df.shape[0] if hasattr(sample_cache_manager.df, 'shape') else 0
                scratchpad_stats = sample_scratchpad.get_stats()
                print(f"\n📊 样例 {sample_num} 最终统计:")
                print(f"  - 缓存问答对: {cache_stats}")
                print(f"  - 暂存区数据: {scratchpad_stats['total']} 条 (实时: {scratchpad_stats['realtime']}, 历史: {scratchpad_stats['historical']})")
                
                print(f"\n✅ 样例 {sample_num} 处理完成")
                
                with open(memory_test_output_path, 'a', encoding='utf-8') as out_f:
                    out_f.write(f"\n📊 样例 {sample_num} 最终统计:\n")
                    out_f.write(f"  - 缓存问答对: {cache_stats}\n")
                    out_f.write(f"  - 暂存区数据: {scratchpad_stats['total']} 条 (实时: {scratchpad_stats['realtime']}, 历史: {scratchpad_stats['historical']})\n")
                    out_f.write(f"\n✅ 样例 {sample_num} 处理完成\n")
                    out_f.write(f"{'='*60}\n")
                    
            except Exception as e:
                print(f"❌ 样例 {sample_num} 处理出错: {e}")
                with open(memory_test_output_path, 'a', encoding='utf-8') as out_f:
                    out_f.write(f"\n❌ 样例 {sample_num} 处理出错: {e}\n")
                    out_f.write(f"{'='*60}\n")
                continue

    print(f"\n🎉 所有测试完成！结果已保存到: {memory_test_output_path}")

if __name__ == "__main__":
    asyncio.run(main())
