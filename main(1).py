import asyncio
import os
import time
from datetime import datetime
from dotenv import load_dotenv

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain.chat_models import init_chat_model
# 导入记忆模块
from langchain.memory import ConversationBufferWindowMemory # 导入滑动窗口记忆
from langchain_core.messages import HumanMessage, AIMessage # 用于手动构建消息，如果需要的话

from all_tool import all_tools, load_dict_from_json # 确保这些是正确的导入路径
print("main.py中datetime是什么类型？", type(datetime), datetime)

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
    如果用户输入不是一个完整的自然语言查询（如“是的”、“再查一次”、“用01810试试看”），请你结合下面的历史对话内容 chat_history，尽力补全用户的真实意图，然后再按照自然语言扩展规则进行扩展查询。
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

model = ChatTongyi(
    model="qwen-plus",  # 或 qwen-max / qwen-turbo 等
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 配合语义分割以适应多API问题
model_split = ChatTongyi(
    model="qwen-turbo",
    model_kwargs={"seed": 45},
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

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
        ("system", f"""你是一个专业的金融助手，当前时间为 {now}。
    你的任务是帮助用户获取与金融相关的数据、解释、分析或回答问题。
    你可以调用一组 API 工具来获取股票、基金、宏观经济等数据。

    请按照以下规则操作：
    1. 如果用户的问题是一个自然语言查询，例如：
    - “什么是市盈率？”
    - “你是谁？”
    - “请解释一下ETF的分类。”
    这类问题不需要调用 API 工具，请你直接用自己的知识回答用户。

    2. 如果用户的问题需要查找数据、获取历史行情、调用接口等（如“请查找工商银行最近一周的收盘价”），你应当使用工具来完成。

    3. 当用户没有说明查询时间范围时，请默认查找最近的数据；如果最近无数据，可以适当往历史数据中查找。

    你的回答应简洁清晰，使用中文输出。"""),

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

    # RAG 和工具检索部分 (这部分仍然是每次查询都执行，因为工具的选择可能依赖于当前查询)
    begin_time_rag_process = time.time()
    splited_query = rag_llm_chain_split.invoke({"init_query": query})
    splited_query_list = split_numbered_items(splited_query)
    mode = 1
    topk = 5
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

    #对于根本不需要API的一些提问 处理方式一：
    
    if not query_tools_instances:
        print("未检索到合适的API工具，正在尝试由语言模型直接回答...")
        # 使用LLM直接回答，绕过 AgentExecutor
        direct_llm_response = model.invoke(query)

        print("\n语言模型直接回答:")
        print(direct_llm_response.content)
        continue  # 跳过后续 agent 执行，进入下一轮对话
    

    # 4. 创建 Agent 和 AgentExecutor
    # AgentExecutor 只需要创建一次，因为它会内部管理记忆
    # 每次调用 invoke 时，它会自动更新记忆并将其传递给Agent
    agent = create_tool_calling_agent(model, query_tools_instances, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=query_tools_instances, # 传入本次查询筛选出的工具
        verbose=True,
        memory=memory # 传入记忆对象
    )

    begin_time_llm_response = time.time()
    # 调用 AgentExecutor，它会自动处理记忆
    response = agent_executor.invoke({"input": query})
    end_time_llm_response = time.time()
    print(f"LLM回答时间: {end_time_llm_response - begin_time_llm_response} 秒")

    print("\nAgent的最终回答:")
    print(response["output"]) # 访问 AgentExecutor 的输出
    # 记忆会自动更新，无需手动添加


