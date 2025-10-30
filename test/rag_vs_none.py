import sys
import os
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取当前脚本所在的目录
current_script_dir = os.path.dirname(current_script_path)

# 获取上一级目录的路径
# os.path.pardir 是 '..' 的别名
parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))

# 将上一级目录添加到 Python 的模块搜索路径中
sys.path.append(parent_dir)
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

# 确保这些是正确的导入路径
from all_tool import all_tools, load_dict_from_json
print("main.py中datetime是什么类型？", type(datetime), datetime)

now = datetime.now()

# 加载api_tool_dic
API_TOOL_dic_path = "../api_dic.json"
API_TOOL_dic = load_dict_from_json(API_TOOL_dic_path)
load_dotenv(override=True)


# 加载FASSI数据库
DashScope_API_KEY = os.getenv("DASHSCOPE_API_KEY")

try:
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=DashScope_API_KEY
    )
    print("DashScopeEmbeddings model initialized successfully.")
except Exception as e:
    print(f"Error initializing DashScopeEmbeddings: {e}")
    exit()  # 如果初始化失败，程序退出

# 加载完整数据库
all_database_path = "../faiss_index"

try:
    all_database = FAISS.load_local(all_database_path, embeddings, allow_dangerous_deserialization=True)
    print(f"FAISS 数据库已从 {all_database_path} 成功加载。")
except Exception as e:
    print(f"加载 FAISS 数据库失败: {e}")
    print("请确保创建database的py文件已运行且数据库文件存在。")
    exit()

# 加载关键词数据库
key_database_path = "../key_faiss_index"

try:
    key_database = FAISS.load_local(key_database_path, embeddings, allow_dangerous_deserialization=True)
    print(f"FAISS 数据库已从 {key_database_path} 成功加载。")
except Exception as e:
    print(f"加载 FAISS 数据库失败: {e}")
    print("请确保创建key_database的py文件已运行且数据库文件存在。")
    exit()

# 加载分类数据库
index_list = ["AM", "CN", "HK", "OT"]
faiss_databases = {}
for index in index_list:

   database_path = f"../{index}_APISPECS_faiss_index"
   try:
       faiss_databases[index] = FAISS.load_local(database_path, embeddings, allow_dangerous_deserialization=True)
       print(f"{index}_FAISS 数据库已从 {database_path} 成功加载。")
   except Exception as e:
       print(f"加载 FAISS 数据库失败: {e}")
       print("请确保创建database的py文件已运行且数据库文件存在。")
       exit()

faiss_key_databases = {}
for index in index_list:

    database_path = f"../{index}_APISPECS_key_faiss_index"
    try:
        faiss_key_databases[index] = FAISS.load_local(database_path, embeddings, allow_dangerous_deserialization=True)
        print(f"{index}_FAISS 数据库已从 {database_path} 成功加载。")
    except Exception as e:
        print(f"加载 FAISS 数据库失败: {e}")
        print("请确保创建database的py文件已运行且数据库文件存在。")
        exit()

key_rag_prompt = PromptTemplate(
    template=
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
)

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
    ("human", "{init_query}")
])

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""你是一个专业的金融助手，当前时间为 {now}。
    你的任务是帮助用户获取与金融相关的数据、解释、分析或回答问题。
    你可以调用一组 API 工具来获取股票、债券、公司财务、宏观经济等数据。

    请按照以下规则操作：
    1. 你应当优先调用 API 工具来获取实时或历史数据，而不是依赖自己的知识编造数据。
    只有当问题属于概念解释、定义说明、通用知识时，才直接回答，不使用工具。

    2. 如果用户的问题是自然语言查询，例如：
    - “什么是市盈率？”
    - “你是谁？”
    - “请解释一下ETF的分类。”
    这类问题不需要调用 API 工具，请直接用自己的知识回答。

    3. 当用户没有说明查询时间范围时，请默认查找最近的数据；如果最近无数据，可以适当往历史数据中查找。

    你的回答应简洁清晰，使用中文输出。"""),

        ("placeholder", "{chat_history}"),   # 保留记忆
        ("human", "{input}"),                # 用户输入
        ("placeholder", "{agent_scratchpad}"),
    ]
)
rag_llm_chain_split = rag_prompt_split | model_split | StrOutputParser()

def split_numbered_items(s: str) -> list[str]:
    # 1. 找到每个“数字. ”的起始位置
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
        # 4. 去除两端的 **
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
        category = cleaned_response[:2].upper()

        # 可以在这里添加一个可选的验证步骤，确保提取的分类是预期的
        expected_categories = {"AM", "CN", "HK", "OT"}
        if category in expected_categories:
            return category
        else:
            return None  # 如果提取的分类不在预期列表中，则视为无效
    else:
        return None  # 如果字符串太短，无法提取分类
    
# 构建加工链
key_rag_llm_chain = key_rag_prompt.partial(current_time=str(datetime.now())) | model | StrOutputParser()
rag_llm_chain = rag_prompt.partial(current_time=str(datetime.now())) | model | StrOutputParser()

import json
import time
from datetime import datetime

# 输入和输出文件路径
rag_test_file_path = "rag_vs_none_test.txt"
output_file_path = f"rag_vs_none_test.json"

# 保存所有测试结果
all_results = []

# 读取并处理测试文件
try:
    with open(rag_test_file_path, 'r', encoding='utf-8') as file:
        print(f"--- 开始读取文件: {rag_test_file_path} 并执行RAG测试 ---")

        # 逐行读取文件，每次读取一行
        while True:
            query_line = file.readline()
            if query_line == "/":
                continue
            if not query_line:
                break
            query = query_line.strip()
            

            # RAG 和工具检索
            begin_time_rag_process = time.time()
            splited_query = rag_llm_chain_split.invoke({"init_query": query})
            splited_query_list = split_numbered_items(splited_query)

            last_query_list = []
            for single_split_query in splited_query_list:
                last_query_list.append(
                    rag_llm_chain.invoke({"init_query": single_split_query})
                )
            print("扩展查询结果:", last_query_list)

            results = []
            for single_last_query in last_query_list:
                query_index = extract_market_category(single_last_query)
                print("query_index:", query_index)
                results.append(
                    faiss_databases[query_index].similarity_search(single_last_query, k=3)
                )
            end_time_rag_process = time.time()
            print(f"扩展和RAG检索执行时间: {end_time_rag_process - begin_time_rag_process} 秒")

            query_tool_names = []
            for single_result in results:
                for doc in single_result:
                    tool_meta_name = doc.metadata.get('name')
                    if tool_meta_name and tool_meta_name in API_TOOL_dic:
                        tool_name = API_TOOL_dic[tool_meta_name]
                        query_tool_names.append(tool_name)

            query_tool_names = list(set(query_tool_names)) # 去重
            print("本次查询检索到的工具名称:", query_tool_names)

            # 筛选出实际检索到的工具实例
            query_tools_instances = [tool for tool in all_tools if tool.name in query_tool_names]

            # query_tools_instances = all_tools

            # 结果记录
            result_entry = {
                "query": query,
                "rag_tools": query_tool_names,
                "rag_time": end_time_rag_process - begin_time_rag_process,
                "llm_time": None,
                "rag_llm_answer": None,
                "normal_llm_answer": None,
                "timestamp": datetime.now().isoformat()
            }

            # 若不使用RAG
            # result_entry = {
            #     "query": query,
            #     "rag_tools": None,
            #     "rag_time": None,
            #     "llm_time": None,
            #     "rag_llm_answer": None,
            #     "normal_llm_answer": None,
            #     "timestamp": datetime.now().isoformat()
            # }

            # 执行LLM回答
            if not query_tools_instances:
                normal_chain = prompt | model | StrOutputParser()
                direct_llm_response = normal_chain.invoke({"input": query})
                result_entry["rag_llm_answer"] = direct_llm_response


            else:
                # 创建 Agent 和 AgentExecutor
                agent = create_tool_calling_agent(model, query_tools_instances, prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=query_tools_instances,
                    verbose=True,
                )

                begin_time_llm_response = time.time()
                response = agent_executor.invoke({"input": query})
                end_time_llm_response = time.time()

                result_entry["llm_time"] = end_time_llm_response - begin_time_llm_response
                result_entry["all_time"] = end_time_llm_response - begin_time_rag_process
                result_entry["rag_llm_answer"] = str(response)
                print(f"LLM回答时间: {result_entry['llm_time']} 秒")


            # 一般LLM
            normal_chain = prompt | model | StrOutputParser()
            response_normal = normal_chain.invoke({"input": query})
            result_entry["normal_llm_answer"] = response_normal

            # 将本轮结果加入总结果
            all_results.append(result_entry)

except FileNotFoundError:
    print(f"错误: 文件 {rag_test_file_path} 未找到")
except Exception as e:
    print(f"处理文件时发生错误: {str(e)}")
finally:
    # 将所有结果写入 JSON 文件
    try:
        with open(output_file_path, "w", encoding="utf-8") as f_out:
            json.dump(all_results, f_out, ensure_ascii=False, indent=2)
        print(f"✅ 测试结果已保存到 {output_file_path}")
    except Exception as e:
        print(f"保存结果到 JSON 文件失败: {str(e)}")

#转化为txt文件
with open("rag_vs_none_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("rag_vs_none_result.txt", "w", encoding="utf-8") as f:
    for item in data:
        query = item.get("query", "")
        
        # rag_llm_answer 可能是 str，需要先转成 dict
        rag_answer = item.get("rag_llm_answer", "")
        if isinstance(rag_answer, str):
            try:
                rag_answer = eval(rag_answer)
                rag_output = rag_answer.get("output", "")
            except:
                rag_output = rag_answer
        else:
            rag_output = rag_answer.get("output", "")
        
        normal_answer = item.get("normal_llm_answer", "")

        f.write("问题: " + query + "\n\n")
        f.write("RAG回答:\n" + rag_output + "\n\n")
        f.write("普通LLM回答:\n" + normal_answer + "\n")
        f.write("=" * 50 + "\n\n")

