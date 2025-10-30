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
#from all_tool import API_TOOL_dic
#from all_api import database
from all_tool import all_tools
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from all_tool import load_dict_from_json
import time
from langchain.agents import AgentExecutor
from datetime import datetime
from langchain_community.chat_models import ChatTongyi

now = datetime.now()

# 加载api_tool_dic
API_TOOL_dic_path = "../api_dic.json"
API_TOOL_dic = load_dict_from_json(API_TOOL_dic_path)

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



rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        '''你是一个API查询助手，精通金融数据API文档的结构和内容，当前金融API文档分为美股市场、中国大陆市场、港股市场和其他四类，你的第一个任务是根据查询，确定其所需要API的分类，不一定完全看公司地区，也要考虑公司在哪里上市，以及跟目前查询最相关的市场
如果属于美股市场，请在回答时最先输出“AM”；如果属于中国大陆市场，请在回答时最先输出“CN”；如果属于港股市场，请在回答时最先输出“HK”；如果属于其他市场与类型，则在回答时最先输出“OT”，之后你的任务是 根据用户的自然语言查询，生成一个**目标API的描述文本**。这个扩展后的查询旨在帮助向量数据库更好地匹配到最相关的API文档。

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
API描述: HK, "功能描述": 查询指定股票在香港股票市场的实时行情数据，包括最新价、涨跌幅、成交量和成交额, "数据类型": 实时行情数据, "数据粒度": 实时, "关键词": 港股,实时行情数据.
---
用户查询: 我要问东财转2 (123098)的历史转股溢价率和纯债溢价率
API描述: CN "功能描述": 获取指定沪深可转债的历史价值分析数据。此接口提供特定可转债在不同日期的收盘价、纯债价值、转股价值、纯债溢价率和转股溢价率等关键指标, "数据类型": 可转债价值分析，历史估值数据，时间序列数据, "数据粒度": 日频，针对单只可转债, "关键词": 可转债, 沪深可转债, 价值分析,纯债溢价率,转股溢价率,历史数据
---
用户查询: 我要问查询比亚迪公司的最新动态和中文新闻资讯，包括新闻标题、来源、发布日期和链接等，用于公司舆情监控和投资决策。
API描述: HK, "功能描述": 查询指定公司的最新中文新闻资讯, 包含新闻标题、来源、发布日期和链接。 "数据类型": 新闻资讯, "数据粒度": 实时, "关键词": 最新动态,市场新闻,金融新闻,新闻情绪,新闻追踪,舆情分析
---
用户查询: 我想看看阿里巴巴过去一年的股价走势
API描述: HK, "功能描述": 查询指定股票在香港股票市场的历史行情数据, 包括开盘价、收盘价、最高价、最低价、成交量、成交额和K线数据。 "数据类型": 历史行情数据, "数据粒度": 月频, "关键词": 港股,过去一年,历史行情数据
---
用户查询: 告诉我小米的基本情况
API描述: CN, "功能描述": 查询指定公司的详细公司资料和基本信息，包括公司名称、注册地、所属行业、董事长和公司介绍。 "数据类型": 公司信息, "数据粒度": 非时间序列数据, "关键词": 中国公司,详细公司资料,基本信息
---
用户查询: 腾讯有什么新闻？
API描述: HK, "功能描述": 查询指定公司的中文新闻资讯，包括新闻标题、来源、发布日期和链接。 "数据类型": 新闻资讯, "数据粒度": 实时数据, "关键词": 市场新闻,金融新闻,新闻情绪,新闻追踪
---
用户查询: 我想了解华为在国际上的报道。
API描述: CN, "功能描述": 查询指定公司在国际媒体上的英文新闻资讯，包括新闻标题、来源、发布日期和链接。 "数据类型": 新闻资讯, "数据粒度": 实时数据, "关键词": 国际媒体,英文新闻资讯,国际舆情分析
---
用户查询: 告诉我关于特斯拉的英文新闻。
API描述: AM, "功能描述": 查询指定公司在国际媒体上的英文新闻资讯，包括新闻标题、来源、发布日期和链接。 "数据类型": 新闻资讯, "数据粒度": 实时数据, "关键词": 国际媒体,英文新闻资讯,国际舆情监控

示例结束 ---

请只返回最终需要寻找的API描述，不要添加任何解释或说明文字。''',
    ),
    ("human", "{init_query}"),
])
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
        expected_categories = {"AM", "CN", "HK", "OT"}
        if category in expected_categories:
            return category
        else:
            return None  # 如果提取的分类不在预期列表中，则视为无效
    else:
        return None  # 如果字符串太短，无法提取分类

#初始化LLM
model_ds = init_chat_model("deepseek-chat", model_provider="deepseek", seed=2025)
model_qw = model = ChatTongyi(model="qwen-max", api_key=DashScope_API_KEY)

#配合语义分割以适应多API问题
model_split = ChatTongyi(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

rag_prompt_split = PromptTemplate(
    template='''请判断下面这个查询查询几项内容，并且只将所查询的内容按照‘1.\n 2.\n’的格式列出，尽可能的让每个单独的查询内容都拥有准确的时间空间信息，注意我们只保留对金融部分的查询。

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
分割查询: ''',
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

#构建加工链
rag_llm_chain = rag_prompt.partial(current_time=str(datetime.now())) | model_ds | StrOutputParser()

#从.txt文件中获得测试查询
rag_test_file_path = "rag_test_data2.txt"

# 创建一个用于存储所有测试结果的列表
all_test_results = []
#读取文件
try:
    with open(rag_test_file_path, 'r', encoding='utf-8') as file:
        print(f"--- 开始读取文件: {rag_test_file_path} 并执行RAG测试 ---")

        for line_number, query_from_file in enumerate(file, 1):
            query = query_from_file.strip()
            if not query: #跳过空行
                continue

            print(f"\n--- 处理第 {line_number} 个查询: '{query}' ---")
            # 结果字典，包含检索到的tool，以及这些tool的介绍，完善查询所用的时间，检索所用的时间，LLM回答的时间，LLM回答的结果
            current_query_results = {
                "query_id": line_number,
                "query": query,
                "refined_query": None,
                "query_refine_time_ms": 0,
                "retrieval_time_ms": 0,
                "retrieved_tools": [], # 存储工具名称和描述
                "llm_response_time_ms": 0,
                "llm_answer": "N/A"
            }
            # 采用语义分割
            '''
            begin_time_refine = time.time()

            splited_query = rag_llm_chain_split.invoke({"init_query": query})
            splited_query_list = split_numbered_items(splited_query)  # 拆分后的子查询

            last_query_list = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_query = {
                    executor.submit(rag_llm_chain.invoke, {"init_query": sq}): sq
                    for sq in splited_query_list
                }
                for future in as_completed(future_to_query):
                    try:
                        result = future.result()
                        last_query_list.append(result)
                    except Exception as e:
                        print(f"子查询处理失败: {future_to_query[future]}, 错误: {e}")

            end_time_refine = time.time()
            refine_query_time = (end_time_refine - begin_time_refine) * 1000
            current_query_results["refined_query"] = last_query_list
            current_query_results["query_refine_time_ms"] = refine_query_time


            # 并发执行 RAG 检索
            begin_time_retrieval = time.time()
            results = []

            def retrieval_task(single_last_query):
                last_query = rag_llm_chain.invoke({"init_query": single_last_query})
                query_index = extract_market_category(last_query) or "OT"
                return faiss_databases[query_index].similarity_search_with_score(last_query, k=3)

            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_query = {
                    executor.submit(retrieval_task, q): q for q in last_query_list
                }
                for future in as_completed(future_to_query):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"检索失败: {e}")

            end_time_retrieval = time.time()
            retrieval_time = (end_time_retrieval - begin_time_retrieval) * 1000
            current_query_results["retrieval_time_ms"] = retrieval_time

            query_tool_names = [] #应该有多个，此处暂时用最简单的方式，将多个语义的API直接叠加
            retrieved_tool_details = [] # 存储检索到的工具的名称和描述
            for single_result in results:
                for doc, score in single_result:
                    # 从数据库的metadata中获取工具名称
                    tool_meta_name = doc.metadata.get('name')
                    if tool_meta_name and tool_meta_name in API_TOOL_dic:
                        tool_name = API_TOOL_dic[tool_meta_name]
                        query_tool_names.append(tool_name)
                        # 查找对应的Tool对象以获取描述
                        found_tool = next((tool for tool in all_tools if tool.name == tool_name), None)
                        if found_tool:
                            retrieved_tool_details.append({
                                "name": found_tool.name,
                                "description": found_tool.description
                            })
                        else:
                            print(f"  警告: 检索到工具 '{tool_name}' 但未在 all_tools 中找到其完整详情。")
                    else:
                        print(f"  警告: 检索到未知元数据名称: {tool_meta_name}")

            query_tool_names = list(set(query_tool_names)) #去重
            current_query_results["retrieved_tools"] = retrieved_tool_details
            '''
            #不采用语义分割
            begin_time_refine = time.time()
            last_query = rag_llm_chain.invoke({"init_query": query})
            end_time_refine = time.time()
            refine_query_time = (end_time_refine - begin_time_refine) * 1000
            current_query_results["refined_query"] = last_query
            current_query_results["query_refine_time_ms"] = refine_query_time
            begin_time_retrieval = time.time()
            query_index = extract_market_category(last_query) or "OT"
            results = faiss_databases[query_index].similarity_search_with_score(last_query, k=6)
            end_time_retrieval = time.time()
            retrieval_time = (end_time_retrieval - begin_time_retrieval) * 1000
            current_query_results["retrieval_time_ms"] = retrieval_time
            query_tool_names = []
            retrieved_tool_details = [] # 存储检索到的工具的名称和描述
            for doc, score in results:
                # 从数据库的metadata中获取工具名称
                tool_meta_name = doc.metadata.get('name')
                if tool_meta_name and tool_meta_name in API_TOOL_dic:
                    tool_name = API_TOOL_dic[tool_meta_name]
                    query_tool_names.append(tool_name)
                    
                    # 查找对应的Tool对象以获取描述
                    found_tool = next((tool for tool in all_tools if tool.name == tool_name), None)
                    if found_tool:
                        retrieved_tool_details.append({
                            "name": found_tool.name,
                            "description": found_tool.description
                        })
                    else:
                        print(f"  警告: 检索到工具 '{tool_name}' 但未在 all_tools 中找到其完整详情。")
                else:
                    print(f"  警告: 检索到未知元数据名称: {tool_meta_name}")
            current_query_results["retrieved_tools"] = retrieved_tool_details



            # 准备Agent并执行
            # 筛选出实际检索到的工具实例
            query_tools_instances = [tool for tool in all_tools if tool.name in query_tool_names]

            # 构建提示模版
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""你是一个专业的金融助手，当前时间为 {now}。\n你的任务是帮助用户获取与金融相关的数据、解释、分析或回答问题。\n你可以调用一组 API 工具来获取股票、债券、公司财务、宏观经济等数据。\n请按照以下规则操作：\n1. 你应当优先调用 API 工具来获取实时或历史数据，而不是依赖自己的知识编造数据。 只有当问题属于概念解释、定义说明、通用知识时，才直接回答，不使用工具。\n2. 如果用户的问题是自然语言查询，例如： - “什么是市盈率？” - “你是谁？” - “请解释一下ETF的分类。” 这类问题不需要调用 API 工具，请直接用自己的知识回答。\n3. 当用户没有说明查询时间范围时，请默认查找最近的数据；如果最近无数据，可以适当往历史数据中查找。\n你的回答应简洁清晰，使用中文输出。""",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            agent = create_tool_calling_agent(model_qw, query_tools_instances, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=query_tools_instances, verbose=False)

            begin_time_llm = time.time()
            # input 使用原始查询 'query'
            response = agent_executor.invoke({"input": query})
            end_time_llm = time.time()
            llm_response_time = (end_time_llm - begin_time_llm) * 1000 # 转换为毫秒

            current_query_results["llm_response_time_ms"] = llm_response_time
            current_query_results["llm_answer"] = response.get('output', 'N/A')

            all_test_results.append(current_query_results)
        print("\n--- 文件读取完毕，所有查询处理完成 ---")

except FileNotFoundError:
    print(f"错误：文件 '{rag_test_file_path}' 未找到。请检查文件路径是否正确。")
except Exception as e:
    print(f"读取文件时发生未知错误：{e}")


output_json_file = "rag_test_results2_nosplit_append.json"
output_report_file = "rag_test_report2_nosplit_append.txt"
import json

# 将结果输出到 JSON 文件
print(f"\n--- 将结果保存到JSON文件: {output_json_file} ---")
try:
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(all_test_results, f, ensure_ascii=False, indent=4)
    print(f"JSON结果已成功保存到 {output_json_file}")
except Exception as e:
    print(f"保存JSON文件时发生错误: {e}")

# 将结果输出到格式化报告文件
print(f"--- 将结果保存到文本报告文件: {output_report_file} ---")
try:
    with open(output_report_file, 'w', encoding='utf-8') as f:
        f.write(f"RAG系统测试报告 - 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(all_test_results):
            f.write(f"查询 {result['query_id']}:\n")
            f.write(f"  原始问题: '{result['query']}'\n")
            f.write(f"  完善后的查询: '{result['refined_query']}'\n")
            f.write(f"  查询完善耗时: {result['query_refine_time_ms']:.2f} ms\n")
            f.write(f"  工具检索耗时: {result['retrieval_time_ms']:.2f} ms\n")
            
            f.write(f"  检索到的工具 ({len(result['retrieved_tools'])}个):\n")
            if result['retrieved_tools']:
                for tool in result['retrieved_tools']:
                    f.write(f"    - 名称: '{tool['name']}'\n")
                    f.write(f"      描述: '{tool['description']}'\n")
            else:
                f.write("    无工具检索到。\n")
            
            f.write(f"  LLM回答耗时: {result['llm_response_time_ms']:.2f} ms\n")
            f.write(f"  LLM最终回答:\n")
            # 为多行回答添加缩进
            for line in result['llm_answer'].splitlines():
                f.write(f"    {line}\n")
            f.write("-" * 70 + "\n\n")
    
    print(f"文本报告已成功保存到 {output_report_file}")
except Exception as e:
    print(f"保存文本报告文件时发生错误: {e}")

print("\n--- 所有输出操作完成 ---")