# 金融API调用助手

## 运行环境：

python == 3.12.11

python-dotenv==1.1.1

gradio==5.43.1

langchain==0.3.26

langgraph==1.0.2

langchain-community==0.3.27

langchain-deepseek==0.1.3

faiss-cpu==1.11.0

dashscope==1.23.8

pandas==2.3.1

numpy==2.3.1

requests==2.32.4

akshare==1.17.85

alpha_vantage==3.0.0

newsapi-python==0.2.7

pywebview==6.0

## 安装方式：

推荐使用conda环境进行安装:

```python
conda create -n rag_model python==3.12.11
conda activate rag_model
```

安装所需要的包:（注意下载顺序一致）

```python
pip install langgraph==1.0.2 
pip install python-dotenv==1.1.1 gradio==5.43.1 langchain==0.3.26 langchain-community==0.3.27 langchain-deepseek==0.1.3 faiss-cpu==1.11.0
pip install dashscope==1.23.8 pandas==2.3.1 numpy==2.3.1 requests==2.32.4 akshare==1.17.85 alpha_vantage==3.0.0 newsapi-python==0.2.7 pywebview==6.0
```

## API-KEY 配置：

在文件根目录中创建.env文件。

在.env文件中配置API-KEY。

```python
DEEPSEEK_API_KEY = your-deepseek-api-key
DASHSCOPE_API_KEY = your-dashscope-api-key
ALPHA_VANTAGE_API_KEY = your-alpha-api-key
NEWS_API_KEY = your-news-api-key
```

## 运行方式：

首先运行all_api_section.py文件

```python
python all_api_section.py
```

其次运行all_tool.py文件

```python
python all_tool.py
```

最后运行main.py文件

```python
python main.py
```

在弹出窗口中使用本工具，或在浏览器http://localhost:7860/中使用本工具。（默认端口为7860，可在main.py文件进行更改）
