import os
import pandas as pd
import numpy as np
import sys
import requests
import json
import csv
import re
import akshare as ak
import datetime as dt
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type, Literal
from langchain.tools import tool
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.alphaintelligence import AlphaIntelligence
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from dotenv import load_dotenv
from newsapi import NewsApiClient

# 检查是否在 PyInstaller 打包环境中运行
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # 如果是打包环境，使用 _MEIPASS 作为基础路径
    base_path = sys._MEIPASS
else:
    # 如果是开发环境，使用当前文件所在的目录作为基础路径
    base_path = os.path.dirname(__file__)

dotenv_path = os.path.join(base_path, '.env')

# 显式加载 .env 文件
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"成功从内部加载 .env 文件: {dotenv_path}")
else:
    # 如果内部路径找不到，作为备用，尝试从当前工作目录加载 (开发环境可能需要)
    print(f"内部 .env 文件未找到: {dotenv_path}，尝试从当前工作目录加载...")
    # 尝试默认加载方式
    load_dotenv(override=True)

Alpha_Vantage_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
News_API_KEY = os.getenv("NEWS_API_KEY")
newsapi = NewsApiClient(api_key=News_API_KEY)
ts = TimeSeries(key=Alpha_Vantage_API_KEY, output_format='pandas')
fd = FundamentalData(key=Alpha_Vantage_API_KEY, output_format="pandas")
ns = AlphaIntelligence(key=Alpha_Vantage_API_KEY, output_format='pandas')
cc = CryptoCurrencies(key=Alpha_Vantage_API_KEY, output_format='pandas')
MAX_ROWS_TO_DISPLAY = 10

@tool
def get_intraday_time_series(
    symbol: str,
    interval: str,
    adjusted: Optional[str] = None,
    extended_hours: Optional[str] = None,
    month: Optional[str] = None,
    outputsize: Optional[str] = None,
    datatype: Optional[str] = None,
    hours_ago: Optional[int] = None
) -> str:
    """
    这个 Alpha Vantage 的 API 接口能让你获取指定股票的盘中（一天之内）的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据。
    它不仅提供当前数据，还能追溯到 20 多年前的历史数据。
    这个数据涵盖了包括盘前和盘后交易时段在内的所有交易时间（例如，美国市场从凌晨 4 点到晚上 8 点）。
    你既可以获取未经任何调整的原始交易数据，也可以获取经过股票拆分和股息派发调整后的数据，以便更好地进行历史分析。
    在金融领域，这种 OHLCV 数据通常也被称为“蜡烛图”数据。

    Args:
        symbol (str): 股票代码，例如：IBM。
        interval (str): 时间序列中两个连续数据点之间的时间间隔。支持的值有：'1min', '5min', '15min', '30min', '60min'。
        adjusted (Optional[str]): 是否返回经过历史拆分和股息事件调整后的值。设置为 'true' 返回调整后的值（默认），设置为 'false' 返回原始（交易时）盘中值。
        extended_hours (Optional[str]): 是否包含盘前和盘后交易时间（例如，美国市场东部时间凌晨 4:00 至晚上 8:00）。设置为 'true' 包含（默认），设置为 'false' 仅查询常规交易时间（美国东部时间上午 9:30 至下午 4:00）。
        month (Optional[str]): 默认情况下不设置此参数，API 将返回最近几天的盘中数据。你可以使用此参数（YYYY-MM 格式）查询历史上的特定月份，例如 '2009-01'。支持 2000 年 1 月以来 20 多年中的任何月份。
        outputsize (Optional[str]): 默认情况下，'compact' 返回盘中时间序列中最新的 100 个数据点；'full' 返回最近 30 天的盘中数据（如果未指定 'month' 参数），或者如果指定了 'month' 参数，则返回特定月份的完整盘中数据。建议使用 'compact' 选项以减少每次 API 调用的数据大小。
        datatype (Optional[str]): 返回数据的格式，'json' 或 'csv'。默认为 'json'。
        hours_ago (Optional[int]): 指定返回从最新可用数据点时间起，过去多少小时内的所有数据。例如，设置为 24 将返回从最新数据点倒推 24 小时的数据。
                                    重要提示 Alpha Vantage API 的 `outputsize='compact'` 模式仅返回最新的 100 个数据点。如果 `hours_ago` 指定的时间范围超出了这 100 个数据点所能覆盖的时间（例如，对于 '1min' 间隔，100 个点只能覆盖约 1.67 小时），则可能无法获取到完整的数据。在这种情况下，请将 `outputsize` 参数明确设置为 `'full'` 以获取更长时间范围的数据。

    Returns:
        str: 盘中 OHLCV（开盘价、最高价、最低价、收盘价、成交量）数据，通常为最近的几条数据或指定时间段内的数据以供简洁展示。
    """
    api_params = {
        "symbol": symbol,
        "interval": interval,
        "adjusted": adjusted,
        "extended_hours": extended_hours,
        "month": month,
        "outputsize": outputsize, 
        "datatype": datatype
    }
    
    # 移除字典中值为 None 的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = ts.get_intraday(**filtered_params)

        if data.empty:
            return f"无法获取股票 {symbol} 在 {interval} 间隔下的盘中数据。可能是无效的股票代码、时间间隔，或没有可用数据。"
        
        display_data = data

        # 处理hours_ago参数
        if hours_ago is not None and hours_ago > 0:
            # 获取数据集中最新的时间点
            latest_data_time = data.index[0] 
            cutoff_time = latest_data_time - timedelta(hours=hours_ago)
            
            display_data = data[data.index >= cutoff_time]

            if display_data.empty:
                return f"股票 {symbol.upper()} 在最新数据点（{latest_data_time.strftime('%Y-%m-%d %H:%M:%S')}）倒推 {hours_ago} 小时内没有 {interval} 间隔的盘中数据。请检查 `outputsize` 参数是否设置为 'full' 以获取更长时间范围的数据。"
            
            return f"股票 {symbol.upper()} 的盘中时间序列 (间隔: {interval}, 从 {latest_data_time.strftime('%Y-%m-%d %H:%M:%S')} 倒推 {hours_ago} 小时内的数据):\n{display_data.to_string()}"
        
        # 如果hours_ago未设置或无效，则返回最新5条数据
        else:
            display_data = data.head(10)
            return f"股票 {symbol.upper()} 的盘中时间序列 (间隔: {interval}, 最新 5 条数据):\n{display_data.to_string()}"

    except Exception as e:
        return f"获取股票 {symbol} 盘中数据时发生错误: {e}"
    
@tool
def get_daily_time_series(
    symbol: str,
    outputsize: Optional[str] = None,
    datatype: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """
    这个 Alpha Vantage 的 API 返回指定全球股票的原始（按交易数据）日频时间序列数据，
    包含日期、每日开盘价、最高价、最低价、收盘价、成交量等信息，覆盖超过 20 年的历史数据。
    在金融文献中，这种 OHLCV 数据（开盘 Open、最高 High、最低 Low、收盘 Close、成交量 Volume）有时也被称为“蜡烛图数据”。
    数据覆盖时间：只覆盖交易日的收盘数据（不包含盘前盘后）。
    适用场景：中长期回测、技术分析。

    Args:
        symbol (str): 股票代码，例如：IBM。
        outputsize (Optional[str]): 返回数据大小。默认为只返回最新 100 条数据（'compact'）。可选 'compact' 或 'full'，建议使用 'compact' 选项以减少每次 API 调用的数据大小。。
        datatype (Optional[str]): 返回数据格式，'json' 或 'csv'。默认为 'json'。
        date (Optional[str]): 指定要查询的单个日期，格式为 'YYYY-MM-DD'。例如：'2025-08-05'。如果设置此参数，将返回该日期的 OHLCV 数据。
                              **重要提示：** 如果未显式设置 `outputsize='full'`，且查询的 `date` 不在 Alpha Vantage API 默认 `compact` 模式（最新 100 个交易日）返回的数据范围内，则可能无法找到该日期的数据。如需查询更早的历史日期，请务必将 `outputsize` 参数设置为 `'full'`。

    Returns:
        str: 股票日频 OHLCV 数据的字符串表示，通常返回最新部分数据摘要或指定日期的数据。
    """
    api_params = {
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": datatype,
    }

    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = ts.get_daily(**filtered_params)

        if data.empty:
            return f"无法获取股票 {symbol} 的日线数据，可能是无效的股票代码或暂无数据。"

        # 处理date参数
        if date is not None:
            try:
                # 将查询日期字符串转换为datetime对象，并标准化
                query_datetime = pd.to_datetime(date).normalize()
                
                # 过滤DataFrame，查找匹配的日期
                daily_data = data[data.index.normalize() == query_datetime]
                
                if not daily_data.empty:
                    # 返回指定日期的数据
                    return f"股票 {symbol.upper()} 在 {date} 的日线数据:\n{daily_data.to_string()}"
                else:
                    return (
                        f"无法找到股票 {symbol.upper()} 在 {date} 的日线数据。该日期可能不是交易日，"
                        f"或当前 `outputsize` 设置（{outputsize if outputsize else 'compact'}）无法覆盖该历史范围。 "
                        f"如需查询更早日期，请尝试将 `outputsize` 参数设置为 'full'。"
                    )
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM-DD' 格式。"
        else:
            # 如果没有指定date，则默认返回最新5条数据
            return f"股票 {symbol.upper()} 的日线时间序列 (最新 5 条数据):\n{data.head(5).to_string()}"
    except Exception as e:
        return f"获取股票 {symbol} 日线数据时发生错误: {e}"
    
@tool
def get_daily_adjusted_time_series(
    symbol: str,
    outputsize: Optional[str] = None,
    datatype: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """
    这个 Alpha Vantage 的 API 接口能让你获取指定全球股票的日频开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据。
    其最显著的特点是提供了经过股票拆分和股息派发事件调整后的收盘价 (adjusted close values)。
    这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。
    同时，它也包含了原始（按交易数据）的 OHLCV 值，并能追溯到 20 多年的历史数据。
    在金融领域，这种 OHLCV 数据通常也被称为“蜡烛图”数据。

    Args:
        symbol (str): 您选择的股票名称。例如：IBM。
        outputsize (Optional[str]): 返回数据大小。默认情况下，'compact' 只返回最新的 100 个数据点；'full' 返回 20 多年历史数据的完整时间序列。建议使用 'compact' 选项以减少每次 API 调用的数据大小。
        datatype (Optional[str]): 返回数据的格式，'json' 或 'csv'。默认情况下，'json'。
        date (Optional[str]): 指定要查询的单个日期，格式为 'YYYY-MM-DD'。例如：'2025-08-05'。如果设置此参数，将返回该日期的 OHLCV 数据。
                              **重要提示：** 如果未显式设置 `outputsize='full'`，且查询的 `date` 不在 Alpha Vantage API 默认 `compact` 模式（最新 100 个交易日）返回的数据范围内，则可能无法找到该日期的数据。如需查询更早的历史日期，请务必将 `outputsize` 参数设置为 `'full'`。

    Returns:
        str: 日频 OHLCV（开盘价、最高价、最低价、收盘价、成交量）数据，特别包含经过调整的收盘价和历史拆分/股息事件的详细信息。通常返回最新部分数据摘要或指定日期的数据。
    """

    api_params = {
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": datatype,
    }

    # 移除字典中值为None的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = ts.get_daily_adjusted(**filtered_params)

        if data.empty:
            return f"无法获取股票 {symbol} 的日线调整数据，可能是无效的股票代码或暂无数据。"

        # 处理date参数
        if date is not None:
            try:
                # 将查询日期字符串转换为 datetime 对象，并标准化
                query_datetime = pd.to_datetime(date).normalize()
                
                # 过滤DataFrame查找匹配的日期
                daily_data = data[data.index.normalize() == query_datetime]
                
                if not daily_data.empty:
                    # 返回指定日期的数据
                    return f"股票 {symbol.upper()} 在 {date} 的日线调整数据:\n{daily_data.to_string()}"
                else:
                    return (
                        f"无法找到股票 {symbol.upper()} 在 {date} 的日线调整数据。该日期可能不是交易日，"
                        f"或当前 `outputsize` 设置（{outputsize if outputsize else 'compact'}）无法覆盖该历史范围。 "
                        f"如需查询更早日期，请尝试将 `outputsize` 参数设置为 'full'。"
                    )
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM-DD' 格式。"
        else:
            # 如果没有指定date，则默认返回最新5条数据
            return f"股票 {symbol.upper()} 的日线调整时间序列 (最新 5 条数据):\n{data.head(5).to_string()}"
    except Exception as e:
        return f"获取股票 {symbol} 日线调整数据时发生错误: {e}"
    
@tool
def get_weekly_time_series(
    symbol: str,
    datatype: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """
    这个 Alpha Vantage 的 API 接口能让你获取指定全球股票的周线时间序列数据，
    包括每周最后一个交易日的价格、周开盘价、周最高价、周最低价、周收盘价和周成交量。
    它涵盖了 20 多年的历史数据。

    Args:
        symbol (str): 您选择的股票名称。例如：IBM。
        datatype (Optional[str]): 返回数据的格式，'json' 或 'csv'。默认情况下，'json'。
        date (Optional[str]): 指定要查询的单个日期，格式为 'YYYY-MM-DD'。例如：'2025-08-05'。
                              如果设置此参数，工具将尝试查找包含该日期的那一周（通常以周五为周结束日）的周线数据。
                              如果该周数据不存在，将返回相应提示。

    Returns:
        str: 周线 OHLCV（开盘价、最高价、最低价、收盘价、成交量）数据。通常返回最新部分数据摘要或指定日期的数据。
    """
    api_params = {
        "symbol": symbol,
        "datatype": datatype,
    }

    # 移除字典中值为None的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = ts.get_weekly(**filtered_params)

        if data.empty:
            return f"无法获取股票 {symbol} 的周线数据，可能是无效的股票代码或暂无数据。"

        # 处理date参数
        if date is not None:
            try:
                # 将查询日期字符串转换为datetime对象
                query_datetime = pd.to_datetime(date)
                
                # 计算该日期所属周的周五
                target_friday = query_datetime.to_period('W-FRI').end_time.normalize()
                
                # 过滤DataFrame，查找匹配的周五日期
                weekly_data = data[data.index.normalize() == target_friday]
                
                if not weekly_data.empty:
                    # 返回指定日期所属周的数据
                    return (
                        f"股票 {symbol.upper()} 在 {date} 所属周（周线数据索引为 {target_friday.strftime('%Y-%m-%d')}）的周线数据:\n"
                        f"{weekly_data.to_string()}"
                    )
                else:
                    return (
                        f"无法找到股票 {symbol.upper()} 在 {date} 所属周（周线数据索引为 {target_friday.strftime('%Y-%m-%d')}）的周线数据。"
                        f"该周可能没有交易数据，或超出了可用历史范围。"
                    )
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM-DD' 格式。"
        else:
            # 如果没有指定date，则默认返回最新5条数据
            return f"股票 {symbol.upper()} 的周线时间序列 (最新 5 条数据):\n{data.head(5).to_string()}"
    except Exception as e:
        return f"获取股票 {symbol} 周线数据时发生错误: {e}"
    
@tool
def get_weekly_adjusted_time_series(
    symbol: str,
    datatype: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """
    这个 Alpha Vantage 的 API 接口能让你获取指定全球股票的周线时间序列数据。
    其显著特点是提供了经过股票拆分和股息派发事件调整后的数据，
    包括每周最后一个交易日的价格、周开盘价、周最高价、周最低价、周收盘价、
    周调整收盘价 (weekly adjusted close)、周成交量和周股息 (weekly dividend)。
    这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。
    它涵盖了 20 多年的历史数据。

    Args:
        symbol (str): 您选择的股票名称。例如：IBM。
        datatype (Optional[str]): 返回数据的格式，'json' 或 'csv'。默认情况下，'json'。
        date (Optional[str]): 指定要查询的单个日期，格式为 'YYYY-MM-DD'。例如：'2025-08-05'。
                              如果设置此参数，工具将尝试查找包含该日期的那一周（通常以周五为周结束日）的周线调整数据。
                              如果该周数据不存在，将返回相应提示。

    Returns:
        str: 周线 OHLCV（开盘价、最高价、最低价、收盘价、成交量）数据，特别包含经过调整的收盘价和周股息信息。通常返回最新部分数据摘要或指定日期的数据。
    """
    api_params = {
        "symbol": symbol,
        "datatype": datatype,
    }

    # 移除字典中值为None的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = ts.get_weekly_adjusted(**filtered_params)

        if data.empty:
            return f"无法获取股票 {symbol} 的周线调整数据，可能是无效的股票代码或暂无数据。"

        # 处理date参数
        if date is not None:
            try:
                # 将查询日期字符串转换为datetime对象
                query_datetime = pd.to_datetime(date)
                
                # 计算该日期所属周的周五
                target_friday = query_datetime.to_period('W-FRI').end_time.normalize()
                
                # 过滤DataFrame，查找匹配的周五日期
                weekly_data = data[data.index.normalize() == target_friday]
                
                if not weekly_data.empty:
                    # 返回指定日期所属周的数据
                    return (
                        f"股票 {symbol.upper()} 在 {date} 所属周（周线数据索引为 {target_friday.strftime('%Y-%m-%d')}）的周线调整数据:\n"
                        f"{weekly_data.to_string()}"
                    )
                else:
                    return (
                        f"无法找到股票 {symbol.upper()} 在 {date} 所属周（周线数据索引为 {target_friday.strftime('%Y-%m-%d')}）的周线调整数据。"
                        f"该周可能没有交易数据，或超出了可用历史范围。"
                    )
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM-DD' 格式。"
        else:
            # 如果没有指定date，则默认返回最新5条数据
            return f"股票 {symbol.upper()} 的周线调整时间序列 (最新 5 条数据):\n{data.head(5).to_string()}"
    except Exception as e:
        return f"获取股票 {symbol} 周线调整数据时发生错误: {e}"


@tool
def get_monthly_time_series(
    symbol: str,
    datatype: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """
    这个 Alpha Vantage 的 API 接口能让你获取指定全球股票的月线时间序列数据，
    包括每月最后一个交易日的价格、月开盘价、月最高价、月最低价、月收盘价和月成交量。
    它涵盖了 20 多年的历史数据。

    Args:
        symbol (str): 您选择的股票名称。例如：IBM。
        datatype (Optional[str]): 返回数据的格式，'json' 或 'csv'。默认情况下，'json'。
        date (Optional[str]): 指定要查询的单个日期，格式为 'YYYY-MM-DD' 或 'YYYY-MM'。例如：'2025-08-05' 或 '2025-08'。
                              如果设置此参数，工具将尝试查找该日期所属月份的月线数据（通常以该月最后一个交易日为索引）。
                              如果该月数据不存在，将返回相应提示。

    Returns:
        str: 月线 OHLCV（开盘价、最高价、最低价、收盘价、成交量）数据。通常返回最新部分数据摘要或指定日期的数据。
    """
    api_params = {
        "symbol": symbol,
        "datatype": datatype,
    }

    # 移除字典中值为None的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = ts.get_monthly(**filtered_params)

        if data.empty:
            return f"无法获取股票 {symbol} 的月线数据，可能是无效的股票代码或暂无数据。"

        # 处理date参数
        if date is not None:
            try:
                # 将查询日期字符串转换为datetime对象
                query_datetime = pd.to_datetime(date)
                
                # 计算该日期所属月份的月末日期
                target_month_end = query_datetime.to_period('M').end_time.normalize()
                
                # 过滤DataFrame，查找匹配的月末日期
                monthly_data = data[data.index.normalize() == target_month_end]
                
                if not monthly_data.empty:
                    # 返回指定日期所属月份的数据
                    return (
                        f"股票 {symbol.upper()} 在 {date} 所属月份（月线数据索引为 {target_month_end.strftime('%Y-%m-%d')}）的月线数据:\n"
                        f"{monthly_data.to_string()}"
                    )
                else:
                    return (
                        f"无法找到股票 {symbol.upper()} 在 {date} 所属月份（月线数据索引为 {target_month_end.strftime('%Y-%m-%d')}）的月线数据。"
                        f"该月可能没有交易数据，或超出了可用历史范围。"
                    )
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM-DD' 或 'YYYY-MM' 格式。"
        else:
            # 如果没有指定date，则默认返回最新5条数据
            return f"股票 {symbol.upper()} 的月线时间序列 (最新 5 条数据):\n{data.head(5).to_string()}"
    except Exception as e:
        return f"获取股票 {symbol} 月线数据时发生错误: {e}"

@tool
def get_monthly_adjusted_time_series(
    symbol: str,
    datatype: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """
    这个 Alpha Vantage 的 API 接口能让你获取指定全球股票的月线时间序列数据。
    其显著特点是提供了经过股票拆分和股息派发事件调整后的数据，
    包括每月最后一个交易日的价格、月开盘价、月最高价、月最低价、月收盘价、
    月调整收盘价 (monthly adjusted close)、月成交量和月股息 (monthly dividend)。
    这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。
    它涵盖了 20 多年的历史数据。

    Args:
        symbol (str): 您选择的股票名称。例如：IBM。
        datatype (Optional[str]): 返回数据的格式，'json' 或 'csv'。默认情况下，'json'。
        date (Optional[str]): 指定要查询的单个日期，格式为 'YYYY-MM-DD' 或 'YYYY-MM'。例如：'2025-08-05' 或 '2025-08'。
                              如果设置此参数，工具将尝试查找该日期所属月份的月线调整数据（通常以该月最后一个交易日为索引）。
                              如果该月数据不存在，将返回相应提示。

    Returns:
        str: 月线 OHLCV（开盘价、最高价、最低价、收盘价、成交量）数据，特别包含经过调整的收盘价和月股息信息。通常返回最新部分数据摘要或指定日期的数据。
    """
    api_params = {
        "symbol": symbol,
        "datatype": datatype,
    }

    # 移除字典中值为None的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = ts.get_monthly_adjusted(**filtered_params)

        if data.empty:
            return f"无法获取股票 {symbol} 的月线调整数据，可能是无效的股票代码或暂无数据。"

        # 处理date参数
        if date is not None:
            try:
                # 将查询日期字符串转换为 datetime 对象
                query_datetime = pd.to_datetime(date)
                
                # 计算该日期所属月份的月末日期
                target_month_end = query_datetime.to_period('M').end_time.normalize()
                
                # 过滤DataFrame，查找匹配的月末日期
                monthly_data = data[data.index.normalize() == target_month_end]
                
                if not monthly_data.empty:
                    # 返回指定日期所属月份的数据
                    return (
                        f"股票 {symbol.upper()} 在 {date} 所属月份（月线数据索引为 {target_month_end.strftime('%Y-%m-%d')}）的月线调整数据:\n"
                        f"{monthly_data.to_string()}"
                    )
                else:
                    return (
                        f"无法找到股票 {symbol.upper()} 在 {date} 所属月份（月线数据索引为 {target_month_end.strftime('%Y-%m-%d')}）的月线调整数据。"
                        f"该月可能没有交易数据，或超出了可用历史范围。"
                    )
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM-DD' 或 'YYYY-MM' 格式。"
        else:
            # 如果没有指定date，则默认返回最新5条数据
            return f"股票 {symbol.upper()} 的月线调整时间序列 (最新 5 条数据):\n{data.head(5).to_string()}"
    except Exception as e:
        return f"获取股票 {symbol} 月线调整数据时发生错误: {e}"

    
@tool
def get_quote_endpoint(
    symbol: str,
    datatype: Optional[str] = None,
) -> str:
    """
    这个 API 接口用于获取指定股票代码的最新价格和成交量信息。
    每次 API 请求只能指定一个股票代码。
    如果您需要批量查询大量股票代码，可以考虑使用 Realtime Bulk Quotes API，
    该 API 每次请求最多支持 100 个股票代码。

    Args:
        symbol (str): 您选择的全球股票代码。例如：IBM。
        datatype (Optional[str]): 返回数据的格式，'json' 或 'csv'。默认情况下，'json'。

    Returns:
        str: 指定股票的最新价格和成交量信息。
    """
    api_params = {
        "symbol": symbol,
        "datatype": datatype,
    }

    # 移除字典中值为None的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = ts.get_quote_endpoint(**filtered_params)

        if data.empty:
             return f"无法获取股票 {symbol} 的最新报价数据，可能是无效的股票代码或暂无数据。"
        
        # 如果返回的是字典，需要转换为字符串
        if isinstance(data, dict):
            # 格式化字典输出
            formatted_data = "\n".join([f"{k}: {v}" for k, v in data.items()])
            return f"股票 {symbol.upper()} 的最新报价信息:\n{formatted_data}"
        elif isinstance(data, pd.DataFrame):
            # 如果是DataFrame，只有一行数据，直接返回其字符串表示
            return f"股票 {symbol.upper()} 的最新报价信息:\n{data.to_string()}"
        else:
            return f"股票 {symbol.upper()} 的最新报价信息:\n{str(data)}"

    except Exception as e:
        return f"获取股票 {symbol} 最新报价数据时发生错误: {e}"

@tool
def get_market_status() -> str:
    """
    这个 API 接口返回全球主要股票、外汇和加密货币交易场所的当前市场状态（开放或关闭）。
    它提供了一个实时的快照，显示哪些市场正在交易，哪些已经休市。
    此函数不需要任何参数，因为它查询的是全局市场状态。

    Returns:
        str: 全球主要交易市场的当前开放/关闭状态信息。
    """
    try:
        if 'Alpha_Vantage_API_KEY' not in globals() or not Alpha_Vantage_API_KEY:
            return "错误：Alpha Vantage API 密钥未配置。请确保设置了 'Alpha_Vantage_API_KEY'。"

        API_KEY = Alpha_Vantage_API_KEY
        url = f"https://www.alphavantage.co/query?function=MARKET_STATUS&apikey={API_KEY}"

        # 调用 API
        response = requests.get(url)
        # 检查HTTP请求是否成功
        response.raise_for_status()
        data = response.json()

        if not data or "markets" not in data:
            return "无法获取市场状态信息，API返回数据为空或格式不正确。"

        output_lines = ["全球市场开放/关闭状态:"]
        output_lines.append("-" * 40)

        for market in data.get("markets", []):
            market_type = market.get("market_type", "N/A")
            region = market.get("region", "N/A")
            current_status = market.get("current_status", "N/A").upper()
            primary_exchanges = market.get("primary_exchanges", "N/A")
            local_open = market.get("local_open", "N/A")
            local_close = market.get("local_close", "N/A")
            notes = market.get("notes", "")
            output_lines.append(f"市场类型: {market_type}")
            output_lines.append(f"区域: {region}")
            output_lines.append(f"当前状态: {current_status}")
            output_lines.append(f"主要交易所: {primary_exchanges}")
            output_lines.append(f"本地开放时间: {local_open}")
            output_lines.append(f"本地关闭时间: {local_close}")
            # 如果有备注信息，则添加
            if notes:
                output_lines.append(f"备注: {notes}")
            output_lines.append("-" * 40)

        return "\n".join(output_lines)

    except requests.exceptions.RequestException as req_err:
        return f"获取全球市场状态时发生网络或API请求错误: {req_err}"
    except json.JSONDecodeError as json_err:
        return f"获取全球市场状态时解析JSON响应失败: {json_err}"
    except Exception as e:
        return f"获取全球市场状态时发生未知错误: {e}"

@tool
def get_historical_options_chain(
    symbol: str,
    date: Optional[str] = None,
    datatype: Optional[str] = None,
) -> str:
    """
    这个 Alpha Vantage 的 API 接口能让你获取指定股票在特定日期的完整历史期权链数据。
    它涵盖了 15 年以上的历史数据，并返回期权的隐含波动率 (IV) 和常见的希腊字母（如 Delta, Gamma, Theta, Vega, Rho）。
    期权链数据会按到期日按时间顺序排序，同一到期日内的合约则按行权价从低到高排序。

    Args:
        symbol (str): 您选择的股票名称。例如：IBM。
        date (Optional[str]): 查询期权链数据的特定日期，格式为 YYYY-MM-DD。例如：2017-11-15。
                              如果未提供，API 将返回上一个交易日的数据。接受 2008-01-01 或之后的日期。
        datatype (Optional[str]): 返回数据的格式，'json' 或 'csv'。默认情况下，'json'。

    Returns:
        str: 指定股票在特定日期的历史期权链数据的摘要，包含隐含波动率 (IV) 和希腊字母信息。
             由于数据量可能很大，通常只返回部分关键信息作为摘要。
    """
    try:
        if 'Alpha_Vantage_API_KEY' not in globals() or not Alpha_Vantage_API_KEY:
            return "错误：Alpha Vantage API 密钥未配置。请确保设置了 'Alpha_Vantage_API_KEY'。"

        base_url = "https://www.alphavantage.co/query?"
        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "apikey": Alpha_Vantage_API_KEY,
        }
        if date:
            params["date"] = date
        
        # 确保datatype为json
        params["datatype"] = datatype if datatype in ["json", "csv"] else "json"

        # 调用 API
        response = requests.get(base_url, params=params)
        # 检查HTTP请求是否成功
        response.raise_for_status()
        data = response.json()
        print("data:", data)
        # 检查错误信息
        if data['message'] != "success":
            if "Error Message" in data:
                return f"Alpha Vantage API 错误: {data['Error Message']}"
            elif "Note" in data:
                return f"Alpha Vantage API 提示: {data['Note']}"
            else:
                return f"Alpha Vantage API 调用失败。"

        # 取出实际数据
        raw_options_data = data.get("data", [])
        if not raw_options_data:
            return f"无法获取股票 {symbol} 在 {date if date else '最近交易日'} 的历史期权链数据，API返回数据为空或无期权合约。"

        # 按期权类型进行分组
        grouped_options = {}
        for contract in raw_options_data:
            expiration_date = contract.get("expiration")
            contract_type = contract.get("type")

            if expiration_date not in grouped_options:
                grouped_options[expiration_date] = {"calls": [], "puts": []}
            
            if contract_type == "call":
                grouped_options[expiration_date]["calls"].append(contract)
            elif contract_type == "put":
                grouped_options[expiration_date]["puts"].append(contract)

        # 按到期日排序，确保输出顺序一致
        sorted_expirations = sorted(grouped_options.keys())

        output_lines = [f"股票 {symbol.upper()} 在 {date if date else '最近交易日'} 的历史期权链摘要:"]
        output_lines.append("=" * 60)

        # 遍历最多前MAX_ROWS_TO_DISPLAY个到期日的数据，以提供摘要
        for i, expiration in enumerate(sorted_expirations):
            if i >= MAX_ROWS_TO_DISPLAY:
                remaining_exp = len(sorted_expirations) - i
                output_lines.append(f"\n... (还有 {remaining_exp} 个更多到期日，请自行查询以获取完整数据)")
                break

            expiry_data = grouped_options[expiration]
            calls = expiry_data.get("calls", [])
            puts = expiry_data.get("puts", [])

            output_lines.append(f"\n到期日: {expiration}")
            
            # 打印看涨期权摘要
            output_lines.append(f"  看涨期权 ({len(calls)} 份合约):")
            for j, call in enumerate(calls):
                if j >= MAX_ROWS_TO_DISPLAY:
                    output_lines.append("    ... (更多看涨期权)")
                    break
                output_lines.append(
                    f"    - 行权价: {call.get('strike', 'N/A')}, "
                    f"最后价格: {call.get('last', 'N/A')}, "
                    f"隐含波动率: {call.get('implied_volatility', 'N/A')}, "
                    f"Delta: {call.get('delta', 'N/A')}"
                    f"Gamma: {call.get('gamma', 'N/A')}, "
                    f"Theta: {call.get('theta', 'N/A')}, "
                    f"Vega: {call.get('vega', 'N/A')}, "
                    f"Rho: {call.get('rho', 'N/A')}"
                )
            
            # 打印看跌期权摘要
            output_lines.append(f"  看跌期权 ({len(puts)} 份合约):")
            for j, put in enumerate(puts):
                if j >= MAX_ROWS_TO_DISPLAY:
                    output_lines.append("    ... (更多看跌期权)")
                    break
                output_lines.append(
                    f"    - 行权价: {put.get('strike', 'N/A')}, "
                    f"最后价格: {put.get('last', 'N/A')}, "
                    f"隐含波动率: {put.get('implied_volatility', 'N/A')}, "
                    f"Delta: {put.get('delta', 'N/A')}"
                    f"Gamma: {call.get('gamma', 'N/A')}, "
                    f"Theta: {call.get('theta', 'N/A')}, "
                    f"Vega: {call.get('vega', 'N/A')}, "
                    f"Rho: {call.get('rho', 'N/A')}"
                )
            output_lines.append("-" * 60)

        return "\n".join(output_lines)

    except requests.exceptions.RequestException as req_err:
        return f"获取历史期权链数据时发生网络或API请求错误: {req_err}"
    except json.JSONDecodeError as json_err:
        return f"获取历史期权链数据时解析JSON响应失败: {json_err}. 原始响应可能如下: {response.text[:500]}..."
    except Exception as e:
        return f"获取历史期权链数据时发生未知错误: {e}"

@tool
def get_sse_market_summary() -> str:
    """
    获取上海证券交易所 (SSE) 股票市场的整体概况数据。
    该接口返回最近一个交易日的股票市场总貌统计信息，包括流通股本、总市值、平均市盈率、上市公司数量等。
    当前交易日的数据通常在交易所收盘后才会统计并更新。

    Args:
        无。此函数不需要任何参数。

    Returns:
        str: 一个格式化后的字符串，包含上海证券交易所股票市场总貌的表格数据。
             如果获取失败，则返回错误信息。
    """
    try:
        # 调用 API
        df = ak.stock_sse_summary()

        if df.empty:
            return "未能获取上海证券交易所股票市场总貌数据，返回数据为空。"

        # 将DataFrame转换为字符串
        # 提取报告时间，并将其从DataFrame中移除
        report_time = None
        if '项目' in df.columns and '报告时间' in df['项目'].values:
            report_row = df[df['项目'] == '报告时间']
            if not report_row.empty:
                report_time = report_row['股票'].iloc[0] 
                df = df[df['项目'] != '报告时间']

        output_lines = []
        if report_time:
            output_lines.append(f"上海证券交易所股票市场总貌 (报告时间: {report_time}):\n")
        else:
            output_lines.append("上海证券交易所股票市场总貌:\n")

        # 将DataFrame转换为字符串，并去除索引
        df_string = df.to_string(index=False)
        output_lines.append(df_string)
        
        return "\n".join(output_lines)

    except Exception as e:
        return f"获取上海证券交易所股票市场总貌时发生错误: {e}"
    
@tool
def get_szse_security_category_summary(date: str) -> str:
    """
    获取深圳证券交易所 (SZSE) 指定日期的市场总貌中证券类别统计数据。
    该接口返回各类证券（如股票、基金、债券等）的数量、成交金额、总市值和流通市值等统计信息。
    当前交易日的数据通常在交易所收盘后才会统计并更新。

    Args:
        date (str): 查询数据的日期，格式为 'YYYYMMDD'。例如：'20200619'。
                    当前交易日的数据需要交易所收盘后才能统计。

    Returns:
        str: 一个格式化后的字符串，包含深圳证券交易所证券类别统计的表格数据。
             如果获取失败，则返回错误信息。
    """
    try:
        # 调用 API
        df = ak.stock_szse_summary(date=date)

        if df.empty:
            return f"未能获取深圳证券交易所 {date} 的证券类别统计数据，返回数据为空。请检查日期是否正确或是否为非交易日。"

        output_lines = [f"深圳证券交易所 {date} 证券类别统计概览:\n"]
        output_lines.append("=" * 60)

        # 将 DataFrame 转换为字符串，并去除索引
        df_string = df.to_string(index=False)
        output_lines.append(df_string)
        output_lines.append("=" * 60)
        output_lines.append("注意：数量单位为 '只'，成交金额单位为 '元'。")
        
        return "\n".join(output_lines)

    except Exception as e:
        return f"获取深圳证券交易所 {date} 证券类别统计数据时发生错误: {e}"

@tool
def get_szse_regional_trading_summary(date: str) -> str:
    """
    获取深圳证券交易所 (SZSE) 指定月份的市场总貌中地区交易排序数据。
    该接口返回按地区划分的总交易额、占市场份额、股票交易额、基金交易额和债券交易额等统计信息。
    这些数据通常用于分析不同地区在深圳证券交易所的交易活跃度。

    Args:
        date (str): 查询数据的年月，格式为 'YYYYMM'。例如：'202203'。

    Returns:
        str: 一个格式化后的字符串，包含深圳证券交易所地区交易排序的表格数据。
             如果获取失败，则返回错误信息。
    """
    # 日期格式验证
    if not (len(date) == 6 and date.isdigit()):
        return f"错误：日期格式不正确。请提供 'YYYYMM' 格式的年月，例如 '202203'。"

    try:
        # 调用 API
        df = ak.stock_szse_area_summary(date=date)
        print("df:", df)
        if df.empty:
            return f"未能获取深圳证券交易所 {date} 月份的地区交易排序数据，返回数据为空。请检查日期是否正确或该月份无数据。"

        output_lines = [f"深圳证券交易所 {date} 月份地区交易排序概览:\n"]
        output_lines.append("=" * 60)

        # 遍历所有浮点数列进行格式化
        for col in ['总交易额', '股票交易额', '基金交易额', '债券交易额']:
            if col in df.columns:
                # 将科学计数法转换为普通数字字符串，并添加单位
                df[col] = df[col].apply(lambda x: f"{x:.2f} 元" if pd.notna(x) else "N/A")
        
        if '占市场' in df.columns:
             df['占市场'] = df['占市场'].apply(lambda x: f"{x:.3f} %" if pd.notna(x) else "N/A")

        # 将DataFrame转换为字符串，并去除索引
        df_string = df.to_string(index=False)
        output_lines.append(df_string)
        output_lines.append("=" * 60)
        output_lines.append("注意：交易额单位为 '元'，占市场份额单位为 '%'。")
        
        return "\n".join(output_lines)

    except Exception as e:
        return f"获取深圳证券交易所 {date} 月份地区交易排序数据时发生错误: {e}"

@tool
def get_szse_sector_trading_summary(symbol: str, date: str) -> str:
    """
    获取深圳证券交易所 (SZSE) 指定时间段内（当月或当年）的股票行业成交数据。
    该接口返回按行业划分的交易天数、成交金额（人民币元）、成交金额占比、成交股数、成交股数占比、
    成交笔数和成交笔数占比等统计信息。
    这些数据可用于分析不同行业在深圳证券交易所的交易活跃度和市场份额。

    Args:
        symbol (str): 查询的时间范围。可选值：'当月'（查询指定月份的数据）或 '当年'（查询指定月份所在年份的累计数据）。
                      请务必选择 '当月' 或 '当年'。
        date (str): 查询数据的年月，格式为 'YYYYMM'。例如：'202405'。
                    请注意，只能查询已过去的月份数据，无法查询未来月份的数据。

    Returns:
        str: 一个格式化后的字符串，包含深圳证券交易所股票行业成交的表格数据。
             如果获取失败，则返回错误信息。
    """
    # 验证symbol参数
    valid_symbols = ["当月", "当年"]
    if symbol not in valid_symbols:
        return f"错误：symbol 参数无效。请选择 '当月' 或 '当年'。"

    # 验证date参数格式和是否为未来日期
    if not (len(date) == 6 and date.isdigit()):
        return f"错误：日期格式不正确。请提供 'YYYYMM' 格式的年月，例如 '202405'。"
    
    try:
        query_year = int(date[:4])
        query_month = int(date[4:])
        query_date_obj = datetime(query_year, query_month, 1)
        
        current_date_obj = datetime.now()
        # 检查是否是未来月份
        if query_date_obj > current_date_obj.replace(day=1):
            return f"错误：无法获取未来月份（{date}）的股票行业成交数据。请提供已过去的月份。"

    except ValueError:
        return f"错误：日期 '{date}' 无法解析。请确保是有效的 'YYYYMM' 格式。"

    try:
        # 调用 API
        df = ak.stock_szse_sector_summary(symbol=symbol, date=date)

        if df.empty:
            return f"未能获取深圳证券交易所 {date} {symbol} 的股票行业成交数据，返回数据为空。请检查日期是否正确或该时间段无数据。"

        output_lines = [f"深圳证券交易所 {date} {symbol} 股票行业成交数据概览:\n"]
        output_lines.append("=" * 80)

        # 成交金额和成交股数可能很大，保留两位小数，并明确单位
        for col in ['成交金额-人民币元', '成交股数-股数', '成交笔数-笔']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A") # 添加千位分隔符
        
        # 格式化百分比列
        for col in ['成交金额-占总计', '成交股数-占总计', '成交笔数-占总计']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

        # 将DataFrame转换为字符串，并去除索引
        df_string = df.to_string(index=False)
        output_lines.append(df_string)
        output_lines.append("=" * 80)
        output_lines.append("注意：成交金额单位为 '人民币元'，成交股数单位为 '股'，成交笔数单位为 '笔'。")
        
        return "\n".join(output_lines)

    except Exception as e:
        return f"获取深圳证券交易所 {date} {symbol} 股票行业成交数据时发生错误: {e}"
    
@tool
def get_sse_daily_deal_summary(date: str) -> str:
    """
    获取上海证券交易所 (SSE) 指定日期的每日股票成交概况数据。
    该接口提供包括挂牌数、市价总值、流通市值、成交金额、成交量、平均市盈率、
    换手率和流通换手率等详细信息，并按股票总计、主板A股、主板B股、科创板和股票回购等类别进行细分。

    Args:
        date (str): 查询数据的日期，格式为 'YYYYMMDD'。例如：'20250221'。
                    请注意：
                    1. 只能获取 20211227（包含）之后的数据。
                    2. 无法查询未来日期的数据。
                    3. 当前交易日的数据需在交易所收盘后获取。

    Returns:
        str: 一个格式化后的字符串，包含上海证券交易所每日股票成交概况的表格数据。
             如果获取失败，则返回错误信息。
    """
    # 验证日期格式
    if not (len(date) == 8 and date.isdigit()):
        return f"错误：日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20240520'。"

    # 验证日期范围
    min_date_str = "20211227"
    try:
        query_date_obj = datetime.strptime(date, "%Y%m%d")
        min_date_obj = datetime.strptime(min_date_str, "%Y%m%d")
        current_date_obj = datetime.now()

        if query_date_obj < min_date_obj:
            return f"错误：该接口仅支持获取 {min_date_str}（包含）之后的数据。您查询的日期为 {date}。"

        # 如果是当前日期，则提示可能未更新
        if query_date_obj > current_date_obj:
            return f"错误：无法获取未来日期（{date}）的每日概况数据。请提供已过去的日期。"
        elif query_date_obj.date() == current_date_obj.date():
            # 提示用户当前日期数据可能未更新
            pass

    except ValueError:
        return f"错误：日期 '{date}' 无法解析。请确保是有效的 'YYYYMMDD' 格式。"

    try:
        # 调用 API
        df = ak.stock_sse_deal_daily(date=date)

        if df.empty:
            return f"未能获取上海证券交易所 {date} 的每日股票成交概况数据，返回数据为空。请检查日期是否正确、是否为非交易日，或数据尚未发布。"

        output_lines = [f"上海证券交易所 {date} 每日股票成交概况:\n"]
        output_lines.append("=" * 80)

        # 遍历所有数值列进行格式化，保留两位小数，并处理 NaN
        for col in df.columns:
            # 排除非数值列
            if col != '单日情况':
                df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
        
        # 将DataFrame转换为字符串，并去除索引
        df_string = df.to_string(index=False)
        output_lines.append(df_string)
        output_lines.append("=" * 80)
        output_lines.append("注意：市价总值、流通市值、成交金额单位可能为 '万元' 或 '亿元'，具体请参考交易所官方数据说明。")
        output_lines.append("成交量单位可能为 '万股' 或 '亿股'。")
        
        return "\n".join(output_lines)

    except Exception as e:
        return f"获取上海证券交易所 {date} 每日股票成交概况数据时发生错误: {e}"
    
@tool
def get_eastmoney_individual_stock_info(symbol: str) -> str:
    """
    获取东方财富网站上指定股票代码的个股详细信息。
    该接口返回包括股票最新价格、代码、简称、总股本、流通股本、总市值、流通市值、
    所属行业和上市时间等关键数据。

    Args:
        symbol (str): 需要查询的股票代码，例如 '603777' 或 '000001'。请提供完整的股票代码。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的详细信息。
             如果获取失败，则返回错误信息。
    """
    # 验证股票代码格式
    if not (symbol.isdigit() and len(symbol) == 6):
        return f"错误：股票代码 '{symbol}' 格式不正确。请提供6位数字的股票代码，例如 '600000'。"

    try:
        # 调用 API
        df = ak.stock_individual_info_em(symbol=symbol)

        if df.empty:
            return f"未能获取股票代码 {symbol} 的个股信息，返回数据为空。请检查股票代码是否正确或数据源暂无此信息。"

        output_lines = [f"股票代码 {symbol} 的详细信息 (数据来源: 东方财富):\n"]
        output_lines.append("=" * 40)

        # 遍历 DataFrame，将 'item' 和 'value' 列转换为键值对格式
        for index, row in df.iterrows():
            item = row['item']
            value = row['value']
            
            # 对特定数值进行格式化
            if item in ['最新', '总市值', '流通市值']:
                try:
                    # 转换为浮点数并格式化为带千位分隔符和两位小数
                    value = float(value)
                    value_str = f"{value:,.2f}"
                except ValueError:
                    # 无法转换则保持原样
                    value_str = str(value)
            elif item in ['总股本', '流通股']:
                try:
                    value = float(value)
                    # 格式化为带千位分隔符
                    value_str = f"{int(value):,}"
                except ValueError:
                    value_str = str(value)
            else:
                # 其他项目直接转换为字符串
                value_str = str(value)

            output_lines.append(f"{item}: {value_str}")
        
        output_lines.append("=" * 40)
        output_lines.append("注意：市值和股本的单位通常为 '元' 或 '股'，具体请参考东方财富官网。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票代码 {symbol} 的个股信息时发生错误: {e}"
    
@tool
def get_eastmoney_stock_bid_ask_quote(symbol: str) -> str:
    """
    获取东方财富网站上指定股票代码的实时行情报价数据。
    该接口提供包括买卖盘五档报价（价格和数量）、最新价、均价、涨跌幅、总手、成交金额、
    换手率、量比、最高价、最低价、今开、昨收、涨停价、跌停价、外盘和内盘等详细信息。

    Args:
        symbol (str): 需要查询的股票代码，例如 '000001'。请提供完整的6位数字股票代码。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的实时行情报价数据。
             如果获取失败，则返回错误信息。
    """
    # 验证股票代码格式
    if not (symbol.isdigit() and len(symbol) == 6):
        return f"错误：股票代码 '{symbol}' 格式不正确。请提供6位数字的股票代码，例如 '600000'。"

    try:
        # 调用 API
        df = ak.stock_bid_ask_em(symbol=symbol)

        if df.empty:
            return f"未能获取股票代码 {symbol} 的行情报价数据，返回数据为空。请检查股票代码是否正确，或当前非交易时间/数据源暂无此信息。"

        output_lines = [f"股票代码 {symbol} 的实时行情报价 (数据来源: 东方财富):\n"]
        output_lines.append("=" * 40)

        # 遍历DataFrame，将 'item' 和 'value' 列转换为键值对格式
        for index, row in df.iterrows():
            item = str(row['item'])
            value = row['value']

            # 将NaN值转换为字符串 "N/A"
            if pd.isna(value):
                value_str = "N/A"
            else:
                # 根据item字段进行格式化
                if 'sell_' in item or 'buy_' in item:
                    # 交易量
                    if 'vol' in item:
                        value_str = f"{int(value):,} 手"
                    # 价格
                    else:
                        value_str = f"{value:.2f} 元"
                elif item in ['最新', '均价', '最高', '最低', '今开', '昨收', '涨停', '跌停']:
                    value_str = f"{value:.2f} 元"
                elif item == '涨幅':
                    value_str = f"{value:.2f}%"
                elif item == '涨跌':
                    value_str = f"{value:+.2f} 元"
                elif item == '总手':
                    value_str = f"{int(value):,} 手"
                elif item == '金额':
                    # 金额通常很大，可以转换为亿元或万元显示
                    if value >= 1_0000_0000:
                        value_str = f"{value / 1_0000_0000:.2f} 亿元"
                    elif value >= 1_0000:
                        value_str = f"{value / 1_0000:.2f} 万元"
                    else:
                        value_str = f"{value:,.2f} 元"
                elif item in ['换手', '量比']:
                    value_str = f"{value:.2f}%"
                elif item in ['外盘', '内盘']:
                    value_str = f"{int(value):,} 手"
                else:
                    # 其他项目直接转换为字符串
                    value_str = str(value)

            output_lines.append(f"{item}: {value_str}")
        
        output_lines.append("=" * 40)
        output_lines.append("注意：成交量、外盘、内盘单位通常为 '手'，金额单位可能为 '元'、'万元' 或 '亿元'，具体请参考东方财富官网。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票代码 {symbol} 的行情报价数据时发生错误: {e}"

@tool
def get_eastmoney_all_a_shares_realtime_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有沪深京 A 股上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率、总市值、流通市值、涨速、5分钟涨跌、
    60日涨跌幅和年初至今涨跌幅等全面的实时市场指标。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定沪深京 A 股的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的沪深京 A 股的股票代码（如 '000001'）
                                          或公司名称（如 '平安银行' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含沪深京 A 股实时行情数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_zh_a_spot_em()

        if df.empty:
            return "未能获取沪深京 A 股实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '最新价': lambda x: f"{x:.2f}",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '今开': lambda x: f"{x:.2f}",
            '昨收': lambda x: f"{x:.2f}",
            '量比': lambda x: f"{x:.2f}",
            '换手率': lambda x: f"{x:.2f}%",
            '市盈率-动态': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '市净率': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '总市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '流通市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '涨速': lambda x: f"{x:.2f}%",
            '5分钟涨跌': lambda x: f"{x:+.2f}%",
            '60日涨跌幅': lambda x: f"{x:+.2f}%",
            '年初至今涨跌幅': lambda x: f"{x:+.2f}%"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        all_display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in all_display_cols:
            all_display_cols.remove('序号')

        # 如果提供了股票代码或名称，则筛选特定股票
        if stock_identifier:
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            if '代码' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['代码'].astype(str) == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的沪深京 A 股实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"沪深京 A 股 '{stock_info['名称']}' ({stock_info['代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in all_display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为沪深京 A 股 '{stock_info['名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的沪深京 A 股，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(MAX_ROWS_TO_DISPLAY).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > MAX_ROWS_TO_DISPLAY:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - MAX_ROWS_TO_DISPLAY} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有沪深京A股的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条沪深京 A 股实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '总市值']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有 A 股的实时行情数据概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询股票代码000001的最新价和涨跌幅。' 或 '查询平安银行的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取沪深京 A 股实时行情数据时发生错误: {e}"
    
@tool
def get_sh_a_shares_realtime_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有沪 A 股上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率、总市值、流通市值、涨速、5分钟涨跌、
    60日涨跌幅和年初至今涨跌幅等全面的实时市场指标。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定沪 A 股的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的沪 A 股的股票代码（如 '600000'）
                                          或公司名称（如 '浦发银行' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含沪 A 股实时行情数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_sh_a_spot_em()

        if df.empty:
            return "未能获取沪 A 股实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '最新价': lambda x: f"{x:.2f}",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '今开': lambda x: f"{x:.2f}",
            '昨收': lambda x: f"{x:.2f}",
            '量比': lambda x: f"{x:.2f}",
            '换手率': lambda x: f"{x:.2f}%",
            '市盈率-动态': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '市净率': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '总市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '流通市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '涨速': lambda x: f"{x:.2f}%",
            '5分钟涨跌': lambda x: f"{x:+.2f}%",
            '60日涨跌幅': lambda x: f"{x:+.2f}%",
            '年初至今涨跌幅': lambda x: f"{x:+.2f}%"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        all_display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in all_display_cols:
            all_display_cols.remove('序号')

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            if '代码' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['代码'].astype(str) == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的沪 A 股实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"沪 A 股 '{stock_info['名称']}' ({stock_info['代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in all_display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为沪 A 股 '{stock_info['名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的沪 A 股，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(MAX_ROWS_TO_DISPLAY).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > MAX_ROWS_TO_DISPLAY:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - MAX_ROWS_TO_DISPLAY} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有沪 A 股的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条沪 A 股实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '总市值']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有沪 A 股的实时行情数据概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询股票代码600000的最新价和涨跌幅。' 或 '查询浦发银行的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取沪 A 股实时行情数据时发生错误: {e}"

@tool
def get_sz_a_shares_realtime_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有深 A 股上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率、总市值、流通市值、涨速、5分钟涨跌、
    60日涨跌幅和年初至今涨跌幅等全面的实时市场指标。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定深 A 股的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的深 A 股的股票代码（如 '000001'）
                                          或公司名称（如 '平安银行' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含深 A 股实时行情数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_sz_a_spot_em()

        if df.empty:
            return "未能获取深 A 股实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '最新价': lambda x: f"{x:.2f}",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '今开': lambda x: f"{x:.2f}",
            '昨收': lambda x: f"{x:.2f}",
            '量比': lambda x: f"{x:.2f}",
            '换手率': lambda x: f"{x:.2f}%",
            '市盈率-动态': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '市净率': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '总市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '流通市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '涨速': lambda x: f"{x:.2f}%",
            '5分钟涨跌': lambda x: f"{x:+.2f}%",
            '60日涨跌幅': lambda x: f"{x:+.2f}%",
            '年初至今涨跌幅': lambda x: f"{x:+.2f}%"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        all_display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in all_display_cols:
            all_display_cols.remove('序号')

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            if '代码' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['代码'].astype(str) == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的深 A 股实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"深 A 股 '{stock_info['名称']}' ({stock_info['代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in all_display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为深 A 股 '{stock_info['名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的深 A 股，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(MAX_ROWS_TO_DISPLAY).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > MAX_ROWS_TO_DISPLAY:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - MAX_ROWS_TO_DISPLAY} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有深 A 股的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条深 A 股实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '总市值']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有深 A 股的实时行情数据概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询股票代码000001的最新价和涨跌幅。' 或 '查询平安银行的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取深 A 股实时行情数据时发生错误: {e}"
    
@tool
def get_bj_a_shares_realtime_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有京 A 股上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率、总市值、流通市值、涨速、5分钟涨跌、
    60日涨跌幅和年初至今涨跌幅等全面的实时市场指标。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定京 A 股的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的京 A 股的股票代码（如 '836149'）
                                          或公司名称（如 '翰博高新' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含京 A 股实时行情数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_bj_a_spot_em()

        if df.empty:
            return "未能获取京 A 股实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '最新价': lambda x: f"{x:.2f}",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '今开': lambda x: f"{x:.2f}",
            '昨收': lambda x: f"{x:.2f}",
            '量比': lambda x: f"{x:.2f}",
            '换手率': lambda x: f"{x:.2f}%",
            '市盈率-动态': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '市净率': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '总市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '流通市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '涨速': lambda x: f"{x:.2f}%",
            '5分钟涨跌': lambda x: f"{x:+.2f}%",
            '60日涨跌幅': lambda x: f"{x:+.2f}%",
            '年初至今涨跌幅': lambda x: f"{x:+.2f}%"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        all_display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in all_display_cols:
            all_display_cols.remove('序号')

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            if '代码' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['代码'].astype(str) == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的京 A 股实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"京 A 股 '{stock_info['名称']}' ({stock_info['代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in all_display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为京 A 股 '{stock_info['名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的京 A 股，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(MAX_ROWS_TO_DISPLAY).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > MAX_ROWS_TO_DISPLAY:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - MAX_ROWS_TO_DISPLAY} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有京 A 股的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条京 A 股实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '总市值']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有京 A 股的实时行情数据概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询股票代码836149的最新价和涨跌幅。' 或 '查询翰博高新的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取京 A 股实时行情数据时发生错误: {e}"

@tool
def get_eastmoney_new_a_shares_realtime_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有新股上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率、上市时间、总市值、流通市值、涨速、
    5分钟涨跌、60日涨跌幅和年初至今涨跌幅等全面的实时市场指标。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定新股的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的新股的股票代码（如 '688583'）
                                          或公司名称（如 '中芯聚源' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含新股实时行情数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_new_a_spot_em()

        if df.empty:
            return "未能获取新股实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '最新价': lambda x: f"{x:.2f}",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '今开': lambda x: f"{x:.2f}",
            '昨收': lambda x: f"{x:.2f}",
            '量比': lambda x: f"{x:.2f}",
            '换手率': lambda x: f"{x:.2f}%",
            '市盈率-动态': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '市净率': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '总市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '流通市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '涨速': lambda x: f"{x:.2f}%",
            '5分钟涨跌': lambda x: f"{x:+.2f}%",
            '60日涨跌幅': lambda x: f"{x:+.2f}%",
            '年初至今涨跌幅': lambda x: f"{x:+.2f}%",
            '上市时间': lambda x: str(x) # 上市时间直接转字符串
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        all_display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in all_display_cols:
            all_display_cols.remove('序号')

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 尝试按代码精确匹配
            if '代码' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['代码'].astype(str) == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的新股实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"新股 '{stock_info['名称']}' ({stock_info['代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in all_display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为新股 '{stock_info['名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的新股，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(MAX_ROWS_TO_DISPLAY).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > MAX_ROWS_TO_DISPLAY:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - MAX_ROWS_TO_DISPLAY} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有新股的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条新股实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '上市时间', '总市值']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有新股的实时行情数据概览。")
            output_lines.append("如果您需要查询特定新股的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询股票代码688583的最新价和上市时间。' 或 '查询中芯聚源的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取新股实时行情数据时发生错误: {e}"

@tool
def get_cy_a_shares_realtime_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的创业板上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率、总市值、流通市值、涨速、5分钟涨跌、
    60日涨跌幅和年初至今涨跌幅等全面的实时市场指标。

    用户可以提供一个可选的股票代码或名称来查询特定股票的详细行情。
    如果未提供股票标识符，工具将返回所有创业板股票的实时行情摘要（前MAX_ROWS_TO_DISPLAY行）。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询实时行情数据的创业板股票代码（如 '300001'）或股票名称（如 '特锐德'）。
                                          如果未提供，将返回所有创业板股票的摘要。

    Returns:
        str: 一个格式化后的字符串，包含所有创业板实时行情数据的摘要，或特定股票的详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_cy_a_spot_em()

        if df.empty:
            return "未能获取创业板实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 确保 '代码' 和 '名称' 列存在且为字符串类型
        if '代码' not in df.columns or '名称' not in df.columns:
            return "错误：获取到的创业板行情数据缺少 '代码' 或 '名称' 列，无法进行查询或展示。"
        df['代码'] = df['代码'].astype(str)
        df['名称'] = df['名称'].astype(str)

        # 定义需要格式化的列及其格式函数
        # 格式化辅助函数
        def format_value(value, unit="", precision=2, is_percent=False, is_money=False, is_volume=False):
            if pd.isna(value):
                return "N/A"
            
            if is_percent:
                # 百分比带正负号
                return f"{value:+.{precision}f}{unit}"
            elif is_money:
                if value >= 1_0000_0000:
                    return f"{value / 1_0000_0000:,.{precision}f}亿{unit}"
                elif value >= 1_0000:
                    return f"{value / 1_0000:,.{precision}f}万{unit}"
                else:
                    return f"{value:,.{precision}f}{unit}"
            elif is_volume:
                return f"{int(value):,}{unit}"
            else:
                return f"{value:,.{precision}f}{unit}"

        # 格式化规则字典
        format_rules = {
            '最新价': lambda x: format_value(x, unit="元", precision=2),
            '涨跌幅': lambda x: format_value(x, unit="%", precision=2, is_percent=True),
            '涨跌额': lambda x: format_value(x, unit="元", precision=2, is_percent=True),
            '成交量': lambda x: format_value(x, unit="手", precision=0, is_volume=True),
            '成交额': lambda x: format_value(x, unit="元", precision=2, is_money=True),
            '振幅': lambda x: format_value(x, unit="%", precision=2),
            '最高': lambda x: format_value(x, unit="元", precision=2),
            '最低': lambda x: format_value(x, unit="元", precision=2),
            '今开': lambda x: format_value(x, unit="元", precision=2),
            '昨收': lambda x: format_value(x, unit="元", precision=2),
            '量比': lambda x: format_value(x, precision=2),
            '换手率': lambda x: format_value(x, unit="%", precision=2),
            '市盈率-动态': lambda x: format_value(x, precision=2),
            '市净率': lambda x: format_value(x, precision=2),
            '总市值': lambda x: format_value(x, unit="元", precision=2, is_money=True),
            '流通市值': lambda x: format_value(x, unit="元", precision=2, is_money=True),
            '涨速': lambda x: format_value(x, unit="%", precision=2),
            '5分钟涨跌': lambda x: format_value(x, unit="%", precision=2, is_percent=True),
            '60日涨跌幅': lambda x: format_value(x, unit="%", precision=2, is_percent=True),
            '年初至今涨跌幅': lambda x: format_value(x, unit="%", precision=2, is_percent=True)
        }

        # 如果提供了股票标识符，查询特定股票
        if stock_identifier:
            resolved_stock_code = None
            resolved_stock_name = None

            # 判断 stock_identifier 是代码还是名称
            if stock_identifier.isdigit() and len(stock_identifier) == 6:
                resolved_stock_code = stock_identifier
                # 从已获取的DataFrame中找到名称
                matched_name_df = df[df['代码'] == resolved_stock_code]
                if not matched_name_df.empty:
                    resolved_stock_name = matched_name_df.iloc[0]['名称']
            else:
                # 通过名称查找股票代码
                # 优先精确匹配名称
                matched_stocks = df[df['名称'] == stock_identifier]

                # 如果没有精确匹配，则进行模糊匹配
                if matched_stocks.empty:
                    matched_stocks = df[df['名称'].str.contains(stock_identifier, case=False, na=False)]

                if matched_stocks.empty:
                    return f"未能在创业板实时行情中找到与 '{stock_identifier}' 匹配的股票代码或名称。请检查输入是否正确。"
                elif len(matched_stocks) > 1:
                    output_lines = [f"找到多个与 '{stock_identifier}' 匹配的创业板股票，请提供更具体的股票名称或直接提供股票代码："]
                    output_lines.append("=" * 100)
                    for index, row in matched_stocks.head(10).iterrows():
                        output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                    if len(matched_stocks) > 10:
                        output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
                    output_lines.append("=" * 100)
                    return "\n".join(output_lines)
                else:
                    resolved_stock_code = matched_stocks.iloc[0]['代码']
                    resolved_stock_name = matched_stocks.iloc[0]['名称']

            if not resolved_stock_code:
                return f"无法为 '{stock_identifier}' 解析出有效的创业板股票代码。请确保输入正确或尝试直接提供股票代码。"

            # 过滤出指定股票的数据
            target_stock_data = df[df['代码'] == resolved_stock_code]

            if target_stock_data.empty:
                return f"未能在创业板实时行情中找到 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 的实时行情数据。可能该股票已退市或代码有误。"
            
            # 取第一个匹配
            stock_info = target_stock_data.iloc[0]

            output_lines = [f"创业板股票 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 的实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)

            # 遍历所有列，应用格式化规则并输出
            for col in df.columns:
                if col in ['序号']:
                    continue
                display_name = col
                if col == '市盈率-动态': display_name = '市盈率(动态)'
                elif col == '5分钟涨跌': display_name = '5分钟涨跌幅'
                elif col == '60日涨跌幅': display_name = '60日涨跌幅'
                elif col == '年初至今涨跌幅': display_name = '年初至今涨跌幅'
                
                value = stock_info[col]
                formatted_value = format_rules.get(col, lambda x: str(x))(value)
                output_lines.append(f"{display_name}: {formatted_value}")
                
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为创业板股票 '{resolved_stock_name or stock_identifier}' 的实时行情数据。")

            return "\n".join(output_lines)

        # 如果未提供股票标识符，返回所有股票的摘要
        else:
            output_lines = [f"成功获取 {len(df)} 条创业板实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分列
            display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '总市值']
            
            # 确保要显示的列都在 DataFrame 中
            display_cols = [col for col in display_cols if col in df.columns]

            # 应用格式化规则到要显示的列
            df_display_summary = df[display_cols].copy()
            for col in display_cols:
                if col in format_rules:
                    df_display_summary[col] = df_display_summary[col].apply(lambda x: format_rules[col](x) if pd.notna(x) else "N/A")

            df_string = df_display_summary.head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有创业板的实时行情数据概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询特锐德的实时行情。' 或 '查询股票代码300001的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取创业板实时行情数据时发生错误: {e}"
    
@tool
def get_kc_a_shares_realtime_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有科创板上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率、总市值、流通市值、涨速、5分钟涨跌、
    60日涨跌幅和年初至今涨跌幅等全面的实时市场指标。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定科创板股票的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的科创板股票的股票代码（如 '688001'）
                                          或公司名称（如 '中芯国际' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含科创板实时行情数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_kc_a_spot_em()

        if df.empty:
            return "未能获取科创板实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '最新价': lambda x: f"{x:.2f}",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '今开': lambda x: f"{x:.2f}",
            '昨收': lambda x: f"{x:.2f}",
            '量比': lambda x: f"{x:.2f}",
            '换手率': lambda x: f"{x:.2f}%",
            '市盈率-动态': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '市净率': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '总市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '流通市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '涨速': lambda x: f"{x:.2f}%",
            '5分钟涨跌': lambda x: f"{x:+.2f}%",
            '60日涨跌幅': lambda x: f"{x:+.2f}%",
            '年初至今涨跌幅': lambda x: f"{x:+.2f}%"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in display_cols:
            display_cols.remove('序号')

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            if '代码' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['代码'].astype(str) == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的科创板实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"科创板股票 '{stock_info['名称']}' ({stock_info['代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为科创板股票 '{stock_info['名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的科创板股票，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(MAX_ROWS_TO_DISPLAY).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > MAX_ROWS_TO_DISPLAY:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - MAX_ROWS_TO_DISPLAY} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有科创板股票的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条科创板实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '总市值']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有科创板的实时行情数据概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询股票代码688001的最新价和涨跌幅。' 或 '查询中芯国际的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取科创板实时行情数据时发生错误: {e}"

@tool
def get_ab_shares_comparison_realtime_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有 AB 股比价的实时行情数据。
    该接口返回包括 B 股代码、B 股名称、B 股最新价、B 股涨跌幅、A 股代码、A 股名称、
    A 股最新价、A 股涨跌幅以及 A 股与 B 股的比价等全面的实时市场指标。

    用户可以选择提供一个股票代码（A股或B股）或公司名称（A股或B股）来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定 AB 股对的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的 AB 股对的股票代码（如 '200026' 或 '000026'）
                                          或公司名称（如 '飞亚达B' 或 '飞亚达A' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含 AB 股比价实时行情数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_zh_ab_comparison_em()

        if df.empty:
            return "未能获取 AB 股比价实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式
        format_rules = {
            '最新价B': lambda x: f"{x:.2f}",
            '涨跌幅B': lambda x: f"{x:+.2f}%",
            '最新价A': lambda x: f"{x:.2f}",
            '涨跌幅A': lambda x: f"{x:+.2f}%",
            '比价': lambda x: f"{x:.2f}"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in display_cols:
            display_cols.remove('序号')

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按 B股代码 或 A股代码 精确匹配
            if 'B股代码' in df_formatted.columns and 'A股代码' in df_formatted.columns:
                filtered_df = df_formatted[
                    (df_formatted['B股代码'].astype(str) == stock_identifier) |
                    (df_formatted['A股代码'].astype(str) == stock_identifier)
                ]

            # 如果没有代码匹配，按 B股名称 或 A股名称 精确匹配
            if filtered_df.empty:
                if 'B股名称' in df_formatted.columns:
                    filtered_df = df_formatted[df_formatted['B股名称'] == stock_identifier]
                if filtered_df.empty and 'A股名称' in df_formatted.columns:
                    filtered_df = df_formatted[df_formatted['A股名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按 B股名称 或 A股名称 模糊匹配
            if filtered_df.empty:
                if 'B股名称' in df_formatted.columns and 'A股名称' in df_formatted.columns:
                    filtered_df = df_formatted[
                        df_formatted['B股名称'].str.contains(stock_identifier, case=False, na=False) |
                        df_formatted['A股名称'].str.contains(stock_identifier, case=False, na=False)
                    ]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的 AB 股比价实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"AB 股对 '{stock_info['B股名称']}' ({stock_info['B股代码']}) / '{stock_info['A股名称']}' ({stock_info['A股代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为 AB 股对 '{stock_info['B股名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的 AB 股对，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(MAX_ROWS_TO_DISPLAY).iterrows():
                    output_lines.append(f"  - B股: {row['B股名称']} ({row['B股代码']}), A股: {row['A股名称']} ({row['A股代码']})")
                if len(filtered_df) > MAX_ROWS_TO_DISPLAY:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - MAX_ROWS_TO_DISPLAY} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有 AB 股的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条 AB 股比价实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['B股代码', 'B股名称', '最新价B', '涨跌幅B', 'A股代码', 'A股名称', '最新价A', '涨跌幅A', '比价']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有 AB 股比价的实时行情数据概览。")
            output_lines.append("如果您需要查询特定 AB 股的详细信息，请提供其 A 股或 B 股代码或名称。")
            output_lines.append("例如，您可以询问 '查询B股代码200026和A股代码000026的比价。' 或 '查询飞亚达的比价。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取 AB 股比价实时行情数据时发生错误: {e}"

@tool
def get_stock_historical_data(
    symbol: str,
    start_date: str,
    end_date: str,
    period: str = "daily",
    adjust: str = ""
) -> str:
    """
    获取东方财富网提供的指定沪深京 A 股上市公司的历史行情数据。
    用户可以指定股票代码、查询周期（日、周、月）、开始日期、结束日期以及复权方式。

    Args:
        symbol (str): 需要查询的股票代码（例如：'000001' 代表平安银行，'600000' 代表浦发银行）。
        start_date (str): 查询的开始日期，格式为 'YYYYMMDD'（例如：'20230101'）。
        end_date (str): 查询的结束日期，格式为 'YYYYMMDD'（例如：'20240101'）。
        period (str, optional): 数据周期，可选值包括 'daily' (日线), 'weekly' (周线), 'monthly' (月线)。
                                默认为 'daily'。
        adjust (str, optional): 复权类型，可选值包括 '' (不复权), 'qfq' (前复权), 'hfq' (后复权)。
                                默认为 '' (不复权)。
                                - 不复权 ('')：返回原始价格数据，不考虑除权除息。
                                - 前复权 ('qfq')：保持当前价格不变，将历史价格进行调整，使股价连续。
                                - 后复权 ('hfq')：保持历史价格不变，在每次股票权益事件发生后，调整当前股票价格，反映真实收益率。

    Returns:
        str: 一个格式化后的字符串，包含指定股票历史行情数据的完整详情。
             如果获取失败，则返回错误信息。
    """
    # 校验日期格式
    for date_str in [start_date, end_date]:
        try:
            datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            return f"错误：日期格式不正确。请使用 'YYYYMMDD' 格式，例如 '20230101'。您输入的日期是 '{date_str}'。"

    # 校验周期参数
    valid_periods = {"daily", "weekly", "monthly"}
    if period not in valid_periods:
        return f"错误：'period' 参数无效。可选值包括 {', '.join(valid_periods)}。您输入的是 '{period}'。"

    # 校验复权参数
    valid_adjusts = {"", "qfq", "hfq"}
    if adjust not in valid_adjusts:
        return f"错误：'adjust' 参数无效。可选值包括 {', '.join(valid_adjusts)}。您输入的是 '{adjust}'。"

    try:
        # 调用 API
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        if df.empty:
            return f"未能获取股票代码 '{symbol}' 从 {start_date} 到 {end_date} 的历史行情数据。可能该股票在该时间段内无交易数据，或代码有误。"

        output_lines = [f"成功获取股票代码 '{symbol}' 从 {start_date} 到 {end_date} 的 {period} 历史行情数据 (共 {len(df)} 条记录，复权方式: {adjust if adjust else '不复权'})。\n"]
        output_lines.append("=" * 100)

        # 格式化数值列
        df_display = df.copy()

        # 定义需要格式化的列及其格式
        format_rules = {
            '开盘': lambda x: f"{x:.2f}",
            '收盘': lambda x: f"{x:.2f}",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.2f}",
            '换手率': lambda x: f"{x:.2f}%"
        }

        for col, func in format_rules.items():
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: func(x) if pd.notna(x) else "N/A")

        # 仅显示部分列
        display_cols = ['日期', '开盘', '收盘', '涨跌幅', '成交额', '换手率']
        
        # 确保要显示的列都在 DataFrame 中
        display_cols = [col for col in display_cols if col in df_display.columns]

        # 直接返回整个 DataFrame 的字符串表示
        output_lines.append("历史行情数据详情:")
        df_string = df_display[display_cols].to_string(index=False)
        output_lines.append(df_string)

        output_lines.append("=" * 100)
        output_lines.append(f"提示：此为股票 '{symbol}' 从 {start_date} 到 {end_date} 的完整历史行情数据。")
        output_lines.append("如果您需要查询特定日期或特定指标的详细信息，请明确提出。")
        output_lines.append("例如，您可以询问 '股票000001在2024年5月28日的收盘价和成交量是多少？'")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票代码 '{symbol}' 历史行情数据时发生错误: {e}"

@tool
def get_stock_minute_data_em(
    symbol: str,
    start_date: str = "1979-09-01 09:32:00",
    end_date: str = "2222-01-01 09:32:00",
    period: str = "5",
    adjust: str = "",
    date: Optional[str] = None,
    full_day_data: bool = False
) -> str:
    """
    获取东方财富网提供的指定股票的分时行情数据。
    该接口可以获取指定股票、频率、复权调整和时间区间的分时数据。
    请注意以下限制：
    - 1 分钟数据 (period='1') 只返回近 5 个交易日数据且不复权 (adjust 必须为 '')。
    - 其他频率 (5, 15, 30, 60 分钟) 支持更长的历史数据和复权调整。

    Args:
        symbol (str): 股票代码（例如：'000001'）。
        start_date (str, optional): 查询的开始日期时间，格式为 'YYYY-MM-DD HH:MM:SS'。
                                    默认为 '1979-09-01 09:32:00'，表示从最早可用数据开始。
                                    当 'date' 参数被提供时，此参数会被忽略。
        end_date (str, optional): 查询的结束日期时间，格式为 'YYYY-MM-DD HH:MM:SS'。
                                  默认为 '2222-01-01 09:32:00'，表示到最新可用数据结束。
                                  当 'date' 参数被提供时，此参数会被忽略。
        period (str, optional): 分时数据周期，可选值包括 '1', '5', '15', '30', '60' (分钟)。
                                默认为 '5'。
        adjust (str, optional): 复权类型，可选值包括 '' (不复权), 'qfq' (前复权), 'hfq' (后复权)。
                                请注意，当 period='1' 时，adjust 必须为 ''。默认为 ''。
        date (Optional[str]): 指定要查询的单个日期，格式为 'YYYY-MM-DD'。例如：'2024-03-20'。
                              如果提供此参数，工具将返回该日期内的分时数据，并忽略 'start_date' 和 'end_date' 参数。
        full_day_data (bool, optional): 仅当 'date' 参数被提供时有效。
                                        如果设置为 True，则返回指定日期内的所有分时数据。
                                        如果设置为 False (默认)，则只返回该日期内的前5条和后5条分时数据作为摘要。

    Returns:
        str: 一个格式化后的字符串，包含指定股票分时行情数据的完整详情或摘要。
             如果获取失败或参数不符合要求，则返回错误信息。
    """
    # 校验分时数据周期参数
    valid_periods = {"1", "5", "15", "30", "60"}
    if period not in valid_periods:
        return f"错误：'period' 参数无效。可选值包括 {', '.join(valid_periods)}。您输入的是 '{period}'。"

    # 校验复权类型参数
    valid_adjusts = {"", "qfq", "hfq"}
    if adjust not in valid_adjusts:
        return f"错误：'adjust' 参数无效。可选值包括 {', '.join(valid_adjusts)}。您输入的是 '{adjust}'。"

    # 确保分时数据周期参数为1时不复权
    if period == "1" and adjust != "":
        return "错误：当 'period' 为 '1' 分钟时，'adjust' 参数必须为 '' (不复权)。"

    actual_start_date_for_api = start_date
    actual_end_date_for_api = end_date
    
    if date is not None:
        try:
            # 分离时间
            datetime.strptime(date, '%Y-%m-%d')
            # 查询指定日期
            actual_start_date_for_api = f"{date} 00:00:00"
            actual_end_date_for_api = f"{date} 23:59:59"
        except ValueError:
            return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2024-03-20'。您输入的是 '{date}'。"
    else:
        # date参数未给出, 使用原来的时间区间
        date_time_format = '%Y-%m-%d %H:%M:%S'
        for dt_str in [start_date, end_date]:
            try:
                datetime.strptime(dt_str, date_time_format)
            except ValueError:
                return f"错误：日期时间格式不正确。请使用 '{date_time_format}' 格式，例如 '2024-03-20 09:30:00'。您输入的是 '{dt_str}'。"


    try:
        # 调用 API
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol,
            start_date=actual_start_date_for_api,
            end_date=actual_end_date_for_api,
            period=period,
            adjust=adjust
        )

        if df.empty:
            if date is not None:
                return f"未能获取股票代码 '{symbol}' 在 {date} 的 {period} 分钟分时行情数据。可能该日期无交易数据或数据源暂无数据。"
            else:
                return f"未能获取股票代码 '{symbol}' 从 {start_date} 到 {end_date} 期间的 {period} 分钟分时行情数据。可能该时间段无交易数据或数据源暂无数据。"

        output_lines = []
        if date is not None:
            output_lines.append(f"成功获取股票代码 '{symbol}' 在 {date} 的 {period} 分钟分时行情数据 (共 {len(df)} 条记录，复权方式: {adjust if adjust else '不复权'})。\n")
        else:
            output_lines.append(f"成功获取股票代码 '{symbol}' 从 {start_date} 到 {end_date} 的 {period} 分钟分时行情数据 (共 {len(df)} 条记录，复权方式: {adjust if adjust else '不复权'})。\n")
        
        output_lines.append("=" * 100)

        # 格式化数值列
        df_display = df.copy()

        # 定义需要格式化的列及其格式
        common_format_rules = {
            '开盘': lambda x: f"{x:.2f}",
            '收盘': lambda x: f"{x:.2f}",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
        }

        for col, func in common_format_rules.items():
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: func(x) if pd.notna(x) else "N/A")

        # 根据不同分时数据周期, 显示不同的列
        if period == "1":
            specific_format_rules = {
                '均价': lambda x: f"{x:.2f}"
            }
            display_cols = ['时间', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '均价']
        else:
            specific_format_rules = {
                '涨跌幅': lambda x: f"{x:+.2f}%",
                '涨跌额': lambda x: f"{x:+.2f}",
                '振幅': lambda x: f"{x:.2f}%",
                '换手率': lambda x: f"{x:.2f}%"
            }
            display_cols = ['时间', '开盘', '收盘', '涨跌幅', '成交额', '换手率']

        for col, func in specific_format_rules.items():
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: func(x) if pd.notna(x) else "N/A")

        # 确保要显示的列都在 DataFrame 中
        display_cols = [col for col in display_cols if col in df_display.columns]

        if date is not None and full_day_data:
            # 返回指定日期内的所有分时数据
            output_lines.append(f"分时行情数据详情 (日期: {date}, 完整数据):")
            df_string = df_display[display_cols].to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{symbol}' 在 {date} 的完整 {period} 分钟分时行情数据。")
            output_lines.append("如果您需要查询特定时间点或特定指标的详细信息，请明确提出。")
        else:
            # 返回该日期内的前5条和后5条分时数据作为摘要
            output_lines.append("数据概览 (前5行):")
            df_string_head = df_display[display_cols].head(5).to_string(index=False)
            output_lines.append(df_string_head)

            if len(df) > 5:
                output_lines.append("\n数据概览 (后5行):")
                df_string_tail = df_display[display_cols].tail(5).to_string(index=False)
                output_lines.append(df_string_tail)

            output_lines.append("=" * 100)
            if date is not None:
                output_lines.append(f"提示：此为股票 '{symbol}' 在 {date} 的 {period} 分钟分时行情数据概览。")
                output_lines.append(f"如果需要查看该日期的所有分时数据，请在请求中明确说明，例如：'获取股票000001在{date}的1分钟分时数据，并显示完整数据。'")
            else:
                output_lines.append(f"提示：此为股票 '{symbol}' 的 {period} 分钟分时行情数据概览。")
            output_lines.append("如果您需要查询特定时间点或特定指标的详细信息，请明确提出。")
            output_lines.append("例如，您可以询问 '股票000001在2024-03-20 14:00:00的收盘价是多少？'")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票代码 '{symbol}' 分时行情数据时发生错误: {e}"
    
@tool
def get_stock_intraday_data_em(
    symbol: str,
    full_data: Optional[bool] = False,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> str:
    """
    获取东方财富网提供的指定股票最近一个交易日的分时数据，包含盘前数据。
    这些数据包括时间、成交价、成交手数以及买卖盘性质（买盘、卖盘、中性盘）。

    用户可以选择获取数据的摘要（前几行、后几行和总记录数），或者获取完整的日内分时数据。
    此外，可以指定一个具体的时间范围来过滤数据。

    Args:
        symbol (str): 必填参数。需要查询的股票代码（例如：'000001'）。
        full_data (Optional[bool]): 可选参数。如果为 True，则返回指定股票最近一个交易日的所有分时数据；
                                     如果为 False 或未提供，则返回数据的摘要（前5行、后5行和总记录数）。
                                     默认为 False。
        start_time (Optional[str]): 可选参数。查询的开始时间，格式为 'HH:MM:SS'（例如：'09:30:00'）。
                                     如果提供，数据将从该时间点开始过滤。
        end_time (Optional[str]): 可选参数。查询的结束时间，格式为 'HH:MM:SS'（例如：'14:00:00'）。
                                   如果提供，数据将过滤到该时间点结束。

    Returns:
        str: 一个格式化后的字符串，包含指定股票最近一个交易日分时数据。
             如果获取失败，则返回错误信息。
    """
    time_format = '%H:%M:%S'

    # 校验时间格式
    if start_time:
        try:
            datetime.strptime(start_time, time_format).time()
        except ValueError:
            return f"错误：'start_time' 参数格式不正确。请使用 '{time_format}' 格式，例如 '09:30:00'。您输入的是 '{start_time}'。"
    if end_time:
        try:
            datetime.strptime(end_time, time_format).time()
        except ValueError:
            return f"错误：'end_time' 参数格式不正确。请使用 '{time_format}' 格式，例如 '15:00:00'。您输入的是 '{end_time}'。"
    
    # 检查开始时间是否晚于结束时间
    if start_time and end_time:
        if datetime.strptime(start_time, time_format).time() >= datetime.strptime(end_time, time_format).time():
            return "错误：'start_time' 必须早于 'end_time'。"

    try:
        # 调用 API
        df = ak.stock_intraday_em(symbol=symbol)

        if df.empty:
            return f"未能获取股票代码 '{symbol}' 最近一个交易日的分时数据。可能当前非交易时间或数据源暂无数据。"

        # 应用时间范围过滤
        if start_time or end_time:
            # 将 DataFrame 的 '时间' 列转换为 time 对象
            df['_time_obj'] = df['时间'].apply(lambda x: datetime.strptime(x, time_format).time())

            filter_start_time = datetime.strptime(start_time, time_format).time() if start_time else datetime.min.time()
            filter_end_time = datetime.strptime(end_time, time_format).time() if end_time else datetime.max.time()

            df = df[(df['_time_obj'] >= filter_start_time) & (df['_time_obj'] <= filter_end_time)].copy()
            df = df.drop(columns=['_time_obj'])

            if df.empty:
                time_range_str = ""
                if start_time and end_time:
                    time_range_str = f"在 {start_time} 到 {end_time} 之间"
                elif start_time:
                    time_range_str = f"从 {start_time} 开始"
                elif end_time:
                    time_range_str = f"到 {end_time} 结束"
                return f"未能获取股票代码 '{symbol}' 最近一个交易日 {time_range_str} 的分时数据。该时间段内可能无交易数据。"

        # 格式化数值列
        df_display = df.copy()

        # 定义需要格式化的列及其格式
        format_rules = {
            '成交价': lambda x: f"{x:.2f}",
            '手数': lambda x: f"{int(x):,}"
        }

        for col, func in format_rules.items():
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: func(x) if pd.notna(x) else "N/A")

        # 仅显示部分列
        display_cols = ['时间', '成交价', '手数', '买卖盘性质']
        
        # 确保要显示的列都在DataFrame中
        display_cols = [col for col in display_cols if col in df_display.columns]

        # 构建输出消息头部
        time_range_desc = ""
        if start_time and end_time:
            time_range_desc = f"从 {start_time} 到 {end_time}"
        elif start_time:
            time_range_desc = f"从 {start_time} 开始"
        elif end_time:
            time_range_desc = f"到 {end_time} 结束"
        
        output_lines = [f"成功获取股票代码 '{symbol}' 最近一个交易日 {time_range_desc} 的分时数据 (共 {len(df_display)} 条记录)。\n"]
        output_lines.append("=" * 100)

        if full_data:
            # 返回完整数据
            output_lines.append(f"以下是股票 '{symbol}' 完整的日内分时数据{time_range_desc}：")
            df_string = df_display[display_cols].to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{symbol}' 完整的日内分时数据{time_range_desc}。")
        else:
            # 返回摘要数据
            output_lines.append("数据概览 (前5行):")
            df_string_head = df_display[display_cols].head(5).to_string(index=False)
            output_lines.append(df_string_head)

            # 如果数据超过5行，显示最后5行
            if len(df_display) > 5:
                output_lines.append("\n数据概览 (后5行):")
                df_string_tail = df_display[display_cols].tail(5).to_string(index=False)
                output_lines.append(df_string_tail)

            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{symbol}' 最近一个交易日的日内分时数据概览{time_range_desc}。")
            output_lines.append("如果您需要查询特定时间点或特定成交信息的详细数据，请明确提出。")
            
            # 提示如何获取完整数据或指定时间范围
            # 如果没有时间过滤，提示可以加时间过滤
            if not start_time and not end_time:
                output_lines.append(f"如果您需要获取所有分时数据，请在请求中明确说明，例如：'获取股票000001的完整分时数据'。")
                output_lines.append(f"如果您需要特定时间段的数据，例如：'获取股票000001在09:30:00到10:00:00之间的分时数据'。")
            # 如果有时间过滤但不是完整数据，提示可以获取完整数据
            elif not full_data:
                 output_lines.append(f"如果您需要获取该时间段内的所有分时数据，请在请求中明确说明，例如：'获取股票000001在{start_time if start_time else ''}到{end_time if end_time else ''}之间的完整分时数据'。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票代码 '{symbol}' 日内分时数据时发生错误: {e}"

@tool
def get_a_stock_pre_market_min_data_em(
    symbol: str, 
    start_time: Optional[str] = "09:00:00", 
    end_time: Optional[str] = "15:40:00",
    return_full_data: bool = False
) -> str:
    """
    获取东方财富网-股票行情中，指定A股股票最近一个交易日的分钟级数据，包含盘前分钟数据。
    该接口返回每分钟的开盘价、收盘价、最高价、最低价、成交量（手）、成交额和最新价。

    用户必须提供A股股票代码。
    可选地，可以提供时间范围来过滤数据，格式为 "HH:MM:SS"。
    默认查询时间范围为09:00:00到15:40:00，覆盖盘前和正常交易时段。
    可以通过 'return_full_data' 参数选择返回指定时间范围内的全部数据，或返回数据概览和示例。

    Args:
        symbol (str): 必需参数。A股股票代码，例如 "000001" (平安银行) 或 "600000" (浦发银行)。
        start_time (Optional[str]): 可选参数。查询的起始时间，格式为 "HH:MM:SS"，例如 "09:00:00"。
                                     默认为 "09:00:00"。
        end_time (Optional[str]): 可选参数。查询的结束时间，格式为 "HH:MM:SS"，例如 "15:40:00"。
                                   默认为 "15:40:00"。
        return_full_data (bool): 可选参数。默认为 False。
                                 如果设置为 True，将返回指定时间范围内的所有分钟数据（以字符串形式）。
                                 如果设置为 False，将返回数据的概览信息以及前5分钟和后5分钟的示例数据。

    Returns:
        str: 一个格式化后的字符串，包含指定A股股票最近一个交易日的盘前分钟数据摘要或全部数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证股票代码格式
    if not symbol.isdigit() or len(symbol) != 6:
        return f"错误：股票代码 '{symbol}' 格式不正确。A股代码通常是6位数字，例如 '000001' 或 '600000'。"

    # 验证时间格式
    def is_valid_time(time_str):
        try:
            datetime.strptime(time_str, "%H:%M:%S").time()
            return True
        except ValueError:
            return False

    if start_time and not is_valid_time(start_time):
        return f"错误：'start_time' 参数格式不正确。请提供 'HH:MM:SS' 格式的时间，例如 '09:00:00'。"
    if end_time and not is_valid_time(end_time):
        return f"错误：'end_time' 参数格式不正确。请提供 'HH:MM:SS' 格式的时间，例如 '15:40:00'。"

    try:
        # 调用 API
        df = ak.stock_zh_a_hist_pre_min_em(
            symbol=symbol, 
            start_time=start_time, 
            end_time=end_time
        )

        if df.empty:
            return f"未能获取股票代码 '{symbol}' 在指定时间范围 ({start_time}-{end_time}) 的盘前分钟数据。可能该股票无数据、非交易日或数据源暂无数据。"

        # 提取数据日期
        data_date = pd.to_datetime(df['时间'].iloc[0]).strftime('%Y年%m月%d日')

        output_lines = [f"A股股票 '{symbol}' 在 {data_date} 的分钟级行情数据 (数据来源: 东方财富网)。\n"]
        output_lines.append("=" * 100)

        if return_full_data:
            # 返回全部数据
            output_lines.append(f"以下是 {start_time} 至 {end_time} 期间的所有分钟数据：")
            # 限制输出的列
            display_cols = ['时间', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '最新价']
            output_lines.append(df[display_cols].to_string(index=False))
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{symbol}' 在 {data_date} 的全部分钟级行情数据。")
            output_lines.append("成交量单位为 '手' (1手=100股)。")
        else:
            # 返回数据概览和示例
            total_minutes = len(df)
            period_open = df['开盘'].iloc[0]
            period_close = df['收盘'].iloc[-1]
            period_high = df['最高'].max()
            period_low = df['最低'].min()
            total_volume = df['成交量'].sum()
            total_amount = df['成交额'].sum()

            output_lines.append(f"查询时间范围: {start_time} 至 {end_time}")
            output_lines.append(f"总分钟数: {total_minutes} 分钟")
            output_lines.append(f"期间开盘价: {period_open:.2f}")
            output_lines.append(f"期间收盘价: {period_close:.2f}")
            output_lines.append(f"期间最高价: {period_high:.2f}")
            output_lines.append(f"期间最低价: {period_low:.2f}")
            output_lines.append(f"总成交量: {total_volume:,.0f} 手 ({total_volume * 100:,.0f} 股)")
            output_lines.append(f"总成交额: {total_amount:,.2f} 元")
            
            output_lines.append("\n--- 数据示例 (前5分钟) ---")
            # 仅显示部分列
            display_cols = ['时间', '开盘', '收盘', '最高', '最低', '成交量', '成交额']
            df_string = df[display_cols].head(5).to_string(index=False)
            output_lines.append(df_string)

            # 如果数据量大，显示尾部
            if total_minutes > 10:
                output_lines.append("\n--- 数据示例 (后5分钟) ---")
                df_string_tail = df[display_cols].tail(5).to_string(index=False)
                output_lines.append(df_string_tail)

            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{symbol}' 在 {data_date} 的分钟级行情数据概览。")
            output_lines.append("成交量单位为 '手' (1手=100股)。如需获取全部数据，请将 'return_full_data' 参数设置为 True。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票 '{symbol}' 盘前分钟数据时发生错误: {e}"
    
@tool
def get_b_shares_realtime_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有 B 股上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率、总市值、流通市值、涨速、5分钟涨跌、
    60日涨跌幅和年初至今涨跌幅等全面的实时市场指标。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定 B 股的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的 B 股的股票代码（如 '900901'）
                                          或公司名称（如 '上工B股' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含 B 股实时行情数据。
             如果获取失败，则返回错误信息。
    """
    try:
        # 调用 API
        df = ak.stock_zh_b_spot_em()

        if df.empty:
            return "未能获取 B 股实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式
        format_rules = {
            '最新价': lambda x: f"{x:.3f}",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.3f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '最高': lambda x: f"{x:.3f}",
            '最低': lambda x: f"{x:.3f}",
            '今开': lambda x: f"{x:.3f}",
            '昨收': lambda x: f"{x:.3f}",
            '量比': lambda x: f"{x:.2f}",
            '换手率': lambda x: f"{x:.2f}%",
            '市盈率-动态': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '市净率': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '总市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '流通市值': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '涨速': lambda x: f"{x:.2f}%",
            '5分钟涨跌': lambda x: f"{x:+.2f}%",
            '60日涨跌幅': lambda x: f"{x:+.2f}%",
            '年初至今涨跌幅': lambda x: f"{x:+.2f}%"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in display_cols:
            display_cols.remove('序号')

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            if '代码' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['代码'].astype(str) == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的 B 股实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"B 股 '{stock_info['名称']}' ({stock_info['代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为 B 股 '{stock_info['名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的 B 股，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(MAX_ROWS_TO_DISPLAY).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > MAX_ROWS_TO_DISPLAY:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - MAX_ROWS_TO_DISPLAY} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有 B 股的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条 B 股实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '总市值']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有 B 股的实时行情数据概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询股票代码900901的最新价和涨跌幅。' 或 '查询上工B股的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取 B 股实时行情数据时发生错误: {e}"

@tool
def get_b_stock_spot_data_sina(
    stock_identifier: Optional[str] = None
) -> str:
    """
    获取新浪财经-B股市场的实时行情数据。
    该接口返回所有B股股票的最新价格、涨跌额、涨跌幅、买入价、卖出价、
    昨日收盘价、今日开盘价、最高价、最低价、成交量（股）和成交额（元）。

    用户可以选择提供一个股票代码或名称来查询特定B股的行情。
    如果未提供 'stock_identifier'，则返回所有B股的实时数据概览。
    警告：频繁调用此接口可能会导致IP暂时被封禁，建议增加调用间隔。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的B股代码（如 'sh900901'）或公司名称（如 '云赛B股'）。
                                          如果未提供，则返回所有B股的实时数据概览。

    Returns:
        str: 一个格式化后的字符串，包含指定B股或所有B股的实时行情数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        if col_name in ['最新价', '买入', '卖出', '昨收', '今开', '最高', '最低']:
            return f"{float(value):.3f}"
        elif col_name == '涨跌额':
            return f"{float(value):+.3f}"
        elif col_name == '涨跌幅':
            return f"{float(value):+.2f}%"
        elif col_name == '成交量':
            return f"{int(value):,}"
        elif col_name == '成交额':
            return f"{float(value):,.2f}"
        return str(value)

    try:
        # 调用 API
        df = ak.stock_zh_b_spot()

        if df.empty:
            return "未能获取B股市场的实时行情数据。可能当前无数据或数据源暂无数据。"

        # 根据 stock_identifier 参数进行过滤或返回所有数据
        if stock_identifier:
            identifier_lower = stock_identifier.strip().lower()

            # 按代码精确匹配
            matched_by_code = df[df['代码'].str.lower() == identifier_lower]

            # 如果没有代码匹配，按名称精确匹配
            matched_by_name = df[df['名称'].str.lower() == identifier_lower]

            # 合并匹配
            matched_stocks = pd.concat([matched_by_code, matched_by_name]).drop_duplicates()

            # 如果没有代码匹配，按名称精确匹配
            if matched_stocks.empty:
                matched_stocks = df[df['名称'].str.lower().str.contains(identifier_lower, na=False)]

            if matched_stocks.empty:
                return f"未能找到与 '{stock_identifier}' 匹配的B股股票。请检查股票代码或名称是否正确。"
            elif len(matched_stocks) > 1:
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的B股，请提供更具体的股票名称或直接提供股票代码："]
                output_lines.append("=" * 100)
                for index, row in matched_stocks.head(10).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(matched_stocks) > 10:
                    output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                return "\n".join(output_lines)
            else:
                # 找到唯一匹配
                target_stock_series = matched_stocks.iloc[0]
                
                output_lines = [f"B股股票 '{target_stock_series['名称']}' ({target_stock_series['代码']}) 的实时行情数据 (数据来源: 新浪财经)。\n"]
                output_lines.append("=" * 100)

                # 显示指定列
                single_stock_display_order = [
                    '最新价', '涨跌额', '涨跌幅', '今开', '最高', '最低', '昨收',
                    '买入', '卖出', '成交量', '成交额'
                ]
                
                # 逐行输出特定股票的指定列的详细信息
                for item_key in single_stock_display_order:
                    if item_key in target_stock_series.index:
                        output_lines.append(f"{item_key}: {format_value(target_stock_series[item_key], item_key)}")
                
                # 添加额外信息
                for item_key, value in target_stock_series.items():
                    if item_key not in single_stock_display_order and item_key not in ['代码', '名称']:
                        output_lines.append(f"{item_key}: {format_value(value, item_key)}")

                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为B股股票 '{target_stock_series['名称']}' ({target_stock_series['代码']}) 的实时行情数据。")
                output_lines.append("成交量单位为 '股'，成交额单位为 '元'。")
                output_lines.append("警告：频繁调用此接口可能会导致IP暂时被封禁，建议增加调用间隔。")
                
                return "\n".join(output_lines)

        else:
            # 如果未提供股票代码或名称，则返回所有 B 股的实时数据概览
            output_lines = ["B股市场实时行情数据 (数据来源: 新浪财经)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("警告：频繁调用此接口可能会导致IP暂时被封禁，建议增加调用间隔。")
            output_lines.append(f"共找到 {len(df)} 只 B股股票的实时行情数据。")
            output_lines.append("=" * 100)

            # 仅显示部分关键列用于概览
            display_cols_summary = [
                '代码', '名称', '最新价', '涨跌额', '涨跌幅', '成交量', '成交额'
            ]

            available_cols_summary = [col for col in display_cols_summary if col in df.columns]
            
            # 格式化Dataframe
            formatted_df_summary = df[available_cols_summary].copy()
            for col in available_cols_summary:
                if col not in ['代码', '名称']:
                    formatted_df_summary[col] = formatted_df_summary[col].apply(lambda x: format_value(x, col))

            # 限制输出行数
            num_rows_to_show = MAX_ROWS_TO_DISPLAY
            
            if len(formatted_df_summary) > 0:
                output_lines.append("\n--- 数据示例 (前MAX_ROWS_TO_DISPLAY只股票) ---")
                output_lines.append(formatted_df_summary.head(num_rows_to_show).to_string(index=False))

                if len(formatted_df_summary) > num_rows_to_show * 2:
                    output_lines.append("\n--- 数据示例 (后MAX_ROWS_TO_DISPLAY只股票) ---")
                    output_lines.append(formatted_df_summary.tail(num_rows_to_show).to_string(index=False))
            else:
                output_lines.append("未获取到任何B股实时行情数据。")

            output_lines.append("=" * 100)
            output_lines.append("提示：数据包含股票代码、名称、最新价、涨跌额、涨跌幅、买卖价、开盘价、最高价、最低价、成交量(股)和成交额(元)。")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取B股实时行情数据时发生错误: {e}"

@tool
def get_b_shares_historical_data(
    symbol: str,
    start_date: str = "",
    end_date: str = "",
    adjust: str = ""
) -> str:
    """
    获取指定 B 股上市公司的历史行情日频率数据或复权因子。
    数据来源于新浪财经，历史数据按日频率更新。

    Args:
        symbol (str): 股票代码，必须包含市场标识，例如 'sh900901' (上海B股) 或 'sz200002' (深圳B股)。
        start_date (str, optional): 查询的开始日期，格式为 'YYYYMMDD'（例如：'20230101'）。
                                    仅当 'adjust' 为历史行情数据类型时有效。
                                    默认为空字符串。
        end_date (str, optional): 查询的结束日期，格式为 'YYYYMMDD'（例如：'20240101'）。
                                  仅当 'adjust' 为历史行情数据类型时有效。
                                  默认为空字符串。
        adjust (str, optional): 复权类型。
                                默认为 '' (不复权)。
                                可选值包括：
                                - '' (不复权): 返回原始价格数据。
                                - 'qfq' (前复权): 保持当前价格不变，调整历史价格。
                                - 'hfq' (后复权): 保持历史价格不变，调整当前价格，反映真实收益率。
                                - 'qfq-factor' (前复权因子): 返回前复权因子数据。
                                - 'hfq-factor' (后复权因子)。

    Returns:
        str: 一个格式化后的字符串，包含指定股票历史行情数据或复权因子的摘要。
             如果获取失败或参数不符合要求，则返回错误信息。
    """
    # 校验 symbol 格式
    if not (symbol.startswith("sh") or symbol.startswith("sz")) or not symbol[2:].isdigit():
        return f"错误：股票代码 '{symbol}' 格式不正确。B股代码应包含市场标识，例如 'sh900901' 或 'sz200002'。"

    valid_adjusts = {"", "qfq", "hfq", "qfq-factor", "hfq-factor"}
    if adjust not in valid_adjusts:
        return f"错误：'adjust' 参数无效。可选值包括 {', '.join(valid_adjusts)}。您输入的是 '{adjust}'。"

    # 只有在获取历史行情数据时才需要校验日期
    is_historical_data_query = adjust in {"", "qfq", "hfq"}
    if is_historical_data_query:
        if not start_date or not end_date:
            return "错误：查询历史行情数据时，'start_date' 和 'end_date' 不能为空。"
        for date_str in [start_date, end_date]:
            try:
                datetime.strptime(date_str, '%Y%m%d')
            except ValueError:
                return f"错误：日期格式不正确。请使用 'YYYYMMDD' 格式，例如 '20230101'。您输入的日期是 '{date_str}'。"

    try:
        # 调用 API
        df = ak.stock_zh_b_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        if df.empty:
            if is_historical_data_query:
                return f"未能获取股票代码 '{symbol}' 从 {start_date} 到 {end_date} 的历史行情数据。可能该股票在该时间段内无交易数据，或代码有误。"
            else:
                return f"未能获取股票代码 '{symbol}' 的复权因子数据 ({adjust})。可能数据源暂无数据或代码有误。"

        output_lines = []
        if is_historical_data_query:
            output_lines.append(f"成功获取股票代码 '{symbol}' 从 {start_date} 到 {end_date} 的历史行情数据 (共 {len(df)} 条记录，复权方式: {adjust if adjust else '不复权'})。\n")
            output_lines.append("=" * 100)

            df_display = df.copy()

            # 将每列重命名
            df_display.rename(columns={
                'date': '日期',
                'close': '收盘',
                'high': '最高',
                'low': '最低',
                'open': '开盘',
                'volume': '成交量',
                'outstanding_share': '流通股本',
                'turnover': '换手率'
            }, inplace=True)

            format_rules = {
                '开盘': lambda x: f"{x:.3f}",
                '收盘': lambda x: f"{x:.3f}",
                '最高': lambda x: f"{x:.3f}",
                '最低': lambda x: f"{x:.3f}",
                '成交量': lambda x: f"{int(x):,}股",
                '流通股本': lambda x: f"{int(x):,}股",
                '换手率': lambda x: f"{x:.4f}%"
            }

            for col, func in format_rules.items():
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: func(x) if pd.notna(x) else "N/A")

            display_cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '换手率']
            # 确保显示的列存在
            display_cols = [col for col in display_cols if col in df_display.columns]

            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")
            df_string_head = df_display[display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string_head)

            if len(df) > MAX_ROWS_TO_DISPLAY:
                output_lines.append("\n数据概览 (后MAX_ROWS_TO_DISPLAY行):")
                df_string_tail = df_display[display_cols].tail(MAX_ROWS_TO_DISPLAY).to_string(index=False)
                output_lines.append(df_string_tail)

            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{symbol}' 的历史行情数据概览。")
            output_lines.append("如果您需要查询特定日期或特定指标的详细信息，请明确提出。")
            output_lines.append("例如，您可以询问 '股票sh900901在2024-07-22的收盘价和成交量是多少？'")

        # 复权因子数据概览
        else:
            factor_col_name = 'qfq_factor' if adjust == 'qfq-factor' else 'hfq_factor'
            output_lines.append(f"成功获取股票代码 '{symbol}' 的 {adjust} 复权因子数据 (共 {len(df)} 条记录)。\n")
            output_lines.append("=" * 100)

            df_display = df.copy()
            # 重命名列
            df_display.rename(columns={'date': '日期', factor_col_name: '复权因子'}, inplace=True)

            # 格式化复权因子
            if '复权因子' in df_display.columns:
                df_display['复权因子'] = df_display['复权因子'].apply(lambda x: f"{x:.16f}" if pd.notna(x) else "N/A")

            display_cols = ['日期', '复权因子']
            display_cols = [col for col in display_cols if col in df_display.columns]

            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")
            df_string_head = df_display[display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string_head)

            if len(df) > MAX_ROWS_TO_DISPLAY:
                output_lines.append("\n数据概览 (后MAX_ROWS_TO_DISPLAY行):")
                df_string_tail = df_display[display_cols].tail(MAX_ROWS_TO_DISPLAY).to_string(index=False)
                output_lines.append(df_string_tail)

            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{symbol}' 的复权因子数据概览。")
            output_lines.append("如果您需要查询特定日期的复权因子，请明确提出。")
            output_lines.append("例如，您可以询问 '股票sh900901在2024-06-28的前复权因子是多少？'")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票代码 '{symbol}' 历史行情数据或复权因子时发生错误: {e}"

@tool
def get_b_shares_minute_data(
    symbol: str,
    period: str,
    adjust: str = "",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    full_data: Optional[bool] = False
) -> str:
    """
    获取新浪财经提供的指定 B 股股票或指数的最近一个交易日的历史分时行情数据。
    用户可以指定股票代码、数据频率（1, 5, 15, 30, 60 分钟）以及复权方式。

    Args:
        symbol (str): 股票代码，必须包含市场标识，例如 'sh900901' (上海B股) 或 'sz200002' (深圳B股)。
        period (str): 分时数据周期，可选值包括 '1', '5', '15', '30', '60' (分钟)。
                      例如：'1' 代表 1 分钟数据，'5' 代表 5 分钟数据。
        adjust (str, optional): 复权类型，可选值包括 '' (不复权), 'qfq' (前复权), 'hfq' (后复权)。
                                默认为 ''。
        start_time (Optional[str]): 可选参数。查询的开始时间，格式为 'HH:MM:SS'（例如：'09:30:00'）。
                                     如果提供，数据将从该时间点开始过滤。
        end_time (Optional[str]): 可选参数。查询的结束时间，格式为 'HH:MM:SS'（例如：'14:00:00'）。
                                   如果提供，数据将过滤到该时间点结束。
        full_data (Optional[bool]): 可选参数。如果为 True，则返回指定股票在给定时间范围内的所有分时数据；
                                     如果为 False 或未提供，则返回数据的摘要（前MAX_ROWS_TO_DISPLAY行、后MAX_ROWS_TO_DISPLAY行和总记录数）。
                                     默认为 False。

    Returns:
        str: 一个格式化后的字符串，包含指定股票最近一个交易日分时行情数据的完整详情或摘要。
             如果获取失败或参数不符合要求，则返回错误信息。
    """
    # 校验 symbol 格式
    if not (symbol.startswith("sh") or symbol.startswith("sz")) or not symbol[2:].isdigit():
        return f"错误：股票代码 '{symbol}' 格式不正确。B股代码应包含市场标识，例如 'sh900901' 或 'sz200002'。"

    # 校验 period 参数
    valid_periods = {"1", "5", "15", "30", "60"}
    if period not in valid_periods:
        return f"错误：'period' 参数无效。可选值包括 {', '.join(valid_periods)}。您输入的是 '{period}'。"

    # 校验 adjust 参数
    valid_adjusts = {"", "qfq", "hfq"}
    if adjust not in valid_adjusts:
        return f"错误：'adjust' 参数无效。可选值包括 {', '.join(valid_adjusts)}。您输入的是 '{adjust}'。"

    time_format = '%H:%M:%S'
    # 校验时间格式
    if start_time:
        try:
            datetime.strptime(start_time, time_format).time()
        except ValueError:
            return f"错误：'start_time' 参数格式不正确。请使用 '{time_format}' 格式，例如 '09:30:00'。您输入的是 '{start_time}'。"
    if end_time:
        try:
            datetime.strptime(end_time, time_format).time()
        except ValueError:
            return f"错误：'end_time' 参数格式不正确。请使用 '{time_format}' 格式，例如 '15:00:00'。您输入的是 '{end_time}'。"
    
    # 检查开始时间是否晚于结束时间
    if start_time and end_time:
        if datetime.strptime(start_time, time_format).time() >= datetime.strptime(end_time, time_format).time():
            return "错误：'start_time' 必须早于 'end_time'。"

    try:
        # 调用 API
        df = ak.stock_zh_b_minute(
            symbol=symbol,
            period=period,
            adjust=adjust
        )

        if df.empty:
            return f"未能获取股票代码 '{symbol}' 最近一个交易日的 {period} 分钟分时行情数据。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列以提高可读性
        df_display = df.copy()

        # 重命名列
        df_display.rename(columns={
            'day': '时间',
            'open': '开盘',
            'high': '最高',
            'low': '最低',
            'close': '收盘',
            'volume': '成交量'
        }, inplace=True)

        # 如果存在时间参数,则按给定时间过滤
        if start_time or end_time:
            # 将"时间"列转换为datetime类型
            df_display['时间_time_obj'] = pd.to_datetime(df_display['时间']).dt.time

            filter_start_time = datetime.strptime(start_time, time_format).time() if start_time else datetime.min.time()
            filter_end_time = datetime.strptime(end_time, time_format).time() if end_time else datetime.max.time()

            df_display = df_display[(df_display['时间_time_obj'] >= filter_start_time) & 
                                    (df_display['时间_time_obj'] <= filter_end_time)].copy()
            df_display = df_display.drop(columns=['时间_time_obj'])

            if df_display.empty:
                time_range_str = ""
                if start_time and end_time:
                    time_range_str = f"在 {start_time} 到 {end_time} 之间"
                elif start_time:
                    time_range_str = f"从 {start_time} 开始"
                elif end_time:
                    time_range_str = f"到 {end_time} 结束"
                return f"未能获取股票代码 '{symbol}' 最近一个交易日 {time_range_str} 的 {period} 分钟分时数据。该时间段内可能无交易数据。"

        # 定义格式化规则
        format_rules = {
            '开盘': lambda x: f"{x:.3f}",
            '最高': lambda x: f"{x:.3f}",
            '最低': lambda x: f"{x:.3f}",
            '收盘': lambda x: f"{x:.3f}",
            '成交量': lambda x: f"{int(x):,}股"
        }

        for col, func in format_rules.items():
            if col in df_display.columns:
                # 将"时间"格式化为字符串
                if col == '时间':
                    df_display[col] = pd.to_datetime(df_display[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    df_display[col] = df_display[col].apply(lambda x: func(x) if pd.notna(x) else "N/A")

        # 显示部分列
        display_cols = ['时间', '开盘', '收盘', '最高', '最低', '成交量']
        display_cols = [col for col in display_cols if col in df_display.columns]

        # 输出的背景信息
        time_range_desc = ""
        if start_time and end_time:
            time_range_desc = f"从 {start_time} 到 {end_time}"
        elif start_time:
            time_range_desc = f"从 {start_time} 开始"
        elif end_time:
            time_range_desc = f"到 {end_time} 结束"
        
        output_lines = [
            f"成功获取股票代码 '{symbol}' 最近一个交易日 {time_range_desc} 的 {period} 分钟分时行情数据 (共 {len(df_display)} 条记录，复权方式: {adjust if adjust else '不复权'})。\n"
        ]
        output_lines.append("=" * 100)

        if full_data:
            # 返回所有数据
            output_lines.append(f"以下是股票 '{symbol}' 完整的日内分时数据{time_range_desc}：")
            df_string = df_display[display_cols].to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{symbol}' 完整的日内分时数据{time_range_desc}。")
        else:
            # 返回数据概览
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")
            df_string_head = df_display[display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string_head)

            if len(df_display) > MAX_ROWS_TO_DISPLAY:
                output_lines.append("\n数据概览 (后MAX_ROWS_TO_DISPLAY行):")
                df_string_tail = df_display[display_cols].tail(MAX_ROWS_TO_DISPLAY).to_string(index=False)
                output_lines.append(df_string_tail)

            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{symbol}' 最近一个交易日的 {period} 分钟分时行情数据概览{time_range_desc}。")
            output_lines.append("如果您需要查询特定时间点或特定指标的详细信息，请明确提出。")
            
            # 提出引导建议
            if not start_time and not end_time:
                output_lines.append(f"如果您需要获取所有分时数据，请在请求中明确说明，例如：'获取股票{symbol}的完整{period}分钟分时数据'。")
                output_lines.append(f"如果您需要特定时间段的数据，例如：'获取股票{symbol}在09:30:00到10:00:00之间的{period}分钟分时数据'。")
            elif not full_data:
                 output_lines.append(f"如果您需要获取该时间段内的所有分时数据，请在请求中明确说明，例如：'获取股票{symbol}在{start_time if start_time else ''}到{end_time if end_time else ''}之间的完整{period}分钟分时数据'。")


        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票代码 '{symbol}' 分时行情数据时发生错误: {e}"

@tool
def get_risk_warning_stocks_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的当前交易日风险警示板（ST/*ST 股票）所有 A 股上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率等全面的实时市场指标。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定风险警示股票的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的风险警示股票的股票代码（如 '600000'）
                                          或公司名称（如 '*ST天山' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含风险警示板 A 股实时行情数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_zh_a_st_em()

        if df.empty:
            return "未能获取风险警示板股票的实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '最新价': lambda x: f"{x:.2f}",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '今开': lambda x: f"{x:.2f}",
            '昨收': lambda x: f"{x:.2f}",
            '量比': lambda x: f"{x:.2f}",
            '换手率': lambda x: f"{x:.2f}%",
            '市盈率-动态': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '市净率': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in display_cols:
            display_cols.remove('序号')

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            if '代码' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['代码'].astype(str) == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的风险警示股票实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"风险警示股票 '{stock_info['名称']}' ({stock_info['代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为风险警示股票 '{stock_info['名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的风险警示股票，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(5).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > 5:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - 5} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有风险警示股票的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条风险警示板 A 股实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '市盈率-动态']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为风险警示板股票的实时行情数据概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询*ST天山的最新价和涨跌幅。' 或 '查询股票代码600000的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取风险警示板股票实时行情数据时发生错误: {e}"

@tool
def get_new_a_shares_quotes(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的当前交易日新股板块所有 A 股上市公司的实时行情数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、涨跌额、成交量、成交额、振幅、最高价、最低价、
    今开、昨收、量比、换手率、市盈率（动态）、市净率等全面的实时市场指标。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定新股的详细实时行情。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的新股的股票代码（如 '301559'）
                                          或公司名称（如 'N中集环' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含新股 A 股实时行情数据。
             如果获取失败，则返回错误信息。
    """
    try:
        # 调用 API
        df = ak.stock_zh_a_new_em()

        if df.empty:
            return "未能获取新股板块的实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '最新价': lambda x: f"{x:.2f}",
            '涨跌幅': lambda x: f"{x:+.2f}%",
            '涨跌额': lambda x: f"{x:+.2f}",
            '成交量': lambda x: f"{int(x):,}手",
            '成交额': lambda x: f"{x / 1_0000_0000:.2f}亿元" if x >= 1_0000_0000 else f"{x / 1_0000:.2f}万元",
            '振幅': lambda x: f"{x:.2f}%",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '今开': lambda x: f"{x:.2f}",
            '昨收': lambda x: f"{x:.2f}",
            '量比': lambda x: f"{x:.2f}",
            '换手率': lambda x: f"{x:.2f}%",
            '市盈率-动态': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A",
            '市净率': lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "N/A"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        # 包含所有原始列
        display_cols = df_formatted.columns.tolist()
        # 移除 '序号' 列
        if '序号' in display_cols:
            display_cols.remove('序号')

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            if '代码' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['代码'].astype(str) == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty and '名称' in df_formatted.columns:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]


            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的新股实时行情数据。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"新股 '{stock_info['名称']}' ({stock_info['代码']}) 的实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为新股 '{stock_info['名称']}' 的详细实时行情。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的新股，请提供更具体的股票代码或名称："]
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(5).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > 5:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - 5} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有新股的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条新股板块 A 股实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '市盈率-动态']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为新股板块的实时行情数据概览。")
            output_lines.append("如果您需要查询特定新股的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询N中集环的最新价和涨跌幅。' 或 '查询股票代码301559的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取新股板块实时行情数据时发生错误: {e}"

@tool
def get_ah_shares_realtime_quotes(stock_code: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有 A+H 上市公司的实时行情数据。
    数据包括 A 股和 H 股的最新价、涨跌幅、代码，以及 A/H 股的比价和溢价。
    请注意，此数据更新可能存在约 15 分钟的延迟。

    用户可以选择提供一个股票代码（可以是 A 股代码或 H 股代码）来查询其对应的实时行情。
    如果未提供股票代码，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码，此函数将返回该特定股票的详细实时行情。

    Args:
        stock_code (Optional[str]): 可选参数。要查询的 A+H 股的股票代码（可以是 A 股代码或 H 股代码）。
                                    例如：'01385' (H股) 或 '688385' (A股)。

    Returns:
        str: 一个格式化后的字符串，包含 A+H 股实时行情数据。
             如果获取失败，则返回错误信息。
    """
    try:
        # 调用 API
        df = ak.stock_zh_ah_spot_em()

        if df.empty:
            return "未能获取 A+H 股实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '最新价-HKD': lambda x: f"{x:.2f} HKD",
            'H股-涨跌幅': lambda x: f"{x:+.2f}%",
            '最新价-RMB': lambda x: f"{x:.2f} RMB",
            'A股-涨跌幅': lambda x: f"{x:+.2f}%",
            '比价': lambda x: f"{x:.2f}",
            '溢价': lambda x: f"{x:+.2f}%"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义要显示的列，以便在两种模式下保持一致
        display_cols = ['名称', 'H股代码', '最新价-HKD', 'H股-涨跌幅', 'A股代码', '最新价-RMB', 'A股-涨跌幅', '比价', '溢价']
        # 确保要显示的列都在 DataFrame 中
        display_cols = [col for col in display_cols if col in df_formatted.columns]


        if stock_code:
            # 如果提供了股票代码，则筛选特定股票
            # 匹配 H股代码 或 A股代码
            filtered_df = df_formatted[(df_formatted['H股代码'] == stock_code) | (df_formatted['A股代码'] == stock_code)]

            if filtered_df.empty:
                return f"未找到股票代码 '{stock_code}' 对应的 A+H 股实时行情数据。请检查代码是否正确。"
            else:
                stock_info = filtered_df.iloc[0]
                output_lines = [f"股票代码 '{stock_code}' 的 A+H 股实时行情数据 (数据来源: 东方财富)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为股票 '{stock_info['名称']}' ({stock_info['H股代码']}/{stock_info['A股代码']}) 的详细实时行情。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码，则返回所有 A+H 股的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条 A+H 股实时行情数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            df_string = df_formatted[display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有 A+H 股的实时行情数据概览。")
            output_lines.append("如果您需要查询特定 A+H 股的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询复旦微电的 A 股和 H 股最新价及溢价。' 或 '查询股票代码01385的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取 A+H 股实时行情数据时发生错误: {e}"

@tool
def get_ah_shares_historical_data(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = ""
) -> str:
    """
    获取腾讯财经提供的指定 A+H 上市公司的历史行情日频率数据。
    用户可以指定港股股票代码、查询开始日期、结束日期以及复权方式。

    Args:
        symbol (str): 港股股票代码，例如 '02318'。
        start_date (str): 查询的开始日期，格式为 'YYYY-MM-DD'（例如：'2023-01-01'）。
        end_date (str): 查询的结束日期，格式为 'YYYY-MM-DD'（例如：'2024-03-31'）。
        adjust (str, optional): 复权类型。
                                默认为 '' (不复权)。
                                可选值包括：
                                - '' (不复权): 返回原始价格数据。
                                - 'qfq' (前复权): 保持当前价格不变，调整历史价格。
                                - 'hfq' (后复权): 保持历史价格不变，调整当前价格，反映真实收益率。

    Returns:
        str: 一个格式化后的字符串，包含指定股票历史行情数据的完整详情。
             如果获取失败或参数不符合要求，则返回错误信息。
    """
    # 校验 symbol 格式
    if not symbol.isdigit() or len(symbol) != 5:
        return f"错误：股票代码 '{symbol}' 格式不正确。港股代码通常为5位数字，例如 '02318'。"

    date_format = '%Y-%m-%d'

    # 校验日期格式和逻辑
    try:
        parsed_start_date = datetime.strptime(start_date, date_format)
        parsed_end_date = datetime.strptime(end_date, date_format)
    except ValueError:
        return f"错误：日期格式不正确。请使用 '{date_format}' 格式，例如 '2023-01-01'。您输入的是 '{start_date}' 或 '{end_date}'。"

    if parsed_start_date > parsed_end_date:
        return f"错误：开始日期 '{start_date}' 不能晚于结束日期 '{end_date}'。"

    current_year = datetime.now().year
    if not (1990 <= parsed_start_date.year <= current_year) or not (1990 <= parsed_end_date.year <= current_year):
        return f"错误：日期年份不在有效范围内 (1990年至今)。您输入的是 '{start_date}' 到 '{end_date}'。"

    valid_adjusts = {"", "qfq", "hfq"}
    if adjust not in valid_adjusts:
        return f"错误：'adjust' 参数无效。可选值包括 {', '.join(valid_adjusts)}。您输入的是 '{adjust}'。"

    try:
        # 提取年份用于调用 AKShare API
        api_start_year = str(parsed_start_date.year)
        api_end_year = str(parsed_end_date.year)

        # 调用 API
        df = ak.stock_zh_ah_daily(
            symbol=symbol,
            start_year=api_start_year,
            end_year=api_end_year,
            adjust=adjust
        )

        if df.empty:
            return f"未能获取港股代码 '{symbol}' 从 {api_start_year} 年到 {api_end_year} 年的历史行情数据。可能该股票在该时间段内无交易数据，或代码有误。"

        # 将'日期'列转换为datetime对象以便进行精确过滤
        df['日期'] = pd.to_datetime(df['日期'])

        # 根据用户输入的完整日期范围进行过滤
        df_filtered = df[(df['日期'] >= parsed_start_date) & (df['日期'] <= parsed_end_date)].copy()

        if df_filtered.empty:
            return f"未能获取港股代码 '{symbol}' 从 {start_date} 到 {end_date} 的历史行情数据。该时间段内可能无交易数据。"

        output_lines = [
            f"成功获取港股代码 '{symbol}' 从 {start_date} 到 {end_date} 的历史行情数据 (共 {len(df_filtered)} 条记录，复权方式: {adjust if adjust else '不复权'})。\n"
        ]
        output_lines.append("=" * 100)

        # 格式化数值列
        df_display = df_filtered.copy()

        # 重命名列
        df_display.rename(columns={
            '日期': '日期',
            '开盘': '开盘',
            '收盘': '收盘',
            '最高': '最高',
            '最低': '最低',
            '成交量': '成交量'
        }, inplace=True)

        # 定义格式化规则
        format_rules = {
            '开盘': lambda x: f"{x:.2f}",
            '收盘': lambda x: f"{x:.2f}",
            '最高': lambda x: f"{x:.2f}",
            '最低': lambda x: f"{x:.2f}",
            '成交量': lambda x: f"{int(x):,}股"
        }

        for col, func in format_rules.items():
            if col in df_display.columns:
                # 将"日期"列格式化为字符串类型
                if col == '日期':
                    df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')
                else:
                    df_display[col] = df_display[col].apply(lambda x: func(x) if pd.notna(x) else "N/A")

        # 显示部分列
        display_cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
        display_cols = [col for col in display_cols if col in df_display.columns]

        # 直接返回整个 DataFrame 的字符串表示
        output_lines.append("历史行情数据详情:")
        df_string = df_display[display_cols].to_string(index=False)
        output_lines.append(df_string)

        output_lines.append("=" * 100)
        output_lines.append(f"提示：此为港股代码 '{symbol}' 从 {start_date} 到 {end_date} 的完整历史行情数据。")
        output_lines.append("如果您需要查询特定日期或特定指标的详细信息，请明确提出。")
        output_lines.append("例如，您可以询问 '港股02318在2024-01-29的收盘价和成交量是多少？'")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股代码 '{symbol}' 历史行情数据时发生错误: {e}"
    
@tool
def get_ah_shares_names(company_name: Optional[str] = None) -> str:
    """
    获取腾讯财经提供的所有 A+H 上市公司的股票代码和名称。
    这些数据可以用于查询 A+H 股的历史行情数据等其他相关接口。

    用户可以选择提供一个公司名称来查询其对应的港股股票代码。
    如果未提供公司名称，工具将返回所有 A+H 股的代码和名称的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了公司名称，工具将尝试查找匹配的股票代码。

    Args:
        company_name (Optional[str]): 可选参数。要查询的 A+H 股公司名称（或部分名称）。
                                      例如：'比亚迪股份' 或 '中国移动'。
                                      如果提供此参数，工具将返回匹配的股票代码；
                                      如果未提供，则返回所有 A+H 股的概览。

    Returns:
        str: 一个格式化后的字符串。
             如果提供了 'company_name' 且找到唯一匹配，则返回该公司的港股代码和名称。
             如果提供了 'company_name' 但未找到匹配，则返回未找到信息。
             如果提供了 'company_name' 但找到多个匹配，则列出所有匹配项的代码和名称，并提示用户提供更具体的名称。
             如果未提供 'company_name'，则返回所有 A+H 股的代码和名称的摘要。
             如果获取失败，则返回错误信息。
    """
    try:
        # 调用 API
        df = ak.stock_zh_ah_name()

        if df.empty:
            return "未能获取 A+H 股代码和名称数据，返回数据为空。可能数据源暂无数据。"

        if company_name:
            # 如果提供了公司名称，则进行搜索
            matching_stocks = df[df['名称'].str.contains(company_name, case=False, na=False)]

            if matching_stocks.empty:
                return f"未找到名称包含 '{company_name}' 的 A+H 股信息。请检查公司名称是否正确。"
            elif len(matching_stocks) == 1:
                # 找到唯一匹配
                stock_code = matching_stocks.iloc[0]['代码']
                stock_name = matching_stocks.iloc[0]['名称']
                return f"找到匹配的 A+H 股信息：名称: {stock_name}, 港股代码: {stock_code}。"
            else:
                # 找到多个匹配
                output_lines = [f"找到多个名称包含 '{company_name}' 的 A+H 股信息，请提供更具体的名称："]
                for index, row in matching_stocks.head(10).iterrows(): # 最多显示前10个
                    output_lines.append(f"  - 名称: {row['名称']}, 港股代码: {row['代码']}")
                if len(matching_stocks) > 10:
                    output_lines.append(f"  ... (还有 {len(matching_stocks) - 10} 个更多匹配)")
                output_lines.append("\n提示：请尝试提供更精确的公司全称或更长的部分名称以缩小范围。")
                return "\n".join(output_lines)
        else:
            # 如果未提供公司名称，则返回所有 A+H 股的概览
            output_lines = [f"成功获取 {len(df)} 条 A+H 股的代码和名称数据 (数据来源: 腾讯财经)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分列以保持简洁，并确保关键信息可见
            display_cols = ['代码', '名称']
            
            # 确保要显示的列都在 DataFrame 中
            display_cols = [col for col in display_cols if col in df.columns]

            df_string = df[display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有 A+H 股的代码和名称数据概览。")
            output_lines.append("如果您需要查询特定股票的代码或名称，请提供相关信息。")
            output_lines.append("例如，您可以询问 '比亚迪股份的港股代码是多少？' 或 '查询名称中包含“中国”的A+H股票。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取 A+H 股代码和名称数据时发生错误: {e}"

@tool
def get_all_a_shares_code_name(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有沪深京 A 股上市公司的股票代码和股票简称数据。
    该接口返回包括股票代码（code）和股票名称（name）两列数据。

    用户可以选择提供一个股票代码或公司名称来查询其对应的代码和名称。
    如果未提供股票代码或名称，此函数将返回数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定 A 股的代码和名称。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的 A 股的股票代码（如 '000001'）
                                          或公司名称（如 '平安银行' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含 A 股股票代码和简称数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_info_a_code_name()

        if df.empty:
            return "未能获取沪深京 A 股股票代码和名称数据，返回数据为空。可能数据源暂无数据。"

        # 确保'code'和'name'列存在
        if 'code' not in df.columns or 'name' not in df.columns:
            return "获取到的数据缺少 'code' 或 'name' 列，无法进行查询。"
        # 确保代码是字符串类型以便精确匹配
        df['code'] = df['code'].astype(str)

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            filtered_df = df[df['code'] == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty:
                filtered_df = df[df['name'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty:
                filtered_df = df[df['name'].str.contains(stock_identifier, case=False, na=False)]

            if filtered_df.empty:
                return f"未找到与 '{stock_identifier}' 匹配的 A 股股票代码或名称。请检查输入是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                return (
                    f"查询到 A 股股票信息 (数据来源: 沪深京三个交易所):\n"
                    f"====================================================================================================\n"
                    f"代码: {stock_info['code']}\n"
                    f"名称: {stock_info['name']}\n"
                    f"====================================================================================================\n"
                    f"提示：此为股票 '{stock_info['name']}' 的代码和名称。"
                )
            else:
                # 找到多个匹配
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的 A 股股票，请提供更具体的股票代码或名称："]
                output_lines.append("=" * 100)
                # 限制显示数量，避免过长
                for index, row in filtered_df.head(10).iterrows():
                    output_lines.append(f"  - 代码: {row['code']}, 名称: {row['name']}")
                if len(filtered_df) > 10:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有 A 股的概览
            output_lines = [f"成功获取 {len(df)} 条沪深京 A 股股票代码和名称数据 (数据来源: 沪深京三个交易所)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示 'code' 和 'name' 列
            display_df = df[['code', 'name']].head(MAX_ROWS_TO_DISPLAY)
            
            df_string = display_df.to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为所有 A 股股票代码和名称的概览。")
            output_lines.append("您可以根据此列表中的股票代码或名称来查询具体的股票行情。")
            output_lines.append("例如，您可以询问 '查询股票代码000001对应的股票名称。' 或 '查询平安银行的股票代码。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取沪深京 A 股股票代码和名称数据时发生错误: {e}"

@tool
def get_sh_stock_info(symbol: str, stock_identifier: Optional[str] = None) -> str:
    """
    获取上海证券交易所上市公司的股票代码和简称数据。
    该接口返回包括证券代码、证券简称、公司全称和上市日期等详细信息。

    用户必须提供 `symbol` 参数来指定查询的股票类型，可选值为 '主板A股', '主板B股', '科创板'。
    用户可以选择提供一个股票代码或公司名称来查询其对应的详细信息。
    如果未提供股票代码或名称，此函数将返回指定 `symbol` 类型下所有股票数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定上海 A 股的详细信息。

    Args:
        symbol (str): 必选参数。指定要查询的上海证券交易所股票类型。
                      可选值包括: '主板A股', '主板B股', '科创板'。
        stock_identifier (Optional[str]): 可选参数。要查询的上海 A 股的证券代码（如 '600000'）
                                          或公司简称（如 '浦发银行' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含上海 A 股股票信息。
             如果获取失败或 `symbol` 参数无效，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    allowed_symbols = ["主板A股", "主板B股", "科创板"]
    if symbol not in allowed_symbols:
        return f"错误：无效的 symbol 参数 '{symbol}'。可选值包括: {', '.join(allowed_symbols)}。"

    try:
        # 调用 API
        df = ak.stock_info_sh_name_code(symbol=symbol)

        if df.empty:
            return f"未能获取上海证券交易所 {symbol} 的股票信息，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 确保 '证券代码' 和 '证券简称' 列存在且为字符串类型以便精确匹配和模糊匹配
        if '证券代码' not in df.columns or '证券简称' not in df.columns:
            return "获取到的数据缺少 '证券代码' 或 '证券简称' 列，无法进行查询。"
        
        df['证券代码'] = df['证券代码'].astype(str)
        df['证券简称'] = df['证券简称'].astype(str)

        # 定义所有可能的显示列
        all_display_cols = ['证券代码', '证券简称', '公司全称', '上市日期']
        # 确保要显示的列都在 DataFrame 中
        all_display_cols = [col for col in all_display_cols if col in df.columns]

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按证券代码精确匹配
            filtered_df = df[df['证券代码'] == stock_identifier]

            # 如果没有代码匹配，按证券简称精确匹配
            if filtered_df.empty:
                filtered_df = df[df['证券简称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按证券简称模糊匹配
            if filtered_df.empty:
                filtered_df = df[df['证券简称'].str.contains(stock_identifier, case=False, na=False)]

            if filtered_df.empty:
                return f"在上海证券交易所 {symbol} 中，未找到与 '{stock_identifier}' 匹配的股票信息。请检查证券代码或简称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"上海证券交易所 {symbol} 股票 '{stock_info['证券简称']}' ({stock_info['证券代码']}) 的详细信息 (数据来源: 上海证券交易所)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in all_display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为上海证券交易所 {symbol} 股票 '{stock_info['证券简称']}' 的详细信息。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"在上海证券交易所 {symbol} 中，找到多个与 '{stock_identifier}' 匹配的股票，请提供更具体的证券代码或简称："]
                output_lines.append("=" * 100)
                # 限制显示数量，避免过长
                # 仅显示代码和简称
                for index, row in filtered_df.head(10).iterrows():
                    output_lines.append(f"  - 证券代码: {row['证券代码']}, 证券简称: {row['证券简称']}")
                if len(filtered_df) > 10:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                output_lines.append("\n提示：请尝试提供精确的证券代码或完整的公司简称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有股票的概览
            output_lines = [f"成功获取 {len(df)} 条上海证券交易所 {symbol} 股票信息 (数据来源: 上海证券交易所)。\n"]
            output_lines.append("=" * 100)
            output_lines.append(f"数据概览 (前MAX_ROWS_TO_DISPLAY行，类型: {symbol}):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['证券代码', '证券简称', '上市日期']
            summary_display_cols = [col for col in summary_display_cols if col in df.columns]

            df_string = df[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为上海证券交易所 {symbol} 的股票信息概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其证券代码或简称。")
            output_lines.append(f"例如，您可以询问 '查询上海主板A股中股票代码600000的详细信息。' 或 '查询上海科创板中浦发银行的股票信息。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取上海证券交易所股票信息时发生错误: {e}"

@tool
def get_sz_stock_info(symbol: str, stock_identifier: Optional[str] = None) -> str:
    """
    获取深圳证券交易所上市公司的股票代码和股票简称数据。
    该接口返回包括板块、A股代码、A股简称、A股上市日期、A股总股本、A股流通股本、所属行业等详细信息。

    用户必须提供 `symbol` 参数来指定查询的股票类型，可选值为 'A股列表', 'B股列表', 'CDR列表', 'AB股列表'。
    用户可以选择提供一个股票代码或公司名称来查询其对应的详细信息。
    如果未提供股票代码或名称，此函数将返回指定 `symbol` 类型下所有股票数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定深交所股票的详细信息。

    Args:
        symbol (str): 必选参数。指定要查询的深圳证券交易所股票类型。
                      可选值包括: 'A股列表', 'B股列表', 'CDR列表', 'AB股列表'。
        stock_identifier (Optional[str]): 可选参数。要查询的深圳证券交易所股票的A股代码（如 '000001'）
                                          或A股简称（如 '平安银行' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含深圳证券交易所股票信息。
             如果获取失败或 `symbol` 参数无效，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    allowed_symbols = ["A股列表", "B股列表", "CDR列表", "AB股列表"]
    if symbol not in allowed_symbols:
        return f"错误：无效的 symbol 参数 '{symbol}'。可选值包括: {', '.join(allowed_symbols)}。"

    try:
        # 调用 API
        df = ak.stock_info_sz_name_code(symbol=symbol)

        if df.empty:
            return f"未能获取深圳证券交易所 {symbol} 的股票信息，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 确保关键列存在且为字符串类型以便精确匹配和模糊匹配
        required_cols = ['A股代码', 'A股简称']
        for col in required_cols:
            if col not in df.columns:
                return f"获取到的数据缺少 '{col}' 列，无法进行查询。请检查数据源或 AKShare 版本。"
            df[col] = df[col].astype(str) # 确保代码和简称是字符串类型

        # 定义所有可能的显示列
        all_display_cols = ['板块', 'A股代码', 'A股简称', 'A股上市日期', 'A股总股本', 'A股流通股本', '所属行业']
        # 确保要显示的列都在 DataFrame 中
        all_display_cols = [col for col in all_display_cols if col in df.columns]

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按A股代码精确匹配
            filtered_df = df[df['A股代码'] == stock_identifier]

            # 如果没有代码匹配，按A股简称精确匹配
            if filtered_df.empty:
                filtered_df = df[df['A股简称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按A股简称模糊匹配
            if filtered_df.empty:
                filtered_df = df[df['A股简称'].str.contains(stock_identifier, case=False, na=False)]

            if filtered_df.empty:
                return f"在深圳证券交易所 {symbol} 中，未找到与 '{stock_identifier}' 匹配的股票信息。请检查证券代码或简称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"深圳证券交易所 {symbol} 股票 '{stock_info['A股简称']}' ({stock_info['A股代码']}) 的详细信息 (数据来源: 深圳证券交易所)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in all_display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为深圳证券交易所 {symbol} 股票 '{stock_info['A股简称']}' 的详细信息。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"在深圳证券交易所 {symbol} 中，找到多个与 '{stock_identifier}' 匹配的股票，请提供更具体的证券代码或简称："]
                output_lines.append("=" * 100)
                # 限制显示数量，避免过长
                # 仅显示代码和简称
                for index, row in filtered_df.head(10).iterrows():
                    output_lines.append(f"  - A股代码: {row['A股代码']}, A股简称: {row['A股简称']}")
                if len(filtered_df) > 10:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                output_lines.append("\n提示：请尝试提供精确的证券代码或完整的公司简称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有股票的概览
            output_lines = [f"成功获取 {len(df)} 条深圳证券交易所 {symbol} 股票信息 (数据来源: 深圳证券交易所)。\n"]
            output_lines.append("=" * 100)
            output_lines.append(f"数据概览 (前MAX_ROWS_TO_DISPLAY行，类型: {symbol}):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['板块', 'A股代码', 'A股简称', 'A股上市日期', '所属行业']
            summary_display_cols = [col for col in summary_display_cols if col in df.columns]

            df_string = df[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为深圳证券交易所 {symbol} 的股票信息概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其A股代码或A股简称。")
            output_lines.append(f"例如，您可以询问 '查询深圳A股列表中股票代码000001的详细信息。' 或 '查询深圳A股列表中平安银行的股票信息。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取深圳证券交易所股票信息时发生错误: {e}"

@tool
def get_bj_stock_info(stock_identifier: Optional[str] = None) -> str:
    """
    获取北京证券交易所所有上市公司的股票代码和简称数据。
    该接口返回包括证券代码、证券简称、总股本、流通股本、上市日期、所属行业、地区和报告日期等详细信息。

    用户可以选择提供一个股票代码或公司名称来查询其对应的详细信息。
    如果未提供股票代码或名称，此函数将返回所有北交所股票数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定北交所股票的详细信息。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的北京证券交易所股票的证券代码（如 '430017'）
                                          或证券简称（如 '星昊医药' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含北京证券交易所股票信息。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_info_bj_name_code()

        if df.empty:
            return "未能获取北京证券交易所股票信息，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 确保关键列存在且为字符串类型以便精确匹配和模糊匹配
        required_cols = ['证券代码', '证券简称']
        for col in required_cols:
            if col not in df.columns:
                return f"获取到的数据缺少 '{col}' 列，无法进行查询。请检查数据源或 AKShare 版本。"
            # 确保代码和简称是字符串类型
            df[col] = df[col].astype(str)

        # 格式化数值列的辅助函数
        def format_value(value, rule):
            if pd.notna(value):
                return rule(value)
            return "N/A"

        # 定义需要格式化的列及其格式规则
        format_rules = {
            '总股本': lambda x: f"{int(x):,}股",
            '流通股本': lambda x: f"{int(x):,}股"
        }

        # 应用格式化规则到DataFrame的副本
        df_formatted = df.copy()
        for col, func in format_rules.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, func))

        # 定义所有可能的显示列
        all_display_cols = ['证券代码', '证券简称', '总股本', '流通股本', '上市日期', '所属行业', '地区', '报告日期']
        # 确保要显示的列都在 DataFrame 中
        all_display_cols = [col for col in all_display_cols if col in df_formatted.columns]

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按证券代码精确匹配
            filtered_df = df_formatted[df_formatted['证券代码'] == stock_identifier]

            # 如果没有代码匹配，按证券简称精确匹配
            if filtered_df.empty:
                filtered_df = df_formatted[df_formatted['证券简称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按证券简称模糊匹配
            if filtered_df.empty:
                filtered_df = df_formatted[df_formatted['证券简称'].str.contains(stock_identifier, case=False, na=False)]

            if filtered_df.empty:
                return f"在北京证券交易所中，未找到与 '{stock_identifier}' 匹配的股票信息。请检查证券代码或简称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"北京证券交易所股票 '{stock_info['证券简称']}' ({stock_info['证券代码']}) 的详细信息 (数据来源: 北京证券交易所)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in all_display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为北京证券交易所股票 '{stock_info['证券简称']}' 的详细信息。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"在北京证券交易所中，找到多个与 '{stock_identifier}' 匹配的股票，请提供更具体的证券代码或简称："]
                output_lines.append("=" * 100)
                # 限制显示数量，避免过长
                # 仅显示代码和简称
                for index, row in filtered_df.head(10).iterrows():
                    output_lines.append(f"  - 证券代码: {row['证券代码']}, 证券简称: {row['证券简称']}")
                if len(filtered_df) > 10:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                output_lines.append("\n提示：请尝试提供精确的证券代码或完整的公司简称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有股票的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条北京证券交易所股票信息 (数据来源: 北京证券交易所)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['证券代码', '证券简称', '总股本', '上市日期', '所属行业']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为北京证券交易所的股票信息概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其证券代码或简称。")
            output_lines.append("例如，您可以询问 '查询北京证券交易所中股票代码430017的详细信息。' 或 '查询北京证券交易所中星昊医药的股票信息。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取北京证券交易所股票信息时发生错误: {e}"

@tool
def get_sz_stock_name_change(symbol: str, stock_identifier: Optional[str] = None) -> str:
    """
    获取深圳证券交易所股票的名称变更历史数据。
    该接口返回包括变更日期、证券代码、证券简称，以及根据变更类型（全称或简称）对应的变更前/后名称信息。

    用户必须提供 `symbol` 参数来指定查询的变更类型，可选值为 '全称变更' 或 '简称变更'。
    用户可以选择提供一个股票代码、当前证券简称、变更前名称或变更后名称来查询其对应的名称变更历史。
    如果未提供股票代码或名称，此函数将返回指定 `symbol` 类型下所有名称变更数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定深交所股票的名称变更历史。

    Args:
        symbol (str): 必选参数。指定要查询的名称变更类型。
                      可选值包括: '全称变更', '简称变更'。
        stock_identifier (Optional[str]): 可选参数。要查询的深圳证券交易所股票的证券代码（如 '000004'）、
                                          证券简称（如 '国华网安'）、变更前名称或变更后名称。
                                          这里的'名称'具体指全称或简称，取决于symbol参数。

    Returns:
        str: 一个格式化后的字符串，包含深圳证券交易所股票名称变更历史。
             如果获取失败或 `symbol` 参数无效，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    allowed_symbols = ["全称变更", "简称变更"]
    if symbol not in allowed_symbols:
        return f"错误：无效的 symbol 参数 '{symbol}'。可选值包括: {', '.join(allowed_symbols)}。"

    try:
        # 调用 API
        df = ak.stock_info_sz_change_name(symbol=symbol)

        if df.empty:
            return f"未能获取深圳证券交易所 {symbol} 的股票名称变更信息，返回数据为空。可能数据源暂无数据。"

        # 根据 symbol 动态确定变更前/后的列名
        if symbol == "全称变更":
            prev_name_col = "变更前全称"
            curr_name_col = "变更后全称"
        else: # symbol == "简称变更"
            prev_name_col = "变更前简称"
            curr_name_col = "变更后简称"

        # 确保关键列存在且为字符串类型以便精确匹配和模糊匹配
        required_cols = ['证券代码', '证券简称', prev_name_col, curr_name_col]
        for col in required_cols:
            if col not in df.columns:
                return f"获取到的数据缺少 '{col}' 列，无法进行查询。请检查数据源或 AKShare 版本。"
            # 将所有相关列转换为字符串类型，并用空字符串填充NaN值
            df[col] = df[col].astype(str).fillna('')

        # 定义所有可能的显示列
        # 动态构建显示列，确保顺序和内容正确
        all_display_cols = ['变更日期', '证券代码', '证券简称', prev_name_col, curr_name_col]
        # 确保要显示的列都在 DataFrame 中
        all_display_cols = [col for col in all_display_cols if col in df.columns]

        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            # 构建一个布尔掩码
            mask = pd.Series([False] * len(df))

            # 按证券代码精确匹配
            mask |= (df['证券代码'] == stock_identifier)

            # 按证券简称精确匹配
            mask |= (df['证券简称'] == stock_identifier)
            
            # 按变更前名称精确匹配
            mask |= (df[prev_name_col] == stock_identifier)

            # 按变更后名称精确匹配
            mask |= (df[curr_name_col] == stock_identifier)

            # 如果没有精确匹配，进行模糊匹配
            if not mask.any():
                fuzzy_mask = pd.Series([False] * len(df))
                fuzzy_mask |= df['证券简称'].str.contains(stock_identifier, case=False, na=False)
                fuzzy_mask |= df[prev_name_col].str.contains(stock_identifier, case=False, na=False)
                fuzzy_mask |= df[curr_name_col].str.contains(stock_identifier, case=False, na=False)
                # 将模糊匹配结果合并到主掩码中
                mask |= fuzzy_mask

            # 进行筛选，并确保只保留相关列，然后去重
            filtered_df = df.loc[mask, all_display_cols].drop_duplicates()

            if filtered_df.empty:
                return f"在深圳证券交易所 {symbol} 变更记录中，未找到与 '{stock_identifier}' 匹配的股票名称变更信息。请检查证券代码或名称是否正确。"
            else:
                output_lines = [f"深圳证券交易所 {symbol} 股票名称变更历史中与 '{stock_identifier}' 相关的记录 (数据来源: 深圳证券交易所)。\n"]
                output_lines.append("=" * 100)
                
                # 限制显示数量，避免过长
                display_limit = 10
                if len(filtered_df) > display_limit:
                    output_lines.append(f"找到 {len(filtered_df)} 条记录，以下显示前 {display_limit} 条：")
                else:
                    output_lines.append(f"找到 {len(filtered_df)} 条记录：")

                # 直接使用 all_display_cols 来确保显示所有相关列
                df_string = filtered_df.head(display_limit).to_string(index=False)
                output_lines.append(df_string)
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为深圳证券交易所 {symbol} 股票名称变更历史中与 '{stock_identifier}' 相关的记录。")
                if len(filtered_df) > display_limit:
                    output_lines.append("如果需要更多记录，请考虑更精确的查询或分批请求。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有变更记录的概览
            output_lines = [f"成功获取 {len(df)} 条深圳证券交易所 {symbol} 股票名称变更历史数据 (数据来源: 深圳证券交易所)。\n"]
            output_lines.append("=" * 100)
            output_lines.append(f"数据概览 (前MAX_ROWS_TO_DISPLAY行，类型: {symbol}):")

            # 仅显示部分关键列用于概览
            df_string = df[all_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为深圳证券交易所 {symbol} 的股票名称变更历史概览。")
            output_lines.append("如果您需要查询特定股票的名称变更历史，请提供其证券代码或名称。")
            output_lines.append(f"例如，您可以询问 '查询深圳全称变更中股票代码000004的名称变更历史。' 或 '查询深圳简称变更中平安银行的名称变更历史。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取深圳证券交易所股票名称变更信息时发生错误: {e}"

@tool
def get_main_stock_holders(stock_identifier: str, holder_name: Optional[str] = None) -> str:
    """
    获取指定股票在新浪财经-股本股东-主要股东页面上的所有历史数据。
    该接口返回包括编号、股东名称、持股数量、持股比例、股本性质、截至日期、公告日期、股东说明、股东总数和平均持股数等详细信息。

    用户必须提供一个股票代码或公司名称来查询其对应的主要股东历史数据。
    如果提供了股票名称，工具将尝试查找对应的股票代码。
    用户还可以选择提供一个 `holder_name` 参数来进一步筛选特定股东的持股历史。

    Args:
        stock_identifier (str): 必需参数。要查询主要股东历史数据的股票代码（如 '600004'）或公司名称（如 '白云机场'）。
        holder_name (Optional[str]): 可选参数。要筛选的股东名称（如 '上海国际集团有限公司' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的主要股东历史数据。
             如果获取失败、未找到股票或找到多个匹配，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    resolved_stock_code = None
    resolved_stock_name = None

    # 判断 stock_identifier 是代码还是名称
    if stock_identifier.isdigit() and len(stock_identifier) == 6:
        resolved_stock_code = stock_identifier
        # 获取股票名称，以用于输出
        try:
            all_a_shares_df = ak.stock_info_a_code_name()
            if not all_a_shares_df.empty and 'code' in all_a_shares_df.columns and 'name' in all_a_shares_df.columns:
                matched_name_df = all_a_shares_df[all_a_shares_df['code'] == resolved_stock_code]
                if not matched_name_df.empty:
                    resolved_stock_name = matched_name_df.iloc[0]['name']
        except Exception:
            # 如果获取名称失败，就用代码代替
            resolved_stock_name = stock_identifier
    else:
        # 通过名称查找股票代码
        try:
            all_a_shares_df = ak.stock_info_a_code_name()
            if all_a_shares_df.empty:
                return "错误：无法获取A股列表以解析股票名称。请尝试直接提供股票代码。"

            # 确保 'code' 和 'name' 列存在且为字符串类型
            if 'code' not in all_a_shares_df.columns or 'name' not in all_a_shares_df.columns:
                return "错误：获取到的A股列表数据缺少 'code' 或 'name' 列，无法解析股票名称。"
            all_a_shares_df['code'] = all_a_shares_df['code'].astype(str)
            all_a_shares_df['name'] = all_a_shares_df['name'].astype(str)

            # 优先精确匹配名称
            matched_stocks = all_a_shares_df[all_a_shares_df['name'] == stock_identifier]

            # 如果没有精确匹配，模糊匹配
            if matched_stocks.empty:
                matched_stocks = all_a_shares_df[all_a_shares_df['name'].str.contains(stock_identifier, case=False, na=False)]

            if matched_stocks.empty:
                return f"未能在A股列表中找到与 '{stock_identifier}' 匹配的股票代码。请检查股票名称是否正确或尝试直接提供股票代码。"
            elif len(matched_stocks) > 1:
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的股票，请提供更具体的股票名称或直接提供股票代码："]
                output_lines.append("=" * 100)
                for index, row in matched_stocks.head(10).iterrows():
                    output_lines.append(f"  - 代码: {row['code']}, 名称: {row['name']}")
                if len(matched_stocks) > 10:
                    output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                return "\n".join(output_lines)
            else:
                resolved_stock_code = matched_stocks.iloc[0]['code']
                resolved_stock_name = matched_stocks.iloc[0]['name']

        except Exception as e:
            return f"解析股票名称 '{stock_identifier}' 时发生错误: {e}。请尝试直接提供股票代码。"

    if not resolved_stock_code:
        return f"无法为 '{stock_identifier}' 解析出有效的股票代码。请确保输入正确或尝试直接提供股票代码。"

    try:
        # 调用 API
        df = ak.stock_main_stock_holder(stock=resolved_stock_code)

        if df.empty:
            return f"未能获取股票代码 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 的主要股东历史数据，返回数据为空。可能数据源暂无数据或该股票无主要股东信息。"

        # 如果提供了 holder_name，则进行筛选
        if holder_name:
            if '股东名称' not in df.columns:
                return f"获取到的数据缺少 '股东名称' 列，无法筛选股东 '{holder_name}'。请检查数据源或 AKShare 版本。"
            # 确保股东名称是字符串
            df['股东名称'] = df['股东名称'].astype(str).fillna('')

            # 精确匹配股东名称
            filtered_df = df[df['股东名称'] == holder_name]

            # 如果没有精确匹配，模糊匹配
            if filtered_df.empty:
                filtered_df = df[df['股东名称'].str.contains(holder_name, case=False, na=False)]
            
            if filtered_df.empty:
                return f"在股票 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 的主要股东历史中，未找到与 '{holder_name}' 匹配的股东信息。请检查股东名称是否正确或尝试更宽泛的名称。"
            # 更新df为筛选后的结果
            df = filtered_df

        # 定义所有可能的显示列
        all_display_cols = ['编号', '股东名称', '持股数量', '持股比例', '股本性质', '截至日期', '公告日期', '股东说明', '股东总数', '平均持股数']
        # 确保要显示的列都在 DataFrame 中
        all_display_cols = [col for col in all_display_cols if col in df.columns]

        # 格式化数值列
        def format_number(value, suffix="", precision=0):
            if pd.notna(value):
                # 转换为整数，如果小数点后全为0
                if value == int(value):
                    return f"{int(value):,}{suffix}"
                else:
                    return f"{value:,.{precision}f}{suffix}"
            return "N/A"

        df_formatted = df[all_display_cols].copy() # 复制一份进行格式化

        if '持股数量' in df_formatted.columns:
            df_formatted['持股数量'] = df_formatted['持股数量'].apply(lambda x: format_number(x, '股'))
        if '持股比例' in df_formatted.columns:
            df_formatted['持股比例'] = df_formatted['持股比例'].apply(lambda x: format_number(x, '%', 2))
        if '股东总数' in df_formatted.columns:
            df_formatted['股东总数'] = df_formatted['股东总数'].apply(lambda x: format_number(x))
        if '平均持股数' in df_formatted.columns:
            df_formatted['平均持股数'] = df_formatted['平均持股数'].apply(lambda x: format_number(x, '股'))
        
        # 格式化日期列，并处理NaN
        for col in ['截至日期', '公告日期']:
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].astype(str).replace('NaT', 'N/A').replace('nan', 'N/A')
        
        # 股东说明可能为空，替换NaN
        if '股东说明' in df_formatted.columns:
            df_formatted['股东说明'] = df_formatted['股东说明'].fillna('无')

        # 构建输出字符串
        output_lines = []
        if holder_name:
            output_lines.append(f"股票 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 中股东 '{holder_name}' 的持股历史数据 (数据来源: 新浪财经)。\n")
        else:
            output_lines.append(f"股票 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 的主要股东历史数据 (数据来源: 新浪财经)。\n")
        
        output_lines.append("=" * 100)
        
        # 限制显示数量，避免过长
        display_limit = MAX_ROWS_TO_DISPLAY
        if len(df_formatted) > display_limit:
            output_lines.append(f"找到 {len(df_formatted)} 条记录，以下显示最近的 {display_limit} 条：")
            df_string = df_formatted.head(display_limit).to_string(index=False)
        else:
            output_lines.append(f"找到 {len(df_formatted)} 条记录：")
            df_string = df_formatted.to_string(index=False)
            
        output_lines.append(df_string)
        output_lines.append("=" * 100)
        
        if holder_name:
            output_lines.append(f"提示：此为股票 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 中股东 '{holder_name}' 的持股历史数据。")
        else:
            output_lines.append(f"提示：此为股票 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 的主要股东历史数据。")
        
        if len(df_formatted) > display_limit:
            output_lines.append("如果需要更多记录，请考虑更精确的查询。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 的主要股东信息时发生错误: {e}"

@tool
def get_hk_stock_spot_em(stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网提供的所有港股的实时行情数据。请注意，此数据有 15 分钟延时。
    该接口返回包括序号、代码、名称、最新价、涨跌额、涨跌幅、今开、最高、最低、昨收、成交量和成交额等详细信息。

    用户可以选择提供一个股票代码或公司名称来查询其对应的实时行情数据。
    如果未提供股票代码或名称，此函数将返回所有港股行情数据的摘要（前MAX_ROWS_TO_DISPLAY行和总记录数）。
    如果提供了股票代码或名称，此函数将返回该特定港股的实时行情数据。

    Args:
        stock_identifier (Optional[str]): 可选参数。要查询的港股代码（如 '00593'）或名称（如 '梦东方' 或其部分名称）。

    Returns:
        str: 一个格式化后的字符串，包含港股实时行情信息。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_hk_spot_em()

        if df.empty:
            return "未能获取港股实时行情数据，返回数据为空。可能当前非交易时间或数据源暂无数据。"

        # 确保关键列存在且为字符串类型以便精确匹配和模糊匹配
        required_cols = ['代码', '名称']
        for col in required_cols:
            if col not in df.columns:
                return f"获取到的数据缺少 '{col}' 列，无法进行查询。请检查数据源或 AKShare 版本。"
            # 确保代码和名称是字符串类型
            df[col] = df[col].astype(str)

        # 定义所有可能的显示列
        all_display_cols = ['序号', '代码', '名称', '最新价', '涨跌额', '涨跌幅', '今开', '最高', '最低', '昨收', '成交量', '成交额']
        # 确保要显示的列都在 DataFrame 中
        all_display_cols = [col for col in all_display_cols if col in df.columns]

        # 格式化数值列的辅助函数
        def format_value(value, unit="", precision=2):
            if pd.notna(value):
                if unit == "股":
                    return f"{int(value):,}{unit}"
                elif unit == "%":
                    return f"{value:,.{precision}f}{unit}"
                elif unit == "港元":
                    return f"{value:,.{precision}f}{unit}"
                else:
                    return f"{value:,.{precision}f}"
            return "N/A"
        # 复制一份进行格式化
        df_formatted = df.copy()

        # 应用格式化规则
        if '最新价' in df_formatted.columns:
            df_formatted['最新价'] = df_formatted['最新价'].apply(lambda x: format_value(x, '港元', 3))
        if '涨跌额' in df_formatted.columns:
            df_formatted['涨跌额'] = df_formatted['涨跌额'].apply(lambda x: format_value(x, '港元', 3))
        if '涨跌幅' in df_formatted.columns:
            df_formatted['涨跌幅'] = df_formatted['涨跌幅'].apply(lambda x: format_value(x, '%', 2))
        if '今开' in df_formatted.columns:
            df_formatted['今开'] = df_formatted['今开'].apply(lambda x: format_value(x, '', 3))
        if '最高' in df_formatted.columns:
            df_formatted['最高'] = df_formatted['最高'].apply(lambda x: format_value(x, '', 3))
        if '最低' in df_formatted.columns:
            df_formatted['最低'] = df_formatted['最低'].apply(lambda x: format_value(x, '', 3))
        if '昨收' in df_formatted.columns:
            df_formatted['昨收'] = df_formatted['昨收'].apply(lambda x: format_value(x, '', 3))
        if '成交量' in df_formatted.columns:
            df_formatted['成交量'] = df_formatted['成交量'].apply(lambda x: format_value(x, '股', 0))
        if '成交额' in df_formatted.columns:
            df_formatted['成交额'] = df_formatted['成交额'].apply(lambda x: format_value(x, '港元', 2))


        if stock_identifier:
            # 如果提供了股票代码或名称，则筛选特定股票
            filtered_df = pd.DataFrame()

            # 按代码精确匹配
            filtered_df = df_formatted[df_formatted['代码'] == stock_identifier]

            # 如果没有代码匹配，按名称精确匹配
            if filtered_df.empty:
                filtered_df = df_formatted[df_formatted['名称'] == stock_identifier]
            
            # 如果仍然没有精确匹配，按名称模糊匹配
            if filtered_df.empty:
                filtered_df = df_formatted[df_formatted['名称'].str.contains(stock_identifier, case=False, na=False)]

            if filtered_df.empty:
                return f"在港股实时行情数据中，未找到与 '{stock_identifier}' 匹配的股票信息。请检查股票代码或名称是否正确。"
            elif len(filtered_df) == 1:
                # 找到唯一匹配
                stock_info = filtered_df.iloc[0]
                output_lines = [f"港股实时行情 '{stock_info['名称']}' ({stock_info['代码']}) 的详细信息 (数据来源: 东方财富网，数据有15分钟延时)。\n"]
                output_lines.append("=" * 100)
                
                # 逐行输出特定股票的详细信息
                for col in all_display_cols:
                    if col in stock_info.index:
                        output_lines.append(f"{col}: {stock_info[col]}")
                
                output_lines.append("=" * 100)
                output_lines.append(f"提示：此为港股实时行情 '{stock_info['名称']}' 的详细信息。")
                return "\n".join(output_lines)
            else:
                # 找到多个匹配
                output_lines = [f"在港股实时行情数据中，找到多个与 '{stock_identifier}' 匹配的股票，请提供更具体的股票代码或名称："]
                output_lines.append("=" * 100)
                # 限制显示数量，避免过长
                # 仅显示代码和名称
                for index, row in filtered_df.head(10).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(filtered_df) > 10:
                    output_lines.append(f"  ... (还有 {len(filtered_df) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                output_lines.append("\n提示：请尝试提供精确的股票代码或完整的公司名称以获取唯一结果。")
                return "\n".join(output_lines)
        else:
            # 如果未提供股票代码或名称，则返回所有股票的概览
            output_lines = [f"成功获取 {len(df_formatted)} 条港股实时行情数据 (数据来源: 东方财富网，数据有15分钟延时)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前MAX_ROWS_TO_DISPLAY行):")

            # 仅显示部分关键列用于概览
            summary_display_cols = ['代码', '名称', '最新价', '涨跌幅', '成交量', '成交额']
            summary_display_cols = [col for col in summary_display_cols if col in df_formatted.columns]

            df_string = df_formatted[summary_display_cols].head(MAX_ROWS_TO_DISPLAY).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append("提示：此为港股实时行情概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其代码或名称。")
            output_lines.append(f"例如，您可以询问 '查询港股代码00593的实时行情。' 或 '查询港股梦东方的实时行情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股实时行情信息时发生错误: {e}"

@tool
def get_hk_stock_hist(stock_identifier: str,
    period: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = ""
) -> str:
    """
    获取指定港股的历史行情数据，数据来源于东方财富网。
    用户可以指定查询的股票（通过代码或名称）、时间周期（日、周、月）、日期范围以及是否进行复权处理。

    如果用户未指定日期范围，工具将默认返回最近一段时间的历史数据（日线默认近1年，周线默认近5年，月线默认近10年），
    以避免返回过大的数据集。如果需要查询所有历史数据，请明确指定 `start_date='19700101'` 和 `end_date='22220101'`。

    Args:
        stock_identifier (str): 必需参数。要查询历史行情数据的港股代码（如 '00593'）或股票名称（如 '梦东方'）。
        period (str): 必需参数。数据的时间周期。可选值: 'daily' (日线), 'weekly' (周线), 'monthly' (月线)。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYYMMDD' (如 '20230101')。
                                     如果未提供，将根据 `period` 默认查询最近一段时间。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYYMMDD' (如 '20240101')。
                                   如果未提供，将默认查询到最新日期。
        adjust (str): 可选参数。复权类型。可选值: '' (不复权，默认), 'qfq' (前复权), 'hfq' (后复权)。

    Returns:
        str: 一个格式化后的字符串，包含港股历史行情数据。
             如果获取失败、未找到股票或找到多个匹配，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    allowed_periods = ["daily", "weekly", "monthly"]
    if period not in allowed_periods:
        return f"错误：无效的 period 参数 '{period}'。可选值包括: {', '.join(allowed_periods)}。"

    allowed_adjusts = ["", "qfq", "hfq"]
    if adjust not in allowed_adjusts:
        return f"错误：无效的 adjust 参数 '{adjust}'。可选值包括: {', '.join(allowed_adjusts)}。"

    resolved_stock_code = None
    resolved_stock_name = None

    # 解析股票代码或名称
    # 判断 stock_identifier 是代码还是名称
    if stock_identifier.isdigit() and len(stock_identifier) >= 5:
        resolved_stock_code = stock_identifier
        # 获取股票名称，以用于输出
        try:
            hk_spot_df = ak.stock_hk_spot_em()
            if not hk_spot_df.empty and '代码' in hk_spot_df.columns and '名称' in hk_spot_df.columns:
                matched_name_df = hk_spot_df[hk_spot_df['代码'].astype(str) == resolved_stock_code]
                if not matched_name_df.empty:
                    resolved_stock_name = matched_name_df.iloc[0]['名称']
        except Exception:
            # 如果获取名称失败，就用代码代替
            resolved_stock_name = stock_identifier
    else:
        # 通过名称查找股票代码
        try:
            hk_spot_df = ak.stock_hk_spot_em()
            if hk_spot_df.empty:
                return "错误：无法获取港股列表以解析股票名称。请尝试直接提供股票代码。"

            # 确保 '代码' 和 '名称' 列存在且为字符串类型
            if '代码' not in hk_spot_df.columns or '名称' not in hk_spot_df.columns:
                return "错误：获取到的港股列表数据缺少 '代码' 或 '名称' 列，无法解析股票名称。"
            hk_spot_df['代码'] = hk_spot_df['代码'].astype(str)
            hk_spot_df['名称'] = hk_spot_df['名称'].astype(str)

            # 优先精确匹配名称
            matched_stocks = hk_spot_df[hk_spot_df['名称'] == stock_identifier]

            # 如果没有精确匹配，模糊匹配
            if matched_stocks.empty:
                matched_stocks = hk_spot_df[hk_spot_df['名称'].str.contains(stock_identifier, case=False, na=False)]

            if matched_stocks.empty:
                return f"未能在港股列表中找到与 '{stock_identifier}' 匹配的股票代码。请检查股票名称是否正确或尝试直接提供股票代码。"
            elif len(matched_stocks) > 1:
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的港股，请提供更具体的股票名称或直接提供股票代码："]
                output_lines.append("=" * 100)
                for index, row in matched_stocks.head(10).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(matched_stocks) > 10:
                    output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                return "\n".join(output_lines)
            else:
                resolved_stock_code = matched_stocks.iloc[0]['代码']
                resolved_stock_name = matched_stocks.iloc[0]['名称']

        except Exception as e:
            return f"解析股票名称 '{stock_identifier}' 时发生错误: {e}。请尝试直接提供股票代码。"

    if not resolved_stock_code:
        return f"无法为 '{stock_identifier}' 解析出有效的港股代码。请确保输入正确或尝试直接提供股票代码。"

    # 处理日期参数的默认值
    # print("datetime是什么类型？", type(datetime), datetime)
    today_date_obj = datetime.now().date()
    today_str = today_date_obj.strftime("%Y%m%d")
    
    if end_date is None:
        end_date_param = today_str
    else:
        end_date_param = end_date

    if start_date is None:
        if period == "daily":
            # 默认查询最近一年
            start_date_param = (today_date_obj - timedelta(days=365)).strftime("%Y%m%d")
        elif period == "weekly":
            # 默认查询最近五年
            start_date_param = (today_date_obj - timedelta(days=365 * 5)).strftime("%Y%m%d")
        elif period == "monthly":
            # 默认查询最近十年
            start_date_param = (today_date_obj - timedelta(days=365 * 10)).strftime("%Y%m%d")
        else:
            # 取所有有效数据
            start_date_param = "19700101"
    else:
        start_date_param = start_date

    try:
        # 调用 API
        df = ak.stock_hk_hist(
            symbol=resolved_stock_code,
            period=period,
            start_date=start_date_param,
            end_date=end_date_param,
            adjust=adjust
        )

        if df.empty:
            return f"未能获取港股 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 在 {start_date_param} 至 {end_date_param} 期间的 {period} 历史行情数据。可能该日期范围内无数据或数据源暂无数据。"

        # 格式化输出
        # 定义所有可能的显示列
        all_display_cols = [
            '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额',
            '振幅', '涨跌幅', '涨跌额', '换手率'
        ]
        # 确保要显示的列都在 DataFrame 中
        all_display_cols = [col for col in all_display_cols if col in df.columns]

        # 格式化数值列的辅助函数
        def format_value(value, unit="", precision=2):
            if pd.notna(value):
                # 检查是否为整数，如果是，则不显示小数位
                if isinstance(value, (int, float)) and value == int(value):
                    return f"{int(value):,}{unit}"
                else:
                    return f"{value:,.{precision}f}{unit}"
            return "N/A"
        # 复制一份进行格式化
        df_formatted = df[all_display_cols].copy()

        # 应用格式化规则
        price_cols = ['开盘', '收盘', '最高', '最低', '涨跌额']
        for col in price_cols:
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, '港元', 3))

        if '成交量' in df_formatted.columns:
            df_formatted['成交量'] = df_formatted['成交量'].apply(lambda x: format_value(x, '股', 0))
        if '成交额' in df_formatted.columns:
            df_formatted['成交额'] = df_formatted['成交额'].apply(lambda x: format_value(x, '港元', 2))
        
        percent_cols = ['振幅', '涨跌幅', '换手率']
        for col in percent_cols:
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(lambda x: format_value(x, '%', 2))
        
        # 确保日期列是字符串
        if '日期' in df_formatted.columns:
            df_formatted['日期'] = df_formatted['日期'].astype(str)

        # 构建输出字符串
        output_lines = [f"港股 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 的 {period} 历史行情数据 (数据来源: 东方财富网)。\n"]
        output_lines.append(f"查询日期范围: {start_date_param} 至 {end_date_param}，复权类型: {adjust if adjust else '不复权'}。\n")
        output_lines.append("=" * 100)
        
        # 限制显示数量，避免过长
        display_limit = MAX_ROWS_TO_DISPLAY
        if len(df_formatted) > display_limit:
            output_lines.append(f"找到 {len(df_formatted)} 条记录，以下显示最近的 {display_limit} 条：")
            # 历史数据看最新的几条
            df_string = df_formatted.tail(display_limit).to_string(index=False)
        else:
            output_lines.append(f"找到 {len(df_formatted)} 条记录：")
            df_string = df_formatted.to_string(index=False)
            
        output_lines.append(df_string)
        output_lines.append("=" * 100)
        output_lines.append(f"提示：此为港股 '{resolved_stock_name or stock_identifier}' 的历史行情数据。")
        if len(df_formatted) > display_limit:
            output_lines.append("如果需要更多记录，请考虑更精确的日期范围查询。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 历史行情信息时发生错误: {e}"

@tool
def get_hk_company_profile_em(stock_identifier: str) -> str:
    """
    获取指定港股的公司资料，数据来源于东方财富网。
    该接口返回包括公司名称、英文名称、注册地、公司成立日期、所属行业、董事长、公司秘书、员工人数、办公地址、公司网址、E-MAIL、年结日、联系电话、核数师、传真和公司介绍等详细信息。

    用户必须提供一个港股代码或公司名称来查询其对应的公司资料。
    如果提供了股票名称，工具将尝试查找对应的股票代码。

    Args:
        stock_identifier (str): 必需参数。要查询公司资料的港股代码（如 '03900'）或公司名称（如 '腾讯控股'）。

    Returns:
        str: 一个格式化后的字符串，包含指定港股的公司资料。
             如果获取失败、未找到股票或找到多个匹配，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    resolved_stock_code = None
    resolved_stock_name = None

    # 解析股票代码或名称
    # 判断 stock_identifier 是代码还是名称
    if stock_identifier.isdigit() and len(stock_identifier) >= 5:
        resolved_stock_code = stock_identifier
        # 获取股票名称，以用于输出
        try:
            hk_spot_df = ak.stock_hk_spot_em()
            if not hk_spot_df.empty and '代码' in hk_spot_df.columns and '名称' in hk_spot_df.columns:
                matched_name_df = hk_spot_df[hk_spot_df['代码'].astype(str) == resolved_stock_code]
                if not matched_name_df.empty:
                    resolved_stock_name = matched_name_df.iloc[0]['名称']
        except Exception:
            # 如果获取名称失败，就用代码代替
            resolved_stock_name = stock_identifier
    else:
        # 通过名称查找股票代码
        try:
            hk_spot_df = ak.stock_hk_spot_em()
            if hk_spot_df.empty:
                return "错误：无法获取港股列表以解析股票名称。请尝试直接提供股票代码。"

            # 确保 '代码' 和 '名称' 列存在且为字符串类型
            if '代码' not in hk_spot_df.columns or '名称' not in hk_spot_df.columns:
                return "错误：获取到的港股列表数据缺少 '代码' 或 '名称' 列，无法解析股票名称。"
            hk_spot_df['代码'] = hk_spot_df['代码'].astype(str)
            hk_spot_df['名称'] = hk_spot_df['名称'].astype(str)

            # 优先精确匹配名称
            matched_stocks = hk_spot_df[hk_spot_df['名称'] == stock_identifier]

            # 如果没有精确匹配，模糊匹配
            if matched_stocks.empty:
                matched_stocks = hk_spot_df[hk_spot_df['名称'].str.contains(stock_identifier, case=False, na=False)]

            if matched_stocks.empty:
                return f"未能在港股列表中找到与 '{stock_identifier}' 匹配的股票代码。请检查股票名称是否正确或尝试直接提供股票代码。"
            elif len(matched_stocks) > 1:
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的港股，请提供更具体的股票名称或直接提供股票代码："]
                output_lines.append("=" * 100)
                for index, row in matched_stocks.head(10).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(matched_stocks) > 10:
                    output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                return "\n".join(output_lines)
            else:
                resolved_stock_code = matched_stocks.iloc[0]['代码']
                resolved_stock_name = matched_stocks.iloc[0]['名称']

        except Exception as e:
            return f"解析股票名称 '{stock_identifier}' 时发生错误: {e}。请尝试直接提供股票代码。"

    if not resolved_stock_code:
        return f"无法为 '{stock_identifier}' 解析出有效的港股代码。请确保输入正确或尝试直接提供股票代码。"

    try:
        # 调用 API
        df = ak.stock_hk_company_profile_em(symbol=resolved_stock_code)

        if df.empty:
            return f"未能获取港股 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 的公司资料，返回数据为空。可能数据源暂无数据。"

        # 格式化输出
        # 公司资料直接取第一行数据
        company_info = df.iloc[0]

        output_lines = [f"港股 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 的公司资料 (数据来源: 东方财富网)。\n"]
        output_lines.append("=" * 100)

        # 定义要显示的字段
        display_fields = {
            '公司名称': '公司名称',
            '英文名称': '英文名称',
            '注册地': '注册地',
            '公司成立日期': '公司成立日期',
            '所属行业': '所属行业',
            '董事长': '董事长',
            '公司秘书': '公司秘书',
            '员工人数': '员工人数',
            '办公地址': '办公地址',
            '公司网址': '公司网址',
            'E-MAIL': 'E-MAIL',
            '年结日': '年结日',
            '联系电话': '联系电话',
            '核数师': '核数师',
            '传真': '传真',
            '公司介绍': '公司介绍'
        }

        for col, display_name in display_fields.items():
            if col in company_info.index:
                value = company_info[col]
                if pd.isna(value) or value == '':
                    formatted_value = "N/A"
                elif col == '员工人数' and isinstance(value, (int, float)):
                    # 员工人数格式化为整数并加逗号
                    formatted_value = f"{int(value):,}"
                else:
                    formatted_value = str(value).strip()
                output_lines.append(f"{display_name}: {formatted_value}")
            
        output_lines.append("=" * 100)
        output_lines.append(f"提示：此为港股 '{resolved_stock_name or stock_identifier}' 的公司资料。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 公司资料时发生错误: {e}"

@tool
def get_hk_security_profile_em(stock_identifier: str) -> str:
    """
    获取指定港股的证券资料，数据来源于东方财富网。
    该接口返回包括证券代码、证券简称、上市日期、证券类型、发行价、发行量(股)、每手股数、每股面值、交易所、板块、年结日、ISIN（国际证券识别编码）、是否沪港通标的、是否深港通标的等详细信息。

    用户必须提供一个港股代码或公司名称来查询其对应的证券资料。
    如果提供了股票名称，工具将尝试查找对应的股票代码。

    Args:
        stock_identifier (str): 必需参数。要查询证券资料的港股代码（如 '03900'）或公司名称（如 '绿城中国'）。

    Returns:
        str: 一个格式化后的字符串，包含指定港股的证券资料。
             如果获取失败、未找到股票或找到多个匹配，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    resolved_stock_code = None
    resolved_stock_name = None

    # 解析股票代码或名称
    # 判断 stock_identifier 是代码还是名称
    if stock_identifier.isdigit() and len(stock_identifier) >= 5:
        resolved_stock_code = stock_identifier
        # 获取股票名称，以用于输出
        try:
            hk_spot_df = ak.stock_hk_spot_em()
            if not hk_spot_df.empty and '代码' in hk_spot_df.columns and '名称' in hk_spot_df.columns:
                matched_name_df = hk_spot_df[hk_spot_df['代码'].astype(str) == resolved_stock_code]
                if not matched_name_df.empty:
                    resolved_stock_name = matched_name_df.iloc[0]['名称']
        except Exception:
            # 如果获取名称失败，就用代码代替
            resolved_stock_name = stock_identifier
    else:
        # 通过名称查找股票代码
        try:
            hk_spot_df = ak.stock_hk_spot_em()
            if hk_spot_df.empty:
                return "错误：无法获取港股列表以解析股票名称。请尝试直接提供股票代码。"

            # 确保 '代码' 和 '名称' 列存在且为字符串类型
            if '代码' not in hk_spot_df.columns or '名称' not in hk_spot_df.columns:
                return "错误：获取到的港股列表数据缺少 '代码' 或 '名称' 列，无法解析股票名称。"
            hk_spot_df['代码'] = hk_spot_df['代码'].astype(str)
            hk_spot_df['名称'] = hk_spot_df['名称'].astype(str)

            # 优先精确匹配名称
            matched_stocks = hk_spot_df[hk_spot_df['名称'] == stock_identifier]

            # 如果没有精确匹配，模糊匹配
            if matched_stocks.empty:
                matched_stocks = hk_spot_df[hk_spot_df['名称'].str.contains(stock_identifier, case=False, na=False)]

            if matched_stocks.empty:
                return f"未能在港股列表中找到与 '{stock_identifier}' 匹配的股票代码。请检查股票名称是否正确或尝试直接提供股票代码。"
            elif len(matched_stocks) > 1:
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的港股，请提供更具体的股票名称或直接提供股票代码："]
                output_lines.append("=" * 100)
                for index, row in matched_stocks.head(10).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(matched_stocks) > 10:
                    output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                return "\n".join(output_lines)
            else:
                resolved_stock_code = matched_stocks.iloc[0]['代码']
                resolved_stock_name = matched_stocks.iloc[0]['名称']

        except Exception as e:
            return f"解析股票名称 '{stock_identifier}' 时发生错误: {e}。请尝试直接提供股票代码。"

    if not resolved_stock_code:
        return f"无法为 '{stock_identifier}' 解析出有效的港股代码。请确保输入正确或尝试直接提供股票代码。"

    try:
        # 调用 API
        df = ak.stock_hk_security_profile_em(symbol=resolved_stock_code)

        if df.empty:
            return f"未能获取港股 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 的证券资料，返回数据为空。可能数据源暂无数据。"

        # 格式化输出
        # 证券资料直接取第一行数据
        security_info = df.iloc[0]

        output_lines = [f"港股 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 的证券资料 (数据来源: 东方财富网)。\n"]
        output_lines.append("=" * 100)

        # 定义要显示的字段
        display_fields = {
            '证券代码': '证券代码',
            '证券简称': '证券简称',
            '上市日期': '上市日期',
            '证券类型': '证券类型',
            '发行价': '发行价',
            '发行量(股)': '发行量',
            '每手股数': '每手股数',
            '每股面值': '每股面值',
            '交易所': '交易所',
            '板块': '所属板块',
            '年结日': '年结日',
            'ISIN（国际证券识别编码）': 'ISIN编码',
            '是否沪港通标的': '沪港通标的',
            '是否深港通标的': '深港通标的'
        }

        # 股价及发行量格式化函数
        def format_value(value, unit="", precision=2):
            if pd.notna(value) and value != '':
                if isinstance(value, (int, float)):
                    if unit == "股":
                        return f"{int(value):,}{unit}"
                    elif unit == "港元":
                        return f"{value:,.{precision}f}{unit}"
                    else:
                        return f"{value:,.{precision}f}"
                else:
                    return str(value).strip()
            return "N/A"

        for col, display_name in display_fields.items():
            if col in security_info.index:
                value = security_info[col]
                if col == '发行价':
                    formatted_value = format_value(value, '港元', 3)
                elif col in ['发行量(股)', '每手股数']:
                    formatted_value = format_value(value, '股', 0)
                elif col == '上市日期':
                    # 格式化日期
                    if isinstance(value, str) and ' ' in value:
                        formatted_value = value.split(' ')[0]
                    else:
                        formatted_value = str(value).strip()
                else:
                    # 其他字段直接转字符串
                    formatted_value = format_value(value)
                output_lines.append(f"{display_name}: {formatted_value}")
            
        output_lines.append("=" * 100)
        output_lines.append(f"提示：此为港股 '{resolved_stock_name or stock_identifier}' 的证券资料。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 证券资料时发生错误: {e}"

@tool
def get_hk_famous_spot_em(stock_identifier: str) -> str:
    """
    获取东方财富网-行情中心-港股市场-知名港股中，指定股票的实时行情数据。
    该接口返回包括证券代码、名称、最新价、涨跌额、涨跌幅、今开、最高、最低、昨收、成交量和成交额等实时市场数据。

    用户必须提供一个港股代码或公司名称来查询其对应的知名港股实时行情。
    如果提供了股票名称，工具将尝试查找对应的股票代码。
    请注意，此工具仅查询东方财富网定义的“知名港股”列表中的股票。如果查询的股票不在该列表中，即使股票存在，也可能返回未找到信息。

    Args:
        stock_identifier (str): 必需参数。要查询实时行情数据的港股代码（如 '00700'）或公司名称（如 '腾讯控股'）。

    Returns:
        str: 一个格式化后的字符串，包含指定知名港股的实时行情数据。
             如果获取失败、未找到股票或找到多个匹配，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    resolved_stock_code = None
    resolved_stock_name = None

    # 解析股票代码或名称
    # 判断 stock_identifier 是代码还是名称
    if stock_identifier.isdigit() and len(stock_identifier) >= 5:
        resolved_stock_code = stock_identifier
        # 获取股票名称，以用于输出
        try:
            hk_spot_df = ak.stock_hk_spot_em()
            if not hk_spot_df.empty and '代码' in hk_spot_df.columns and '名称' in hk_spot_df.columns:
                matched_name_df = hk_spot_df[hk_spot_df['代码'].astype(str) == resolved_stock_code]
                if not matched_name_df.empty:
                    resolved_stock_name = matched_name_df.iloc[0]['名称']
        except Exception:
            # 如果获取名称失败，就用代码代替
            resolved_stock_name = stock_identifier
    else:
        # 通过名称查找股票代码
        try:
            hk_spot_df = ak.stock_hk_spot_em()
            if hk_spot_df.empty:
                return "错误：无法获取港股列表以解析股票名称。请尝试直接提供股票代码。"

            # 确保 '代码' 和 '名称' 列存在且为字符串类型
            if '代码' not in hk_spot_df.columns or '名称' not in hk_spot_df.columns:
                return "错误：获取到的港股列表数据缺少 '代码' 或 '名称' 列，无法解析股票名称。"
            hk_spot_df['代码'] = hk_spot_df['代码'].astype(str)
            hk_spot_df['名称'] = hk_spot_df['名称'].astype(str)

            # 优先精确匹配名称
            matched_stocks = hk_spot_df[hk_spot_df['名称'] == stock_identifier]

            # 如果没有精确匹配，模糊匹配
            if matched_stocks.empty:
                matched_stocks = hk_spot_df[hk_spot_df['名称'].str.contains(stock_identifier, case=False, na=False)]

            if matched_stocks.empty:
                return f"未能在港股列表中找到与 '{stock_identifier}' 匹配的股票代码。请检查股票名称是否正确或尝试直接提供股票代码。"
            elif len(matched_stocks) > 1:
                output_lines = [f"找到多个与 '{stock_identifier}' 匹配的港股，请提供更具体的股票名称或直接提供股票代码："]
                output_lines.append("=" * 100)
                for index, row in matched_stocks.head(10).iterrows():
                    output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                if len(matched_stocks) > 10:
                    output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
                output_lines.append("=" * 100)
                return "\n".join(output_lines)
            else:
                resolved_stock_code = matched_stocks.iloc[0]['代码']
                resolved_stock_name = matched_stocks.iloc[0]['名称']

        except Exception as e:
            return f"解析股票名称 '{stock_identifier}' 时发生错误: {e}。请尝试直接提供股票代码。"

    if not resolved_stock_code:
        return f"无法为 '{stock_identifier}' 解析出有效的港股代码。请确保输入正确或尝试直接提供股票代码。"

    try:
        # 调用 API
        famous_spot_df = ak.stock_hk_famous_spot_em()

        if famous_spot_df.empty:
            return "未能获取知名港股的实时行情数据，返回数据为空。可能数据源暂无数据或网络问题。"
        
        # 确保 '代码' 列是字符串类型以便比较
        if '代码' in famous_spot_df.columns:
            famous_spot_df['代码'] = famous_spot_df['代码'].astype(str)
        else:
            return "错误：获取到的知名港股列表数据缺少 '代码' 列，无法进行查询。"

        # 过滤出指定股票的数据
        # 处理港股代码中的.HK后缀
        target_code_variants = [resolved_stock_code, f"{resolved_stock_code}.HK"]
        
        filtered_df = famous_spot_df[
            famous_spot_df['代码'].isin(target_code_variants)
        ]

        if filtered_df.empty:
            return f"未在东方财富网的“知名港股”列表中找到 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 的实时行情数据。该股票可能不在此知名列表中。"
        
        # 取第一个匹配
        stock_data = filtered_df.iloc[0]

        # 格式化输出
        output_lines = [f"港股 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 的知名港股实时行情数据 (数据来源: 东方财富网)。\n"]
        output_lines.append("=" * 100)
        
        # 格式化数值列的辅助函数
        def format_value(value, unit="", precision=2):
            if pd.notna(value):
                if unit == "股":
                    return f"{int(value):,}{unit}"
                elif unit == "%":
                    return f"{value:,.{precision}f}{unit}"
                elif unit == "港元":
                    # 对于较大的金额，转换为万或亿
                    if value >= 1_000_000_000:
                        return f"{value / 1_000_000_000:,.{precision}f}亿{unit}"
                    elif value >= 1_000_000:
                        return f"{value / 1_000_000:,.{precision}f}百万{unit}"
                    elif value >= 10_000:
                        return f"{value / 10_000:,.{precision}f}万{unit}"
                    else:
                        return f"{value:,.{precision}f}{unit}"
                else:
                    return f"{value:,.{precision}f}"
            return "N/A"

        # 定义要显示的字段名称和格式
        display_fields = {
            '代码': {'name': '证券代码', 'format': lambda x: str(x)},
            '名称': {'name': '证券简称', 'format': lambda x: str(x)},
            '最新价': {'name': '最新价', 'format': lambda x: format_value(x, '港元', 3)},
            '涨跌额': {'name': '涨跌额', 'format': lambda x: format_value(x, '港元', 3)},
            '涨跌幅': {'name': '涨跌幅', 'format': lambda x: format_value(x, '%', 2)},
            '今开': {'name': '今开', 'format': lambda x: format_value(x, '港元', 3)},
            '最高': {'name': '最高', 'format': lambda x: format_value(x, '港元', 3)},
            '最低': {'name': '最低', 'format': lambda x: format_value(x, '港元', 3)},
            '昨收': {'name': '昨收', 'format': lambda x: format_value(x, '港元', 3)},
            '成交量': {'name': '成交量', 'format': lambda x: format_value(x, '股', 0)},
            '成交额': {'name': '成交额', 'format': lambda x: format_value(x, '港元', 2)}
        }

        for col, info in display_fields.items():
            if col in stock_data.index:
                formatted_value = info['format'](stock_data[col])
                output_lines.append(f"{info['name']}: {formatted_value}")
            
        output_lines.append("=" * 100)
        output_lines.append(f"提示：此为港股 '{resolved_stock_name or stock_identifier}' 的实时行情数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股 '{resolved_stock_code}' ({resolved_stock_name or stock_identifier}) 知名港股实时行情信息时发生错误: {e}"

@tool
def overview(symbol: str) -> str:
    """
    获取公司概况：市值、PE 等关键指标。
    此API只能查找除中国外的公司。

    Args:
        symbol (str): 股票代码，例如 IBM。

    Returns:
        str: 公司概况信息。
    """
    if not symbol:
        return "参数 symbol 为必填项，请提供有效的股票代码。"
    try:
        # 调用 API
        data, _ = fd.get_company_overview(symbol)
        info = data.iloc[0]
        # 返回公司概况
        return (f"{symbol.upper()} 概况：\n"
                f"Name: {info.get('Name')}\n"
                f"MarketCapitalization: {info.get('MarketCapitalization')}\n"
                f"PERatio: {info.get('PERatio')}\n"
                f"EPS: {info.get('EPS')}\n")
    except Exception as e:
        return f"获取公司概况时发生错误: {e}"

@tool
def income_statement(symbol: str, year: Optional[int] = None, quarter_end_date: Optional[str] = None) -> str:
    """
    获取指定年度或季度的损益表数据，或最新的年度数据。
    此API只能查找除中国外的公司。

    Args:
        symbol (str): 股票代码，例如 IBM。
        year (Optional[int]): 可选参数，指定要查询的年度，例如 2023。
        quarter_end_date (Optional[str]): 可选参数，指定要查询的季度末日期，格式为 'YYYY-MM-DD'，例如 '2023-03-31'。

    Returns:
        str: 指定年度或季度，或最新年度的收入与净利润。
    """
    if not symbol:
        return "参数 symbol 为必填项，请提供有效的股票代码。"

    try:
        result_data = None
        period_label = ""

        if quarter_end_date:
            # 获取季度数据并按精确日期过滤
            quarterly_data, _ = fd.get_income_statement_quarterly(symbol)
            if quarterly_data.empty:
                return f"未能获取 {symbol.upper()} 的季度损益表数据。"

            # 过滤出与 quarter_end_date 完全匹配的数据
            filtered_data = quarterly_data[quarterly_data['fiscalDateEnding'] == quarter_end_date]

            if not filtered_data.empty:
                result_data = filtered_data.iloc[0]
                period_label = f"{quarter_end_date} 季度"
            else:
                return f"未能找到 {symbol.upper()} 在 {quarter_end_date} 的季度损益表数据。请检查日期格式或数据可用性。"

        elif year:
            # 获取年度数据并按年份过滤
            annual_data, _ = fd.get_income_statement_annual(symbol)
            if annual_data.empty:
                return f"未能获取 {symbol.upper()} 的年度损益表数据。"

            # 从 'fiscalDate' 字符串中提取年份进行过滤
            # 确保 'fiscalDate' 列是字符串类型，以便进行字符串操作
            annual_data['fiscalYear'] = annual_data['fiscalDateEnding'].apply(lambda x: int(x.split('-')[0]))
            filtered_data = annual_data[annual_data['fiscalYear'] == year]

            if not filtered_data.empty:
                result_data = filtered_data.iloc[0]
                period_label = f"{year} 年度"
            else:
                return f"未能找到 {symbol.upper()} 在 {year} 年度的损益表数据。请检查年份或数据可用性。"
        else:
            # 当 year 和 quarter_end_date 都为空时，返回最新年度数据
            annual_data, _ = fd.get_income_statement_annual(symbol)
            if annual_data.empty:
                return f"未能获取 {symbol.upper()} 的最新年度损益表数据。"
            result_data = annual_data.iloc[0]
            period_label = "最新年度"

        if result_data is None:
            # 不应到达此处
            return f"未能获取 {symbol.upper()} 的损益表数据。"

        return (f"{symbol.upper()} {result_data.get('fiscalDate')} {period_label} 损益表：\n" # 添加 fiscalDate 方便确认
                f"TotalRevenue: {result_data.get('totalRevenue')}\n"
                f"NetIncome: {result_data.get('netIncome')}\n")

    except Exception as e:
        return f"获取损益表时发生错误: {e}"

@tool
def balance_sheet(symbol: str, year: Optional[int] = None, quarter_end_date: Optional[str] = None) -> str:
    """
    获取指定年度或季度的资产负债表数据，或最新的年度数据。
    此API只能查找除中国外的公司。

    Args:
        symbol (str): 股票代码，例如 IBM。
        year (Optional[int]): 可选参数，指定要查询的年度，例如 2023。
        quarter_end_date (Optional[str]): 可选参数，指定要查询的季度末日期，格式为 'YYYY-MM-DD'，例如 '2023-03-31'。

    Returns:
        str: 指定年度或季度，或最新年度的资产、负债和股东权益。
    """
    if not symbol:
        return "参数 symbol 为必填项，请提供有效的股票代码。"

    try:
        result_data = None
        period_label = ""

        if quarter_end_date:
            # 获取季度数据并按精确日期过滤
            quarterly_data, _ = fd.get_balance_sheet_quarterly(symbol)
            if quarterly_data.empty:
                return f"未能获取 {symbol.upper()} 的季度资产负债表数据。"

            # 过滤出与 quarter_end_date 完全匹配的数据
            filtered_data = quarterly_data[quarterly_data['fiscalDateEnding'] == quarter_end_date]

            if not filtered_data.empty:
                result_data = filtered_data.iloc[0]
                period_label = f"{quarter_end_date} 季度"
            else:
                return f"未能找到 {symbol.upper()} 在 {quarter_end_date} 的季度资产负债表数据。请检查日期格式或数据可用性。"

        elif year:
            # 获取年度数据并按年份过滤
            annual_data, _ = fd.get_balance_sheet_annual(symbol)
            if annual_data.empty:
                return f"未能获取 {symbol.upper()} 的年度资产负债表数据。"

            # 从 'fiscalDate' 字符串中提取年份进行过滤
            # 确保 'fiscalDate' 列是字符串类型，以便进行字符串操作
            annual_data['fiscalYear'] = annual_data['fiscalDateEnding'].apply(lambda x: int(x.split('-')[0]))
            filtered_data = annual_data[annual_data['fiscalYear'] == year]

            if not filtered_data.empty:
                result_data = filtered_data.iloc[0]
                period_label = f"{year} 年度"
            else:
                return f"未能找到 {symbol.upper()} 在 {year} 年度的资产负债表数据。请检查年份或数据可用性。"
        else:
            # 当 year 和 quarter_end_date 都为空时，返回最新年度数据
            annual_data, _ = fd.get_balance_sheet_annual(symbol)
            if annual_data.empty:
                return f"未能获取 {symbol.upper()} 的最新年度资产负债表数据。"
            result_data = annual_data.iloc[0]
            period_label = "最新年度"

        if result_data is None:
            # 不应该到此处
            return f"未能获取 {symbol.upper()} 的资产负债表数据。"

        return (f"{symbol.upper()} {result_data.get('fiscalDate')} {period_label} 资产负债表：\n"
                f"TotalAssets: {result_data.get('totalAssets')}\n"
                f"TotalLiabilities: {result_data.get('totalLiabilities')}\n"
                f"ShareholderEquity: {result_data.get('totalShareholderEquity')}\n")

    except Exception as e:
        return f"获取资产负债表时发生错误: {e}"

@tool
def cash_flow(symbol: str, year: Optional[int] = None, quarter_end_date: Optional[str] = None) -> str:
    """
    获取指定年度或季度的现金流量表数据，或最新的年度数据。
    此API只能查找除中国外的公司。

    Args:
        symbol (str): 股票代码，例如 IBM。
        year (Optional[int]): 可选参数，指定要查询的年度，例如 2023。
        quarter_end_date (Optional[str]): 可选参数，指定要查询的季度末日期，格式为 'YYYY-MM-DD'，例如 '2023-03-31'。

    Returns:
        str: 指定年度或季度，或最新年度的经营、投资和自由现金流。
    """
    if not symbol:
        return "参数 symbol 为必填项，请提供有效的股票代码。"


    try:
        result_data = None
        period_label = ""

        if quarter_end_date:
            # 获取季度数据并按精确日期过滤
            quarterly_data, _ = fd.get_cash_flow_quarterly(symbol)
            if quarterly_data.empty:
                return f"未能获取 {symbol.upper()} 的季度现金流量表数据。"

            # 过滤出与 quarter_end_date 完全匹配的数据
            filtered_data = quarterly_data[quarterly_data['fiscalDateEnding'] == quarter_end_date]

            if not filtered_data.empty:
                result_data = filtered_data.iloc[0]
                period_label = f"{quarter_end_date} 季度"
            else:
                return f"未能找到 {symbol.upper()} 在 {quarter_end_date} 的季度现金流量表数据。请检查日期格式或数据可用性。"

        elif year:
            # 获取年度数据并按年份过滤
            annual_data, _ = fd.get_cash_flow_annual(symbol)
            if annual_data.empty:
                return f"未能获取 {symbol.upper()} 的年度现金流量表数据。"

            # 从 'fiscalDate' 字符串中提取年份进行过滤
            # 确保 'fiscalDate' 列是字符串类型，以便进行字符串操作
            annual_data['fiscalYear'] = annual_data['fiscalDateEnding'].apply(lambda x: int(x.split('-')[0]))
            filtered_data = annual_data[annual_data['fiscalYear'] == year]

            if not filtered_data.empty:
                result_data = filtered_data.iloc[0]
                period_label = f"{year} 年度"
            else:
                return f"未能找到 {symbol.upper()} 在 {year} 年度的现金流量表数据。请检查年份或数据可用性。"
        else:
            # 当 year 和 quarter_end_date 都为空时，返回最新年度数据
            annual_data, _ = fd.get_cash_flow_annual(symbol)
            if annual_data.empty:
                return f"未能获取 {symbol.upper()} 的最新年度现金流量表数据。"
            result_data = annual_data.iloc[0]
            period_label = "最新年度"

        if result_data is None:
            # 不应该到此处
            return f"未能获取 {symbol.upper()} 的现金流量表数据。"

        return (f"{symbol.upper()} {result_data.get('fiscalDate')} {period_label} 现金流量表：\n"
                f"OperatingCashflow: {result_data.get('operatingCashflow')}\n"
                f"CapitalExpenditures: {result_data.get('capitalExpenditures')}\n"
                f"FreeCashFlow: {result_data.get('freeCashflow')}\n")

    except Exception as e:
        return f"获取现金流量表时发生错误: {e}"

@tool
def earnings(symbol: str, year: Optional[int] = None, quarter_end_date: Optional[str] = None) -> str:
    """
    获取指定年度或季度的每股收益及分析师预估数据，或最新的年度数据。
    此API只能查找除中国外的公司。

    Args:
        symbol (str): 股票代码，例如 IBM。
        year (Optional[int]): 可选参数，指定要查询的年度，例如 2023。
        quarter_end_date (Optional[str]): 可选参数，指定要查询的季度末日期，格式为 'YYYY-MM-DD'，例如 '2023-03-31'。

    Returns:
        str: 指定年度或季度，或最新年度的每股收益、预测和盈余意外。
    """
    if not symbol:
        return "参数 symbol 为必填项，请提供有效的股票代码。"


    try:
        result_data = None
        period_label = ""

        if quarter_end_date:
            # 获取季度数据并按精确日期过滤
            quarterly_data, _ = fd.get_earnings_quarterly(symbol)
            if quarterly_data.empty:
                return f"未能获取 {symbol.upper()} 的季度每股收益数据。"

            # 过滤出与 quarter_end_date 完全匹配的数据
            filtered_data = quarterly_data[quarterly_data['fiscalDateEnding'] == quarter_end_date]

            if not filtered_data.empty:
                result_data = filtered_data.iloc[0]
                period_label = f"{quarter_end_date} 季度"
            else:
                return f"未能找到 {symbol.upper()} 在 {quarter_end_date} 的季度每股收益数据。请检查日期格式或数据可用性。"

        elif year:
            # 获取年度数据并按年份过滤
            annual_data, _ = fd.get_earnings_annual(symbol)
            if annual_data.empty:
                return f"未能获取 {symbol.upper()} 的年度每股收益数据。"

            # 从 'fiscalDate' 字符串中提取年份进行过滤
            # 确保 'fiscalDate' 列是字符串类型，以便进行字符串操作
            annual_data['fiscalYear'] = annual_data['fiscalDateEnding'].apply(lambda x: int(x.split('-')[0]))
            filtered_data = annual_data[annual_data['fiscalYear'] == year]

            if not filtered_data.empty:
                result_data = filtered_data.iloc[0]
                period_label = f"{year} 年度"
            else:
                return f"未能找到 {symbol.upper()} 在 {year} 年度的每股收益数据。请检查年份或数据可用性。"
        else:
            # 当 year 和 quarter_end_date 都为空时，返回最新年度数据
            annual_data, _ = fd.get_earnings_annual(symbol) # 确保 fd 在此作用域内可用
            if annual_data.empty:
                return f"未能获取 {symbol.upper()} 的最新年度每股收益数据。"
            result_data = annual_data.iloc[0]
            period_label = "最新年度"

        if result_data is None:
            # 不应该到此处
            return f"未能获取 {symbol.upper()} 的每股收益数据。"

        return (f"{symbol.upper()} {result_data.get('fiscalDate')} {period_label} 每股收益：\n"
                f"ReportedEPS: {result_data.get('reportedEPS')}\n"
                f"EstimatedEPS: {result_data.get('estimatedEPS')}\n"
                f"Surprise: {result_data.get('surprise')}\n")

    except Exception as e:
        return f"获取每股收益时发生错误: {e}"

@tool
def dividends(symbol: str, year: Optional[int] = None, quarter_end_date: Optional[str] = None) -> str:
    """
    获取指定年度或季度的历史分红信息，或最近三次分红。
    此API只能查找除中国外的公司。

    Args:
        symbol (str): 股票代码，例如 IBM。
        year (Optional[int]): 可选参数，指定要查询的年度，例如 2023。
        quarter_end_date (Optional[str]): 可选参数，指定要查询的季度末日期，格式为 'YYYY-MM-DD'，例如 '2023-03-31'。
                                        将返回该季度内的所有分红。

    Returns:
        str: 指定年度或季度内的所有分红数据，或最近三次分红。
    """
    if not symbol:
        return "参数 symbol 为必填项，请提供有效的股票代码。"

    try:
        # 获取尽可能多的历史分红数据，以便进行过滤
        url = f"https://www.alphavantage.co/query?function=DIVIDENDS&symbol={symbol}&outputsize=full&apikey={Alpha_Vantage_API_KEY}"
        # 调用 API
        r = requests.get(url)
        data = r.json()

        # 检查错误信息
        if "Note" in data:
            return f"获取分红信息时发生错误: {data['Note']}"
        if "Error Message" in data:
            return f"获取分红信息时发生错误: {data['Error Message']}"
        if "data" not in data or not data["data"]:
            return f"{symbol.upper()} 没有可用的分红数据。"

        all_dividends = data["data"]

        filtered_dividends = []
        period_label = ""

        if quarter_end_date:
            try:
                print("1")
                # 解析季度末日期
                end_date = datetime.strptime(quarter_end_date, '%Y-%m-%d').date()
                # 根据季度末日期确定该季度的开始日期
                # 第一季度
                if end_date.month in [1, 2, 3]:
                    start_date = datetime.date(datetime(end_date.year, 1, 1))
                # 第二季度
                elif end_date.month in [4, 5, 6]:
                    start_date = datetime.date(datetime(end_date.year, 4, 1))
                # 第三季度
                elif end_date.month in [7, 8, 9]:
                    start_date = datetime.date(datetime(end_date.year, 7, 1))
                # 第四季度
                elif end_date.month in [10, 11, 12]:
                    start_date = datetime.date(datetime(end_date.year, 10, 1))
                else:
                    return "无效的季度末日期格式或月份。请使用 'YYYY-MM-DD' 格式。"

                period_label = f"{quarter_end_date} 季度"
                # 过滤出在该季度内的分红
                for div in all_dividends:
                    ex_date_str = div.get("ex_dividend_date")
                    # 确保 ex_date_str 是字符串
                    if isinstance(ex_date_str, str):
                        try:
                            ex_date = datetime.strptime(ex_date_str, '%Y-%m-%d').date()
                            if start_date <= ex_date <= end_date:
                                filtered_dividends.append(div)
                        except ValueError:
                            # 如果日期字符串格式不正确，则跳过此条分红数据
                            continue
                # 确保按日期降序排序，以显示最近的分红在前
                filtered_dividends.sort(key=lambda x: x.get("ex_dividend_date", ""), reverse=True)

            except ValueError:
                return "无效的季度末日期格式。请使用 'YYYY-MM-DD' 格式。"
            except Exception as e:
                return f"处理季度分红数据时发生错误: {e}"

        elif year:
            period_label = f"{year} 年度"
            # 过滤出指定年份的分红
            for div in all_dividends:
                ex_date_str = div.get("ex_dividend_date")
                # 确保 ex_date_str 是字符串
                if isinstance(ex_date_str, str) and ex_date_str.startswith(str(year)):
                    filtered_dividends.append(div)
            # 确保按日期降序排序
            filtered_dividends.sort(key=lambda x: x.get("ex_dividend_date", ""), reverse=True)

        else:
            # 当 year 和 quarter_end_date 都为空时，返回最近三次分红
            filtered_dividends = all_dividends[:3]
            period_label = "最近三次"

        if not filtered_dividends:
            if year:
                return f"未能找到 {symbol.upper()} 在 {year} 年度的分红数据。"
            elif quarter_end_date:
                return f"未能找到 {symbol.upper()} 在 {quarter_end_date} 季度的分红数据。"
            else:
                # 如果 all_dividends 为空
                return f"{symbol.upper()} 没有可用的分红数据。"

        lines = []
        for d in filtered_dividends:
            date = "ex_dividend_date:" + d.get("ex_dividend_date", "未知日期")
            amount = "amount:" + d.get("amount", "金额数据缺失")
            lines.append(f"{date}: {amount}")

        return f"{symbol.upper()} {period_label} 分红：\n" + "\n".join(lines)

    except requests.exceptions.RequestException as req_err:
        return f"网络请求错误: {req_err}"
    except Exception as e:
        return f"获取分红信息时发生意外错误: {e}"

@tool
def splits(symbol: str, year: Optional[int] = None) -> str:
    """
    获取历史拆股事件：指定年度或最近三次。
    此API只能查找除中国外的公司。

    Args:
        symbol (str): 股票代码，例如 IBM。
        year (Optional[int]): 可选参数，指定要查询的年度，例如 2023。

    Returns:
        str: 指定年度内的所有拆股日期和比例，或最近三次拆股日期和比例。
    """
    if not symbol:
        return "参数 symbol 为必填项，请提供有效的股票代码。"

    try:
        # 获取尽可能多的历史拆股数据，以便进行过滤
        url = f"https://www.alphavantage.co/query?function=SPLITS&symbol={symbol}&outputsize=full&apikey={Alpha_Vantage_API_KEY}"
        # 调用 API
        r = requests.get(url)
        data = r.json()

        # 检查错误信息
        if "Note" in data:
            return f"获取拆股信息时发生错误: {data['Note']}"
        if "Error Message" in data:
            return f"获取拆股信息时发生错误: {data['Error Message']}"
        if "data" not in data or not data["data"]:
            return f"{symbol.upper()} 没有可用的拆股数据。"

        all_splits = data["data"]

        filtered_splits = []
        period_label = ""

        if year:
            period_label = f"{year} 年度"
            # 过滤出指定年份的拆股
            for s in all_splits:
                effective_date_str = s.get("effective_date")
                # 确保 effective_date_str 是字符串且以指定年份开头
                if isinstance(effective_date_str, str) and effective_date_str.startswith(str(year)):
                    filtered_splits.append(s)
            # 确保按日期降序排序
            filtered_splits.sort(key=lambda x: x.get("effective_date", ""), reverse=True)
        else:
            # 当 year 为空时，返回最近三次拆股
            filtered_splits = all_splits[:3]
            period_label = "最近三次"

        if not filtered_splits:
            if year:
                return f"未能找到 {symbol.upper()} 在 {year} 年度的拆股数据。"
            else:
                # 如果 all_splits 为空
                return f"{symbol.upper()} 没有可用的拆股数据。"

        lines = []
        for s in filtered_splits:
            date = s.get("effective_date", "未知日期")
            ratio = s.get("split_factor", "比率缺失")
            lines.append(f"{date}: {ratio}")

        return f"{symbol.upper()} {period_label} 拆股记录：\n" + "\n".join(lines)

    except requests.exceptions.RequestException as req_err:
        return f"网络请求错误: {req_err}"
    except Exception as e:
        return f"获取拆股信息时发生错误: {e}"

@tool
def earnings_calendar(
        symbol: Optional[str] = None,
        horizon: Optional[str] = None
) -> str:
    """
       获取未来 3、6 或 12 个月内预计发布财报的公司列表。
       此API只能查找除中国外的公司。

       Args:
           symbol (Optional[str]): 股票代码，例如 IBM。若为空，则返回所有公司列表。
           horizon (Optional[str]): '3month','6month','12month'。

       Returns:
           str: 预期发布财报的公司及日期列表。
       """
    if symbol and horizon:
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={symbol}&horizon={horizon}&apikey={Alpha_Vantage_API_KEY}"
    elif horizon:
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon={horizon}&apikey={Alpha_Vantage_API_KEY}"
    elif symbol:
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symobl={symbol}&apikey={Alpha_Vantage_API_KEY}"
    else:
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={Alpha_Vantage_API_KEY}"

    try:
        # 调用 API
        r = requests.get(url)
        # 如果不是 JSON，尝试作为 CSV 解析
        content = r.content.decode('utf-8')
        lines = list(csv.reader(content.splitlines(), delimiter=','))
        # 第一行为表头，后续为数据
        if len(lines) <= 1:
            return "未获取到可用的财报日历数据。"
        # 展示表头和最多5条数据
        header = lines[0]
        # 取最多5条数据行
        display_rows = lines[1:6]
        sentences = []
        for row in display_rows:
            sym = row[0]
            name = row[1]
            report_date = row[2]
            period_end = row[3]
            eps = row[4]
            curr = row[5] if len(row) > 5 else ''
            sentence = (f"{sym}（{name}）预计于 {report_date} 发布截至 {period_end} 的财报，"
                        f"预估每股收益为 {eps} {curr}。\n")
            sentences.append(sentence)
        return "".join(sentences)
    except Exception as e:
        return f"获取财报日历时发生错误: {e}"

@tool
def get_eastmoney_stock_trading_halt_info(date: str, symbol: Optional[str] = None) -> str:
    """
    获取东方财富网数据中心指定日期的A股停复牌信息。
    该接口返回包括股票代码、名称、停牌时间、停牌截止时间、停牌期限、
    停牌原因、所属市场以及预计复牌时间等详细信息。

    Args:
        date (str): 需要查询停复牌信息的日期，格式为 'YYYYMMDD'，例如 '20240426'。
        symbol (Optional[str]): 可选参数。指定股票代码，例如  '600737' (A股)。
                                如果提供，将只返回该股票的调研统计数据。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的停复牌信息摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20240426'。"

    try:
        # 调用 API
        df = ak.stock_tfp_em(date=date)

        if df.empty:
            return f"在日期 {date} 未能获取到停复牌信息。请检查日期是否正确或该日期无数据。"

        output_lines = [f"东方财富网 {date} 的A股停复牌信息:\n"]
        output_lines.append("=" * 80)

        # 如果用户指定了股票代码
        if symbol:
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的停复牌信息。请检查股票代码是否正确或该日期无该股票停复牌数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的停复牌信息。")

        output_lines.append(f"总计找到 {len(df)} 条停复牌记录。")

        # 选择并重新排序关键列，以便在输出中显示
        # 移除 '序号' 列
        display_columns = [
            '代码', '名称', '停牌时间', '停牌截止时间', '停牌期限', '停牌原因', '所属市场', '预计复牌时间'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 对预计复牌时间进行处理
        if '预计复牌时间' in display_df.columns:
            display_df['预计复牌时间'] = display_df['预计复牌时间'].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '待定'
            )

        # 显示 DataFrame 的尾部数据
        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示最近 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.tail(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：停牌信息可能包含未来日期的数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的停复牌信息时发生错误: {e}"

@tool
def get_cninfo_stock_profile(symbol: str, fields: Optional[List[str]] = None) -> str:
    """
    获取巨潮资讯网指定股票代码的公司概况信息。
    该接口返回包括公司名称、英文名称、曾用简称、股票代码（A/B/H股）、
    所属市场、所属行业、法人代表、注册资金、成立日期、上市日期、官方网站、
    联系方式、注册地址、办公地址、主营业务、经营范围和机构简介等详细信息。

    Args:
        symbol (str): 需要查询公司概况的股票代码，例如 '600030'。
        fields (Optional[List[str]]): 可选参数，指定需要返回的公司概况字段列表。
                                       如果为空或None，则返回所有可用信息。
                                       可用的字段包括：'公司名称', '英文名称', '曾用简称',
                                       'A股代码', 'A股简称', 'B股代码', 'B股简称',
                                       'H股代码', 'H股简称', '入选指数', '所属市场',
                                       '所属行业', '法人代表', '注册资金', '成立日期',
                                       '上市日期', '官方网站', '电子邮箱', '联系电话',
                                       '传真', '注册地址', '办公地址', '邮政编码',
                                       '主营业务', '经营范围', '机构简介'。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的公司概况详细信息。
             如果获取失败，则返回错误信息。
    """
    if not symbol:
        return "参数 symbol 为必填项，请提供有效的股票代码。"
    try:
        # 调用 API
        data = ak.stock_profile_cninfo(symbol=symbol)
        if data.empty:
            return "抱歉, 没有查找到相关信息。"
        info = data.iloc[0].to_dict()

        # 确定各个可以查找的信息
        all_fields_map = {
            '公司名称': '公司名称', '英文名称': '英文名称', '曾用简称': '曾用简称',
            'A股代码': 'A股代码', 'A股简称': 'A股简称', 'B股代码': 'B股代码',
            'B股简称': 'B股简称', 'H股代码': 'H股代码', 'H股简称': 'H股简称',
            '入选指数': '入选指数', '所属市场': '所属市场', '所属行业': '所属行业',
            '法人代表': '法人代表', '注册资金': '注册资金', '成立日期': '成立日期',
            '上市日期': '上市日期', '官方网站': '官方网站', '电子邮箱': '电子邮箱',
            '联系电话': '联系电话', '传真': '传真', '注册地址': '注册地址',
            '办公地址': '办公地址', '邮政编码': '邮政编码', '主营业务': '主营业务',
            '经营范围': '经营范围', '机构简介': '机构简介'
        }

        # 确保公司名称始终作为概况的开头
        company_name = info.get('公司名称', '未知公司')
        result_parts = [f"{company_name} 概况："]

        # 如果fields为空， 返回所有信息
        if not fields:
            for key, display_name in all_fields_map.items():
                if key == '公司名称':
                    continue
                value = info.get(key)
                # 只有当值不为空字符串、None或NaN时才添加
                if value is not None and str(value).strip() != '' and str(value).strip().lower() != 'nan':
                    result_parts.append(f"{display_name}: {value}")
        # 返回要查找的指定信息
        else:
            # 确定查找字段是否存在
            valid_fields = [f for f in fields if f in all_fields_map]
            if not valid_fields:
                return f"请求的字段 '{', '.join(fields)}' 均无效或不存在。请检查字段名称。"

            for field_name in valid_fields:
                if field_name == '公司名称':
                    continue
                display_name = all_fields_map.get(field_name, field_name)
                value = info.get(field_name)
                # 只有当值不为空字符串、None或NaN时才添加，否则显示(无)
                if value is not None and str(value).strip() != '' and str(value).strip().lower() != 'nan':
                    result_parts.append(f"{display_name}: {value}")
                else:
                    result_parts.append(f"{display_name}: (无)")

        # 除了标题外没有其他信息
        if len(result_parts) == 1:
            return f"{company_name} 概况：没有找到或请求的有效信息。"

        return "\n".join(result_parts)

    except Exception as e:
        return f"在获取股票信息时出现错误: {e}"

@tool
def get_eastmoney_institutional_research_stats(date: str, symbol: Optional[str] = None) -> str:
    """
    获取东方财富网数据中心指定日期的机构调研统计数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、接待机构数量、接待方式、接待人员、
    接待地点、接待日期和公告日期等详细信息。

    Args:
        date (str): 需要查询的机构调研统计数据的日期，格式为 'YYYYMMDD'，例如 '20210128'。
        symbol (Optional[str]): 可选参数。指定股票代码，例如 '00005' (港股) 或 '600737' (A股)。
                                如果提供，将只返回该股票的调研统计数据。

    Returns:
        str: 一个格式化后的字符串，包含指定日期机构调研统计的摘要信息和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20210128'。"

    try:
        # 调用 API
        df = ak.stock_jgdy_tj_em(date=date)

        if df.empty:
            return f"在日期 {date} 未能获取到机构调研统计数据。请检查日期是否正确或该日期无数据。"

        output_lines = [f"东方财富网 {date} 的机构调研统计数据:\n"]
        output_lines.append("=" * 60)

        # 如果用户指定了股票代码
        if symbol:
            original_len = len(df)
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的调研统计数据。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的调研统计数据。")

        output_lines.append(f"总计找到 {len(df)} 条机构调研记录。")

        # 选择并重新排序关键列，以便在输出中显示
        # 移除 '序号' 列
        display_columns = [
            '代码', '名称', '最新价', '涨跌幅', '接待机构数量', '接待日期', '公告日期', '接待方式'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 格式化数值型
        if '最新价' in display_df.columns:
            display_df['最新价'] = display_df['最新价'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '-')
        if '涨跌幅' in display_df.columns:
            # 涨跌幅单位为 %，直接格式化为两位小数
            display_df['涨跌幅'] = display_df['涨跌幅'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '-')
        if '接待机构数量' in display_df.columns:
            display_df['接待机构数量'] = display_df['接待机构数量'].apply(lambda x: f"{int(x)}" if pd.notna(x) else '-')

        # 显示前几条记录
        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 60)

        output_lines.append(display_df.tail(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 60)
        output_lines.append("提示：涨跌幅单位为 %。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的机构调研统计数据时发生错误: {e}"

@tool
def get_a_share_goodwill_market_overview() -> str:
    """
    获取东方财富网数据中心-特色数据-商誉-A股商誉市场概况的历史数据。
    该接口返回每个报告期的商誉、商誉减值、净资产、商誉占净资产比例、
    商誉减值占净资产比例、净利润规模以及商誉减值占净利润比例等关键财务指标。

    Args:
        无。此函数不接受任何输入参数，直接返回所有历史数据。

    Returns:
        str: 一个格式化后的字符串，包含A股商誉市场概况的详细历史数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_sy_profile_em()

        if df.empty:
            return "未能获取A股商誉市场概况数据。请检查数据源是否可用。"

        output_lines = ["东方财富网-A股商誉市场概况历史数据:\n"]
        output_lines.append("=" * 80)

        # 复制DataFrame进行格式化
        formatted_df = df.copy()

        # 定义需要格式化为“亿元”的列
        monetary_cols = ['商誉', '商誉减值', '净资产', '净利润规模']
        for col in monetary_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x / 1e8:,.2f}" if pd.notna(x) else "-"
                )

        # 定义需要格式化为百分比的列
        percentage_cols = ['商誉占净资产比例', '商誉减值占净资产比例', '商誉减值占净利润比例']
        for col in percentage_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x * 100:,.2f}%" if pd.notna(x) else "-"
                )

        # 将格式化后的DataFrame转换为字符串
        output_lines.append(formatted_df.to_string(index=False))

        output_lines.append("=" * 80)
        output_lines.append("注意：")
        output_lines.append(f"1. 商誉、商誉减值、净资产、净利润规模的数值单位为 '亿元'。")
        output_lines.append(f"2. 比例数据已转换为百分比形式。")
        output_lines.append(f"3. 'NaN' 值表示数据缺失。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取A股商誉市场概况数据时发生错误: {e}"

@tool
def get_baidu_stock_trading_halt_info(date: str, symbol: Optional[str] = None) -> str:
    """
    获取百度股市通指定日期的停复牌信息，主要提供港股及其他非A股市场的停复牌数据。
    该接口返回包括股票代码、股票简称、交易所、停牌时间、复牌时间以及停牌事项说明等详细信息。

    Args:
        date (str): 需要查询停复牌信息的日期，格式为 'YYYYMMDD'，例如 '20241107'。
        symbol (Optional[str]): 可选参数。指定股票代码，例如 '00005' (港股) 。
                                如果提供，将只返回该股票的分红派息信息。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的停复牌信息摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20241107'。"

    try:
        # 调用 API
        df = ak.news_trade_notify_suspend_baidu(date=date)

        if df.empty:
            return f"在日期 {date} 未能获取到百度股市通的停复牌信息。请检查日期是否正确或该日期无数据。"

        output_lines = [f"百度股市通 {date} 的停复牌信息 (主要为港股):\n"]
        output_lines.append("=" * 80)

        # 如果用户指定了股票代码
        if symbol:
            original_len = len(df)
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的停复牌信息。请检查股票代码是否正确或该日期无该股票停复牌数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的停复牌信息。")

        output_lines.append(f"总计找到 {len(df)} 条停复牌记录。")

        # 选择并重新排序关键列
        display_columns = [
            '股票代码', '股票简称', '交易所', '停牌时间', '复牌时间', '停牌事项说明'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 对时间列进行处理
        for col in ['停牌时间', '复牌时间']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '待定/未披露'
                )

        # 显示指定数量行
        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：此数据主要提供港股及其他非A股市场的停复牌信息。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的百度股市通停复牌信息时发生错误: {e}"

@tool
def get_baidu_dividend_info(date: str, symbol: Optional[str] = None) -> str:
    """
    获取百度股市通指定日期的分红派息信息，主要提供港股及其他非A股市场的分红数据。
    该接口返回包括股票代码、除权日、分红、送股、转增、实物派发、交易所、
    股票简称和报告期等详细信息。

    Args:
        date (str): 需要查询分红派息信息的日期，格式为 'YYYYMMDD'，例如 '20241107'。
        symbol (Optional[str]): 可选参数。指定股票代码，例如 '00005' (港股) 或 '600737' (A股)。
                                如果提供，将只返回该股票的分红派息信息。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的分红派息信息摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20241107'。"

    try:
        # 调用 API
        df = ak.news_trade_notify_dividend_baidu(date=date)

        # 检查返回的 DataFrame 是否为空
        if df.empty:
            return f"在日期 {date} 未能获取到百度股市通的分红派息信息。请检查日期是否正确或该日期无数据。"

        output_lines = [f"百度股市通 {date} 的分红派息信息 (主要为港股):\n"]
        output_lines.append("=" * 80)

        # 如果用户指定了股票代码
        if symbol:
            # 过滤 DataFrame，只保留指定股票代码的记录
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            # 未找到该股票的数据
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的分红派息信息。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的分红派息信息。")

        # 添加总记录数信息
        output_lines.append(f"总计找到 {len(df)} 条分红派息记录。")

        # 定义需要显示在输出中的列名顺序
        display_columns = [
            '股票代码', '股票简称', '交易所', '除权日', '分红', '送股', '转增', '实物', '报告期'
        ]

        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()
        display_df = display_df.fillna('-')

        # 格式化日期列
        for col in ['除权日', '报告期']:
            if col in display_df.columns:
                if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                    # 将日期时间对象格式化为字符串
                    display_df[col] = display_df[col].dt.strftime('%Y-%m-%d').fillna('-')

        # 确定要显示的行数，不超过预设的最大行数
        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80) # 添加分隔线

        # 将要显示的数据帧转换为字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        # 如果总记录数超过了显示的行数，则添加提示信息
        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：此数据主要提供港股及其他非A股市场的股票分红派息信息。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的百度股市通分红派息信息时发生错误: {e}"


@tool
def get_eastmoney_stock_news(symbol: str) -> str:
    """
    获取东方财富网指定股票代码或关键词的最新新闻资讯数据。
    该接口返回指定股票当日最近 100 条新闻资讯的详细信息，包括关键词、新闻标题、
    新闻内容、发布时间、文章来源和新闻链接。

    Args:
        symbol (str): 需要查询新闻资讯的股票代码或相关关键词，例如 '603777'。

    Returns:
        str: 一个格式化后的字符串，包含指定股票或关键词的最新新闻资讯摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：股票代码或关键词不能为空。请提供有效的 symbol 参数。"

    try:
        # 调用 API
        df = ak.stock_news_em(symbol=symbol)

        if df.empty:
            return f"未能获取到股票代码/关键词 '{symbol}' 的新闻资讯数据。请检查代码或关键词是否正确或无相关新闻。"

        output_lines = [f"东方财富网关于 '{symbol}' 的最新新闻资讯:\n"]
        output_lines.append("=" * 80)
        output_lines.append(f"总计找到 {len(df)} 条新闻资讯记录。")

        # 选择并重新排序关键列
        # 移除 '关键词' 列
        display_columns = [
            '新闻标题', '发布时间', '文章来源', '新闻链接'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 考虑到新闻标题和链接可能较长，截断以适应输出
        if '新闻标题' in display_df.columns:
            display_df['新闻标题'] = display_df['新闻标题'].apply(lambda x: x[:50] + '...' if len(x) > 53 else x)
        if '新闻链接' in display_df.columns:
            display_df['新闻链接'] = display_df['新闻链接'].apply(lambda x: x[:60] + '...' if len(x) > 63 else x)

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示最近 {rows_to_show} 条新闻：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条新闻未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append(f"提示：新闻标题和链接可能已截断，完整内容请点击链接查看。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票代码/关键词 '{symbol}' 的新闻资讯数据时发生错误: {e}"

@tool
def get_baidu_financial_report_release_info(date: str, symbol: Optional[str] = None) -> str:
    """
    获取百度股市通指定日期的财报发行信息，主要提供港股及其他非A股市场的财报数据。
    该接口返回包括股票代码、交易所、股票简称和财报期等详细信息。

    Args:
        date (str): 需要查询财报发行信息的日期，格式为 'YYYYMMDD'，例如 '20241107'。
        symbol (Optional[str]): 可选参数。指定股票代码，例如 '00981' (港股) 或 'XRAY' (美股)。
                                如果提供，将只返回该股票的财报发行信息。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的财报发行信息摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20241107'。"

    try:
        # 调用 API
        df = ak.news_report_time_baidu(date=date)

        # 检查返回的 DataFrame 是否为空
        if df.empty:
            return f"在日期 {date} 未能获取到百度股市通的财报发行信息。请检查日期是否正确或该日期无数据。"

        output_lines = [f"百度股市通 {date} 的财报发行信息 (主要为港股及其他市场):\n"]
        output_lines.append("=" * 80)

        # 如果用户指定了股票代码
        if symbol:
            # 过滤 DataFrame，只保留指定股票代码的记录
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            # 未找到该股票的数据
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的财报发行信息。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的财报发行信息。")

        output_lines.append(f"总计找到 {len(df)} 条财报发行记录。")

        # 定义需要显示在输出中的列名顺序
        display_columns = [
            '股票代码', '股票简称', '交易所', '财报期'
        ]

        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        display_df = display_df.fillna('-')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        # 如果总记录数超过了显示的行数，则添加提示信息
        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：此数据主要提供港股及其他非A股市场的财报发行信息。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的百度股市通财报发行信息时发生错误: {e}"

@tool
def get_eastmoney_stock_performance_report(date: str, symbol: Optional[str] = None) -> str:
    """
        获取东方财富网数据中心指定日期的A股业绩报告数据。
        该接口返回包括股票代码、股票简称、每股收益、营业总收入（含同比和环比）、
        净利润（含同比和环比）、每股净资产、净资产收益率、每股经营现金流量、
        销售毛利率、所处行业和最新公告日期等详细财务指标。

        Args:
            date (str): 需要查询业绩报告的日期，格式为 'YYYYMMDD'，
                        必须是季度末日期，例如 '20200331', '20200630', '20200930', 或 '20201231'。
                        数据从 20100331 开始。
            symbol (Optional[str]): 可选参数。指定股票代码，例如 '000833'。
                                    如果提供，将只返回该股票的业绩报告信息。

        Returns:
            str: 一个格式化后的字符串，包含指定日期的业绩报告摘要和部分详细数据。
                如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        dt_obj = datetime.strptime(date, "%Y%m%d")
        # 检查日期是否为有效的季度末日期
        if dt_obj.strftime("%m%d") not in ["0331", "0630", "0930", "1231"]:
            return f"错误：日期 '{date}' 不是有效的季度末日期。请提供 'YYYY0331', 'YYYY0630', 'YYYY0930', 或 'YYYY1231' 格式的日期。"
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20220331'。"

    try:
        # 调用 API
        df = ak.stock_yjbb_em(date=date)

        # 检查返回的 DataFrame 是否为空
        if df.empty:
            return f"在日期 {date} 未能获取到东方财富网的业绩报告数据。请检查日期是否正确或该日期无数据。"

        output_lines = [f"东方财富网 {date} 的A股业绩报告信息:\n"]
        output_lines.append("=" * 80) # 添加分隔线

        # 如果用户指定了股票代码
        if symbol:
            # 过滤 DataFrame，只保留指定股票代码的记录
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            # 未找到该股票的数据
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的业绩报告数据。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的业绩报告。")

        output_lines.append(f"总计找到 {len(df)} 条业绩报告记录。")

        # 定义需要显示在输出中的列名顺序
        display_columns = [
            '股票代码', '股票简称', '每股收益', '营业总收入-营业总收入', '营业总收入-同比增长',
            '净利润-净利润', '净利润-同比增长', '每股净资产', '净资产收益率',
            '销售毛利率', '所处行业', '最新公告日期'
        ]

        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        display_df = display_df.fillna('-')

        # 定义需要格式化为百分比的列
        percentage_cols = ['营业总收入-同比增长', '营业总收入-季度环比增长', '净利润-同比增长', '净利润-季度环比增长',
                           '净资产收益率', '销售毛利率']
        for col in percentage_cols:
            if col in display_df.columns:
                # 将列转换为数值类型
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna('-')
                # 对数值进行格式化
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) and x != '-' else str(x)
                )

        # 定义需要格式化为货币的列
        currency_cols = ['每股收益', '营业总收入-营业总收入', '净利润-净利润', '每股净资产', '每股经营现金流量']
        for col in currency_cols:
            if col in display_df.columns:
                # 将列转换为数值类型
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna('-')
                # 对数值进行格式化
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and x != '-' else str(x)
                )

        # 格式化 '最新公告日期' 列
        if '最新公告日期' in display_df.columns:
            display_df['最新公告日期'] = pd.to_datetime(display_df['最新公告日期'], errors='coerce').dt.strftime(
                '%Y-%m-%d').fillna('-')

        # 确定要显示的行数，不超过预设的最大行数
        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        # 如果总记录数超过了显示的行数，则添加提示信息
        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：金额单位为元，百分比已按原数据格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的东方财富网业绩报告数据时发生错误: {e}"


@tool
def get_eastmoney_stock_performance_express_report(date: str, symbol: Optional[str] = None) -> str:
    """
    获取东方财富网数据中心指定日期的A股业绩快报数据。
    该接口返回包括股票代码、股票简称、每股收益、营业收入（含去年同期、同比和环比）、
    净利润（含去年同期、同比和环比）、每股净资产、净资产收益率、所处行业、
    公告日期、市场板块和证券类型等详细财务指标。

    Args:
        date (str): 需要查询业绩快报的日期，格式为 'YYYYMMDD'，
                    必须是季度末日期，例如 '20200331', '20200630', '20200930', 或 '20201231'。
                    数据从 20100331 开始。
        symbol (Optional[str]): 可选参数。指定股票代码，例如 '000591'。
                                如果提供，将只返回该股票的业绩快报信息。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的业绩快报摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        dt_obj = datetime.strptime(date, "%Y%m%d")
        # 检查日期是否为有效的季度末日期
        if dt_obj.strftime("%m%d") not in ["0331", "0630", "0930", "1231"]:
            return f"错误：日期 '{date}' 不是有效的季度末日期。请提供 'YYYY0331', 'YYYY0630', 'YYYY0930', 或 'YYYY1231' 格式的日期。"
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20200331'。"

    try:
        # 调用 API
        df = ak.stock_yjkb_em(date=date)

        # 检查返回的 DataFrame 是否为空
        if df.empty:
            return f"在日期 {date} 未能获取到东方财富网的业绩快报数据。请检查日期是否正确或该日期无数据。"

        output_lines = [f"东方财富网 {date} 的A股业绩快报信息:\n"]
        output_lines.append("=" * 80)

        # 如果用户指定了股票代码
        if symbol:
            # 过滤 DataFrame，只保留指定股票代码的记录
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            # 未找到该股票的数据
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的业绩快报数据。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的业绩快报。")

        # 添加总记录数信息
        output_lines.append(f"总计找到 {len(df)} 条业绩快报记录。")

        # 定义需要显示在输出中的列名顺序
        display_columns = [
            '股票代码', '股票简称', '每股收益', '营业收入-营业收入', '营业收入-同比增长',
            '净利润-净利润', '净利润-同比增长', '每股净资产', '净资产收益率',
            '所处行业', '公告日期', '市场板块', '证券类型'
        ]

        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        display_df = display_df.fillna('-')

        # 定义需要格式化的数值列及其格式类型
        numeric_cols_to_format = {
            '每股收益': "yuan",
            '营业收入-营业收入': "yuan",
            '营业收入-去年同期': "yuan",
            '营业收入-同比增长': "percent",
            '营业收入-季度环比增长': "percent",
            '净利润-净利润': "yuan",
            '净利润-去年同期': "yuan",
            '净利润-同比增长': "percent",
            '净利润-季度环比增长': "percent",
            '每股净资产': "yuan",
            '净资产收益率': "percent"
        }

        # 遍历并格式化数值列
        for col, unit_type in numeric_cols_to_format.items():
            if col in display_df.columns:
                # 将列转换为数值类型，无法转换的设为 NaN
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                if unit_type == "yuan":
                    # 对金额进行格式化
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:,.2f}" if pd.notna(x) else '-'
                    )
                elif unit_type == "percent":
                    # 对百分比进行格式化
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                    )

        # 格式化 '公告日期' 列
        if '公告日期' in display_df.columns:
            display_df['公告日期'] = pd.to_datetime(display_df['公告日期'], errors='coerce').dt.strftime(
                '%Y-%m-%d').fillna('-')

        # 确定要显示的行数，不超过预设的最大行数
        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        # 如果总记录数超过了显示的行数，则添加提示信息
        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：金额单位为元，百分比已按原数据格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的东方财富网业绩快报数据时发生错误: {e}"

@tool
def get_eastmoney_stock_performance_forecast(date: str, symbol: Optional[str] = None) -> str:
    """
    获取东方财富网数据中心指定日期的A股业绩预告数据。
    该接口返回包括股票代码、股票简称、预测指标（如预增、预减、扭亏等）、
    业绩变动、预测数值（净利润或营收）、业绩变动幅度、业绩变动原因、
    预告类型、上年同期值和公告日期等详细信息。

    Args:
        date (str): 需要查询业绩预告的日期，格式为 'YYYYMMDD'，
                    必须是季度末日期，例如 '20200331', '20200630', '20200930', 或 '20201231'。
                    数据从 20081231 开始。
        symbol (Optional[str]): 可选参数。指定股票代码，例如 '600189'。
                                如果提供，将只返回该股票的业绩预告信息。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的业绩预告摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式和季度末日期
    try:
        dt_obj = datetime.strptime(date, "%Y%m%d")
        if dt_obj.strftime("%m%d") not in ["0331", "0630", "0930", "1231"]:
            return f"错误：日期 '{date}' 不是有效的季度末日期。请提供 'YYYY0331', 'YYYY0630', 'YYYY0930', 或 'YYYY1231' 格式的日期。"
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20190331'。"

    try:
        # 调用 API
        df = ak.stock_yjyg_em(date=date)

        if df.empty:
            return f"在日期 {date} 未能获取到东方财富网的业绩预告数据。请检查日期是否正确或该日期无数据。"

        output_lines = [f"东方财富网 {date} 的A股业绩预告信息:\n"]
        output_lines.append("=" * 80)

        # 股票代码过滤
        if symbol:
            # 确保股票代码列是字符串类型以便进行精确匹配
            df['股票代码'] = df['股票代码'].astype(str)
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的业绩预告数据。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的业绩预告。")

        output_lines.append(f"总计找到 {len(df)} 条业绩预告记录。")

        # 选择并重新排序关键列，以便在输出中显示
        # 移除 '序号' 列
        display_columns = [
            '股票代码', '股票简称', '预测指标', '业绩变动', '预测数值',
            '业绩变动幅度', '上年同期值', '预告类型', '业绩变动原因', '公告日期'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值
        display_df = display_df.fillna('-')

        # 格式化数值列：预测数值和上年同期值
        currency_cols = ['预测数值', '上年同期值']
        for col in currency_cols:
            if col in display_df.columns:
                # 将列转换为数值类型，无法转换的变为 NaN
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                # 对数值进行格式化，并处理 NaN
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else '-'
                )

        # 格式化百分比列：业绩变动幅度
        if '业绩变动幅度' in display_df.columns:
            display_df['业绩变动幅度'] = pd.to_numeric(display_df['业绩变动幅度'], errors='coerce')
            display_df['业绩变动幅度'] = display_df['业绩变动幅度'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
            )

        # 格式化日期列
        if '公告日期' in display_df.columns:
            display_df['公告日期'] = pd.to_datetime(display_df['公告日期'], errors='coerce').dt.strftime(
                '%Y-%m-%d').fillna('-')

        # 截断长文本，尤其是 '业绩变动原因'
        if '业绩变动原因' in display_df.columns:
            display_df['业绩变动原因'] = display_df['业绩变动原因'].apply(
                lambda x: (x[:100] + '...' if len(str(x)) > 103 else x) if x != '-' else '-'
            )

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：金额单位为元，百分比已按原数据格式化。部分长文本信息可能已截断。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的东方财富网业绩预告数据时发生错误: {e}"

@tool
def get_cninfo_ipo_summary(symbol: str) -> str:
    """
    获取巨潮资讯网指定股票代码的上市相关数据。
    该接口返回包括股票代码、招股公告日期、中签率公告日、每股面值、总发行数量、
    发行前/后每股净资产、摊薄发行市盈率、募集资金净额、上网发行日期、上市日期、
    发行价格、发行费用总额、上网发行中签率和主承销商等详细信息。

    Args:
        symbol (str): 需要查询上市相关信息的股票代码，例如 '600030'。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的上市相关详细信息。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：股票代码不能为空。请提供有效的 symbol 参数。"

    try:
        # 调用 API
        df = ak.stock_ipo_summary_cninfo(symbol=symbol)

        if df.empty:
            return f"未能获取到股票代码 '{symbol}' 的上市相关信息。请检查股票代码是否正确或无相关数据。"

        ipo_data = df.iloc[0].to_dict()

        output_lines = [f"巨潮资讯关于股票 '{symbol}' 的上市相关信息:\n"]
        output_lines.append("=" * 80)

        # 定义需要显示的字段及其顺序
        display_fields = [
            ("股票代码", "股票代码"),
            ("招股公告日期", "招股公告日期"),
            ("中签率公告日", "中签率公告日"),
            ("上市日期", "上市日期"),
            ("每股面值", "每股面值", "元"),
            ("发行价格", "发行价格", "元"),
            ("总发行数量", "总发行数量", "万股"),
            ("上网发行中签率", "上网发行中签率", "%"),
            ("发行前每股净资产", "发行前每股净资产", "元"),
            ("发行后每股净资产", "发行后每股净资产", "元"),
            ("摊薄发行市盈率", "摊薄发行市盈率", ""),
            ("募集资金净额", "募集资金净额", "万元"),
            ("发行费用总额", "发行费用总额", "万元"),
            ("主承销商", "主承销商", "")
        ]

        # 格式化输出
        for item in display_fields:
            display_name = item[0]
            col_name = item[1]
            unit = item[2] if len(item) > 2 else ""

            value = ipo_data.get(col_name)

            formatted_value = "N/A"
            if pd.notna(value) and value is not None and str(value).strip() != "":
                if isinstance(value, (float, int)):
                    if col_name in ["每股面值", "发行价格", "发行前每股净资产", "发行后每股净资产", "摊薄发行市盈率",
                                    "上网发行中签率"]:
                        formatted_value = f"{value:.2f}"
                    elif col_name in ["总发行数量", "募集资金净额", "发行费用总额"]:
                        formatted_value = f"{value:,.2f}"
                    else:
                        formatted_value = str(value)
                elif isinstance(value, datetime):
                    formatted_value = value.strftime('%Y-%m-%d')
                else:
                    formatted_value = str(value).strip()

            output_lines.append(f"{display_name}: {formatted_value} {unit}".strip())

        output_lines.append("=" * 80)

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票代码 '{symbol}' 的上市相关信息时发生错误: {e}"

@tool
def get_eastmoney_bj_balance_sheet(date: str, symbol: Optional[str] = None) -> str:
    """
    获取东方财富网数据中心指定日期的北交所 (北京证券交易所) 资产负债表数据。
    该接口返回包括股票代码、股票简称、货币资金、应收账款、存货、总资产（含同比）、
    应付账款、预收账款、总负债（含同比）、资产负债率、股东权益合计和公告日期等详细财务指标。

    Args:
        date (str): 需要查询资产负债表的日期，格式为 'YYYYMMDD'，
                    必须是季度末日期，例如 '20240331', '20240630', '20240930', 或 '20241231'。
                    数据从 20081231 开始。
        symbol (Optional[str]): 可选参数。指定北交所股票代码，例如 '873223'。
                                如果提供，将只返回该股票的资产负债表信息。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的资产负债表摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式和季度末日期
    try:
        dt_obj = datetime.strptime(date, "%Y%m%d")
        if dt_obj.strftime("%m%d") not in ["0331", "0630", "0930", "1231"]:
            return f"错误：日期 '{date}' 不是有效的季度末日期。请提供 'YYYY0331', 'YYYY0630', 'YYYY0930', 或 'YYYY1231' 格式的日期。"
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20240331'。"

    try:
        # 调用 API
        df = ak.stock_zcfz_bj_em(date=date)

        if df.empty:
            return f"在日期 {date} 未能获取到东方财富网北交所的资产负债表数据。请检查日期是否正确或该日期无数据。"

        output_lines = [f"东方财富网 {date} 的北交所资产负债表信息:\n"]
        output_lines.append("=" * 80)

        # 股票代码过滤
        if symbol:
            # 确保股票代码列是字符串类型以便进行精确匹配
            df['股票代码'] = df['股票代码'].astype(str)
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的北交所资产负债表数据。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的资产负债表。")

        output_lines.append(f"总计找到 {len(df)} 条资产负债表记录。")

        # 选择并重新排序关键列，以便在输出中显示
        # 移除 '序号' 列
        display_columns = [
            '股票代码', '股票简称', '资产-货币资金', '资产-应收账款', '资产-存货',
            '资产-总资产', '资产-总资产同比', '负债-应付账款', '负债-预收账款',
            '负债-总负债', '负债-总负债同比', '资产负债率', '股东权益合计', '公告日期'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值
        display_df = display_df.fillna('-')

        # 格式化数值列
        currency_cols = [
            '资产-货币资金', '资产-应收账款', '资产-存货', '资产-总资产',
            '负债-应付账款', '负债-预收账款', '负债-总负债', '股东权益合计'
        ]
        for col in currency_cols:
            if col in display_df.columns:
                # 将列转换为数值类型，无法转换的变为 NaN
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                # 对数值进行格式化，并处理 NaN
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else '-'
                )

        # 格式化百分比列
        percentage_cols = ['资产-总资产同比', '负债-总负债同比', '资产负债率']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 格式化日期列
        if '公告日期' in display_df.columns:
            display_df['公告日期'] = pd.to_datetime(display_df['公告日期'], errors='coerce').dt.strftime(
                '%Y-%m-%d').fillna('-')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：金额单位为元，百分比已按原数据格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的东方财富网北交所资产负债表数据时发生错误: {e}"

@tool
def get_eastmoney_profit_statement(date: str, symbol: Optional[str] = None) -> str:
    """
    获取东方财富网数据中心指定日期的A股利润表数据。
    该接口返回包括股票代码、股票简称、净利润（含同比）、营业总收入（含同比）、
    营业总支出（含销售、管理、财务费用）、营业利润、利润总额和公告日期等详细财务指标。

    Args:
        date (str): 需要查询利润表的日期，格式为 'YYYYMMDD'，
                    必须是季度末日期，例如 '20240331', '20240630', '20240930', 或 '20241231'。
                    数据从 20120331 开始。
        symbol (Optional[str]): 可选参数。指定股票代码，例如 '603156'。
                                如果提供，将只返回该股票的利润表信息。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的利润表摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式和季度末日期
    try:
        dt_obj = datetime.strptime(date, "%Y%m%d")
        if dt_obj.strftime("%m%d") not in ["0331", "0630", "0930", "1231"]:
            return f"错误：日期 '{date}' 不是有效的季度末日期。请提供 'YYYY0331', 'YYYY0630', 'YYYY0930', 或 'YYYY1231' 格式的日期。"
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20240331'。"

    try:
        # 调用 API
        df = ak.stock_lrb_em(date=date)

        if df.empty:
            return f"在日期 {date} 未能获取到东方财富网的利润表数据。请检查日期是否正确或该日期无数据。"

        output_lines = [f"东方财富网 {date} 的A股利润表信息:\n"]
        output_lines.append("=" * 80)

        # 股票代码过滤
        if symbol:
            # 确保股票代码列是字符串类型以便进行精确匹配
            df['股票代码'] = df['股票代码'].astype(str)
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的利润表数据。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的利润表。")

        output_lines.append(f"总计找到 {len(df)} 条利润表记录。")

        # 选择并重新排序关键列，以便在输出中显示
        # 移除 '序号' 列
        display_columns = [
            '股票代码', '股票简称', '净利润', '净利润同比', '营业总收入', '营业总收入同比',
            '营业总支出-营业支出', '营业总支出-销售费用', '营业总支出-管理费用',
            '营业总支出-财务费用', '营业总支出-营业总支出', '营业利润', '利润总额', '公告日期'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值
        display_df = display_df.fillna('-')

        # 格式化数值列
        currency_cols = [
            '净利润', '营业总收入', '营业总支出-营业支出', '营业总支出-销售费用',
            '营业总支出-管理费用', '营业总支出-财务费用', '营业总支出-营业总支出',
            '营业利润', '利润总额'
        ]
        for col in currency_cols:
            if col in display_df.columns:
                # 尝试将列转换为数值类型，无法转换的变为 NaN
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                # 对数值进行格式化，并处理 NaN
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else '-'
                )

        # 格式化百分比列
        percentage_cols = ['净利润同比', '营业总收入同比']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 格式化日期列
        if '公告日期' in display_df.columns:
            display_df['公告日期'] = pd.to_datetime(display_df['公告日期'], errors='coerce').dt.strftime(
                '%Y-%m-%d').fillna('-')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：金额单位为元，百分比已按原数据格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的东方财富网利润表数据时发生错误: {e}"

@tool
def get_eastmoney_cash_flow_statement(date: str, symbol: Optional[str] = None) -> str:
    """
    获取东方财富网数据中心指定日期的A股现金流量表数据。
    该接口返回包括股票代码、股票简称、净现金流（含同比）、经营性现金流（含净现金流占比）、
    投资性现金流（含净现金流占比）、融资性现金流（含净现金流占比）和公告日期等详细财务指标。

    Args:
        date (str): 需要查询现金流量表的日期，格式为 'YYYYMMDD'，
                    必须是季度末日期，例如 '20200331', '20200630', '20200930', 或 '20201231'。
                    数据从 20081231 开始。
        symbol (Optional[str]): 可选参数。指定股票代码，例如 '603156'。
                                如果提供，将只返回该股票的现金流量表信息。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的现金流量表摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式和季度末日期
    try:
        dt_obj = datetime.strptime(date, "%Y%m%d")
        if dt_obj.strftime("%m%d") not in ["0331", "0630", "0930", "1231"]:
            return f"错误：日期 '{date}' 不是有效的季度末日期。请提供 'YYYY0331', 'YYYY0630', 'YYYY0930', 或 'YYYY1231' 格式的日期。"
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20240331'。"

    try:
        # 调用 API
        df = ak.stock_xjll_em(date=date)

        if df.empty:
            return f"在日期 {date} 未能获取到东方财富网的现金流量表数据。请检查日期是否正确或该日期无数据。"

        output_lines = [f"东方财富网 {date} 的A股现金流量表信息:\n"]
        output_lines.append("=" * 80)

        # 股票代码过滤
        if symbol:
            # 确保股票代码列是字符串类型以便进行精确匹配
            df['股票代码'] = df['股票代码'].astype(str)
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的现金流量表数据。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的现金流量表。")

        output_lines.append(f"总计找到 {len(df)} 条现金流量表记录。")

        # 选择并重新排序关键列，以便在输出中显示
        # 移除 '序号' 列
        display_columns = [
            '股票代码', '股票简称', '净现金流-净现金流', '净现金流-同比增长',
            '经营性现金流-现金流量净额', '经营性现金流-净现金流占比',
            '投资性现金流-现金流量净额', '投资性现金流-净现金流占比',
            '融资性现金流-现金流量净额', '融资性现金流-净现金流占比',
            '公告日期'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值
        display_df = display_df.fillna('-')

        # 格式化数值列
        currency_cols = [
            '净现金流-净现金流', '经营性现金流-现金流量净额',
            '投资性现金流-现金流量净额', '融资性现金流-现金流量净额'
        ]
        for col in currency_cols:
            if col in display_df.columns:
                # 将列转换为数值类型，无法转换的变为 NaN
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                # 对数值进行格式化，并处理 NaN
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else '-'
                )

        # 格式化百分比列
        percentage_cols = [
            '净现金流-同比增长', '经营性现金流-净现金流占比',
            '投资性现金流-净现金流占比', '融资性现金流-净现金流占比'
        ]
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 格式化日期列
        if '公告日期' in display_df.columns:
            display_df['公告日期'] = pd.to_datetime(display_df['公告日期'], errors='coerce').dt.strftime(
                '%Y-%m-%d').fillna('-')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：金额单位为元，百分比已按原数据格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的东方财富网现金流量表数据时发生错误: {e}"

@tool
def get_eastmoney_executive_shareholding_changes(
    change_type: str = '全部',
    stock_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    获取东方财富网数据中心的高管及股东持股变动数据。
    该接口可以查询所有类型的变动，或仅查询股东增持或股东减持的数据。
    返回信息包括股票代码、股票名称、最新价、涨跌幅、股东名称、持股变动信息（增减、变动数量、占总股本/流通股比例）、
    变动后持股情况（持股总数、占总股本/流通股比例）、变动开始日、变动截止日和公告日等。

    Args:
        change_type (str): 需要查询的持股变动类型，可选值包括 '全部' (所有增减持记录),
                           '股东增持' (仅增持记录), '股东减持' (仅减持记录)。默认为‘全部'。
        stock_code (Optional[str]): 可选参数。指定股票代码，例如 '600030'。
                                  如果提供，将只返回该股票的增减持信息。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYY-MM-DD'。
                                  将过滤公告日期在此日期或之后的数据。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYY-MM-DD'。
                                  将过滤公告日期在此日期或之前的数据。

    Returns:
        str: 一个格式化后的字符串，包含指定类型的持股变动摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    valid_change_types = ["全部", "股东增持", "股东减持"]
    if change_type not in valid_change_types:
        return f"错误：'change_type' 参数无效。请选择 {', '.join(valid_change_types)} 中的一个。"

    # 解析日期参数
    parsed_start_date = None
    parsed_end_date = None
    if start_date:
        try:
            parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            return "错误：'start_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    if end_date:
        try:
            parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            return "错误：'end_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"

    if parsed_start_date and parsed_end_date and parsed_start_date > parsed_end_date:
        return "错误：'start_date' 不能晚于 'end_date'。"

    try:
        # 调用 API
        df = ak.stock_ggcg_em(symbol=change_type)

        if df.empty:
            return f"未能获取到类型为 '{change_type}' 的高管及股东持股变动数据。请检查数据源或稍后再试。"

        output_lines = [f"东方财富网 '{change_type}' 类型的高管及股东持股变动信息:\n"]
        output_lines.append("=" * 80)

        # 股票代码过滤
        if stock_code:
            # 确保 '代码' 列是字符串类型以便进行精确匹配
            df['代码'] = df['代码'].astype(str)
            df = df[df['代码'] == stock_code.strip()].copy()
            if df.empty:
                return f"在类型为 '{change_type}' 的数据中，未能找到股票代码 '{stock_code}' 的增减持信息。请检查股票代码是否正确或无相关数据。"
            output_lines.append(f"已过滤，只显示股票 '{stock_code}' 的增减持信息。")

        # 日期过滤
        if parsed_start_date or parsed_end_date:
            # 将 '公告日' 列转换为日期时间类型，无效值设为 NaT
            df['公告日_dt'] = pd.to_datetime(df['公告日'], errors='coerce').dt.date

            if parsed_start_date:
                df = df[df['公告日_dt'] >= parsed_start_date].copy()
                output_lines.append(f"已过滤，公告日从 {start_date} 开始。")
            if parsed_end_date:
                df = df[df['公告日_dt'] <= parsed_end_date].copy()
                output_lines.append(f"已过滤，公告日到 {end_date} 结束。")

            # 移除辅助列
            df = df.drop(columns=['公告日_dt'])

            if df.empty:
                date_range_info = ""
                if start_date and end_date:
                    date_range_info = f"在 {start_date} 到 {end_date} 期间"
                elif start_date:
                    date_range_info = f"从 {start_date} 开始"
                elif end_date:
                    date_range_info = f"到 {end_date} 结束"
                return f"在类型为 '{change_type}' 的数据中，{date_range_info} 未能找到相关增减持信息。"

        output_lines.append(f"总计找到 {len(df)} 条记录。")

        # 选择并重新排序关键列
        display_columns = [
            '代码', '名称', '最新价', '涨跌幅', '股东名称', '持股变动信息-增减',
            '持股变动信息-变动数量', '持股变动信息-占总股本比例', '持股变动信息-占流通股比例',
            '变动后持股情况-持股总数', '变动后持股情况-占总股本比例',
            '变动后持股情况-持流通股数', '变动后持股情况-占流通股比例',
            '变动开始日', '变动截止日', '公告日'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值
        display_df = display_df.fillna('-')

        # 格式化数值列
        price_cols = ['最新价']
        for col in price_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else '-'
                )

        percentage_cols = [
            '涨跌幅', '持股变动信息-占总股本比例', '持股变动信息-占流通股比例',
            '变动后持股情况-占总股本比例', '变动后持股情况-占流通股比例'
        ]
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        quantity_cols = [
            '持股变动信息-变动数量', '变动后持股情况-持股总数', '变动后持股情况-持流通股数'
        ]
        for col in quantity_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else '-'
                )

        # 格式化日期列
        date_cols = ['变动开始日', '变动截止日', '公告日']
        for col in date_cols:
            if col in display_df.columns:
                # 确保日期列是字符串，然后转换为日期格式
                display_df[col] = display_df[col].astype(str)
                display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y-%m-%d').fillna('-')

        # 截断长文本
        if '股东名称' in display_df.columns:
            display_df['股东名称'] = display_df['股东名称'].apply(
                lambda x: (x[:15] + '...' if len(str(x)) > 18 else x) if x != '-' else '-'
            )
        if '持股变动信息-增减' in display_df.columns:
            display_df['持股变动信息-增减'] = display_df['持股变动信息-增减'].astype(str)

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：金额和数量已按原数据格式化，百分比已添加 '%' 符号。部分长文本信息可能已截断。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取高管及股东持股变动数据时发生错误: {e}"
    
def _parse_chinese_currency_string(s):
    """将带有'亿'或'万'的中文金额字符串转换为浮点数（以元为单位）"""
    if pd.isna(s) or s == '-' or not isinstance(s, str):
        return np.nan
    s = s.strip()
    if '亿' in s:
        try:
            return float(s.replace('亿', '')) * 100000000
        except ValueError:
            return np.nan
    elif '万' in s:
        try:
            return float(s.replace('万', '')) * 10000
        except ValueError:
            return np.nan
    elif '%' in s:
        try:
            return float(s.replace('%', '')) / 100
        except ValueError:
            return np.nan
    else:
        # 直接转换为浮点数，如果失败则返回 NaN
        try:
            return float(s)
        except ValueError:
            return np.nan

def _format_float_to_chinese_currency(value):
    """将浮点数（以元为单位）格式化为带有'亿'或'万'的中文金额字符串"""
    if pd.isna(value):
        return '-'
    abs_value = abs(value)
    if abs_value >= 100000000:
        return f"{value / 100000000:.2f}亿"
    elif abs_value >= 10000:
        return f"{value / 10000:.2f}万"
    else:
        return f"{value:,.2f}"

@tool
def get_tonghuashun_stock_fund_flow(period_type: str, stock_code: Optional[str] = None) -> str:
    """
    获取同花顺数据中心的个股资金流数据。
    它支持查询即时资金流，以及3日、5日、10日、20日等不同时间周期的资金流排行数据。
    返回信息包括股票代码、股票简称、最新价、涨跌幅、换手率、流入资金、流出资金、净额、成交额等。

    Args:
        period_type (str): 需要查询的资金流周期类型，可选值包括 '即时', '3日排行', '5日排行', '10日排行', '20日排行'。 默认设为'即时'。
        stock_code (Optional[str]): 可选参数。指定股票代码，例如 '300256'。
                                  如果提供，将只返回该股票的资金流信息。

    Returns:
        str: 一个格式化后的字符串，包含指定周期和股票的资金流摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    valid_period_types = ["即时", "3日排行", "5日排行", "10日排行", "20日排行"]
    if period_type not in valid_period_types:
        return f"错误：'period_type' 参数无效。请选择 {', '.join(valid_period_types)} 中的一个。"

    try:
        # 调用 API
        df = ak.stock_fund_flow_individual(symbol=period_type)

        if df.empty:
            return f"未能获取到类型为 '{period_type}' 的个股资金流数据。请检查数据源或稍后再试。"

        output_lines = [f"同花顺 '{period_type}' 个股资金流信息:\n"]
        output_lines.append("=" * 80)

        # 股票代码过滤
        if stock_code:
            original_len = len(df)
            # 确保 '股票代码' 列是字符串类型以便进行精确匹配
            df['股票代码'] = df['股票代码'].astype(str)
            df_filtered = df[df['股票代码'] == stock_code.strip()].copy()
            if df_filtered.empty:
                return f"在类型为 '{period_type}' 的数据中，未能找到股票代码 '{stock_code}' 的资金流信息。请检查股票代码是否正确或无相关数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{stock_code}' 的资金流信息。")

        output_lines.append(f"总计找到 {len(df)} 条记录。")

        # 统一处理 NaN 值
        df = df.fillna('-')

        # 根据 period_type 选择和格式化列
        if period_type == "即时":
            display_columns = [
                '股票代码', '股票简称', '最新价', '涨跌幅', '换手率',
                '流入资金', '流出资金', '净额', '成交额'
            ]

            # 确保所有要显示的列都存在于DataFrame中
            actual_display_columns = [col for col in display_columns if col in df.columns]
            display_df = df[actual_display_columns].copy()

            # 格式化 '最新价'
            if '最新价' in display_df.columns:
                display_df['最新价'] = pd.to_numeric(display_df['最新价'], errors='coerce')
                display_df['最新价'] = display_df['最新价'].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else '-'
                )

            # 资金相关的列先解析再格式化
            currency_cols = ['流入资金', '流出资金', '净额', '成交额']
            for col in currency_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(_parse_chinese_currency_string)
                    display_df[col] = display_df[col].apply(_format_float_to_chinese_currency)

            # 确保字符串类型
            for col in ['涨跌幅', '换手率']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].astype(str).replace('nan', '-')

        # "3日排行", "5日排行", "10日排行", "20日排行"
        else:
            display_columns = [
                '股票代码', '股票简称', '最新价', '阶段涨跌幅', '连续换手率', '资金流入净额'
            ]

            # 确保所有要显示的列都存在于DataFrame中
            actual_display_columns = [col for col in display_columns if col in df.columns]
            display_df = df[actual_display_columns].copy()

            # 格式化 '最新价'
            if '最新价' in display_df.columns:
                display_df['最新价'] = pd.to_numeric(display_df['最新价'], errors='coerce')
                display_df['最新价'] = display_df['最新价'].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else '-'
                )

            # 确保字符串类型
            for col in ['资金流入净额', '阶段涨跌幅', '连续换手率']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].astype(str).replace('nan', '-')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：资金金额已转换为易读的亿/万元格式，百分比已按原数据格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取类型为 '{period_type}' 的个股资金流数据时发生错误: {e}"

@tool
def get_eastmoney_individual_fund_flow(
        stock_code: str,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取东方财富网数据中心指定股票和市场的近100个交易日的个股资金流数据，并可按日期范围过滤。
    它返回包括日期、收盘价、涨跌幅、主力净流入（净额和净占比）、超大单净流入（净额和净占比）、
    大单净流入（净额和净占比）、中单净流入（净额和净占比）、小单净流入（净额和净占比）等详细资金流指标。

    Args:
        stock_code (str): 需要查询的股票代码，例如 '000425'。
        market (str): 股票所属市场，可选值包括 'sh' (上海证券交易所), 'sz' (深圳证券交易所), 'bj' (北京证券交易所)。根据输入的股票所在市场进行变更。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYY-MM-DD'。
                                  将过滤日期在此日期或之后的数据。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYY-MM-DD'。
                                  将过滤日期在此日期或之前的数据。

    Returns:
        str: 一个格式化后的字符串，包含指定股票和市场的资金流数据摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    valid_markets = ["sh", "sz", "bj"]
    if market.lower() not in valid_markets:
        return f"错误：'market' 参数无效。请选择 {', '.join(valid_markets)} 中的一个。"

    if not stock_code or not isinstance(stock_code, str):
        return "错误：'stock_code' 参数无效。请提供有效的股票代码字符串，例如 '000425'。"

    # 解析日期参数
    parsed_start_date = None
    parsed_end_date = None
    if start_date:
        try:
            parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            return "错误：'start_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    if end_date:
        try:
            parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            return "错误：'end_date' 参数格式无效。请使用 'YYYY-M-D' 格式。"

    if parsed_start_date and parsed_end_date and parsed_start_date > parsed_end_date:
        return "错误：'start_date' 不能晚于 'end_date'。"

    try:
        # 调用 API
        df = ak.stock_individual_fund_flow(stock=stock_code, market=market.lower())

        if df.empty:
            return f"未能获取到股票代码 '{stock_code}' ({market.upper()}市场) 的资金流数据。请检查股票代码和市场是否正确或该股票无资金流数据。"

        output_lines = [f"东方财富网股票 '{stock_code}' ({market.upper()}市场) 资金流信息:\n"]
        output_lines.append("=" * 80)

        # 日期过滤
        if parsed_start_date or parsed_end_date:
            # 将 '日期' 列转换为日期时间类型，无效值设为 NaT，然后提取日期部分
            df['日期_dt'] = pd.to_datetime(df['日期'], errors='coerce').dt.date

            if parsed_start_date:
                df = df[df['日期_dt'] >= parsed_start_date].copy()
                output_lines.append(f"已过滤，日期从 {start_date} 开始。")
            if parsed_end_date:
                df = df[df['日期_dt'] <= parsed_end_date].copy()
                output_lines.append(f"已过滤，日期到 {end_date} 结束。")

            # 移除辅助列
            df = df.drop(columns=['日期_dt'])

            if df.empty:
                date_range_info = ""
                if start_date and end_date:
                    date_range_info = f"在 {start_date} 到 {end_date} 期间"
                elif start_date:
                    date_range_info = f"从 {start_date} 开始"
                elif end_date:
                    date_range_info = f"到 {end_date} 结束"
                return f"在指定日期范围 {date_range_info} 内，未能找到股票 '{stock_code}' ({market.upper()}市场) 的资金流数据。"
        else:
            output_lines.append("默认显示近100个交易日的数据。")

        output_lines.append(f"总计找到 {len(df)} 条资金流记录。")

        # 选择并重新排序关键列
        display_columns = [
            '日期', '收盘价', '涨跌幅',
            '主力净流入-净额', '主力净流入-净占比',
            '超大单净流入-净额', '超大单净流入-净占比',
            '大单净流入-净额', '大单净流入-净占比',
            '中单净流入-净额', '中单净流入-净占比',
            '小单净流入-净额', '小单净流入-净占比'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值
        display_df = display_df.fillna('-')

        # 格式化数值列
        # 收盘价
        if '收盘价' in display_df.columns:
            display_df['收盘价'] = pd.to_numeric(display_df['收盘价'], errors='coerce')
            display_df['收盘价'] = display_df['收盘价'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else '-'
            )

        # 涨跌幅和所有净占比列
        percentage_cols = [
            '涨跌幅', '主力净流入-净占比', '超大单净流入-净占比',
            '大单净流入-净占比', '中单净流入-净占比', '小单净流入-净占比'
        ]
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 所有净额列
        currency_cols = [
            '主力净流入-净额', '超大单净流入-净额',
            '大单净流入-净额', '中单净流入-净额', '小单净流入-净额'
        ]
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else '-'
                )

        # 格式化日期列
        if '日期' in display_df.columns:
            # 确保日期列是字符串，然后转换为日期格式
            display_df['日期'] = display_df['日期'].astype(str)
            display_df['日期'] = pd.to_datetime(display_df['日期'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('-')

        # 显示最近的几条
        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示最近 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：金额单位为元，百分比已添加 '%' 符号。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票 '{stock_code}' ({market.upper()}市场) 的资金流数据时发生错误: {e}"

@tool
def get_eastmoney_shareholder_meetings(
        stock_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取东方财富网数据中心的所有A股上市公司的股东大会信息，并可按日期范围过滤。
    它返回包括股票代码、股票简称、股东大会名称、召开开始日、股权登记日、现场登记日、
    网络投票时间（开始和结束）、决议公告日、公告日以及股东大会的提案内容等详细信息。

    Args:
        stock_code (Optional[str]): 可选参数。指定股票代码，例如 '603131'。
                                  如果提供，将只返回该股票的股东大会信息。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYY-MM-DD'。
                                  将过滤公告日期在此日期或之后的数据。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYY-MM-DD'。
                                  将过滤公告日期在此日期或之前的数据。

    Returns:
        str: 一个格式化后的字符串，包含股东大会数据的摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 解析日期参数
    parsed_start_date = None
    parsed_end_date = None
    if start_date:
        try:
            parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            return "错误：'start_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    if end_date:
        try:
            parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            return "错误：'end_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"

    if parsed_start_date and parsed_end_date and parsed_start_date > parsed_end_date:
        return "错误：'start_date' 不能晚于 'end_date'。"

    try:
        # 调用 API
        df = ak.stock_gddh_em()

        if df.empty:
            return "未能获取到东方财富网的股东大会数据。请检查数据源或稍后再试。"

        output_lines = ["东方财富网A股股东大会信息:\n"]
        output_lines.append("=" * 80)

        # 记录原始数据行数
        original_len = len(df)

        # 股票代码过滤
        if stock_code:
            # 确保 '代码' 列是字符串类型以便进行精确匹配
            df['代码'] = df['代码'].astype(str)
            df = df[df['代码'] == stock_code.strip()].copy()
            if df.empty:
                return f"未能找到股票代码 '{stock_code}' 的股东大会信息。请检查股票代码是否正确或无相关数据。"
            output_lines.append(f"已过滤，只显示股票 '{stock_code}' 的股东大会信息。")

        # 日期过滤
        if parsed_start_date or parsed_end_date:
            # 将 '公告日' 列转换为日期时间类型，无效值设为 NaT，然后提取日期部分
            df['公告日_dt'] = pd.to_datetime(df['公告日'], errors='coerce').dt.date

            if parsed_start_date:
                df = df[df['公告日_dt'] >= parsed_start_date].copy()
                output_lines.append(f"已过滤，公告日从 {start_date} 开始。")
            if parsed_end_date:
                df = df[df['公告日_dt'] <= parsed_end_date].copy()
                output_lines.append(f"已过滤，公告日到 {end_date} 结束。")

            # 移除辅助列
            df = df.drop(columns=['公告日_dt'])

            if df.empty:
                date_range_info = ""
                if start_date and end_date:
                    date_range_info = f"在 {start_date} 到 {end_date} 期间"
                elif start_date:
                    date_range_info = f"从 {start_date} 开始"
                elif end_date:
                    date_range_info = f"到 {end_date} 结束"
                return f"在指定日期范围 {date_range_info} 内，未能找到股票 '{stock_code if stock_code else '所有股票'}' 的股东大会信息。"

        output_lines.append(f"总计找到 {len(df)} 条股东大会记录。")

        # 选择并重新排序关键列
        display_columns = [
            '代码', '简称', '股东大会名称', '召开开始日', '股权登记日', '现场登记日',
            '网络投票时间-开始日', '网络投票时间-结束日', '决议公告日', '公告日',
            '序列号', '提案'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值，将其显示为 '-'
        display_df = display_df.fillna('-')

        # 格式化日期列
        date_cols = [
            '召开开始日', '股权登记日', '现场登记日',
            '网络投票时间-开始日', '网络投票时间-结束日',
            '决议公告日', '公告日'
        ]
        for col in date_cols:
            if col in display_df.columns:
                # 确保日期列是字符串，然后转换为日期格式
                display_df[col] = display_df[col].astype(str)
                display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y-%m-%d').fillna('-')

        # 格式化 '提案' 列
        if '提案' in display_df.columns:
            display_df['提案'] = display_df['提案'].apply(
                lambda x: (x[:50] + '...' if len(str(x)) > 50 else x) if x != '-' else '-'
            )

        # 格式化 '股东大会名称'
        if '股东大会名称' in display_df.columns:
            display_df['股东大会名称'] = display_df['股东大会名称'].apply(
                lambda x: (x[:30] + '...' if len(str(x)) > 30 else x) if x != '-' else '-'
            )

        if '公告日' in display_df.columns:
            # 将公告日转换为可排序的格式，然后排序
            display_df['temp_sort_date'] = pd.to_datetime(display_df['公告日'], errors='coerce')
            display_df = display_df.sort_values(by='temp_sort_date', ascending=False).drop(columns='temp_sort_date')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：日期已格式化，部分长文本信息（如提案、股东大会名称）可能已截断。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取东方财富网股东大会数据时发生错误: {e}"

@tool
def get_sina_stock_history_dividend(stock_code: Optional[str] = None) -> str:
    """
    获取新浪财经数据中心所有A股上市公司的历史分红数据。
    它返回包括股票代码、股票名称、上市日期、累计股息（百分比）、年均股息（百分比）、
    分红次数、融资总额（亿元）和融资次数等详细信息。

    Args:
        stock_code (Optional[str]): 可选参数。指定股票代码，例如 '000550'。
                                  如果提供，将只返回该股票的历史分红信息。

    Returns:
        str: 一个格式化后的字符串，包含历史分红数据的摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用 API
        df = ak.stock_history_dividend()

        if df.empty:
            return "未能获取到新浪财经的历史分红数据。请检查数据源或稍后再试。"

        output_lines = ["新浪财经A股历史分红信息:\n"]
        output_lines.append("=" * 80)

        # 股票代码过滤
        if stock_code:
            original_len = len(df)
            # 确保 '代码' 列是字符串类型以便进行精确匹配
            df['代码'] = df['代码'].astype(str)
            df_filtered = df[df['代码'] == stock_code.strip()].copy()
            if df_filtered.empty:
                return f"未能找到股票代码 '{stock_code}' 的历史分红信息。请检查股票代码是否正确或无相关数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{stock_code}' 的历史分红信息。")

        output_lines.append(f"总计找到 {len(df)} 条历史分红记录。")

        # 选择并重新排序关键列，以便在输出中显示
        display_columns = [
            '代码', '名称', '上市日期', '累计股息', '年均股息',
            '分红次数', '融资总额', '融资次数'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值，将其显示为 '-'
        display_df = display_df.fillna('-')

        # 格式化数值列
        # 累计股息, 年均股息
        percentage_cols = ['累计股息', '年均股息']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 融资总额
        if '融资总额' in display_df.columns:
            display_df['融资总额'] = pd.to_numeric(display_df['融资总额'], errors='coerce')
            display_df['融资总额'] = display_df['融资总额'].apply(
                lambda x: f"{x:.4f}亿" if pd.notna(x) else '-'
            )

        # 分红次数, 融资次数
        count_cols = ['分红次数', '融资次数']
        for col in count_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else '-'
                )

        # 格式化日期列
        if '上市日期' in display_df.columns:
            display_df['上市日期'] = pd.to_datetime(display_df['上市日期'], errors='coerce').dt.strftime(
                '%Y-%m-%d').fillna('-')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：累计股息和年均股息已添加 '%' 符号，融资总额已添加 '亿' 单位。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取新浪财经历史分红数据时发生错误: {e}"

@tool
def get_cninfo_stock_dividend_history(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取巨潮资讯网指定股票的历史分红数据，并可按日期范围过滤。
    它返回包括实施方案公告日期、送股比例、转增比例、派息比例、股权登记日、除权日、
    派息日、股份到账日、实施方案分红说明、分红类型和报告时间等详细信息。
    送股比例、转增比例和派息比例的单位均为“每 10 股”。

    Args:
        symbol (str): 需要查询的股票代码，例如 '600009'。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYY-MM-DD'。
                                  将过滤“实施方案公告日期”在此日期或之后的数据。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYY-MM-DD'。
                                  将过滤“实施方案公告日期”在此日期或之前的数据。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的历史分红数据摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol or not isinstance(symbol, str):
        return "错误：'symbol' 参数无效。请提供有效的股票代码字符串，例如 '600009'。"

    # 解析日期参数
    parsed_start_date = None
    parsed_end_date = None
    if start_date:
        try:
            parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            return "错误：'start_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    if end_date:
        try:
            parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            return "错误：'end_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"

    if parsed_start_date and parsed_end_date and parsed_start_date > parsed_end_date:
        return "错误：'start_date' 不能晚于 'end_date'。"

    try:
        # 调用 API
        df = ak.stock_dividend_cninfo(symbol=symbol)

        if df.empty:
            return f"未能获取到股票代码 '{symbol}' 的历史分红数据。请检查股票代码是否正确或该股票无历史分红数据。"

        output_lines = [f"巨潮资讯网股票 '{symbol}' 的历史分红信息:\n"]
        output_lines.append("=" * 80)

        # 日期过滤
        if parsed_start_date or parsed_end_date:
            # 将 '实施方案公告日期' 列转换为日期时间类型，无效值设为 NaT，然后提取日期部分
            df['公告日期_dt'] = pd.to_datetime(df['实施方案公告日期'], errors='coerce').dt.date

            if parsed_start_date:
                df = df[df['公告日期_dt'] >= parsed_start_date].copy()
                output_lines.append(f"已过滤，实施方案公告日期从 {start_date} 开始。")
            if parsed_end_date:
                df = df[df['公告日期_dt'] <= parsed_end_date].copy()
                output_lines.append(f"已过滤，实施方案公告日期到 {end_date} 结束。")

            # 移除辅助列
            df = df.drop(columns=['公告日期_dt'])

            if df.empty:
                date_range_info = ""
                if start_date and end_date:
                    date_range_info = f"在 {start_date} 到 {end_date} 期间"
                elif start_date:
                    date_range_info = f"从 {start_date} 开始"
                elif end_date:
                    date_range_info = f"到 {end_date} 结束"
                return f"在指定日期范围 {date_range_info} 内，未能找到股票 '{symbol}' 的历史分红数据。"

        output_lines.append(f"总计找到 {len(df)} 条历史分红记录。")

        # 定义要显示的列及其顺序
        display_columns = [
            '实施方案公告日期', '分红类型', '送股比例', '转增比例', '派息比例',
            '股权登记日', '除权日', '派息日', '股份到账日',
            '实施方案分红说明', '报告时间'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值，将其显示为 '-'
        display_df = display_df.fillna('-')

        # 格式化数值列
        ratio_cols = ['送股比例', '转增比例', '派息比例']
        for col in ratio_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) and x > 0 else '-'
                )

        # 格式化日期列
        date_cols = [
            '实施方案公告日期', '股权登记日', '除权日', '派息日', '股份到账日'
        ]
        for col in date_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y-%m-%d').fillna('-')

        # 截断长文本列
        if '实施方案分红说明' in display_df.columns:
            display_df['实施方案分红说明'] = display_df['实施方案分红说明'].apply(
                lambda x: (x[:40] + '...' if len(str(x)) > 40 else x) if x != '-' else '-'
            )
        if '报告时间' in display_df.columns:
            display_df['报告时间'] = display_df['报告时间'].apply(
                lambda x: (x[:20] + '...' if len(str(x)) > 20 else x) if x != '-' else '-'
            )

        if '实施方案公告日期' in display_df.columns:
            # 将公告日期转换为可排序的格式，然后排序
            display_df['temp_sort_date'] = pd.to_datetime(display_df['实施方案公告日期'], errors='coerce')
            display_df = display_df.sort_values(by='temp_sort_date', ascending=False).drop(columns='temp_sort_date')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示最近 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：送股比例、转增比例、派息比例单位为“每 10 股”。部分长文本信息可能已截断。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票 '{symbol}' 的历史分红数据时发生错误: {e}"

@tool
def get_eastmoney_limit_up_stocks_pool(query_date: str, symbol: Optional[str] = None) -> str:
    """
    获取东方财富网指定日期的涨停股池数据，并可按股票代码过滤。
    它返回包括股票代码、名称、涨跌幅、最新价、成交额、流通市值、总市值、换手率、
    封板资金、首次封板时间、最后封板时间、炸板次数、涨停统计、连板数和所属行业等详细信息。
    请注意，该接口只能获取近期的数据。

    Args:
        query_date (str): 需要查询的日期，格式为 YYYYMMDD，例如 '20241008'。
        symbol (Optional[str]): 可选参数。指定股票代码，例如 '600030'。
                                  如果提供，将只返回该股票在指定日期的涨停信息。

    Returns:
        str: 一个格式化后的字符串，包含指定日期和股票的涨停股池数据摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not query_date or not isinstance(query_date, str) or len(query_date) != 8 or not query_date.isdigit():
        return "错误：'query_date' 参数无效。请提供有效的日期字符串，格式为 YYYYMMDD，例如 '20241008'。"

    try:
        # 调用 API
        df = ak.stock_zt_pool_em(date=query_date)

        if df.empty:
            return f"未能获取到日期 '{query_date}' 的涨停股池数据。请检查日期是否正确或该日期无涨停股数据。"

        output_lines = [f"东方财富网 '{query_date}' 涨停股池信息:\n"]
        output_lines.append("=" * 80)

        # 股票代码过滤
        if symbol:
            # 确保 '代码' 列是字符串类型以便进行精确匹配
            df['代码'] = df['代码'].astype(str)
            df_filtered = df[df['代码'] == symbol.strip()].copy()
            if df_filtered.empty:
                return f"在日期 '{query_date}' 的涨停股池中，未能找到股票代码 '{symbol}' 的涨停信息。请检查股票代码是否正确或该股票当日未涨停。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的涨停信息。")

        output_lines.append(f"总计找到 {len(df)} 条涨停股记录。")

        # 定义要显示的列及其顺序
        display_columns = [
            '代码', '名称', '涨跌幅', '最新价', '成交额',
            '流通市值', '总市值', '换手率', '封板资金',
            '首次封板时间', '最后封板时间', '炸板次数',
            '涨停统计', '连板数', '所属行业'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值，将其显示为 '-'
        display_df = display_df.fillna('-')

        # 格式化百分比列
        percentage_cols = ['涨跌幅', '换手率']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 格式化金额列
        # 最新价
        if '最新价' in display_df.columns:
            display_df['最新价'] = pd.to_numeric(display_df['最新价'], errors='coerce')
            display_df['最新价'] = display_df['最新价'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else '-'
            )

        # 成交额, 流通市值, 总市值, 封板资金, 转换为亿/万元格式
        currency_cols = ['成交额', '流通市值', '总市值', '封板资金']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x / 100000000:.2f}亿" if pd.notna(x) and x >= 100000000 else (f"{x / 10000:.2f}万" if pd.notna(x) and x >= 10000 else (f"{x:,.2f}" if pd.notna(x) else '-'))
                )

        # 格式化时间列
        time_cols = ['首次封板时间', '最后封板时间']
        for col in time_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{str(x)[:2]}:{str(x)[2:4]}:{str(x)[4:]}" if isinstance(x, str) and len(x) == 6 else x
                )

        # 格式化整数列
        int_cols = ['炸板次数', '连板数']
        for col in int_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else '-'
                )

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：百分比已添加 '%' 符号，金额已转换为易读的亿/万元格式，时间已格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 '{query_date}' 的涨停股池数据时发生错误: {e}"

@tool
def get_eastmoney_limit_down_stocks_pool(query_date: str) -> str:
    """
    获取东方财富网指定日期的跌停股池数据。
    它返回包括股票代码、名称、涨跌幅、最新价、成交额、流通市值、总市值、动态市盈率、
    换手率、封单资金、最后封板时间、板上成交额、连续跌停次数、开板次数和所属行业等详细信息。
    请注意，该接口只能获取近期的数据。

    Args:
        query_date (str): 需要查询的日期，格式为 YYYYMMDD，例如 '20241011'。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的跌停股池数据摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not query_date or not isinstance(query_date, str) or len(query_date) != 8 or not query_date.isdigit():
        return "错误：'query_date' 参数无效。请提供有效的日期字符串，格式为 YYYYMMDD，例如 '20241011'。"

    try:
        # 调用 API
        df = ak.stock_zt_pool_dtgc_em(date=query_date)

        if df.empty:
            return f"未能获取到日期 '{query_date}' 的跌停股池数据。请检查日期是否正确或该日期无跌停股数据。"

        output_lines = [f"东方财富网 '{query_date}' 跌停股池信息:\n"]
        output_lines.append("=" * 80)
        output_lines.append(f"总计找到 {len(df)} 条跌停股记录。")

        # 定义要显示的列及其顺序
        display_columns = [
            '代码', '名称', '涨跌幅', '最新价', '成交额',
            '流通市值', '总市值', '动态市盈率', '换手率',
            '封单资金', '最后封板时间', '板上成交额',
            '连续跌停', '开板次数', '所属行业'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值，将其显示为 '-'
        display_df = display_df.fillna('-')

        # 格式化百分比列
        percentage_cols = ['涨跌幅', '换手率']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 格式化金额列
        # 最新价
        if '最新价' in display_df.columns:
            display_df['最新价'] = pd.to_numeric(display_df['最新价'], errors='coerce')
            display_df['最新价'] = display_df['最新价'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else '-'
            )

        # 成交额, 流通市值, 总市值, 封单资金, 板上成交额, 转换为亿/万元格式
        currency_cols = ['成交额', '流通市值', '总市值', '封单资金', '板上成交额']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x / 100000000:.2f}亿" if pd.notna(x) and x >= 100000000 else \
                        (f"{x / 10000:.2f}万" if pd.notna(x) and x >= 10000 else \
                             (f"{x:,.2f}" if pd.notna(x) else '-'))
                )

        # 格式化动态市盈率
        if '动态市盈率' in display_df.columns:
            display_df['动态市盈率'] = pd.to_numeric(display_df['动态市盈率'], errors='coerce')
            display_df['动态市盈率'] = display_df['动态市盈率'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else '-'
            )

        # 格式化时间列
        if '最后封板时间' in display_df.columns:
            display_df['最后封板时间'] = display_df['最后封板时间'].apply(
                lambda x: f"{str(x)[:2]}:{str(x)[2:4]}:{str(x)[4:]}" if isinstance(x, str) and len(x) == 6 else x
            )

        # 格式化整数列
        int_cols = ['连续跌停', '开板次数']
        for col in int_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else '-'
                )

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：百分比已添加 '%' 符号，金额已转换为易读的亿/万元格式，时间已格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 '{query_date}' 的跌停股池数据时发生错误: {e}"

@tool
def get_eastmoney_strong_stocks_pool(query_date: str) -> str:
    """
    获取东方财富网指定日期的强势股池数据。
    它返回包括股票代码、名称、涨跌幅、最新价、涨停价、成交额、流通市值、总市值、
    换手率、涨速、是否新高、量比、涨停统计、入选理由和所属行业等详细信息。
    请注意，该接口只能获取近期的数据。

    Args:
        query_date (str): 需要查询的日期，格式为 YYYYMMDD，例如 '20241009'。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的强势股池数据摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not query_date or not isinstance(query_date, str) or len(query_date) != 8 or not query_date.isdigit():
        return "错误：'query_date' 参数无效。请提供有效的日期字符串，格式为 YYYYMMDD，例如 '20241009'。"

    try:
        # 调用 API
        df = ak.stock_zt_pool_strong_em(date=query_date)

        if df.empty:
            return f"未能获取到日期 '{query_date}' 的强势股池数据。请检查日期是否正确或该日期无强势股数据。"

        output_lines = [f"东方财富网 '{query_date}' 强势股池信息:\n"]
        output_lines.append("=" * 80)
        output_lines.append(f"总计找到 {len(df)} 条强势股记录。")

        # 定义要显示的列及其顺序
        display_columns = [
            '代码', '名称', '涨跌幅', '最新价', '涨停价', '成交额',
            '流通市值', '总市值', '换手率', '涨速', '是否新高',
            '量比', '涨停统计', '入选理由', '所属行业'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值，将其显示为 '-'
        display_df = display_df.fillna('-')

        # 格式化百分比列
        percentage_cols = ['涨跌幅', '换手率', '涨速']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 格式化价格列
        price_cols = ['最新价', '涨停价']
        for col in price_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else '-'
                )

        # 格式化金额列
        currency_cols = ['成交额', '流通市值', '总市值']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x / 100000000:.2f}亿" if pd.notna(x) and x >= 100000000 else \
                        (f"{x / 10000:.2f}万" if pd.notna(x) and x >= 10000 else \
                             (f"{x:,.2f}" if pd.notna(x) else '-'))
                )

        # 格式化量比
        if '量比' in display_df.columns:
            display_df['量比'] = pd.to_numeric(display_df['量比'], errors='coerce')
            display_df['量比'] = display_df['量比'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else '-'
            )

        # 截断长文本列
        if '入选理由' in display_df.columns:
            display_df['入选理由'] = display_df['入选理由'].apply(
                lambda x: (x[:30] + '...' if len(str(x)) > 30 else x) if x != '-' else '-'
            )

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：百分比已添加 '%' 符号，金额已转换为易读的亿/万元格式，部分长文本信息可能已截断。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 '{query_date}' 的强势股池数据时发生错误: {e}"

@tool
def get_eastmoney_sub_new_stocks_pool(query_date: str) -> str:
    """
    获取东方财富网指定日期的次新股池数据。
    它返回包括股票代码、名称、涨跌幅、最新价、涨停价、成交额、流通市值、总市值、
    转手率、开板几日、开板日期、上市日期、是否新高、涨停统计和所属行业等详细信息。
    请注意，该接口只能获取近期的数据。

    Args:
        query_date (str): 需要查询的日期，格式为 YYYYMMDD，例如 '20241231'。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的次新股池数据摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not query_date or not isinstance(query_date, str) or len(query_date) != 8 or not query_date.isdigit():
        return "错误：'query_date' 参数无效。请提供有效的日期字符串，格式为 YYYYMMDD，例如 '20241231'。"

    try:
        # 调用 API
        df = ak.stock_zt_pool_sub_new_em(date=query_date)

        if df.empty:
            return f"未能获取到日期 '{query_date}' 的次新股池数据。请检查日期是否正确或该日期无次新股数据。"

        output_lines = [f"东方财富网 '{query_date}' 次新股池信息:\n"]
        output_lines.append("=" * 80)
        output_lines.append(f"总计找到 {len(df)} 条次新股记录。")

        # 定义要显示的列及其顺序
        display_columns = [
            '代码', '名称', '涨跌幅', '最新价', '涨停价', '成交额',
            '流通市值', '总市值', '转手率', '开板几日', '开板日期',
            '上市日期', '是否新高', '涨停统计', '所属行业'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值，将其显示为 '-'
        display_df = display_df.fillna('-')

        # 格式化百分比列
        percentage_cols = ['涨跌幅', '转手率']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 格式化价格列
        price_cols = ['最新价', '涨停价']
        for col in display_df.columns:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else '-'
                )

        # 格式化金额列
        currency_cols = ['成交额', '流通市值', '总市值']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x / 100000000:.2f}亿" if pd.notna(x) and x >= 100000000 else \
                        (f"{x / 10000:.2f}万" if pd.notna(x) and x >= 10000 else \
                             (f"{x:,.2f}" if pd.notna(x) else '-'))
                )

        # 格式化日期列
        date_cols = ['开板日期', '上市日期']
        for col in date_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: str(x) if pd.notna(x) else '-'
                )

        # 格式化整数列
        if '开板几日' in display_df.columns:
            display_df['开板几日'] = pd.to_numeric(display_df['开板几日'], errors='coerce')
            display_df['开板几日'] = display_df['开板几日'].apply(
                lambda x: f"{int(x)}" if pd.notna(x) else '-'
            )

        if '是否新高' in display_df.columns:
            display_df['是否新高'] = display_df['是否新高'].apply(
                lambda x: str(x) if pd.notna(x) else '-'
            )

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {MAX_ROWS_TO_DISPLAY} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：百分比已添加 '%' 符号，金额已转换为易读的亿/万元格式。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 '{query_date}' 的次新股池数据时发生错误: {e}"

@tool
def get_eastmoney_broken_limit_up_stocks_pool(query_date: str) -> str:
    """
    获取东方财富网指定日期的炸板股池数据。
    炸板股是指盘中曾触及涨停但最终未能封住涨停的股票。
    它返回包括股票代码、名称、涨跌幅、最新价、涨停价、成交额、流通市值、总市值、
    换手率、涨速、首次封板时间、炸板次数、涨停统计、振幅和所属行业等详细信息。
    请注意，该接口只能获取近期的数据。

    Args:
        query_date (str): 需要查询的日期，格式为 YYYYMMDD，例如 '20241011'。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的炸板股池数据摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not query_date or not isinstance(query_date, str) or len(query_date) != 8 or not query_date.isdigit():
        return "错误：'query_date' 参数无效。请提供有效的日期字符串，格式为 YYYYMMDD，例如 '20241011'。"

    try:
        # 调用 API
        df = ak.stock_zt_pool_zbgc_em(date=query_date)

        if df.empty:
            return f"未能获取到日期 '{query_date}' 的炸板股池数据。请检查日期是否正确或该日期无炸板股数据。"

        output_lines = [f"东方财富网 '{query_date}' 炸板股池信息:\n"]
        output_lines.append("=" * 80)
        output_lines.append(f"总计找到 {len(df)} 条炸板股记录。")

        # 定义要显示的列及其顺序
        display_columns = [
            '代码', '名称', '涨跌幅', '最新价', '涨停价', '成交额',
            '流通市值', '总市值', '换手率', '涨速', '首次封板时间',
            '炸板次数', '涨停统计', '振幅', '所属行业'
        ]

        # 确保所有要显示的列都存在于DataFrame中
        actual_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值，将其显示为 '-'
        display_df = display_df.fillna('-')

        # 格式化百分比列
        percentage_cols = ['涨跌幅', '换手率', '涨速', '振幅']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 格式化价格列
        price_cols = ['最新价', '涨停价']
        for col in price_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else '-'
                )

        # 格式化金额列
        currency_cols = ['成交额', '流通市值', '总市值']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x / 100000000:.2f}亿" if pd.notna(x) and x >= 100000000 else \
                        (f"{x / 10000:.2f}万" if pd.notna(x) and x >= 10000 else \
                             (f"{x:,.2f}" if pd.notna(x) else '-'))
                )

        # 格式化时间列
        if '首次封板时间' in display_df.columns:
            display_df['首次封板时间'] = display_df['首次封板时间'].apply(
                lambda x: f"{str(x)[:2]}:{str(x)[2:4]}:{str(x)[4:]}" if isinstance(x, str) and len(x) == 6 else x
            )

        # 格式化整数列
        if '炸板次数' in display_df.columns:
            display_df['炸板次数'] = pd.to_numeric(display_df['炸板次数'], errors='coerce')
            display_df['炸板次数'] = display_df['炸板次数'].apply(
                lambda x: f"{int(x)}" if pd.notna(x) else '-'
            )

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：百分比已添加 '%' 符号，金额已转换为易读的亿/万元格式，时间已格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 '{query_date}' 的炸板股池数据时发生错误: {e}"

@tool
def get_news_or_sentiment(
        tickers: Optional[str] = None,
        topics: Optional[str] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        sort: Optional[str] = 'LATEST',
        limit: Optional[int] = 10,
) -> str:
    """
    该 API 返回来自世界各地大量且不断增长的顶级新闻机构的实时和历史市场新闻和情绪数据，
    涵盖股票、加密货币、外汇以及财政政策、并购、IPO 等广泛主题。
    该 API 与我们的核心股票 API、基本面数据和技术指标 API 相结合，可以为您提供金融市场和更广泛经济的 360 度视图。

    Args:
        tickers(Optional[str]):用户选择的股票/加密货币/外汇代码。例如：tickers=IBM 将筛选出提及 IBM 股票代码的文章；tickers=COIN,CRYPTO:BTC,FOREX:USD 将筛选出内容中同时提及 Coinbase (COIN)、比特币 (CRYPTO:BTC) 和美元 (FOREX:USD) 的文章。
        topics(Optional[str]):用户选择的新闻主题。例如：topics='technology' 将筛选出关于科技行业的文章；topics='technology,ipo' 将筛选出内容同时涵盖科技和 IPO 的文章。以下是支持的主题完整列表(可选参数)：'blockchain','earnings','ipo','mergers_and_acquisitions','financial_markets','economy_fiscal','economy_monetary','economy_macro','energy_transportation','finance','life_sciences','manufacturing','real_estate','retail_wholesale','technology'
        time_from(Optional[str]):用户要定位的新闻文章的时间范围起点，格式为 YYYYMMDDTHHMM。如果用户没有严格指定时间，此参数不必设置。如果指定了 time_from 但没有指定 time_to，API 将返回在 time_from 值和当前时间之间发布的文章。
        time_to(Optional[str]):用户要定位的新闻文章的时间范围终点，格式为 YYYYMMDDTHHMM。
        sort(Optional[str]):默认情况下，sort='LATEST' 表示 API 将优先返回最新文章。用户也可以根据具体情况设置 sort='EARLIEST' 或 sort='RELEVANCE'。
        limit(Optional[int]):默认情况下，limit=50 表示 API 最多返回 50 条匹配结果。您也可以设置 limit=1000 表示最多输出 1000 条结果。

    Returns:
        str: limit条新闻情绪数据
    """
    # 过滤掉 None 值
    api_params = {
        "tickers": tickers,
        "topics": topics,
        "time_from": time_from,
        "time_to": time_to,
        "sort": sort,
        "limit": limit,
    }

    # 移除字典中值为 None 的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        news_df, meta_data = ns.get_news_sentiment(**filtered_params)

        # 如果没有数据
        if news_df.empty:
            return f"在给定参数 {filtered_params} 下未能获取到任何新闻情绪数据。"

        # 返回前 N 条新闻情绪
        N = filtered_params.get("limit", 50)
        return f"新闻情绪数据 (限制 {N} 条):{news_df.head(N).to_string()}"


    except Exception as e:
        return f"获取新闻情绪时发生错误: {e}"


@tool
def get_exchange_rate(
        from_currency: Optional[str] = None,
        to_currency: Optional[str] = None,
) -> str:
    """
    此 API 返回任意一对货币的实时汇率，可以是数字货币之间，也可以是数字货币与实体货币之间

    Args:
        from_currency(Optional[str]):用户想要获取汇率的货币。可以是实体货币，也可以是数字/加密货币。例如：from_currency=USD 或 from_currency=BTC。
        to_currency(Optional[str]):汇率的目标货币。可以是实体货币，也可以是数字/加密货币。例如：to_currency=USD 或 to_currency=BTC。

    Returns:
        str: 一对数字货币及实体货币的实时汇率
    """
    # 过滤掉 None 值
    api_params = {
        "from_currency": from_currency,
        "to_currency": to_currency,
    }

    # 移除字典中值为 None 的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = cc.get_digital_currency_exchange_rate(**filtered_params)

        # 如果没有数据
        if data.empty:
            return f"在给定参数 {filtered_params} 下未能获取到任何汇率。"

        return f"从{from_currency}到{to_currency}的汇率信息是：\n{data.head(5).to_string()}"


    except Exception as e:
        return f"获取汇率时发生错误: {e}"

@tool
def get_digital_daily(
        symbol: Optional[str] = None,
        market: Optional[str] = None,
) -> str:
    """
    此 API 返回指定市场（例如欧洲市场）中特定数字货币（例如 BTC）的每日历史时间序列，每日午夜（UTC）刷新。价格和交易量以市场特定货币和美元报价。

    Args:
        symbol(Optional[str]):用户选择的数字/加密货币。可以是数字货币列表中的任何货币。例如：symbol=BTC。
        market(Optional[str]):用户选择的交易市场。可以是市场列表中的任意市场。例如：market=EUR。

    Returns:
        str: 特定货币的每日历史时间序列
    """
    # 过滤掉 None 值
    api_params = {
        "symbol": symbol,
        "market": market,
    }

    # 移除字典中值为 None 的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = cc.get_digital_currency_daily(**filtered_params)

        # 如果没有数据
        if data.empty:
            return f"在给定参数 {filtered_params} 下未能获取到任何市场信息。"

        return f"{symbol}在{market}获取到的每日时间序列是：\n{data.head(5).to_string()}"


    except Exception as e:
        return f"获取信息时发生错误: {e}"

@tool
def get_digital_weekly(
        symbol: Optional[str] = None,
        market: Optional[str] = None,
) -> str:
    """
    此 API 返回指定市场（例如欧洲市场）中特定数字货币（例如 BTC）的每周历史时间序列，每日午夜（UTC）刷新。价格和交易量以市场特定货币和美元报价。

    Args:
        symbol(Optional[str]):用户选择的数字/加密货币。可以是数字货币列表中的任何货币。例如：symbol=BTC。
        market(Optional[str]):用户选择的交易市场。可以是市场列表中的任意市场。例如：market=EUR。

    Returns:
        str: 特定货币的每周历史时间序列
    """
    # 过滤掉 None 值
    api_params = {
        "symbol": symbol,
        "market": market,
    }

    # 移除字典中值为 None 的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = cc.get_digital_currency_weekly(**filtered_params)

        # 如果没有数据
        if data.empty:
            return f"在给定参数 {filtered_params} 下未能获取到任何市场信息。"

        return f"{symbol}在{market}获取到的每周时间序列是：\n{data.head(5).to_string()}"


    except Exception as e:
        return f"获取信息时发生错误: {e}"

@tool
def get_digital_monthly(
        symbol: Optional[str] = None,
        market: Optional[str] = None,
) -> str:
    """
    此 API 返回指定市场（例如欧洲市场）中特定数字货币（例如 BTC）的每月历史时间序列，每日午夜（UTC）刷新。价格和交易量以市场特定货币和美元报价。

    Args:
        symbol(Optional[str]):用户选择的数字/加密货币。可以是数字货币列表中的任何货币。例如：symbol=BTC。
        market(Optional[str]):用户选择的交易市场。可以是市场列表中的任意市场。例如：market=EUR。

    Returns:
        str: 特定货币的每周历史时间序列
    """
    # 过滤掉 None 值
    api_params = {
        "symbol": symbol,
        "market": market,
    }

    # 移除字典中值为 None 的键
    filtered_params = {k: v for k, v in api_params.items() if v is not None}

    try:
        # 调用 API
        data, meta_data = cc.get_digital_currency_monthly(**filtered_params)

        # 如果没有数据
        if data.empty:
            return f"在给定参数 {filtered_params} 下未能获取到任何市场信息。"

        return f"{symbol}在{market}获取到的每月时间序列是：\n{data.head(5).to_string()}"


    except Exception as e:
        return f"获取信息时发生错误: {e}"

@tool
def get_chinamoney_bond_info(
        bond_name: str = "",
        bond_code: str = "",
        bond_issue: str = "",
        bond_type: str = "",
        coupon_type: str = "",
        issue_year: str = "",
        underwriter: str = "",
        grade: str = ""
) -> str:
    """
    查询中国外汇交易中心暨全国银行间同业拆借中心（ChinaMoney）的债券信息。
    用户可以通过债券简称、债券代码、发行期数、债券类型、付息方式、发行年份、承销商和债项评级等多个维度进行筛选。
    此工具返回所有匹配条件的债券的详细信息。

    Args:
        bond_name (str): 债券简称，支持模糊查询。例如 '国债'。默认为空字符串，表示不以此条件筛选。
        bond_code (str): 债券代码，精确查询。例如 '041900474'。默认为空字符串，表示不以此条件筛选。
        bond_issue (str): 债券发行期数。具体可用值可通过调用 `ak.bond_info_cm_query()` 接口获取。默认为空字符串。
        bond_type (str): 债券类型。例如 '短期融资券'。具体可用值可通过调用 `ak.bond_info_cm_query()` 接口获取。默认为空字符串。
        coupon_type (str): 付息方式。例如 '零息式'。具体可用值可通过调用 `ak.bond_info_cm_query()` 接口获取。默认为空字符串。
        issue_year (str): 发行年份。例如 '2019'。默认为空字符串。
        underwriter (str): 承销商名称。具体可用值可通过调用 `ak.bond_info_cm_query()` 接口获取。默认为空字符串。
        grade (str): 债项评级。例如 'A-1'。默认为空字符串。

    Returns:
        str: 一个格式化后的字符串，包含所有匹配条件的债券的详细信息。
             如果未找到匹配债券或查询失败，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用API
        df = ak.bond_info_cm(
            bond_name=bond_name,
            bond_code=bond_code,
            bond_issue=bond_issue,
            bond_type=bond_type,
            coupon_type=coupon_type,
            issue_year=issue_year,
            underwriter=underwriter,
            grade=grade
        )

        if df.empty:
            return "未找到符合您查询条件的债券信息。请检查查询参数是否正确或尝试放宽查询条件。"

        output_lines = ["\n以下是符合您查询条件的债券信息 (数据来源: 中国外汇交易中心):\n"]
        output_lines.append("=" * 60)

        # 遍历 DataFrame，将每条债券的信息格式化输出
        for i, row in df.iterrows():
            output_lines.append(f"--- 债券 {i + 1} ---")
            output_lines.append(f"债券简称: {row.get('债券简称', 'N/A')}")
            output_lines.append(f"债券代码: {row.get('债券代码', 'N/A')}")
            output_lines.append(f"发行人/受托机构: {row.get('发行人/受托机构', 'N/A')}")
            output_lines.append(f"债券类型: {row.get('债券类型', 'N/A')}")
            output_lines.append(f"发行日期: {row.get('发行日期', 'N/A')}")
            output_lines.append(f"最新债项评级: {row.get('最新债项评级', 'N/A')}")
            output_lines.append(f"查询代码: {row.get('查询代码', 'N/A')}")
            output_lines.append("-" * 20)

        output_lines.append("=" * 60)
        output_lines.append("提示：如需查询可用参数值（如债券类型、承销商等），请使用 `ak.bond_info_cm_query()` 接口。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"查询债券信息时发生错误: {e}\n请检查您的查询参数或稍后再试。"

@tool
def get_sse_bond_deal_summary(date: str, bond_type: Optional[str] = None) -> str:
    """
    获取上海证券交易所（上登债券信息网）指定交易日的债券成交概览数据。
    该接口提供按债券类型分类的当日成交笔数、当日成交金额（单位：万元）、
    当年累计成交笔数和当年累计成交金额（单位：万元）等关键统计信息。

    Args:
        date (str): 需要查询的交易日期，格式为 'YYYYMMDD'，例如 '20210104'。
        bond_type (Optional[str], optional): 可选的债券类型，用于筛选结果。
                                             例如 '国债'、'地方政府债'、'公司债券' 等。
                                             如果提供此参数，则只返回该类型债券的信息。
                                             默认为 None，即返回所有类型。

    Returns:
        str: 一个格式化后的字符串，包含指定交易日不同债券类型的成交概览数据。
             如果查询失败、日期格式不正确或该日期无数据，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式
    if not (isinstance(date, str) and len(date) == 8 and date.isdigit()):
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20210104'。"

    try:
        # 将日期字符串转换为日期对象
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return f"错误：日期 '{date}' 不是一个有效的日期。请提供正确的 'YYYYMMDD' 格式的日期。"

    try:
        # 调用API
        df = ak.bond_deal_summary_sse(date=date)

        if df.empty:
            return f"未能获取日期 {date} 的债券成交概览信息。可能该日期无数据或非交易日，请检查日期是否正确。"

        # 根据 bond_type 参数筛选 DataFrame
        if bond_type:
            original_df = df.copy()
            df = df[df['债券类型'] == bond_type]

            # 检查筛选后 DataFrame 是否为空
            if df.empty:
                available_types = original_df['债券类型'].unique().tolist()
                return (f"在日期 {date} 的数据中未找到指定的债券类型 '{bond_type}'。\n"
                        f"当日可用的债券类型包括: {', '.join(available_types)}")

        # 优化输出标题
        title = f"上海证券交易所 {date} 债券成交概览"
        if bond_type:
            title += f" (筛选类型: {bond_type})"

        output_lines = [f"{title} (数据来源: 上登债券信息网):\n"]
        output_lines.append("=" * 70)

        # 格式化输出表头
        output_lines.append(
            f"{'债券类型':<15} | {'当日成交笔数':>10} | {'当日成交金额(万元)':>15} | {'当年成交笔数':>10} | {'当年成交金额(万元)':>15}")
        output_lines.append("-" * 70)

        # 遍历 DataFrame，将每行数据格式化输出
        for index, row in df.iterrows():
            b_type = row.get('债券类型', 'N/A')
            daily_trades = row.get('当日成交笔数', 0)
            daily_amount = row.get('当日成交金额', 0.0)
            yearly_trades = row.get('当年成交笔数', 0)
            yearly_amount = row.get('当年成交金额', 0.0)

            # 格式化金额为带千位分隔符和两位小数
            daily_amount_str = f"{daily_amount:,.2f}"
            yearly_amount_str = f"{yearly_amount:,.2f}"

            output_lines.append(
                f"{b_type:<15} | {daily_trades:>10,} | {daily_amount_str:>15} | {yearly_trades:>10,} | {yearly_amount_str:>15}"
            )

        output_lines.append("=" * 70)
        output_lines.append("注意：成交金额单位为 '万元'。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的债券成交概览信息时发生错误: {e}\n请检查您的日期参数或稍后再试。"

@tool
def get_chinamoney_bond_spot_quotes(bond_name: Optional[str] = None) -> str:
    """
    获取中国外汇交易中心暨全国银行间同业拆借中心（ChinaMoney）的现券市场做市报价数据。
    可以返回所有报价，也可以根据提供的债券简称进行筛选。

    Args:
        bond_name (Optional[str], optional): 可选的债券简称，用于筛选结果。
                                             支持部分匹配，例如输入 '国债' 会返回所有简称中包含 '国债' 的债券报价。
                                             默认为 None，即返回所有当前可用的做市报价。

    Returns:
        str: 一个格式化后的字符串，包含符合条件的现券市场做市报价数据。
             如果查询失败、无数据或未找到指定债券，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用API
        df = ak.bond_spot_quote()

        if df.empty:
            return "当前未获取到现券市场做市报价数据。请稍后再试或检查数据源。"

        # 根据 bond_name 参数筛选 DataFrame
        if bond_name:
            original_df_count = len(df)
            df = df[df['债券简称'].str.contains(bond_name, na=False)]

            # 检查筛选后 DataFrame 是否为空
            if df.empty:
                return (f"在当前 {original_df_count} 条做市报价中，未找到简称包含 '{bond_name}' 的债券。\n"
                        f"请检查您的输入或尝试其他关键字。")

        # 优化输出标题
        title = "现券市场做市报价数据"
        if bond_name:
            title += f" (筛选简称: {bond_name})"
        output_lines = [f"\n以下是{title} (数据来源: 中国外汇交易中心):\n"]

        output_lines.append("=" * 80)

        # 定义列名和宽度，用于格式化输出
        columns = {
            "报价机构": 15,
            "债券简称": 15,
            "买入净价": 10,
            "卖出净价": 10,
            "买入收益率": 10,
            "卖出收益率": 10
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns.items()])
        output_lines.append(header)
        output_lines.append("-" * 80)

        # 遍历 DataFrame，格式化每行数据
        for index, row in df.iterrows():
            # 确保所有列都存在，并处理可能的NaN值
            报价机构 = str(row.get('报价机构', 'N/A')).ljust(columns["报价机构"])
            债券简称 = str(row.get('债券简称', 'N/A')).ljust(columns["债券简称"])

            # 格式化净价和收益率
            买入净价_val = row.get('买入净价')
            卖出净价_val = row.get('卖出净价')
            买入收益率_val = row.get('买入收益率')
            卖出收益率_val = row.get('卖出收益率')

            买入净价 = f"{买入净价_val:.2f}".rjust(columns["买入净价"]) if 买入净价_val is not None else "N/A".rjust(
                columns["买入净价"])
            卖出净价 = f"{卖出净价_val:.2f}".rjust(columns["卖出净价"]) if 卖出净价_val is not None else "N/A".rjust(
                columns["卖出净价"])
            买入收益率 = f"{买入收益率_val:.3f}".rjust(
                columns["买入收益率"]) if 买入收益率_val is not None else "N/A".rjust(columns["买入收益率"])
            卖出收益率 = f"{卖出收益率_val:.3f}".rjust(
                columns["卖出收益率"]) if 卖出收益率_val is not None else "N/A".rjust(columns["卖出收益率"])

            output_lines.append(
                f"{报价机构} | {债券简称} | {买入净价} | {卖出净价} | {买入收益率} | {卖出收益率}"
            )

        output_lines.append("=" * 80)
        output_lines.append("注意：净价单位为 '元'，收益率单位为 '%'。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取现券市场做市报价数据时发生错误: {e}\n请稍后再试。"

@tool
def get_chinabond_yield_curve(start_date: str, end_date: str) -> str:
    """
    获取中国债券信息网提供的国债及其他债券收益率曲线数据。
    该接口返回指定日期范围内，不同类型债券在不同期限下的收益率数据。
    请注意，查询的日期范围（从开始日期到结束日期）必须小于一年（即不超过365天）。

    Args:
        start_date (str): 查询的开始日期，格式为 'YYYYMMDD'，例如 '20210201'。
        end_date (str): 查询的结束日期，格式为 'YYYYMMDD'，例如 '20220201'。

    Returns:
        str: 一个格式化后的字符串，包含指定日期范围内不同债券类型的收益率曲线数据。
             如果查询失败、日期格式不正确或日期范围超出限制，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式
    if not (isinstance(start_date, str) and len(start_date) == 8 and start_date.isdigit()):
        return f"错误：开始日期 '{start_date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20210201'。"
    if not (isinstance(end_date, str) and len(end_date) == 8 and end_date.isdigit()):
        return f"错误：结束日期 '{end_date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20220201'。"

    try:
        start_date_obj = datetime.strptime(start_date, "%Y%m%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y%m%d").date()
    except ValueError:
        return f"错误：提供的日期 '{start_date}' 或 '{end_date}' 无效。请确保日期是真实存在的。"

    if start_date_obj > end_date_obj:
        return "错误：开始日期不能晚于结束日期。"

    # 验证日期范围小于一年
    if (end_date_obj - start_date_obj).days > 365:
        return f"错误：查询日期范围过大。从 {start_date} 到 {end_date} 超过一年（365天）。请将日期范围缩小至一年以内。"

    try:
        # 调用API
        df = ak.bond_china_yield(start_date=start_date, end_date=end_date)

        if df.empty:
            return f"未能获取从 {start_date} 到 {end_date} 的债券收益率曲线数据。可能该日期范围内无数据或非交易日，请检查日期是否正确。"

        output_lines = [f"\n中国债券信息网 {start_date} 至 {end_date} 收益率曲线数据:\n"]
        output_lines.append("=" * 100)

        # 定义列名和宽度，用于格式化输出
        columns = {
            "曲线名称": 25,
            "日期": 10,
            "3月": 8,
            "6月": 8,
            "1年": 8,
            "3年": 8,
            "5年": 8,
            "7年": 8,
            "10年": 8,
            "30年": 8
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns.items()])
        output_lines.append(header)
        output_lines.append("-" * 100)

        # 遍历 DataFrame，格式化每行数据
        for index, row in df.iterrows():
            曲线名称 = str(row.get('曲线名称', 'N/A')).ljust(columns["曲线名称"])
            日期 = str(row.get('日期', 'N/A')).ljust(columns["日期"])

            # 格式化收益率数据，处理 NaN
            yield_values = []
            for col_name in ["3月", "6月", "1年", "3年", "5年", "7年", "10年", "30年"]:
                value = row.get(col_name)
                if pd.notna(value):
                    yield_values.append(f"{value:.4f}".rjust(columns[col_name]))
                else:
                    yield_values.append("N/A".rjust(columns[col_name]))

            output_lines.append(
                f"{曲线名称} | {日期} | " + " | ".join(yield_values)
            )

        output_lines.append("=" * 100)
        output_lines.append("注意：收益率单位为 '%'。部分期限可能无数据，显示为 'N/A'。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取债券收益率曲线数据时发生错误: {e}\n请检查您的日期参数或稍后再试。"

@tool
def get_hs_bond_realtime_quotes(
        start_page: str = "1",
        end_page: str = "10",
        bond_code: Optional[str] = None,
        bond_name: Optional[str] = None
) -> str:
    """
    获取新浪财经提供的沪深债券实时行情数据。
    该接口返回所有沪深债券的最新价、涨跌额、涨跌幅、买入价、卖出价、昨收价、今开价、最高价、最低价、
    成交量和成交额等实时交易信息。数据通过分页获取，用户可以指定开始和结束的页面范围，每页包含80条数据。
    支持通过债券代码或债券名称进行筛选。

    Args:
        start_page (str): 开始获取数据的页面编号，默认为 '1'。
        end_page (str): 结束获取数据的页面编号，默认为 '10'。
        bond_code (Optional[str], optional): 可选的债券代码，用于精确筛选结果。例如 '110001'。
                                             如果提供此参数，则只返回该代码债券的信息。默认为 None。
        bond_name (Optional[str], optional): 可选的债券名称，用于模糊筛选结果。例如 '国债'。
                                             支持部分匹配，如果提供此参数，则只返回名称中包含该关键字的债券信息。默认为 None。

    Returns:
        str: 一个格式化后的字符串，包含指定页面范围内符合筛选条件的沪深债券实时行情数据。
             如果查询失败、页面范围不合法、无数据或未找到指定债券，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证页面参数是否为有效数字
    try:
        start_page_int = int(start_page)
        end_page_int = int(end_page)
    except ValueError:
        return "错误：start_page 和 end_page 参数必须是有效的数字字符串。"

    if not (1 <= start_page_int <= end_page_int):
        return f"错误：页面范围不合法。start_page ({start_page_int}) 必须小于或等于 end_page ({end_page_int})，且都必须大于等于 1。"

    try:
        # 调用API
        df = ak.bond_zh_hs_spot(start_page=str(start_page_int), end_page=str(end_page_int))

        if df.empty:
            return f"未能获取从第 {start_page_int} 页到第 {end_page_int} 页的沪深债券实时行情数据。请检查页面范围或稍后再试。"

        # 根据 bond_code 和 bond_name 参数筛选 DataFrame
        original_df_count = len(df)

        if bond_code:
            # 精确匹配债券代码
            df = df[df['代码'] == bond_code]

        if bond_name:
            df = df[df['名称'].str.contains(bond_name, na=False)]

        # 检查筛选后 DataFrame 是否为空
        if df.empty:
            filter_info = []
            if bond_code:
                filter_info.append(f"代码 '{bond_code}'")
            if bond_name:
                filter_info.append(f"名称包含 '{bond_name}'")

            # 构造筛选条件的描述字符串
            filter_str = ""
            if len(filter_info) == 1:
                filter_str = filter_info[0]
            elif len(filter_info) > 1:
                filter_str = "和".join(filter_info)

            return (f"在从第 {start_page_int} 页到第 {end_page_int} 页的 {original_df_count} 条数据中，"
                    f"未找到符合 {filter_str} 条件的沪深债券。\n"
                    f"请检查您的输入或尝试其他关键字/代码。")

        # 优化输出标题
        title_parts = [f"沪深债券实时行情数据 (从第 {start_page_int} 页到第 {end_page_int} 页)"]
        if bond_code:
            title_parts.append(f"代码: {bond_code}")
        if bond_name:
            title_parts.append(f"名称包含: {bond_name}")

        title_suffix = ""
        # 如果有除了基础标题之外的筛选条件
        if len(title_parts) > 1:
            title_suffix = " (筛选条件: " + ", ".join(title_parts[1:]) + ")"

        output_lines = [
            f"\n{title_parts[0]}{title_suffix} (数据来源: 新浪财经):\n", "=" * 140]

        # 定义列名和宽度，用于格式化输出
        columns = {
            "代码": 10,
            "名称": 15,
            "最新价": 10,
            "涨跌额": 10,
            "涨跌幅": 10,
            "买入": 10,
            "卖出": 10,
            "昨收": 10,
            "今开": 10,
            "最高": 10,
            "最低": 10,
            "成交量(手)": 12,
            "成交额(万)": 12
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns.items()])
        output_lines.append(header)
        output_lines.append("-" * 140)

        # 格式化浮点数
        def format_float_value(val, width, precision=None, suffix=""):
            if pd.notna(val):
                if precision is not None:
                    return f"{val:.{precision}f}{suffix}".rjust(width)
                else:
                    return f"{val}{suffix}".rjust(width)
            return "N/A".rjust(width)

        # 格式化整数
        def format_int_value(val, width):
            if pd.notna(val):
                return f"{int(val):,}".rjust(width)
            return "N/A".rjust(width)

        # 遍历 DataFrame，格式化每行数据
        for index, row in df.iterrows():
            # 获取数据并处理可能的 NaN 值
            代码 = str(row.get('代码', 'N/A')).ljust(columns["代码"])
            名称 = str(row.get('名称', 'N/A')).ljust(columns["名称"])

            最新价 = format_float_value(row.get('最新价'), columns["最新价"], 3)
            涨跌额 = format_float_value(row.get('涨跌额'), columns["涨跌额"], 2)
            涨跌幅 = format_float_value(row.get('涨跌幅'), columns["涨跌幅"], 2, "%")
            买入 = format_float_value(row.get('买入'), columns["买入"], 3)
            卖出 = format_float_value(row.get('卖出'), columns["卖出"], 3)
            昨收 = format_float_value(row.get('昨收'), columns["昨收"], 3)
            今开 = format_float_value(row.get('今开'), columns["今开"], 3)
            最高 = format_float_value(row.get('最高'), columns["最高"], 3)
            最低 = format_float_value(row.get('最低'), columns["最低"], 3)
            成交量 = format_int_value(row.get('成交量'), columns["成交量(手)"])
            成交额 = format_int_value(row.get('成交额'), columns["成交额(万)"])

            output_lines.append(
                f"{代码} | {名称} | {最新价} | {涨跌额} | {涨跌幅} | {买入} | {卖出} | {昨收} | {今开} | {最高} | {最低} | {成交量} | {成交额}"
            )

        output_lines.append("=" * 140)
        output_lines.append("注意：成交量单位为 '手'，成交额单位为 '万'。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取沪深债券实时行情数据时发生错误: {e}\n请检查您的页面参数、筛选参数或稍后再试。"

@tool
def get_hs_bond_daily_historical_data(symbol: str) -> str:
    """
    获取新浪财经提供的沪深债券历史行情数据，数据按日频率更新。
    该接口返回指定债券的所有历史每日行情数据，包括日期、开盘价、最高价、最低价、收盘价和成交量。

    Args:
        symbol (str): 需要查询历史行情的沪深债券代码，例如 'sh010107'。

    Returns:
        str: 一个格式化后的字符串，包含指定沪深债券的所有历史每日行情数据。
             如果查询失败、债券代码不正确或无数据，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：请提供要查询的沪深债券代码 (symbol)。例如：'sh010107'。"

    try:
        # 调用API
        df = ak.bond_zh_hs_daily(symbol=symbol)

        if df.empty:
            return f"未能获取债券代码 '{symbol}' 的历史行情数据。请检查债券代码是否正确或数据源暂无此信息。"

        output_lines = [f"\n沪深债券 '{symbol}' 的历史行情数据 (数据来源: 新浪财经):\n"]
        output_lines.append("=" * 80)

        # 定义列名和宽度
        columns = {
            "date": 12,
            "open": 10,
            "high": 10,
            "low": 10,
            "close": 10,
            "volume": 15
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns.items()])
        output_lines.append(header)
        output_lines.append("-" * 80)

        # 遍历 DataFrame，格式化每行数据
        # 只显示前5行和后5行
        display_df = pd.concat([df.head(5), df.tail(5)])
        if len(df) > 10:
            output_lines.append("... (中间省略) ...")

        for index, row in display_df.iterrows():
            # 确保所有列都存在，并处理可能的NaN值
            date = str(row.get('date', 'N/A')).ljust(columns["date"])

            # 格式化数值，并处理 NaN
            open_price = f"{row.get('open', 0.0):.3f}".rjust(columns["open"]) if pd.notna(
                row.get('open')) else "N/A".rjust(columns["open"])
            high_price = f"{row.get('high', 0.0):.3f}".rjust(columns["high"]) if pd.notna(
                row.get('high')) else "N/A".rjust(columns["high"])
            low_price = f"{row.get('low', 0.0):.3f}".rjust(columns["low"]) if pd.notna(row.get('low')) else "N/A".rjust(
                columns["low"])
            close_price = f"{row.get('close', 0.0):.3f}".rjust(columns["close"]) if pd.notna(
                row.get('close')) else "N/A".rjust(columns["close"])
            volume = f"{int(row.get('volume', 0)):,}".rjust(columns["volume"]) if pd.notna(
                row.get('volume')) else "N/A".rjust(columns["volume"])

            output_lines.append(
                f"{date} | {open_price} | {high_price} | {low_price} | {close_price} | {volume}"
            )

        output_lines.append("=" * 80)
        output_lines.append(f"总计 {len(df)} 条历史数据。成交量单位为 '手'。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取债券代码 '{symbol}' 的历史行情数据时发生错误: {e}\n请检查您的债券代码或稍后再试。"

@tool
def get_sina_convertible_bond_profile(symbol: str) -> str:
    """
    获取新浪财经提供的指定可转债的详细资料数据。
    该接口返回特定可转债的各项基本信息和发行条款，例如债券名称、简称、代码、类型、面值、年限、
    票面利率、到期日、发行规模、上市日期、信用等级等。

    Args:
        symbol (str): 需要查询详情资料的可转债代码，必须带市场标识，例如 'sz128039'。

    Returns:
        str: 一个格式化后的字符串，包含指定可转债的各项详细资料。
             如果查询失败、债券代码不正确或无数据，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：请提供要查询的可转债代码 (symbol)。例如：'sz128039'。"

    # 格式检查
    if not (symbol.startswith("sh") or symbol.startswith("sz")) or len(symbol) < 8:
        return f"错误：债券代码 '{symbol}' 格式不正确。请提供带市场标识的转债代码，例如 'sz128039'。"

    try:
        # 调用API
        df = ak.bond_cb_profile_sina(symbol=symbol)

        if df.empty:
            return f"未能获取可转债代码 '{symbol}' 的详情资料。请检查债券代码是否正确或数据源暂无此信息。"

        output_lines = [f"\n可转债 '{symbol}' 的详情资料 (数据来源: 新浪财经):\n"]
        output_lines.append("=" * 60)

        # 找到最长的 item 字符串长度
        max_item_len = df['item'].apply(len).max()
        if max_item_len < 10:
            max_item_len = 10

        # 遍历 DataFrame，格式化每行数据
        for index, row in df.iterrows():
            item = str(row.get('item', 'N/A'))
            value = str(row.get('value', 'N/A'))
            output_lines.append(f"{item.ljust(max_item_len)} : {value}")

        output_lines.append("=" * 60)

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取可转债代码 '{symbol}' 的详情资料时发生错误: {e}\n请检查您的债券代码或稍后再试。"

@tool
def get_sina_convertible_bond_summary(symbol: str) -> str:
    """
    获取新浪财经提供的指定可转债的债券概况数据。
    该接口返回特定可转债的关键概览信息，例如债券类型、计息方式、付息方式、票面利率、每年付息日、
    发行价格、发行规模、债券面值、债券年限、到期日期、剩余年限等。

    Args:
        symbol (str): 需要查询债券概况的可转债代码，必须带市场标识，例如 'sh155255'。

    Returns:
        str: 一个格式化后的字符串，包含指定可转债的各项概况信息。
             如果查询失败、债券代码不正确或无数据，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：请提供要查询的可转债代码 (symbol)。例如：'sh155255'。"

    # 格式检查
    if not (symbol.startswith("sh") or symbol.startswith("sz")) or len(symbol) < 8:
        return f"错误：债券代码 '{symbol}' 格式不正确。请提供带市场标识的转债代码，例如 'sh155255'。"

    try:
        # 调用API
        df = ak.bond_cb_summary_sina(symbol=symbol)

        if df.empty:
            return f"未能获取可转债代码 '{symbol}' 的债券概况数据。请检查债券代码是否正确或数据源暂无此信息。"

        output_lines = [f"\n可转债 '{symbol}' 的债券概况 (数据来源: 新浪财经):\n"]
        output_lines.append("=" * 60)

        # 找到最长的 item 字符串长度
        max_item_len = df['item'].apply(len).max()
        if max_item_len < 10:
            max_item_len = 10

        # 遍历 DataFrame，格式化每行数据
        for index, row in df.iterrows():
            item = str(row.get('item', 'N/A'))
            value = str(row.get('value', 'N/A'))
            output_lines.append(f"{item.ljust(max_item_len)} : {value}")

        output_lines.append("=" * 60)

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取可转债代码 '{symbol}' 的债券概况数据时发生错误: {e}\n请检查您的债券代码或稍后再试。"

@tool
def get_hs_convertible_bond_realtime_quotes(
        bond_code: Optional[str] = None,
        bond_name: Optional[str] = None
) -> str:
    """
    获取新浪财经提供的所有沪深可转债的实时行情数据。
    该接口无需任何输入参数，将返回所有当前可用的沪深可转债的实时行情数据，
    包括债券代码、名称、最新价、涨跌额、涨跌幅、成交量和成交额等。
    支持通过债券代码或债券名称进行筛选。

    Args:
        bond_code (Optional[str], optional): 可选的债券代码，用于精确筛选结果。例如 '113016'。
                                             如果提供此参数，则只返回该代码债券的信息。默认为 None。
        bond_name (Optional[str], optional): 可选的债券名称，用于模糊筛选结果。例如 '中信'。
                                             支持部分匹配，如果提供此参数，则只返回名称中包含该关键字的债券信息。默认为 None。

    Returns:
        str: 一个格式化后的字符串，包含符合条件的沪深可转债实时行情数据。
             如果查询失败、无数据或未找到指定债券，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用API
        df = ak.bond_zh_hs_cov_spot()

        if df.empty:
            return "当前未获取到沪深可转债实时行情数据。请稍后再试或检查数据源。"

        # 根据 bond_code 和 bond_name 参数筛选 DataFrame
        original_df_count = len(df)  # 记录原始数据条数，用于提示信息

        if bond_code:
            # 精确匹配债券代码
            df = df[df['symbol'] == bond_code]

        if bond_name:
            # 模糊匹配债券名称
            df = df[df['name'].str.contains(bond_name, na=False)]

        # 检查筛选后 DataFrame 是否为空
        if df.empty:
            filter_info = []
            if bond_code:
                filter_info.append(f"代码 '{bond_code}'")
            if bond_name:
                filter_info.append(f"名称包含 '{bond_name}'")

            # 构造筛选条件的描述字符串
            filter_str = ""
            if len(filter_info) == 1:
                filter_str = filter_info[0]
            elif len(filter_info) > 1:
                filter_str = "和".join(filter_info)

            return (f"在总计 {original_df_count} 条可转债数据中，未找到符合 {filter_str} 条件的沪深可转债。\n"
                    f"请检查您的输入或尝试其他关键字/代码。")

        # 优化输出标题
        title_parts = ["沪深可转债实时行情数据"]
        if bond_code:
            title_parts.append(f"代码: {bond_code}")
        if bond_name:
            title_parts.append(f"名称包含: {bond_name}")

        title_suffix = ""
        # 如果有除了基础标题之外的筛选条件
        if len(title_parts) > 1:
            title_suffix = " (筛选条件: " + ", ".join(title_parts[1:]) + ")"

        output_lines = [f"\n以下是{title_parts[0]}{title_suffix} (数据来源: 新浪财经):\n"]

        output_lines.append("=" * 120)

        # 定义要显示的列及其宽度
        columns_to_display = {
            "symbol": 10,
            "name": 15,
            "trade": 10,
            "pricechange": 10,
            "pchange": 10,
            "volume": 12,
            "amount": 12,
            "ticktime": 10
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns_to_display.items()])
        output_lines.append(header)
        output_lines.append("-" * 120)

        # 格式化浮点数
        def format_float_value(val, width, precision=None, suffix=""):
            if pd.notna(val):
                if precision is not None:
                    return f"{val:.{precision}f}{suffix}".rjust(width)
                else:
                    return f"{val}{suffix}".rjust(width)
            return "N/A".rjust(width)

        # 格式化整数
        def format_int_value(val, width):
            if pd.notna(val):
                return f"{int(val):,}".rjust(width)
            return "N/A".rjust(width)

        display_df = df
        # 只有在没有筛选且数据量大的情况下才截断
        if not (bond_code or bond_name) and len(df) > 20:
            display_df = pd.concat([df.head(10), df.tail(10)])
            output_lines.append("... (中间省略) ...")

        for index, row in display_df.iterrows():
            # 获取数据并处理可能的 NaN 值
            symbol_val = str(row.get('symbol', 'N/A')).ljust(columns_to_display["symbol"])
            name_val = str(row.get('name', 'N/A')).ljust(columns_to_display["name"])

            # 数值型数据格式化，并处理 NaN
            trade_val = format_float_value(row.get('trade'), columns_to_display["trade"], 3)
            pricechange_val = format_float_value(row.get('pricechange'), columns_to_display["pricechange"], 2)
            pchange_val = format_float_value(row.get('pchange'), columns_to_display["pchange"], 2, "%")
            volume_val = format_int_value(row.get('volume'), columns_to_display["volume"])
            amount_val = format_int_value(row.get('amount'), columns_to_display["amount"])
            ticktime_val = str(row.get('ticktime', 'N/A')).ljust(columns_to_display["ticktime"])

            output_lines.append(
                f"{symbol_val} | {name_val} | {trade_val} | {pricechange_val} | {pchange_val} | {volume_val} | {amount_val} | {ticktime_val}"
            )

        output_lines.append("=" * 120)
        output_lines.append(f"总计 {len(df)} 条数据。成交量单位为 '手'，成交额单位为 '元'。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取沪深可转债实时行情数据时发生错误: {e}\n请检查您的筛选参数或稍后再试。"

@tool
def get_eastmoney_convertible_bond_list(
        bond_code: Optional[str] = None,
        bond_short_name: Optional[str] = None,
) -> str:
    """
    获取东方财富网数据中心提供的所有可转债数据一览表。
    该接口返回当前交易时刻所有可转债的详细信息，包括债券代码、简称、申购日期、申购代码、
    申购上限、正股信息、转股价值、债现价、转股溢价率、发行规模、中签率、上市时间、信用评级等。

    Args:
        bond_code (Optional[str]): 可选参数，用于精确查询指定债券代码的发行数据。例如 '110083'。
        bond_short_name (Optional[str]): 可选参数，用于模糊查询指定债券简称的发行数据。例如 '苏租转债'。
                                    如果同时提供 bond_code、bond_short_name，则优先级为：
                                    bond_code (精确匹配) > bond_short_name (模糊匹配)

    Returns:
        str: 一个格式化后的字符串，包含所有当前可转债的详细数据。
             如果查询失败、无数据，或者未找到指定债券，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    try:
        # 调用API
        df = ak.bond_zh_cov()

        if df.empty:
            return "当前未能获取到可转债数据一览表。请稍后再试或检查数据源。"

        # 根据 bond_code, bond_short_name 或 bond_name 过滤数据
        filtered_df = df.copy()
        if bond_code:
            filtered_df = filtered_df[filtered_df['债券代码'].astype(str) == bond_code]
            if filtered_df.empty:
                return f"未找到代码为 '{bond_code}' 的可转债数据。"
        elif bond_short_name:
            # 使用 contains 进行模糊匹配
            filtered_df = filtered_df[
                filtered_df['债券简称'].astype(str).str.contains(bond_short_name, na=False, case=False)]
            if filtered_df.empty:
                return f"未找到简称包含 '{bond_short_name}' 的可转债数据。"

        # 如果过滤后 DataFrame 为空
        if filtered_df.empty:
            return "未找到符合条件的可转债数据。"

        output_lines = []
        if bond_code or bond_short_name:
            # 如果是查询特定债券，则只显示该债券的信息
            output_lines.append(f"\n东方财富网可转债数据一览表 - 查询结果:\n")
        else:
            # 否则显示全部或部分数据
            output_lines.append(f"\n东方财富网可转债数据一览表:\n")

        output_lines.append("=" * 180)

        # 定义要显示的列及其宽度
        columns_to_display = {
            "债券代码": 10,
            "债券简称": 15,
            "申购日期": 12,
            "申购上限(万元)": 15,
            "正股简称": 15,
            "正股价": 10,
            "转股价": 10,
            "转股价值": 10,
            "债现价": 10,
            "转股溢价率(%)": 15,
            "发行规模(亿元)": 15,
            "中签率(%)": 12,
            "上市时间": 12,
            "信用评级": 10
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns_to_display.items()])
        output_lines.append(header)
        output_lines.append("-" * 180)

        # 遍历 DataFrame，格式化每行数据
        # 如果是查询特定债券，显示所有匹配结果；否则，只显示前10行和后10行
        if bond_code or bond_short_name:
            # 显示所有过滤后的结果
            display_df = filtered_df
        else:
            display_df = pd.concat([filtered_df.head(10), filtered_df.tail(10)])
            if len(filtered_df) > 20:
                output_lines.append("... (中间省略) ...")

        for index, row in display_df.iterrows():
            # 获取数据并处理可能的 NaN 值
            债券代码_val = str(row.get('债券代码', 'N/A')).ljust(columns_to_display["债券代码"])
            债券简称_val = str(row.get('债券简称', 'N/A')).ljust(columns_to_display["债券简称"])
            申购日期_val = str(row.get('申购日期', 'N/A')).ljust(columns_to_display["申购日期"])

            # 申购上限，处理 NaN 并格式化
            申购上限_val = f"{row.get('申购上限', 0.0):.2f}".rjust(columns_to_display["申购上限(万元)"]) if pd.notna(
                row.get('申购上限')) else "N/A".rjust(columns_to_display["申购上限(万元)"])

            正股简称_val = str(row.get('正股简称', 'N/A')).ljust(columns_to_display["正股简称"])

            # 价格类数据，处理 NaN 并格式化
            正股价_val = f"{row.get('正股价', 0.0):.2f}".rjust(columns_to_display["正股价"]) if pd.notna(
                row.get('正股价')) else "N/A".rjust(columns_to_display["正股价"])
            转股价_val = f"{row.get('转股价', 0.0):.2f}".rjust(columns_to_display["转股价"]) if pd.notna(
                row.get('转股价')) else "N/A".rjust(columns_to_display["转股价"])
            转股价值_val = f"{row.get('转股价值', 0.0):.2f}".rjust(columns_to_display["转股价值"]) if pd.notna(
                row.get('转股价值')) else "N/A".rjust(columns_to_display["转股价值"])
            债现价_val = f"{row.get('债现价', 0.0):.2f}".rjust(columns_to_display["债现价"]) if pd.notna(
                row.get('债现价')) else "N/A".rjust(columns_to_display["债现价"])

            # 溢价率，处理 NaN 并格式化
            转股溢价率_val = f"{row.get('转股溢价率', 0.0):.2f}".rjust(columns_to_display["转股溢价率(%)"]) if pd.notna(
                row.get('转股溢价率')) else "N/A".rjust(columns_to_display["转股溢价率(%)"])

            # 发行规模，处理 NaN 并格式化
            发行规模_val = f"{row.get('发行规模', 0.0):.2f}".rjust(columns_to_display["发行规模(亿元)"]) if pd.notna(
                row.get('发行规模')) else "N/A".rjust(columns_to_display["发行规模(亿元)"])

            # 中签率，处理 NaN 并格式化
            中签率_val = f"{row.get('中签率', 0.0):.4f}".rjust(columns_to_display["中签率(%)"]) if pd.notna(
                row.get('中签率')) else "N/A".rjust(columns_to_display["中签率(%)"])

            上市时间_val = str(row.get('上市时间', 'N/A')).ljust(columns_to_display["上市时间"])
            信用评级_val = str(row.get('信用评级', 'N/A')).ljust(columns_to_display["信用评级"])

            output_lines.append(
                f"{债券代码_val} | {债券简称_val} | {申购日期_val} | {申购上限_val} | {正股简称_val} | {正股价_val} | {转股价_val} | {转股价值_val} | {债现价_val} | {转股溢价率_val} | {发行规模_val} | {中签率_val} | {上市时间_val} | {信用评级_val}"
            )

        output_lines.append("=" * 180)
        output_lines.append(
            f"总计 {len(filtered_df)} 条数据。注意单位：申购上限(万元), 发行规模(亿元), 转股溢价率(%), 中签率(%)。\n完整数据包含 {len(df.columns)} 列，此处仅显示部分核心列。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取可转债数据一览表时发生错误: {e}\n请稍后再试。"

@tool
def get_eastmoney_convertible_bond_details(symbol: str, indicator: str) -> str:
    """
    获取东方财富网数据中心提供的指定可转债的详细资料。
    该接口根据 `indicator` 参数返回不同类型的数据：
    - '基本信息': 包含可转债的重要条款和多达67个字段的详细数据。
    - '中签号': 中签号码信息。
    - '筹资用途': 募集资金用途说明。
    - '重要日期': 可转债的关键日期（如发行日、上市日等）。

    Args:
        symbol (str): 需要查询详情资料的可转债代码，例如 '123121'。
        indicator (str): 指定查询的信息类型。可选值：'基本信息', '中签号', '筹资用途', '重要日期'。

    Returns:
        str: 一个格式化后的字符串，包含指定可转债的请求类型的数据。
             如果查询失败、债券代码不正确、indicator 无效或无数据，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：请提供要查询的可转债代码 (symbol)。例如：'123121'。"

    valid_indicators = {"基本信息", "中签号", "筹资用途", "重要日期"}
    if indicator not in valid_indicators:
        return f"错误：无效的 indicator 参数 '{indicator}'。可选值包括：{', '.join(valid_indicators)}。"

    try:
        # 调用API
        df = ak.bond_zh_cov_info(symbol=symbol, indicator=indicator)

        if df.empty:
            return f"未能获取可转债代码 '{symbol}' 的 '{indicator}' 数据。请检查债券代码或数据源暂无此信息。"

        output_lines = [f"\n可转债 '{symbol}' 的 '{indicator}' 数据 (数据来源: 东方财富网):\n"]
        output_lines.append("=" * 80)

        if indicator == "基本信息":
            # 对于基本信息，将其转换为键值对形式更易读
            if not df.empty:
                first_row = df.iloc[0]
                # 找到最长的 item 字符串长度
                max_item_len = max(len(str(col)) for col in df.columns)
                if max_item_len < 15:
                    max_item_len = 15

                for col_name, value in first_row.items():
                    output_lines.append(f"{str(col_name).ljust(max_item_len)} : {str(value)}")
            else:
                output_lines.append("未找到基本信息。")
        else:
            # 对于其他 indicator,直接转换为字符串
            # 限制行数
            if len(df) > 20:
                output_lines.append(df.head(10).to_string(index=False))
                output_lines.append("\n... (中间省略) ...\n")
                output_lines.append(df.tail(10).to_string(index=False))
            else:
                output_lines.append(df.to_string(index=False))

            output_lines.append(f"\n总计 {len(df)} 条数据。")

        output_lines.append("=" * 80)

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取可转债代码 '{symbol}' 的 '{indicator}' 数据时发生错误: {e}\n请检查您的债券代码或稍后再试。"

@tool
def get_eastmoney_convertible_bond_value_analysis(symbol: str) -> str:
    """
    获取东方财富网行情中心提供的指定可转债的历史价值分析数据。
    该接口返回特定可转债在不同日期的收盘价、纯债价值、转股价值、纯债溢价率和转股溢价率等关键价值指标。

    Args:
        symbol (str): 需要查询价值分析数据的可转债代码，例如 '113527'。

    Returns:
        str: 一个格式化后的字符串，包含指定可转债的所有历史价值分析数据。
             如果查询失败、债券代码不正确或无数据，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：请提供要查询的可转债代码 (symbol)。例如：'113527'。"

    try:
        # 调用API
        df = ak.bond_zh_cov_value_analysis(symbol=symbol)

        if df.empty:
            return f"未能获取可转债代码 '{symbol}' 的价值分析数据。请检查债券代码是否正确或数据源暂无此信息。"

        output_lines = [f"\n可转债 '{symbol}' 的历史价值分析数据 (数据来源: 东方财富网):\n"]
        output_lines.append("=" * 90)

        # 定义要显示的列及其宽度
        columns = {
            "日期": 12,
            "收盘价(元)": 12,
            "纯债价值(元)": 12,
            "转股价值(元)": 12,
            "纯债溢价率(%)": 14,
            "转股溢价率(%)": 14
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns.items()])
        output_lines.append(header)
        output_lines.append("-" * 90)

        # 遍历 DataFrame，格式化每行数据
        # 为了避免输出过长，只显示前5行和后5行
        display_df = pd.concat([df.head(5), df.tail(5)])
        if len(df) > 10:
            output_lines.append("... (中间省略) ...")

        for index, row in display_df.iterrows():
            # 获取数据并处理可能的 NaN 值
            日期 = str(row.get('日期', 'N/A')).ljust(columns["日期"])

            # 数值型数据格式化，并处理 NaN
            收盘价 = f"{row.get('收盘价', 0.0):.3f}".rjust(columns["收盘价(元)"]) if pd.notna(
                row.get('收盘价')) else "N/A".rjust(columns["收盘价(元)"])
            纯债价值 = f"{row.get('纯债价值', 0.0):.3f}".rjust(columns["纯债价值(元)"]) if pd.notna(
                row.get('纯债价值')) else "N/A".rjust(columns["纯债价值(元)"])
            转股价值 = f"{row.get('转股价值', 0.0):.3f}".rjust(columns["转股价值(元)"]) if pd.notna(
                row.get('转股价值')) else "N/A".rjust(columns["转股价值(元)"])
            纯债溢价率 = f"{row.get('纯债溢价率', 0.0):.2f}".rjust(columns["纯债溢价率(%)"]) if pd.notna(
                row.get('纯债溢价率')) else "N/A".rjust(columns["纯债溢价率(%)"])
            转股溢价率 = f"{row.get('转股溢价率', 0.0):.2f}".rjust(columns["转股溢价率(%)"]) if pd.notna(
                row.get('转股溢价率')) else "N/A".rjust(columns["转股溢价率(%)"])

            output_lines.append(
                f"{日期} | {收盘价} | {纯债价值} | {转股价值} | {纯债溢价率} | {转股溢价率}"
            )

        output_lines.append("=" * 90)
        output_lines.append(f"总计 {len(df)} 条历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取可转债代码 '{symbol}' 的价值分析数据时发生错误: {e}\n请检查您的债券代码或稍后再试。"

@tool
def get_china_us_bond_rates(start_date: str) -> str:
    """
    获取东方财富网提供的中国和美国国债收益率历史数据。
    该接口返回从指定 `start_date` 开始的所有交易日的中国和美国各期限（2年、5年、10年、30年）国债收益率，
    以及10年-2年期限利差和各自国家的GDP年增率数据。
    数据最早可追溯到1990年12月19日。

    Args:
        start_date (str): 查询数据的起始日期，格式为 'YYYYMMDD'，例如 '19901219'。

    Returns:
        str: 一个格式化后的字符串，包含从指定日期开始的中美两国国债收益率历史数据。
             如果查询失败、日期格式不正确或无数据，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not start_date:
        return "错误：请提供查询数据的起始日期 (start_date)，格式为 'YYYYMMDD'，例如 '19901219'。"

    # 简单的日期格式检查
    if not (len(start_date) == 8 and start_date.isdigit()):
        return f"错误：日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '19901219'。您输入的是: '{start_date}'"

    try:
        # 调用API
        df = ak.bond_zh_us_rate(start_date=start_date)

        if df.empty:
            return f"未能获取从 '{start_date}' 开始的中美国债收益率数据。请检查日期或数据源暂无此信息。"

        output_lines = [f"\n中美国债收益率历史数据 (从 {start_date} 开始，数据来源: 东方财富网):\n"]
        output_lines.append("=" * 150)

        # 定义要显示的列及其宽度
        columns_to_display = {
            "日期": 12,
            "中国国债收益率2年": 15,
            "中国国债收益率5年": 15,
            "中国国债收益率10年": 16,
            "中国国债收益率30年": 16,
            "中国国债收益率10年-2年": 18,
            "中国GDP年增率": 13,
            "美国国债收益率2年": 15,
            "美国国债收益率5年": 15,
            "美国国债收益率10年": 16,
            "美国国债收益率30年": 16,
            "美国国债收益率10年-2年": 18,
            "美国GDP年增率": 13
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns_to_display.items()])
        output_lines.append(header)
        output_lines.append("-" * 150)

        # 遍历 DataFrame，格式化每行数据
        # 为了避免输出过长，只显示前5行和后5行
        display_df = pd.concat([df.head(5), df.tail(5)])
        if len(df) > 10:
            output_lines.append("... (中间省略) ...")

        for index, row in display_df.iterrows():
            # 获取数据并处理可能的 NaN 值
            日期 = str(row.get('日期', 'N/A')).ljust(columns_to_display["日期"])

            # 数值型数据格式化，并处理 NaN
            中国国债收益率2年 = f"{row.get('中国国债收益率2年', 0.0):.2f}".rjust(
                columns_to_display["中国国债收益率2年"]) if pd.notna(row.get('中国国债收益率2年')) else "N/A".rjust(
                columns_to_display["中国国债收益率2年"])
            中国国债收益率5年 = f"{row.get('中国国债收益率5年', 0.0):.2f}".rjust(
                columns_to_display["中国国债收益率5年"]) if pd.notna(row.get('中国国债收益率5年')) else "N/A".rjust(
                columns_to_display["中国国债收益率5年"])
            中国国债收益率10年 = f"{row.get('中国国债收益率10年', 0.0):.2f}".rjust(
                columns_to_display["中国国债收益率10年"]) if pd.notna(row.get('中国国债收益率10年')) else "N/A".rjust(
                columns_to_display["中国国债收益率10年"])
            中国国债收益率30年 = f"{row.get('中国国债收益率30年', 0.0):.2f}".rjust(
                columns_to_display["中国国债收益率30年"]) if pd.notna(row.get('中国国债收益率30年')) else "N/A".rjust(
                columns_to_display["中国国债收益率30年"])
            中国国债收益率10年_2年 = f"{row.get('中国国债收益率10年-2年', 0.0):.2f}".rjust(
                columns_to_display["中国国债收益率10年-2年"]) if pd.notna(
                row.get('中国国债收益率10年-2年')) else "N/A".rjust(columns_to_display["中国国债收益率10年-2年"])
            中国GDP年增率 = f"{row.get('中国GDP年增率', 0.0):.2f}".rjust(
                columns_to_display["中国GDP年增率"]) if pd.notna(row.get('中国GDP年增率')) else "N/A".rjust(
                columns_to_display["中国GDP年增率"])

            美国国债收益率2年 = f"{row.get('美国国债收益率2年', 0.0):.2f}".rjust(
                columns_to_display["美国国债收益率2年"]) if pd.notna(row.get('美国国债收益率2年')) else "N/A".rjust(
                columns_to_display["美国国债收益率2年"])
            美国国债收益率5年 = f"{row.get('美国国债收益率5年', 0.0):.2f}".rjust(
                columns_to_display["美国国债收益率5年"]) if pd.notna(row.get('美国国债收益率5年')) else "N/A".rjust(
                columns_to_display["美国国债收益率5年"])
            美国国债收益率10年 = f"{row.get('美国国债收益率10年', 0.0):.2f}".rjust(
                columns_to_display["美国国债收益率10年"]) if pd.notna(row.get('美国国债收益率10年')) else "N/A".rjust(
                columns_to_display["美国国债收益率10年"])
            美国国债收益率30年 = f"{row.get('美国国债收益率30年', 0.0):.2f}".rjust(
                columns_to_display["美国国债收益率30年"]) if pd.notna(row.get('美国国债收益率30年')) else "N/A".rjust(
                columns_to_display["美国国债收益率30年"])
            美国国债收益率10年_2年 = f"{row.get('美国国债收益率10年-2年', 0.0):.2f}".rjust(
                columns_to_display["美国国债收益率10年-2年"]) if pd.notna(
                row.get('美国国债收益率10年-2年')) else "N/A".rjust(columns_to_display["美国国债收益率10年-2年"])
            美国GDP年增率 = f"{row.get('美国GDP年增率', 0.0):.2f}".rjust(
                columns_to_display["美国GDP年增率"]) if pd.notna(row.get('美国GDP年增率')) else "N/A".rjust(
                columns_to_display["美国GDP年增率"])

            output_lines.append(
                f"{日期} | {中国国债收益率2年} | {中国国债收益率5年} | {中国国债收益率10年} | {中国国债收益率30年} | {中国国债收益率10年_2年} | {中国GDP年增率} | "
                f"{美国国债收益率2年} | {美国国债收益率5年} | {美国国债收益率10年} | {美国国债收益率30年} | {美国国债收益率10年_2年} | {美国GDP年增率}"
            )

        output_lines.append("=" * 150)
        output_lines.append(f"总计 {len(df)} 条历史数据。收益率和GDP年增率单位为 %。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取中美国债收益率数据时发生错误: {e}\n请检查您的日期或稍后再试。"

@tool
def get_cninfo_treasury_bond_issues(
        start_date: str,
        end_date: str,
        bond_code: Optional[str] = None,
        bond_short_name: Optional[str] = None,
        bond_name: Optional[str] = None
) -> str:
    """
    获取巨潮资讯数据中心提供的指定日期范围内的国债发行历史数据。
    返回的数据包括债券代码、简称、发行起止日、计划和实际发行总量（亿元）、发行价格（元）、
    单位面值（元）、缴款日、增发次数、交易市场、发行方式、发行对象、公告日期以及债券名称等。

    Args:
        start_date (str): 查询数据的起始日期，格式为 'YYYYMMDD'，例如 '20210910'。
        end_date (str): 查询数据的结束日期，格式为 'YYYYMMDD'，例如 '20211109'。
        bond_code (Optional[str]): 可选参数，用于精确查询指定债券代码的发行数据。例如 '210001'。
        bond_short_name (Optional[str]): 可选参数，用于模糊查询指定债券简称的发行数据。例如 '21附息国债01'。
        bond_name (Optional[str]): 可选参数，用于模糊查询指定债券名称的发行数据。例如 '2021年记账式附息(一期)国债'。
                                    如果同时提供 bond_code、bond_short_name 或 bond_name，则优先级为：
                                    bond_code (精确匹配) > bond_short_name (模糊匹配) > bond_name (模糊匹配)。

    Returns:
        str: 一个格式化后的字符串，包含指定日期范围内所有国债发行数据。
             如果查询失败、日期格式不正确、无数据，或者未找到指定债券，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not start_date or not end_date:
        return "错误：请提供查询数据的起始日期 (start_date) 和结束日期 (end_date)，格式为 'YYYYMMDD'。"

    # 日期格式检查
    if not (len(start_date) == 8 and start_date.isdigit()):
        return f"错误：起始日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20210910'。您输入的是: '{start_date}'"
    if not (len(end_date) == 8 and end_date.isdigit()):
        return f"错误：结束日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20211109'。您输入的是: '{end_date}'"

    try:
        # 调用API
        df = ak.bond_treasure_issue_cninfo(start_date=start_date, end_date=end_date)

        if df.empty:
            return f"未能获取从 '{start_date}' 到 '{end_date}' 期间的国债发行数据。请检查日期范围或数据源暂无此信息。"

        # 根据 bond_code, bond_short_name 或 bond_name 过滤数据
        filtered_df = df.copy()
        if bond_code:
            filtered_df = filtered_df[filtered_df['债券代码'].astype(str) == bond_code]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到代码为 '{bond_code}' 的国债发行数据。"
        elif bond_short_name:
            # 使用 contains 进行模糊匹配
            filtered_df = filtered_df[
                filtered_df['债券简称'].astype(str).str.contains(bond_short_name, na=False, case=False)]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到简称包含 '{bond_short_name}' 的国债发行数据。"
        elif bond_name:
            # 使用 contains 进行模糊匹配
            filtered_df = filtered_df[filtered_df['债券名称'].astype(str).str.contains(bond_name, na=False, case=False)]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到名称包含 '{bond_name}' 的国债发行数据。"

        # 如果过滤后 DataFrame 为空
        if filtered_df.empty:
            return f"在 '{start_date}' 到 '{end_date}' 期间，未找到符合条件的国债发行数据。"

        output_lines = []
        if bond_code or bond_short_name or bond_name:
            # 如果是查询特定债券，则只显示该债券的信息
            output_lines.append(f"\n巨潮资讯国债发行数据 (从 {start_date} 到 {end_date}) - 查询结果:\n")
        else:
            # 否则显示全部或部分数据
            output_lines.append(f"\n巨潮资讯国债发行数据 (从 {start_date} 到 {end_date}):\n")

        output_lines.append("=" * 160)

        # 定义要显示的列及其宽度
        columns_to_display = {
            "债券代码": 10,
            "债券简称": 15,
            "发行起始日": 12,
            "发行终止日": 12,
            "计划发行总量(亿元)": 18,
            "实际发行总量(亿元)": 18,
            "发行价格(元)": 12,
            "单位面值(元)": 12,
            "缴款日": 12,
            "增发次数": 8,
            "交易市场": 10,
            "发行方式": 10,
            "发行对象": 10,
            "公告日期": 12,
            "债券名称": 20
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns_to_display.items()])
        output_lines.append(header)
        output_lines.append("-" * 160)

        # 遍历 DataFrame，格式化每行数据
        # 如果是查询特定债券，显示所有匹配结果；否则，只显示前5行和后5行
        if bond_code or bond_short_name or bond_name:
            # 显示所有过滤后的结果
            display_df = filtered_df
        else:
            display_df = pd.concat([filtered_df.head(5), filtered_df.tail(5)])
            if len(filtered_df) > 10:
                output_lines.append("... (中间省略) ...")

        for index, row in display_df.iterrows():
            # 获取数据并处理可能的 NaN 值
            债券代码_val = str(row.get('债券代码', 'N/A')).ljust(columns_to_display["债券代码"])
            债券简称_val = str(row.get('债券简称', 'N/A')).ljust(columns_to_display["债券简称"])
            发行起始日_val = str(row.get('发行起始日', 'N/A')).ljust(columns_to_display["发行起始日"])
            发行终止日_val = str(row.get('发行终止日', 'N/A')).ljust(columns_to_display["发行终止日"])

            # 数值型数据格式化，并处理 NaN
            计划发行总量_val = f"{row.get('计划发行总量', 0.0):.2f}".rjust(
                columns_to_display["计划发行总量(亿元)"]) if pd.notna(row.get('计划发行总量')) else "N/A".rjust(
                columns_to_display["计划发行总量(亿元)"])
            实际发行总量_val = f"{row.get('实际发行总量', 0.0):.2f}".rjust(
                columns_to_display["实际发行总量(亿元)"]) if pd.notna(row.get('实际发行总量')) else "N/A".rjust(
                columns_to_display["实际发行总量(亿元)"])
            发行价格_val = f"{row.get('发行价格', 0.0):.2f}".rjust(columns_to_display["发行价格(元)"]) if pd.notna(
                row.get('发行价格')) else "N/A".rjust(columns_to_display["发行价格(元)"])
            单位面值_val = str(row.get('单位面值', 'N/A')).rjust(columns_to_display["单位面值(元)"])  # int64

            缴款日_val = str(row.get('缴款日', 'N/A')).ljust(columns_to_display["缴款日"])
            增发次数_val = str(row.get('增发次数', 'N/A')).rjust(columns_to_display["增发次数"])  # int64
            交易市场_val = str(row.get('交易市场', 'N/A')).ljust(columns_to_display["交易市场"])
            发行方式_val = str(row.get('发行方式', 'N/A')).ljust(columns_to_display["发行方式"])
            发行对象_val = str(row.get('发行对象', 'N/A')).ljust(columns_to_display["发行对象"])
            公告日期_val = str(row.get('公告日期', 'N/A')).ljust(columns_to_display["公告日期"])

            债券名称_full = str(row.get('债券名称', 'N/A'))
            债券名称_display = (债券名称_full[:columns_to_display["债券名称"] - 3] + "...") if len(债券名称_full) > \
                                                                                               columns_to_display[
                                                                                                   "债券名称"] else 债券名称_full.ljust(
                columns_to_display["债券名称"])

            output_lines.append(
                f"{债券代码_val} | {债券简称_val} | {发行起始日_val} | {发行终止日_val} | {计划发行总量_val} | {实际发行总量_val} | {发行价格_val} | {单位面值_val} | {缴款日_val} | {增发次数_val} | {交易市场_val} | {发行方式_val} | {发行对象_val} | {公告日期_val} | {债券名称_display}"
            )

        output_lines.append("=" * 160)
        output_lines.append(
            f"总计 {len(filtered_df)} 条数据。注意单位：计划发行总量(亿元), 实际发行总量(亿元), 发行价格(元), 单位面值(元)。\n完整数据包含 {len(df.columns)} 列，此处仅显示部分核心列。"
        )

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取国债发行数据时发生错误: {e}\n请检查您的日期范围或稍后再试。"

@tool
def get_cninfo_local_government_bond_issues(
        start_date: str,
        end_date: str,
        bond_code: Optional[str] = None,
        bond_short_name: Optional[str] = None,
        bond_name: Optional[str] = None
) -> str:
    """
    获取巨潮资讯数据中心提供的指定日期范围内的中国地方政府债券发行历史数据。
    返回的数据包括债券代码、简称、发行起止日、计划和实际发行总量（亿元）、发行价格（元）、
    单位面值（元）、缴款日、增发次数、交易市场、发行方式、发行对象、公告日期以及债券名称等。

    Args:
        start_date (str): 查询数据的起始日期，格式为 'YYYYMMDD'，例如 '20210911'。
        end_date (str): 查询数据的结束日期，格式为 'YYYYMMDD'，例如 '20211110'。
        bond_code (Optional[str]): 可选参数，用于精确查询指定债券代码的发行数据。例如 '2105001'。
        bond_short_name (Optional[str]): 可选参数，用于模糊查询指定债券简称的发行数据。例如 '21河南债01'。
        bond_name (Optional[str]): 可选参数，用于模糊查询指定债券名称的发行数据。例如 '2021年河南省政府专项债券(一期)'。
                                    如果同时提供 bond_code、bond_short_name 或 bond_name，则优先级为：
                                    bond_code (精确匹配) > bond_short_name (模糊匹配) > bond_name (模糊匹配)。

    Returns:
        str: 一个格式化后的字符串，包含指定日期范围内所有地方债发行数据。
             如果查询失败、日期格式不正确、无数据，或者未找到指定债券，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not start_date or not end_date:
        return "错误：请提供查询数据的起始日期 (start_date) 和结束日期 (end_date)，格式为 'YYYYMMDD'。"

    # 日期格式检查
    if not (len(start_date) == 8 and start_date.isdigit()):
        return f"错误：起始日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20210911'。您输入的是: '{start_date}'"
    if not (len(end_date) == 8 and end_date.isdigit()):
        return f"错误：结束日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20211110'。您输入的是: '{end_date}'"

    try:
        # 调用API
        df = ak.bond_local_government_issue_cninfo(start_date=start_date, end_date=end_date)

        if df.empty:
            return f"未能获取从 '{start_date}' 到 '{end_date}' 期间的地方债发行数据。请检查日期范围或数据源暂无此信息。"

        # 根据 bond_code, bond_short_name 或 bond_name 过滤数据
        filtered_df = df.copy()
        if bond_code:
            filtered_df = filtered_df[filtered_df['债券代码'].astype(str) == bond_code]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到代码为 '{bond_code}' 的地方政府债发行数据。"
        elif bond_short_name:
            # 使用 contains 进行模糊匹配
            filtered_df = filtered_df[
                filtered_df['债券简称'].astype(str).str.contains(bond_short_name, na=False, case=False)]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到简称包含 '{bond_short_name}' 的地方政府债发行数据。"
        elif bond_name:
            # 使用 contains 进行模糊匹配
            filtered_df = filtered_df[filtered_df['债券名称'].astype(str).str.contains(bond_name, na=False, case=False)]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到名称包含 '{bond_name}' 的地方政府债发行数据。"

        # 如果过滤后 DataFrame 为空
        if filtered_df.empty:
            return f"在 '{start_date}' 到 '{end_date}' 期间，未找到符合条件的地方政府债发行数据。"

        output_lines = []
        if bond_code or bond_short_name or bond_name:
            # 如果是查询特定债券，则只显示该债券的信息
            output_lines.append(f"\n巨潮资讯地方债发行数据 (从 {start_date} 到 {end_date}) - 查询结果:\n")
        else:
            # 否则显示全部或部分数据
            output_lines.append(f"\n巨潮资讯地方债发行数据 (从 {start_date} 到 {end_date}):\n")

        output_lines.append("=" * 160)

        # 定义要显示的列及其宽度
        columns_to_display = {
            "债券代码": 10,
            "债券简称": 15,
            "发行起始日": 12,
            "发行终止日": 12,
            "计划发行总量(亿元)": 18,
            "实际发行总量(亿元)": 18,
            "发行价格(元)": 12,
            "单位面值(元)": 12,
            "缴款日": 12,
            "增发次数": 8,
            "交易市场": 10,
            "发行方式": 10,
            "发行对象": 10,
            "公告日期": 12,
            "债券名称": 20
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns_to_display.items()])
        output_lines.append(header)
        output_lines.append("-" * 160)

        # 遍历 DataFrame，格式化每行数据
        # 如果是查询特定债券，显示所有匹配结果；否则，只显示前5行和后5行
        if bond_code or bond_short_name or bond_name:
            # 显示所有过滤后的结果
            display_df = filtered_df
        else:
            display_df = pd.concat([filtered_df.head(5), filtered_df.tail(5)])
            if len(filtered_df) > 10:
                output_lines.append("... (中间省略) ...")

        for index, row in display_df.iterrows():
            # 获取数据并处理可能的 NaN 值
            债券代码_val = str(row.get('债券代码', 'N/A')).ljust(columns_to_display["债券代码"])
            债券简称_val = str(row.get('债券简称', 'N/A')).ljust(columns_to_display["债券简称"])
            发行起始日_val = str(row.get('发行起始日', 'N/A')).ljust(columns_to_display["发行起始日"])
            发行终止日_val = str(row.get('发行终止日', 'N/A')).ljust(columns_to_display["发行终止日"])

            # 数值型数据格式化，并处理 NaN
            计划发行总量_val = f"{row.get('计划发行总量', 0.0):.2f}".rjust(
                columns_to_display["计划发行总量(亿元)"]) if pd.notna(row.get('计划发行总量')) else "N/A".rjust(
                columns_to_display["计划发行总量(亿元)"])
            实际发行总量_val = f"{row.get('实际发行总量', 0.0):.2f}".rjust(
                columns_to_display["实际发行总量(亿元)"]) if pd.notna(row.get('实际发行总量')) else "N/A".rjust(
                columns_to_display["实际发行总量(亿元)"])
            发行价格_val = f"{row.get('发行价格', 0.0):.2f}".rjust(columns_to_display["发行价格(元)"]) if pd.notna(
                row.get('发行价格')) else "N/A".rjust(columns_to_display["发行价格(元)"])
            单位面值_val = str(row.get('单位面值', 'N/A')).rjust(columns_to_display["单位面值(元)"])  # int64

            缴款日_val = str(row.get('缴款日', 'N/A')).ljust(columns_to_display["缴款日"])
            增发次数_val = str(row.get('增发次数', 'N/A')).rjust(columns_to_display["增发次数"])  # int64
            交易市场_val = str(row.get('交易市场', 'N/A')).ljust(columns_to_display["交易市场"])
            发行方式_val = str(row.get('发行方式', 'N/A')).ljust(columns_to_display["发行方式"])
            发行对象_val = str(row.get('发行对象', 'N/A')).ljust(columns_to_display["发行对象"])
            公告日期_val = str(row.get('公告日期', 'N/A')).ljust(columns_to_display["公告日期"])

            债券名称_full = str(row.get('债券名称', 'N/A'))
            债券名称_display = (债券名称_full[:columns_to_display["债券名称"] - 3] + "...") if len(债券名称_full) > \
                                                                                               columns_to_display[
                                                                                                   "债券名称"] else 债券名称_full.ljust(
                columns_to_display["债券名称"])

            output_lines.append(
                f"{债券代码_val} | {债券简称_val} | {发行起始日_val} | {发行终止日_val} | {计划发行总量_val} | {实际发行总量_val} | {发行价格_val} | {单位面值_val} | {缴款日_val} | {增发次数_val} | {交易市场_val} | {发行方式_val} | {发行对象_val} | {公告日期_val} | {债券名称_display}"
            )

        output_lines.append("=" * 160)
        output_lines.append(
            f"总计 {len(filtered_df)} 条数据。注意单位：计划发行总量(亿元), 实际发行总量(亿元), 发行价格(元), 单位面值(元)。\n完整数据包含 {len(df.columns)} 列，此处仅显示部分核心列。"
        )

        return "\n".join(output_lines)
    except Exception as e:
        return f"获取地方债发行数据时发生错误: {e}\n请检查您的日期范围或稍后再试。"

@tool
def get_cninfo_corporate_bond_issues(
        start_date: str,
        end_date: str,
        bond_code: Optional[str] = None,
        bond_short_name: Optional[str] = None,
        bond_name: Optional[str] = None
) -> str:
    """
    获取巨潮资讯数据中心提供的指定日期范围内的中国企业债券发行历史数据。
    返回的数据包括债券代码、简称、公告日期、交易所网上发行起止日、计划和实际发行总量（万元）、
    发行面值、发行价格（元）、发行方式、发行对象、发行范围、承销方式、最小认购单位（万元）、
    募资用途说明、最低认购额（万元）以及债券名称等详细信息。

    Args:
        start_date (str): 查询数据的起始日期，格式为 'YYYYMMDD'，例如 '20210911'。
        end_date (str): 查询数据的结束日期，格式为 'YYYYMMDD'，例如 '20211110'。
        bond_code (Optional[str]): 可选参数，用于精确查询指定债券代码的发行数据。例如 '184109'。
        bond_short_name (Optional[str]): 可选参数，用于模糊查询指定债券简称的发行数据。例如 '开泰01'。
        bond_name (Optional[str]): 可选参数，用于模糊查询指定债券名称的发行数据。例如 '开泰城建'。
                                    如果同时提供 bond_code、bond_short_name 或 bond_name，则优先级为：
                                    bond_code (精确匹配) > bond_short_name (模糊匹配) > bond_name (模糊匹配)。

    Returns:
        str: 一个格式化后的字符串，包含指定日期范围内所有企业债发行数据。
             如果查询失败、日期格式不正确、无数据，或者未找到指定债券，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not start_date or not end_date:
        return "错误：请提供查询数据的起始日期 (start_date) 和结束日期 (end_date)，格式为 'YYYYMMDD'。"

    # 日期格式检查
    if not (len(start_date) == 8 and start_date.isdigit()):
        return f"错误：起始日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20210911'。您输入的是: '{start_date}'"
    if not (len(end_date) == 8 and end_date.isdigit()):
        return f"错误：结束日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20211110'。您输入的是: '{end_date}'"

    try:
        # 调用API
        df = ak.bond_corporate_issue_cninfo(start_date=start_date, end_date=end_date)

        if df.empty:
            return f"未能获取从 '{start_date}' 到 '{end_date}' 期间的企业债发行数据。请检查日期范围或数据源暂无此信息。"

        # 根据 bond_code, bond_short_name 或 bond_name 过滤数据
        filtered_df = df.copy()
        if bond_code:
            filtered_df = filtered_df[filtered_df['债券代码'].astype(str) == bond_code]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到代码为 '{bond_code}' 的企业债发行数据。"
        elif bond_short_name:
            # 使用 contains 进行模糊匹配
            filtered_df = filtered_df[
                filtered_df['债券简称'].astype(str).str.contains(bond_short_name, na=False, case=False)]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到简称包含 '{bond_short_name}' 的企业债发行数据。"
        elif bond_name:
            # 使用 contains 进行模糊匹配
            filtered_df = filtered_df[filtered_df['债券名称'].astype(str).str.contains(bond_name, na=False, case=False)]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到名称包含 '{bond_name}' 的企业债发行数据。"

        # 如果过滤后 DataFrame 为空
        if filtered_df.empty:
            return f"在 '{start_date}' 到 '{end_date}' 期间，未找到符合条件的企业债发行数据。"

        output_lines = []
        if bond_code or bond_short_name or bond_name:
            # 如果是查询特定债券，则只显示该债券的信息
            output_lines.append(f"\n巨潮资讯企业债发行数据 (从 {start_date} 到 {end_date}) - 查询结果:\n")
        else:
            # 否则显示全部或部分数据
            output_lines.append(f"\n巨潮资讯企业债发行数据 (从 {start_date} 到 {end_date}):\n")

        output_lines.append("=" * 180)

        # 定义要显示的列及其宽度
        columns_to_display = {
            "债券代码": 10,
            "债券简称": 15,
            "公告日期": 12,
            "计划发行总量(万元)": 18,
            "实际发行总量(万元)": 18,
            "发行面值": 10,
            "发行价格(元)": 12,
            "发行方式": 10,
            "最小认购单位(万元)": 18,
            "募资用途说明": 25,
            "债券名称": 25
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns_to_display.items()])
        output_lines.append(header)
        output_lines.append("-" * 180)

        # 遍历 DataFrame，格式化每行数据
        # 如果是查询特定债券，显示所有匹配结果；否则，只显示前5行和后5行
        if bond_code or bond_short_name or bond_name:
            # 显示所有过滤后的结果
            display_df = filtered_df
        else:
            display_df = pd.concat([filtered_df.head(5), filtered_df.tail(5)])
            if len(filtered_df) > 10:
                output_lines.append("... (中间省略) ...")

        for index, row in display_df.iterrows():
            # 获取数据并处理可能的 NaN 值
            债券代码_val = str(row.get('债券代码', 'N/A')).ljust(columns_to_display["债券代码"])
            债券简称_val = str(row.get('债券简称', 'N/A')).ljust(columns_to_display["债券简称"])
            公告日期_val = str(row.get('公告日期', 'N/A')).ljust(columns_to_display["公告日期"])

            # 数值型数据格式化，并处理 NaN
            计划发行总量_val = f"{row.get('计划发行总量', 0.0):.2f}".rjust(
                columns_to_display["计划发行总量(万元)"]) if pd.notna(row.get('计划发行总量')) else "N/A".rjust(
                columns_to_display["计划发行总量(万元)"])
            实际发行总量_val = f"{row.get('实际发行总量', 0.0):.2f}".rjust(
                columns_to_display["实际发行总量(万元)"]) if pd.notna(row.get('实际发行总量')) else "N/A".rjust(
                columns_to_display["实际发行总量(万元)"])
            发行面值_val = f"{row.get('发行面值', 0.0):.2f}".rjust(columns_to_display["发行面值"]) if pd.notna(
                row.get('发行面值')) else "N/A".rjust(columns_to_display["发行面值"])
            发行价格_val = str(row.get('发行价格', 'N/A')).rjust(columns_to_display["发行价格(元)"])  # int64

            发行方式_val = str(row.get('发行方式', 'N/A')).ljust(columns_to_display["发行方式"])

            最小认购单位_val = f"{row.get('最小认购单位', 0.0):.2f}".rjust(
                columns_to_display["最小认购单位(万元)"]) if pd.notna(row.get('最小认购单位')) else "N/A".rjust(
                columns_to_display["最小认购单位(万元)"])

            # 截断募资用途说明和债券名称
            募资用途说明_full = str(row.get('募资用途说明', 'N/A'))
            募资用途说明_display = (募资用途说明_full[:columns_to_display["募资用途说明"] - 3] + "...") if len(
                募资用途说明_full) > columns_to_display["募资用途说明"] else 募资用途说明_full.ljust(
                columns_to_display["募资用途说明"])

            债券名称_full = str(row.get('债券名称', 'N/A'))
            债券名称_display = (债券名称_full[:columns_to_display["债券名称"] - 3] + "...") if len(债券名称_full) > \
                                                                                               columns_to_display[
                                                                                                   "债券名称"] else 债券名称_full.ljust(
                columns_to_display["债券名称"])

            output_lines.append(
                f"{债券代码_val} | {债券简称_val} | {公告日期_val} | {计划发行总量_val} | {实际发行总量_val} | {发行面值_val} | {发行价格_val} | {发行方式_val} | {最小认购单位_val} | {募资用途说明_display} | {债券名称_display}"
            )

        output_lines.append("=" * 180)
        output_lines.append(
            f"总计 {len(filtered_df)} 条数据。注意单位：计划发行总量(万元), 实际发行总量(万元), 发行价格(元), 最小认购单位(万元)。\n完整数据包含 {len(df.columns)} 列，此处仅显示部分核心列。"
        )

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取企业债发行数据时发生错误: {e}\n请检查您的日期范围或稍后再试。"

@tool
def get_cninfo_convertible_bond_issues(
        start_date: str,
        end_date: str,
        bond_code: Optional[str] = None,
        bond_short_name: Optional[str] = None,
        bond_name: Optional[str] = None
) -> str:
    """
    获取巨潮资讯数据中心提供的指定日期范围内的可转债发行历史数据。
    返回的数据包括债券代码、简称、公告日期、发行起止日、计划和实际发行总量（万元）、
    发行面值（元）、发行价格（元）、初始转股价格（元）、转股开始和终止日期、
    网上申购相关信息、优先申购相关信息、交易市场以及债券名称等详细信息。

    Args:
        start_date (str): 查询数据的起始日期，格式为 'YYYYMMDD'，例如 '20210913'。
        end_date (str): 查询数据的结束日期，格式为 'YYYYMMDD'，例如 '20211112'。
        bond_code (Optional[str]): 可选参数，用于精确查询指定债券代码的发行数据。例如 '110083'。
        bond_short_name (Optional[str]): 可选参数，用于模糊查询指定债券简称的发行数据。例如 '苏租转债'。
        bond_name (Optional[str]): 可选参数，用于模糊查询指定债券名称的发行数据。例如 '江苏金融租赁'。
                                    如果同时提供 bond_code、bond_short_name 或 bond_name，则优先级为：
                                    bond_code (精确匹配) > bond_short_name (模糊匹配) > bond_name (模糊匹配)。

    Returns:
        str: 一个格式化后的字符串，包含指定日期范围内所有可转债发行数据。
             如果查询失败、日期格式不正确、无数据，或者未找到指定债券，则返回相应的提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not start_date or not end_date:
        return "错误：请提供查询数据的起始日期 (start_date) 和结束日期 (end_date)，格式为 'YYYYMMDD'。"

    # 日期格式检查
    if not (len(start_date) == 8 and start_date.isdigit()):
        return f"错误：起始日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20210913'。您输入的是: '{start_date}'"
    if not (len(end_date) == 8 and end_date.isdigit()):
        return f"错误：结束日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20211112'。您输入的是: '{end_date}'"

    try:
        # 调用API
        df = ak.bond_cov_issue_cninfo(start_date=start_date, end_date=end_date)

        if df.empty:
            return f"未能获取从 '{start_date}' 到 '{end_date}' 期间的可转债发行数据。请检查日期范围或数据源暂无此信息。"

        # 根据 bond_code, bond_short_name 或 bond_name 过滤数据
        filtered_df = df.copy()
        if bond_code:
            filtered_df = filtered_df[filtered_df['债券代码'].astype(str) == bond_code]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到代码为 '{bond_code}' 的可转债发行数据。"
        elif bond_short_name:
            # 使用 contains 进行模糊匹配
            filtered_df = filtered_df[filtered_df['债券简称'].astype(str).str.contains(bond_short_name, na=False,
                                                                                       case=False)]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到简称包含 '{bond_short_name}' 的可转债发行数据。"
        elif bond_name:
            # 使用 contains 进行模糊匹配
            filtered_df = filtered_df[
                filtered_df['债券名称'].astype(str).str.contains(bond_name, na=False, case=False)]
            if filtered_df.empty:
                return f"在 '{start_date}' 到 '{end_date}' 期间，未找到名称包含 '{bond_name}' 的可转债发行数据。"

        # 如果过滤后 DataFrame 为空
        if filtered_df.empty:
            return f"在 '{start_date}' 到 '{end_date}' 期间，未找到符合条件的可转债发行数据。"

        output_lines = []
        if bond_code or bond_short_name or bond_name:
            # 如果是查询特定债券，则只显示该债券的信息
            output_lines.append(f"\n巨潮资讯可转债发行数据 (从 {start_date} 到 {end_date}) - 查询结果:\n")
        else:
            # 否则显示全部或部分数据
            output_lines.append(f"\n巨潮资讯可转债发行数据 (从 {start_date} 到 {end_date}):\n")

        output_lines.append("=" * 180)

        # 定义要显示的列及其宽度
        columns_to_display = {
            "债券代码": 10,
            "债券简称": 15,
            "公告日期": 12,
            "发行起始日": 12,
            "计划发行总量(万元)": 18,
            "实际发行总量(万元)": 18,
            "发行价格(元)": 12,
            "初始转股价格(元)": 16,
            "转股开始日期": 12,
            "网上申购日期": 12,
            "交易市场": 10,
            "债券名称": 25
        }

        # 格式化表头
        header = " | ".join([col.ljust(width) for col, width in columns_to_display.items()])
        output_lines.append(header)
        output_lines.append("-" * 180)

        # 遍历 DataFrame，格式化每行数据
        # 如果是查询特定债券，显示所有匹配结果；否则，只显示前5行和后5行
        if bond_code or bond_short_name or bond_name:
            # 显示所有过滤后的结果
            display_df = filtered_df
        else:
            display_df = pd.concat([filtered_df.head(5), filtered_df.tail(5)])
            if len(filtered_df) > 10:
                output_lines.append("... (中间省略) ...")

        for index, row in display_df.iterrows():
            # 获取数据并处理可能的 NaN 值
            债券代码_val = str(row.get('债券代码', 'N/A')).ljust(columns_to_display["债券代码"])
            债券简称_val = str(row.get('债券简称', 'N/A')).ljust(columns_to_display["债券简称"])
            公告日期_val = str(row.get('公告日期', 'N/A')).ljust(columns_to_display["公告日期"])
            发行起始日_val = str(row.get('发行起始日', 'N/A')).ljust(columns_to_display["发行起始日"])

            # 数值型数据格式化，并处理 NaN
            计划发行总量_val = f"{row.get('计划发行总量', 0.0):.2f}".rjust(
                columns_to_display["计划发行总量(万元)"]) if pd.notna(row.get('计划发行总量')) else "N/A".rjust(
                columns_to_display["计划发行总量(万元)"])
            实际发行总量_val = f"{row.get('实际发行总量', 0.0):.2f}".rjust(
                columns_to_display["实际发行总量(万元)"]) if pd.notna(row.get('实际发行总量')) else "N/A".rjust(
                columns_to_display["实际发行总量(万元)"])
            发行价格_val = f"{row.get('发行价格', 0.0):.2f}".rjust(columns_to_display["发行价格(元)"]) if pd.notna(
                row.get('发行价格')) else "N/A".rjust(columns_to_display["发行价格(元)"])
            初始转股价格_val = f"{row.get('初始转股价格', 0.0):.2f}".rjust(
                columns_to_display["初始转股价格(元)"]) if pd.notna(row.get('初始转股价格')) else "N/A".rjust(
                columns_to_display["初始转股价格(元)"])

            转股开始日期_val = str(row.get('转股开始日期', 'N/A')).ljust(columns_to_display["转股开始日期"])
            网上申购日期_val = str(row.get('网上申购日期', 'N/A')).ljust(columns_to_display["网上申购日期"])
            交易市场_val = str(row.get('交易市场', 'N/A')).ljust(columns_to_display["交易市场"])

            债券名称_full = str(row.get('债券名称', 'N/A'))
            债券名称_display = (债券名称_full[:columns_to_display["债券名称"] - 3] + "...") if len(债券名称_full) > \
                                                                                               columns_to_display[
                                                                                                   "债券名称"] else 债券名称_full.ljust(
                columns_to_display["债券名称"])

            output_lines.append(
                f"{债券代码_val} | {债券简称_val} | {公告日期_val} | {发行起始日_val} | {计划发行总量_val} | {实际发行总量_val} | {发行价格_val} | {初始转股价格_val} | {转股开始日期_val} | {网上申购日期_val} | {交易市场_val} | {债券名称_display}"
            )

        output_lines.append("=" * 180)
        output_lines.append(
            f"总计 {len(filtered_df)} 条数据。注意单位：计划发行总量(万元), 实际发行总量(万元), 发行价格(元), 初始转股价格(元)。\n完整数据包含 {len(df.columns)} 列，此处仅显示部分核心列。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取可转债发行数据时发生错误: {e}\n请检查您的日期范围或稍后再试。"

@tool
def get_company_news_cn(company_name: str) -> str:
    """
    查询指定国内公司的相关新闻资讯。
    数据来源于 NewsAPI，主要获取语言为中文的新闻。
    请注意：NewsAPI的免费或低级订阅可能对中文新闻源的支持有限，
    因此结果可能主要来自报道中国公司的国际新闻机构。

    Args:
        company_name (str): 必需参数。要查询新闻资讯的国内公司名称（如 '华为', '腾讯', '阿里巴巴'）。

    Returns:
        str: 一个格式化后的字符串，包含相关新闻资讯的标题、来源、发布日期和链接。
             如果未找到新闻，则返回未找到信息。
             如果 NewsAPI Key 未设置或获取失败，则返回错误信息。
    """
    try:
        # 调用API, 查询中文新闻
        articles_response = newsapi.get_everything(
            q=company_name,
            language='zh',
            sort_by='relevancy',
            page_size=10
        )

        articles = articles_response.get('articles', [])

        if not articles:
            return f"未能找到与 '{company_name}' 相关的中文新闻资讯。请尝试其他公司名称或稍后再试。"

        output_lines = [f"找到与 '{company_name}' 相关的中文新闻资讯 (数据来源: NewsAPI):\n"]
        output_lines.append("=" * 100)

        for i, article in enumerate(articles):
            title = article.get('title', '无标题')
            source = article.get('source', {}).get('name', '未知来源')
            published_at = article.get('publishedAt', '未知日期')
            url = article.get('url', '#')

            # 格式化日期，只保留日期部分
            if published_at and 'T' in published_at:
                published_at = published_at.split('T')[0]

            output_lines.append(f"新闻 {i + 1}:")
            output_lines.append(f"  标题: {title}")
            output_lines.append(f"  来源: {source}")
            output_lines.append(f"  日期: {published_at}")
            output_lines.append(f"  链接: {url}\n")
        
        output_lines.append("=" * 100)
        output_lines.append("提示：NewsAPI的免费或低级订阅可能对中文新闻源支持有限，结果可能主要来自报道中国公司的国际新闻机构。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"查询公司 '{company_name}' 新闻资讯时发生错误: {e}"

@tool
def get_company_news_en(company_name: str) -> str:
    """
    查询指定国内公司的相关英文新闻资讯。
    数据来源于 NewsAPI，主要获取语言为英文的新闻。
    此工具适用于查询国际媒体对中国公司的报道。

    Args:
        company_name (str): 必需参数。要查询英文新闻资讯的国内公司名称（如 'Huawei', 'Tencent', 'Alibaba'）。
                            建议使用英文公司名称以获得更准确的匹配。

    Returns:
        str: 一个格式化后的字符串，包含相关英文新闻资讯的标题、来源、发布日期和链接。
             如果未找到新闻，则返回未找到信息。
             如果 NewsAPI Key 未设置或获取失败，则返回错误信息。
    """
    try:
        # 调用API, 查询英文新闻
        articles_response = newsapi.get_everything(
            q=company_name,
            language='en',
            sort_by='relevancy',
            page_size=10
        )

        articles = articles_response.get('articles', [])

        if not articles:
            return f"未能找到与 '{company_name}' 相关的英文新闻资讯。请尝试其他公司名称或稍后再试。"

        output_lines = [f"找到与 '{company_name}' 相关的英文新闻资讯 (数据来源: NewsAPI):\n"]
        output_lines.append("=" * 100)

        for i, article in enumerate(articles):
            title = article.get('title', 'No Title')
            source = article.get('source', {}).get('name', 'Unknown Source')
            published_at = article.get('publishedAt', 'Unknown Date')
            url = article.get('url', '#')

            # 格式化日期，只保留日期部分
            if published_at and 'T' in published_at:
                published_at = published_at.split('T')[0]

            output_lines.append(f"News {i + 1}:")
            output_lines.append(f"  Title: {title}")
            output_lines.append(f"  Source: {source}")
            output_lines.append(f"  Published Date: {published_at}")
            output_lines.append(f"  Link: {url}\n")
        
        output_lines.append("=" * 100)
        output_lines.append("提示：此工具主要用于查询国际媒体对中国公司的英文报道。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"查询公司 '{company_name}' 英文新闻资讯时发生错误: {e}"

@tool
def get_foreign_company_news_en(company_name: str) -> str:
    """
    查询指定国外公司的相关英文新闻资讯。
    数据来源于 NewsAPI，主要获取语言为英文的新闻。
    此工具适用于查询国际媒体对全球非中国公司的报道。

    Args:
        company_name (str): 必需参数。要查询英文新闻资讯的国外公司名称（如 'Apple', 'Google', 'Samsung'）。
                            建议使用公司英文名称以获得更准确的匹配。

    Returns:
        str: 一个格式化后的字符串，包含相关英文新闻资讯的标题、来源、发布日期和链接。
             如果未找到新闻，则返回未找到信息。
             如果 NewsAPI Key 未设置或获取失败，则返回错误信息。
    """
    try:
        # 调用API, 查询英文新闻
        articles_response = newsapi.get_everything(
            q=company_name,
            language='en',
            sort_by='relevancy',
            page_size=10
        )

        articles = articles_response.get('articles', [])

        if not articles:
            return f"未能找到与 '{company_name}' 相关的英文新闻资讯。请尝试其他公司名称或稍后再试。"

        output_lines = [f"找到与 '{company_name}' 相关的英文新闻资讯 (数据来源: NewsAPI):\n"]
        output_lines.append("=" * 100)

        for i, article in enumerate(articles):
            title = article.get('title', 'No Title')
            source = article.get('source', {}).get('name', 'Unknown Source')
            published_at = article.get('publishedAt', 'Unknown Date')
            url = article.get('url', '#')

            # 格式化日期，只保留日期部分
            if published_at and 'T' in published_at:
                published_at = published_at.split('T')[0]

            output_lines.append(f"News {i + 1}:")
            output_lines.append(f"  Title: {title}")
            output_lines.append(f"  Source: {source}")
            output_lines.append(f"  Published Date: {published_at}")
            output_lines.append(f"  Link: {url}\n")
        
        output_lines.append("=" * 100)
        output_lines.append("提示：此工具主要用于查询国际媒体对全球非中国公司的英文报道。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"查询公司 '{company_name}' 英文新闻资讯时发生错误: {e}"

@tool
def get_eastmoney_sh_sz_balance_sheet(date: str, symbol: Optional[str] = None, selected_indicators: Optional[List[str]] = None) -> str:
    """
    获取东方财富网数据中心指定日期的沪深 (上海和深圳证券交易所) 资产负债表数据。
    该接口返回包括股票代码、股票简称、货币资金、应收账款、存货、总资产（含同比）、
    应付账款、预收账款、总负债（含同比）、资产负债率、股东权益合计和公告日期等详细财务指标。

    Args:
        date (str): 需要查询资产负债表的日期，格式为 'YYYYMMDD'，
                    必须是季度末日期，例如 '20240331', '20240630', '20240930', 或 '20241231'。
                    数据从 20081231 开始。
        symbol (Optional[str]): 可选参数。指定沪深股票代码，例如 '600000' 或 '000001'。
                                如果提供，将只返回该股票的资产负债表信息。
        selected_indicators (Optional[List[str]]): 可选参数，指定要返回的财务指标列的列表。
                                                   如果提供此参数，则只返回这些指定的列（以及股票代码和股票简称）。
                                                   例如：['资产-总资产', '负债-总负债', '资产负债率']。

    Returns:
        str: 一个格式化后的字符串，包含指定日期的资产负债表摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式和季度末日期
    try:
        dt_obj = datetime.strptime(date, "%Y%m%d")
        if dt_obj.strftime("%m%d") not in ["0331", "0630", "0930", "1231"]:
            return f"错误：日期 '{date}' 不是有效的季度末日期。请提供 'YYYY0331', 'YYYY0630', 'YYYY0930', 或 'YYYY1231' 格式的日期。"
    except ValueError:
        return f"错误：日期 '{date}' 格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20240331'。"

    try:
        # 调用API
        df = ak.stock_zcfz_em(date=date)

        if df.empty:
            return f"在日期 {date} 未能获取到东方财富网沪深市场的资产负债表数据。请检查日期是否正确或该日期无数据。"

        output_lines = [f"东方财富网 {date} 的沪深市场资产负债表信息:\n"]
        output_lines.append("=" * 80)

        # 股票代码过滤
        if symbol:
            # 确保股票代码列是字符串类型以便进行精确匹配，并去除可能的空格
            df['股票代码'] = df['股票代码'].astype(str).str.strip()
            # 过滤数据
            df_filtered = df[df['股票代码'] == symbol.strip()].copy()
            if df_filtered.empty:
                return f"在日期 {date} 未能找到股票代码 '{symbol}' 的沪深资产负债表数据。请检查股票代码是否正确或该日期无该股票数据。"
            df = df_filtered
            output_lines.append(f"已过滤，只显示股票 '{symbol}' 的资产负债表。")

        output_lines.append(f"总计找到 {len(df)} 条资产负债表记录。")

        actual_display_columns = []
        essential_identifiers = ['股票代码', '股票简称']

        # 包含基本识别信息
        for ident_col in essential_identifiers:
            if ident_col in df.columns:
                actual_display_columns.append(ident_col)

        if selected_indicators:
            # 如果指定了 selected_indicators，则只返回这些列
            found_requested_indicators = []
            not_found_requested_indicators = []
            for req_col in selected_indicators:
                if req_col in df.columns:
                    # 避免重复添加
                    if req_col not in actual_display_columns:
                        actual_display_columns.append(req_col)
                    found_requested_indicators.append(req_col)
                else:
                    not_found_requested_indicators.append(req_col)

            if not_found_requested_indicators:
                output_lines.append(f"注意：您请求的指标 '{', '.join(not_found_requested_indicators)}' 未在数据中找到。")

            if not actual_display_columns:
                return f"错误：未能找到您请求的任何指标，也无法显示基本识别信息（股票代码/简称）。请检查股票代码或指标名称是否正确。"
            else:
                output_lines.append(f"已根据您的请求显示以下指标：{', '.join(actual_display_columns)}")

        else:
            # 如果未指定 selected_indicators
            default_display_columns_priority = [
                '资产-货币资金', '资产-应收账款', '资产-存货',
                '资产-总资产', '资产-总资产同比', '负债-应付账款', '负债-预收账款',
                '负债-总负债', '负债-总负债同比', '资产负债率', '股东权益合计', '公告日期'
            ]
            for col in default_display_columns_priority:
                if col in df.columns and col not in actual_display_columns:
                    actual_display_columns.append(col)

            # 如果预定义的关键列很少，则尝试添加更多通用列以提供一些信息
            # 如果显示的列（包括识别信息）少于5列，且原始数据还有更多列
            if len(actual_display_columns) < 5 and len(df.columns) > len(actual_display_columns):
                # 排除已添加的列和 '序号' 列
                additional_cols = [col for col in df.columns if col not in actual_display_columns and col not in ['序号']][:min(len(df.columns) - len(actual_display_columns), 5)]
                actual_display_columns.extend(additional_cols)
                # 仅当实际添加了额外列时才显示此提示
                if additional_cols:
                    output_lines.append(f"注意：未能找到所有预期的关键指标列，已补充显示 {len(additional_cols)} 列。")

            if not actual_display_columns:
                return f"未能获取到股票代码 '{symbol}' 在 '{date}' 模式下的东方财富资产负债表数据，或数据不包含任何可显示列。"

        display_df = df[actual_display_columns].copy()

        # 统一处理 NaN 值，将其显示为 '-'
        display_df = display_df.fillna('-')

        # 格式化数值列
        currency_cols = [
            '资产-货币资金', '资产-应收账款', '资产-存货', '资产-总资产',
            '负债-应付账款', '负债-预收账款', '负债-总负债', '股东权益合计'
        ]
        for col in currency_cols:
            if col in display_df.columns: # 仅对实际存在的列进行格式化
                # 将列转换为数值类型，无法转换的变为 NaN
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                # 对数值进行格式化，并处理 NaN
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else '-'
                )

        # 格式化百分比列
        percentage_cols = ['资产-总资产同比', '负债-总负债同比', '资产负债率']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else '-'
                )

        # 格式化日期列
        if '公告日期' in display_df.columns:
            display_df['公告日期'] = pd.to_datetime(display_df['公告日期'], errors='coerce').dt.strftime(
                '%Y-%m-%d').fillna('-')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获得格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)")

        output_lines.append("=" * 80)
        output_lines.append("提示：金额单位为元，百分比已按原数据格式化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日期 {date} 的东方财富网沪深市场资产负债表数据时发生错误: {e}"

@tool
def get_ths_balance_sheet(
        symbol: str,
        indicator: str,
        selected_indicators: Optional[List[str]] = None,
        start_date: Optional[str] = None,  # 新增参数：开始日期
        end_date: Optional[str] = None  # 新增参数：结束日期
) -> str:
    """
    获取同花顺-财务指标提供的指定股票的资产负债表历史数据，并可按报告期时间范围过滤。
    该接口可以获取按报告期、按年度或按单季度划分的资产负债表数据，
    包含多达80项详细财务指标。

    Args:
        symbol (str): 股票代码，例如 '000063'。
        indicator (str): 数据粒度，若指定查询时间使用"按报告期"，可选值包括 "按报告期", "按年度", "按单季度"。 且默认为“按报告期”。
        selected_indicators (Optional[List[str]]): 可选参数，指定要返回的财务指标列的列表。
                                                  如果提供此参数，则只返回这些指定的列（以及'报告期'列）。
                                                  例如：['资产总计', '负债总计', '所有者权益合计']。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYY-MM-DD'。
                                    将过滤“报告期”在此日期或之后的数据。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYY-MM-DD'。
                                    将过滤“报告期”在此日期或之前的数据。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的资产负债表摘要和部分详细数据。
             如果获取失败，则返回错误信息。
    """

    def _parse_chinese_currency_string(value):
        """
        解析包含中文货币单位（如“万”、“亿”）的字符串为浮点数。
        处理NaN/None值。
        示例: "1.23亿" -> 123000000.0, "5000万" -> 50000000.0
        """
        if pd.isna(value) or value is None or value == '':
            return float('nan')
        s_value = str(value).strip()

        if '亿' in s_value:
            num_part = s_value.replace('亿', '')
            try:
                return float(num_part) * 100000000
            except ValueError:
                return float('nan')
        elif '万' in s_value:
            num_part = s_value.replace('万', '')
            try:
                return float(num_part) * 10000
            except ValueError:
                return float('nan')
        else:
            try:
                # 直接转换为浮点数，处理可能存在的逗号
                return float(s_value.replace(',', ''))
            except ValueError:
                return float('nan')

    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证 indicator 参数
    valid_indicators = {"按报告期", "按年度", "按单季度"}
    if indicator not in valid_indicators:
        return f"错误：indicator 参数 '{indicator}' 无效。请选择 {', '.join(valid_indicators)} 中的一个。"

    # 解析日期参数
    parsed_start_date_obj = None
    parsed_end_date_obj = None
    parsed_start_timestamp = None
    parsed_end_timestamp = None

    if start_date:
        try:
            parsed_start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            parsed_start_timestamp = pd.Timestamp(parsed_start_date_obj)
        except ValueError:
            return "错误：'start_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    if end_date:
        try:
            parsed_end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
            parsed_end_timestamp = pd.Timestamp(parsed_end_date_obj)
        except ValueError:
            return "错误：'end_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"

    if parsed_start_date_obj and parsed_end_date_obj and parsed_start_date_obj > parsed_end_date_obj:
        return "错误：'start_date' 不能晚于 'end_date'。"

    try:
        # 调用API
        df = ak.stock_financial_debt_ths(symbol=symbol, indicator=indicator)

        if df.empty:
            return f"未能获取到股票代码 '{symbol}' 在 '{indicator}' 模式下的同花顺资产负债表数据。请检查股票代码或指示器是否正确。"

        output_lines = [f"同花顺股票 '{symbol}' ({indicator}) 资产负债表历史数据:\n"]
        output_lines.append("=" * 80)

        # 日期过滤
        if '报告期' in df.columns and (parsed_start_timestamp or parsed_end_timestamp):
            # 将 '报告期' 列转换为日期时间类型，无效值设为 NaT
            df['报告期_dt'] = pd.to_datetime(df['报告期'], errors='coerce', format='%Y%m%d')
            invalid_dates = df['报告期_dt'].isna()
            if invalid_dates.any():
                df.loc[invalid_dates, '报告期_dt'] = pd.to_datetime(df.loc[invalid_dates, '报告期'], errors='coerce',
                                                                    format='%Y-%m-%d')

            if parsed_start_timestamp:
                df = df[df['报告期_dt'] >= parsed_start_timestamp].copy()
                output_lines.append(f"已过滤，报告期从 {start_date} 开始。")
            if parsed_end_timestamp:
                df = df[df['报告期_dt'] <= parsed_end_timestamp].copy()
                output_lines.append(f"已过滤，报告期到 {end_date} 结束。")

            # 移除辅助列
            df = df.drop(columns=['报告期_dt'])

            if df.empty:
                date_range_info = ""
                if start_date and end_date:
                    date_range_info = f"在 {start_date} 到 {end_date} 期间"
                elif start_date:
                    date_range_info = f"从 {start_date} 开始"
                elif end_date:
                    date_range_info = f"到 {end_date} 结束"
                return f"在指定日期范围 {date_range_info} 内，未能找到股票 '{symbol}' 的资产负债表数据。"

        output_lines.append(f"总计找到 {len(df)} 条资产负债表记录，包含 {len(df.columns)} 项指标。")

        actual_display_columns = []
        if selected_indicators:
            # 如果指定了 selected_indicators，则只返回这些列
            if '报告期' in df.columns:
                actual_display_columns.append('报告期')

            found_selected_indicators = []
            not_found_selected_indicators = []
            for requested_col in selected_indicators:
                if requested_col in df.columns:
                    if requested_col not in actual_display_columns:
                        # 避免重复添加
                        actual_display_columns.append(requested_col)
                    found_selected_indicators.append(requested_col)
                else:
                    not_found_selected_indicators.append(requested_col)

            # 如果最终没有列可显示
            if not actual_display_columns:
                return f"错误：未能找到您请求的任何指标。请检查股票代码或指标名称是否正确。"
            # 如果部分请求的指标未找到
            elif not_found_selected_indicators:
                output_lines.append(f"注意：您请求的指标 '{', '.join(not_found_selected_indicators)}' 未在数据中找到。")
                if not found_selected_indicators and '报告期' in df.columns:
                    output_lines.append(f"仅显示 '报告期'。")
                elif found_selected_indicators:
                    output_lines.append(f"已显示找到的指标。")

            output_lines.append(f"已根据您的请求显示以下指标：{', '.join(actual_display_columns)}")

        else:
            # 如果未指定 selected_indicators
            display_columns_priority = [
                '报告期',
                '所有者权益（或股东权益）合计',
                '负债和所有者权益（或股东权益）合计',
                '资产总计',
                '流动资产合计',
                '流动负债合计',
                '货币资金',
                '应收账款',
                '存货',
                '应付账款',
                '资产负债率'
            ]
            actual_display_columns = [col for col in display_columns_priority if col in df.columns]

            # 如果预定义的关键列很少，则显示前几列以提供一些信息
            if len(actual_display_columns) < 3 and len(df.columns) > 0:
                actual_display_columns = df.columns[:min(len(df.columns), 5)].tolist()
                output_lines.append(f"注意：未能找到所有预期的关键指标列，将显示前 {len(actual_display_columns)} 列。")

        # 确保最终要显示的列不为空
        if not actual_display_columns:
            return f"未能获取到股票代码 '{symbol}' 在 '{indicator}' 模式下的同花顺资产负债表数据，或数据不包含任何可显示列。"

        display_df = df[actual_display_columns].copy()

        # 对除“报告期”外的所有列应用解析函数
        for col in display_df.columns:
            if col != '报告期':
                display_df[col] = display_df[col].apply(_parse_chinese_currency_string)

        # 格式化数值列
        for col in display_df.columns:
            if col != '报告期':
                # 检查列在解析后是否包含数值数据
                if pd.api.types.is_numeric_dtype(display_df[col]):
                    # 如果是百分比列，则格式化为百分比
                    if '率' in col or '同比' in col:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x * 100:.2f}%" if pd.notna(x) else '-'
                        )
                    else:  # 否则，格式化为金额
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:,.2f}" if pd.notna(x) else '-'
                        )
                else:
                    # 如果不是数值类型，则用 '-' 填充 NaN
                    display_df[col] = display_df[col].fillna('-')
            else:
                display_df[col] = display_df[col].astype(str).fillna('-')

        # 按 '报告期' 降序排序，以显示最近的记录在前
        if '报告期' in display_df.columns:
            display_df['temp_sort_date'] = pd.to_datetime(display_df['报告期'], errors='coerce', format='%Y%m%d')
            invalid_sort_dates = display_df['temp_sort_date'].isna()
            if invalid_sort_dates.any():
                display_df.loc[invalid_sort_dates, 'temp_sort_date'] = pd.to_datetime(
                    display_df.loc[invalid_sort_dates, '报告期'], errors='coerce', format='%Y-%m-%d')

            display_df = display_df.sort_values(by='temp_sort_date', ascending=False).drop(columns='temp_sort_date')

        rows_to_show = min(len(display_df), MAX_ROWS_TO_DISPLAY)
        output_lines.append(f"显示前 {rows_to_show} 条记录：")
        output_lines.append("-" * 80)

        # 获取格式良好的表格字符串
        output_lines.append(display_df.head(rows_to_show).to_string(index=False))

        if len(df) > MAX_ROWS_TO_DISPLAY:
            output_lines.append(
                f"\n(还有 {len(df) - MAX_ROWS_TO_DISPLAY} 条记录未显示，请在AKShare文档或数据源查看更多。)"
            )
        output_lines.append("=" * 80)
        output_lines.append("提示：金额单位为元，百分比已按原数据格式化。数据包含多项财务指标，此处仅显示部分关键指标。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票 '{symbol}' ({indicator}) 的同花顺资产负债表数据时发生错误: {e}"

@tool
def get_stock_jgdy_detail_em(date: str, stock_identifier: Optional[str] = None) -> str:
    """
    获取东方财富网-数据中心-特色数据-机构调研的详细数据。
    该接口返回包括股票代码、名称、最新价、涨跌幅、调研机构、机构类型、调研人员、
    接待方式、接待人员、接待地点、调研日期和公告日期等详细信息。

    用户必须提供一个查询日期（格式YYYYMMDD）。
    可选地，可以提供股票代码或名称来查询特定股票在指定日期的机构调研数据。
    如果未提供股票标识符，则返回该日期所有机构调研数据的摘要。

    Args:
        date (str): 必需参数。查询的日期，格式为"YYYYMMDD"，例如 "20241211"。
        stock_identifier (Optional[str]): 可选参数。要查询的股票代码（如 '000001'）或股票名称（如 '平安银行'）。
                                          如果未提供，将返回该日期所有机构调研数据的摘要。

    Returns:
        str: 一个格式化后的字符串，包含机构调研详细数据。
             如果获取失败、未找到数据或找到多个匹配，则返回错误或提示信息。
    """
    # 验证日期格式
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return f"错误：日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20241211'。"

    try:
        # 调用API
        df = ak.stock_jgdy_detail_em(date=date)

        if df.empty:
            return f"在日期 '{date}' 未能获取到机构调研详细数据。可能该日期无数据或数据源暂无数据。"

        # 确保 '代码' 和 '名称' 列存在且为字符串类型
        if '代码' not in df.columns or '名称' not in df.columns:
            return "错误：获取到的机构调研数据缺少 '代码' 或 '名称' 列，无法进行查询或展示。"
        df['代码'] = df['代码'].astype(str)
        df['名称'] = df['名称'].astype(str)

        # 定义通用格式化函数
        def format_value(value, unit="", precision=2, is_percent=False, is_price=False):
            if pd.isna(value):
                return "N/A"
            if is_percent:
                return f"{value:+.{precision}f}{unit}"
            elif is_price:
                return f"{value:,.{precision}f}{unit}"
            else:
                return str(value)

        # 如果提供了股票标识符，查询特定股票
        if stock_identifier:
            resolved_stock_code = None
            resolved_stock_name = None

            # 判断 stock_identifier 是代码还是名称
            if stock_identifier.isdigit() and len(stock_identifier) == 6:
                resolved_stock_code = stock_identifier
                # 从已获取的DataFrame中找到名称
                matched_name_df = df[df['代码'] == resolved_stock_code]
                if not matched_name_df.empty:
                    resolved_stock_name = matched_name_df.iloc[0]['名称']
            else:
                # 通过名称查找股票代码
                # 优先精确匹配名称
                matched_stocks = df[df['名称'] == stock_identifier]

                # 如果没有精确匹配，模糊匹配
                if matched_stocks.empty:
                    matched_stocks = df[df['名称'].str.contains(stock_identifier, case=False, na=False)]

                if matched_stocks.empty:
                    return f"未能在日期 '{date}' 的机构调研数据中找到与 '{stock_identifier}' 匹配的股票。请检查输入是否正确或该股票在该日无调研数据。"
                elif len(matched_stocks) > 1:
                    output_lines = [f"在日期 '{date}' 找到多个与 '{stock_identifier}' 匹配的股票，请提供更具体的股票名称或直接提供股票代码："]
                    output_lines.append("=" * 100)
                    for index, row in matched_stocks.head(10).iterrows():
                        output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                    if len(matched_stocks) > 10:
                        output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
                    output_lines.append("=" * 100)
                    return "\n".join(output_lines)
                else:
                    resolved_stock_code = matched_stocks.iloc[0]['代码']
                    resolved_stock_name = matched_stocks.iloc[0]['名称']

            if not resolved_stock_code:
                return f"无法为 '{stock_identifier}' 解析出有效的股票代码。请确保输入正确或尝试直接提供股票代码。"

            # 过滤出指定股票的数据
            target_stock_data = df[df['代码'] == resolved_stock_code]

            if target_stock_data.empty:
                return f"在日期 '{date}' 未能找到股票 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 的机构调研详细数据。该股票在该日可能没有机构调研活动。"
            
            # 如果有多个调研记录，则全部显示
            output_lines = [f"股票 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 在日期 '{date}' 的机构调研详细数据 (数据来源: 东方财富网)。\n"]
            output_lines.append("=" * 100)

            # 定义要显示的字段
            display_fields = {
                '代码': '股票代码',
                '名称': '股票名称',
                '最新价': '最新价',
                '涨跌幅': '涨跌幅',
                '调研机构': '调研机构',
                '机构类型': '机构类型',
                '调研人员': '调研人员',
                '接待方式': '接待方式',
                '接待人员': '接待人员',
                '接待地点': '接待地点',
                '调研日期': '调研日期',
                '公告日期': '公告日期'
            }

            for i, row in target_stock_data.iterrows():
                output_lines.append(f"--- 调研记录 {i+1} ---")
                for col, display_name in display_fields.items():
                    if col in row.index:
                        value = row[col]
                        if col == '最新价':
                            formatted_value = format_value(value, precision=2, is_price=True)
                        elif col == '涨跌幅':
                            formatted_value = format_value(value, unit="%", precision=2, is_percent=True)
                        else:
                            formatted_value = str(value)
                        output_lines.append(f"{display_name}: {formatted_value}")
                output_lines.append("-" * 20)

            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{resolved_stock_name or stock_identifier}' 在日期 '{date}' 的机构调研详细数据。")

            return "\n".join(output_lines)

        # 如果未提供股票标识符，返回所有股票的摘要
        else:
            output_lines = [f"成功获取日期 '{date}' 共 {len(df)} 条机构调研详细数据 (数据来源: 东方财富)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前5行):")

            # 仅显示部分列以保持简洁
            display_cols = ['代码', '名称', '最新价', '涨跌幅', '调研机构', '调研日期', '公告日期']
            
            # 确保要显示的列都在 DataFrame 中
            display_cols = [col for col in display_cols if col in df.columns]

            # 应用格式化规则到要显示的列
            df_display_summary = df[display_cols].copy()
            if '最新价' in df_display_summary.columns:
                df_display_summary['最新价'] = df_display_summary['最新价'].apply(lambda x: format_value(x, precision=2, is_price=True))
            if '涨跌幅' in df_display_summary.columns:
                df_display_summary['涨跌幅'] = df_display_summary['涨跌幅'].apply(lambda x: format_value(x, unit="%", precision=2, is_percent=True))

            df_string = df_display_summary.head(5).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为日期 '{date}' 所有机构调研数据的概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询2024年12月11日平安银行的机构调研详情。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取机构调研详细数据时发生错误: {e}"

@tool
def get_news_report_time_baidu(date: str, stock_identifier: Optional[str] = None) -> str:
    """
    获取百度股市通提供的财报发行数据。
    该接口返回包括股票代码、交易所、股票简称和财报期等信息。
    主要提供港股的财报发行数据，也包含部分美股。

    用户必须提供一个查询日期（格式YYYYMMDD）。
    可选地，可以提供股票代码或名称来查询特定股票在指定日期的财报发行数据。
    如果未提供股票标识符，则返回该日期所有财报发行数据的摘要。

    Args:
        date (str): 必需参数。查询的日期，格式为"YYYYMMDD"，例如 "20241107"。
        stock_identifier (Optional[str]): 可选参数。要查询的股票代码（如 '00700' 或 'AAPL'）或股票名称（如 '腾讯控股' 或 '苹果'）。
                                          如果未提供，将返回该日期所有财报发行数据的摘要。

    Returns:
        str: 一个格式化后的字符串，包含财报发行数据。
             如果获取失败、未找到数据或找到多个匹配，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证日期格式
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return f"错误：日期格式不正确。请提供 'YYYYMMDD' 格式的日期，例如 '20241107'。"

    try:
        # 调用API
        df = ak.news_report_time_baidu(date=date)

        if df.empty:
            return f"在日期 '{date}' 未能获取到财报发行数据。可能该日期无数据或数据源暂无数据。"

        # 确保 '股票代码' 和 '股票简称' 列存在且为字符串类型
        if '股票代码' not in df.columns or '股票简称' not in df.columns:
            return "错误：获取到的财报发行数据缺少 '股票代码' 或 '股票简称' 列，无法进行查询或展示。"
        df['股票代码'] = df['股票代码'].astype(str)
        df['股票简称'] = df['股票简称'].astype(str)

        # 如果提供了股票标识符，查询特定股票
        if stock_identifier:
            resolved_stock_code = None
            resolved_stock_name = None

            # 判断 stock_identifier 是代码还是名称
            if stock_identifier.isdigit() and len(stock_identifier) >= 4:
                resolved_stock_code = stock_identifier
                # 从已获取的DataFrame中找到名称
                matched_name_df = df[df['股票代码'] == resolved_stock_code]
                if not matched_name_df.empty:
                    resolved_stock_name = matched_name_df.iloc[0]['股票简称']
            else: # 假设是股票名称或美股代码
                # 优先精确匹配名称
                matched_stocks = df[df['股票简称'] == stock_identifier]

                # 如果没有精确匹配，模糊匹配名称
                if matched_stocks.empty:
                    matched_stocks = df[df['股票简称'].str.contains(stock_identifier, case=False, na=False)]
                
                # 如果仍然没有匹配，精确匹配股票代码
                if matched_stocks.empty:
                    matched_stocks = df[df['股票代码'].str.upper() == stock_identifier.upper()]


                if matched_stocks.empty:
                    return f"未能在日期 '{date}' 的财报发行数据中找到与 '{stock_identifier}' 匹配的股票。请检查输入是否正确或该股票在该日无财报发行。"
                elif len(matched_stocks) > 1:
                    output_lines = [f"在日期 '{date}' 找到多个与 '{stock_identifier}' 匹配的股票，请提供更具体的股票名称或直接提供股票代码："]
                    output_lines.append("=" * 100)
                    for index, row in matched_stocks.head(10).iterrows():
                        output_lines.append(f"  - 代码: {row['股票代码']}, 简称: {row['股票简称']}, 交易所: {row['交易所']}")
                    if len(matched_stocks) > 10:
                        output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
                    output_lines.append("=" * 100)
                    return "\n".join(output_lines)
                else:
                    resolved_stock_code = matched_stocks.iloc[0]['股票代码']
                    resolved_stock_name = matched_stocks.iloc[0]['股票简称']

            if not resolved_stock_code:
                return f"无法为 '{stock_identifier}' 解析出有效的股票代码。请确保输入正确或尝试直接提供股票代码。"

            # 过滤出指定股票的数据
            target_stock_data = df[df['股票代码'] == resolved_stock_code]

            if target_stock_data.empty:
                return f"在日期 '{date}' 未能找到股票 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 的财报发行数据。该股票在该日可能没有财报发行。"
            
            # 如果有多个财报记录，则全部显示
            output_lines = [f"股票 '{resolved_stock_name or stock_identifier}' ({resolved_stock_code}) 在日期 '{date}' 的财报发行数据 (数据来源: 百度股市通)。\n"]
            output_lines.append("=" * 100)

            # 定义要显示的字段
            display_fields = {
                '股票代码': '股票代码',
                '交易所': '交易所',
                '股票简称': '股票简称',
                '财报期': '财报期'
            }

            for i, row in target_stock_data.iterrows():
                output_lines.append(f"--- 财报记录 {i+1} ---")
                for col, display_name in display_fields.items():
                    if col in row.index:
                        output_lines.append(f"{display_name}: {row[col]}")
                output_lines.append("-" * 20)

            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为股票 '{resolved_stock_name or stock_identifier}' 在日期 '{date}' 的财报发行数据。")

            return "\n".join(output_lines)

        # 如果未提供股票标识符，返回所有股票的摘要
        else:
            output_lines = [f"成功获取日期 '{date}' 共 {len(df)} 条财报发行数据 (数据来源: 百度股市通)。\n"]
            output_lines.append("=" * 100)
            output_lines.append("数据概览 (前5行):")

            # 仅显示部分列以保持简洁
            display_cols = ['股票代码', '股票简称', '交易所', '财报期']
            
            # 确保要显示的列都在 DataFrame 中
            display_cols = [col for col in display_cols if col in df.columns]

            df_string = df[display_cols].head(5).to_string(index=False)
            output_lines.append(df_string)
            output_lines.append("=" * 100)
            output_lines.append(f"提示：此为日期 '{date}' 所有财报发行数据的概览。")
            output_lines.append("如果您需要查询特定股票的详细信息，请提供其股票代码或名称。")
            output_lines.append("例如，您可以询问 '查询2024年11月7日腾讯控股的财报发行时间。' 或 '查询2024年11月7日苹果公司的财报发行时间。'")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取财报发行数据时发生错误: {e}"


@tool
def get_hk_financial_indicators_em(
        stock_identifier: str,
        indicator: str = "年度",
        selected_indicators: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取东方财富网-港股-财务分析-主要指标中，指定港股的财务指标历史数据，并可按报告日期范围过滤。
    该接口返回包括每股收益、营业收入、净利润、毛利率、净资产收益率、资产负债率等全面财务指标。

    用户必须提供一个港股代码或公司名称来查询其对应的财务指标。
    可选择查询“年度”数据或“报告期”（季度）数据，默认为“年度”。
    工具默认展示最近5期的所有关键财务指标，但用户可以通过提供 'selected_indicators' 参数来筛选，或通过日期参数来指定范围。

    Args:
        stock_identifier (str): 必需参数。要查询财务指标的港股代码（如 '00700'）或公司名称（如 '腾讯控股'）。
        indicator (str): 可选参数。指定查询的财务报表类型。可选值：
                         "年度" (默认值): 查询年度财务指标。
                         "报告期": 查询报告期（通常为季度）财务指标。
        selected_indicators (Optional[str]): 可选参数。指定要查询的财务指标列表，多个指标用逗号分隔。
                                             例如："营业收入,归母净利润,毛利率"。
                                             如果提供此参数，则只返回这些指标的信息。
                                             可用的指标包括：
                                             "基本每股收益", "稀释每股收益", "营业收入", "营业收入同比",
                                             "归母净利润", "归母净利润同比", "毛利率", "净利率",
                                             "平均净资产收益率", "总资产报酬率", "资产负债率",
                                             "流动比率", "每股营业收入", "每股净资产", "市盈率(TTM)"。
                                             报告日期和财年信息将始终包含在内（如果数据可用）。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYY-MM-DD'。
                                    将过滤“报告日期”(REPORT_DATE)在此日期或之后的数据。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYY-MM-DD'。
                                  将过滤“报告日期”(REPORT_DATE)在此日期或之前的数据。

    Returns:
        str: 一个格式化后的字符串，包含指定港股的财务指标历史数据。
             如果获取失败、未找到股票或找到多个匹配，则返回错误或提示信息。
    """
    # 检查与参数验证
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证 'indicator' 参数是否有效
    if indicator not in ["年度", "报告期"]:
        return f"错误：'indicator' 参数无效。请选择 '年度' 或 '报告期'。"

    # 日期参数解析与验证
    parsed_start_timestamp = None
    parsed_end_timestamp = None
    if start_date:
        try:
            parsed_start_timestamp = pd.to_datetime(start_date)
        except ValueError:
            return "错误：'start_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    if end_date:
        try:
            parsed_end_timestamp = pd.to_datetime(end_date)
        except ValueError:
            return "错误：'end_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"

    # 验证开始日期是否晚于结束日期
    if parsed_start_timestamp and parsed_end_timestamp and parsed_start_timestamp > parsed_end_timestamp:
        return "错误：'start_date' 不能晚于 'end_date'。"

    resolved_symbol = None
    resolved_name = None

    # 股票代码或名称解析
    # 从港股列表中解析用户输入的 stock_identifier
    try:
        # 获取东方财富网港股实时行情数据，用于查找股票
        hk_spot_df = ak.stock_hk_spot_em()
        if hk_spot_df.empty:
            return "错误：无法获取港股列表以解析股票名称。请尝试直接提供股票代码。"

        # 确保 DataFrame 包含 '代码' 和 '名称' 列，并转换为字符串类型
        if '代码' not in hk_spot_df.columns or '名称' not in hk_spot_df.columns:
            return "错误：获取到的港股列表数据缺少 '代码' 或 '名称' 列，无法解析股票名称。"
        hk_spot_df['代码'] = hk_spot_df['代码'].astype(str)
        hk_spot_df['名称'] = hk_spot_df['名称'].astype(str)

        # 优先精确匹配公司名称
        matched_stocks = hk_spot_df[hk_spot_df['名称'] == stock_identifier]

        # 如果没有精确匹配，则进行模糊匹配
        if matched_stocks.empty:
            matched_stocks = hk_spot_df[hk_spot_df['名称'].str.contains(stock_identifier, case=False, na=False)]

        # 如果名称匹配不到，精确匹配股票代码
        if matched_stocks.empty and stock_identifier.isdigit() and len(stock_identifier) >= 4:
            matched_stocks = hk_spot_df[hk_spot_df['代码'] == stock_identifier]

        # 处理匹配结果
        if matched_stocks.empty:
            return f"未能在港股列表中找到与 '{stock_identifier}' 匹配的股票代码。请检查股票名称是否正确或尝试直接提供股票代码。"
        elif len(matched_stocks) > 1:
            # 如果找到多个匹配项，提示用户提供更具体的输入
            output_lines = [f"找到多个与 '{stock_identifier}' 匹配的港股，请提供更具体的股票名称或直接提供股票代码："]
            output_lines.append("=" * 100)
            for index, row in matched_stocks.head(10).iterrows(): # 只显示前10个匹配项
                output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
            if len(matched_stocks) > 10:
                output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
            output_lines.append("=" * 100)
            return "\n".join(output_lines)
        else:
            # 成功解析出唯一的股票代码和名称
            resolved_symbol = matched_stocks.iloc[0]['代码']
            resolved_name = matched_stocks.iloc[0]['名称']

    except Exception as e:
        return f"解析股票名称 '{stock_identifier}' 时发生错误: {e}。请尝试直接提供股票代码。"

    # 如果未能解析出有效的股票代码，则返回错误
    if not resolved_symbol:
        return f"无法为 '{stock_identifier}' 解析出有效的港股代码。请确保输入正确或尝试直接提供股票代码。"

    # 获取财务指标数据
    try:
        # 调用 akshare 接口获取指定港股的财务分析主要指标数据
        df = ak.stock_financial_hk_analysis_indicator_em(symbol=resolved_symbol, indicator=indicator)

        # 检查返回的 DataFrame 是否为空
        if df.empty:
            return f"未能获取港股 '{resolved_name or stock_identifier}' ({resolved_symbol}) 的{indicator}财务指标数据。可能无数据或数据源暂无数据。"

        # 确保关键日期列存在，否则无法进行后续处理
        if 'REPORT_DATE' not in df.columns and 'FISCAL_YEAR' not in df.columns:
            return "错误：获取到的财务指标数据缺少 'REPORT_DATE' 或 'FISCAL_YEAR' 列，无法进行展示。"

        # 初始化输出列表，添加标题
        output_lines = [
            f"港股 '{resolved_name or stock_identifier}' ({resolved_symbol}) 的{indicator}财务指标历史数据 (数据来源: 东方财富网)。\n"]
        output_lines.append("=" * 100)

        # 日期范围过滤
        date_filter_active = parsed_start_timestamp or parsed_end_timestamp
        if date_filter_active:
            if 'REPORT_DATE' not in df.columns:
                return "错误：数据中缺少 'REPORT_DATE' 列，无法按日期范围进行过滤。"

            # 将 'REPORT_DATE' 列转换为 datetime 类型
            df['REPORT_DATE_dt'] = pd.to_datetime(df['REPORT_DATE'], errors='coerce')

            if parsed_start_timestamp:
                df = df[df['REPORT_DATE_dt'] >= parsed_start_timestamp].copy()
                output_lines.append(f"已过滤，报告日期从 {start_date} 开始。")
            if parsed_end_timestamp:
                df = df[df['REPORT_DATE_dt'] <= parsed_end_timestamp].copy()
                output_lines.append(f"已过滤，报告日期到 {end_date} 结束。")

            # 移除辅助列
            df = df.drop(columns=['REPORT_DATE_dt'])

            # 如果日期过滤后数据为空，则返回提示
            if df.empty:
                date_range_info = ""
                if start_date and end_date:
                    date_range_info = f"在 {start_date} 到 {end_date} 期间"
                elif start_date:
                    date_range_info = f"从 {start_date} 开始"
                elif end_date:
                    date_range_info = f"到 {end_date} 结束"
                return f"在指定日期范围 {date_range_info} 内，未能找到港股 '{resolved_name}' 的财务指标数据。"

        # 定义数值格式化辅助函数
        def format_numeric_value(value, unit="", precision=2, is_percentage=False, is_large_money=False):
            """
            辅助函数，用于格式化数值，处理NaN、百分比和大额金额显示。
            """
            if pd.isna(value):
                return "N/A"
            if is_percentage:
                return f"{value:+.{precision}f}{unit}"
            elif is_large_money:
                # 针对大额金额进行单位转换（亿、万）
                if abs(value) >= 1_0000_0000:
                    return f"{value / 1_0000_0000:,.{precision}f}亿{unit}"
                elif abs(value) >= 1_0000:
                    return f"{value / 1_0000:,.{precision}f}万{unit}"
                else:
                    return f"{value:,.{precision}f}{unit}"
            else:
                return f"{value:,.{precision}f}{unit}"

        # 指标映射与选择
        # 定义指标名称与 DataFrame 实际列名及格式化函数的映射
        indicator_mapping = {
            '报告日期': {'col': 'REPORT_DATE', 'format': lambda x: str(x) if pd.notna(x) else "N/A"},
            '财年': {'col': 'FISCAL_YEAR', 'format': lambda x: str(x) if pd.notna(x) else "N/A"},
            '基本每股收益': {'col': 'BASIC_EPS', 'format': lambda x: format_numeric_value(x, precision=4)},
            '稀释每股收益': {'col': 'DILUTED_EPS', 'format': lambda x: format_numeric_value(x, precision=4)},
            '营业收入': {'col': 'OPERATE_INCOME',
                         'format': lambda x: format_numeric_value(x, '港元', 2, is_large_money=True)},
            '营业收入同比': {'col': 'OPERATE_INCOME_YOY',
                             'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '归母净利润': {'col': 'HOLDER_PROFIT',
                           'format': lambda x: format_numeric_value(x, '港元', 2, is_large_money=True)},
            '归母净利润同比': {'col': 'HOLDER_PROFIT_YOY',
                               'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '毛利率': {'col': 'GROSS_PROFIT_RATIO',
                       'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '净利率': {'col': 'NET_PROFIT_RATIO',
                       'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '平均净资产收益率': {'col': 'ROE_AVG',
                                 'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '总资产报酬率': {'col': 'ROA', 'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '资产负债率': {'col': 'DEBT_ASSET_RATIO',
                           'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '流动比率': {'col': 'CURRENT_RATIO', 'format': lambda x: format_numeric_value(x, precision=2)},
            '每股营业收入': {'col': 'PER_OI', 'format': lambda x: format_numeric_value(x, precision=2)},
            '每股净资产': {'col': 'BPS', 'format': lambda x: format_numeric_value(x, precision=2)},
            '市盈率(TTM)': {'col': 'EPS_TTM', 'format': lambda x: format_numeric_value(x, precision=2)},
        }
        # 显示的关键指标列表
        default_key_indicators = [
            '基本每股收益', '营业收入', '营业收入同比', '归母净利润', '归母净利润同比',
            '毛利率', '净利率', '平均净资产收益率', '资产负债率', '流动比率', '市盈率(TTM)'
        ]

        # 最终要显示的指标列表
        final_display_items: List[tuple] = []
        # 判断用户是否自定义了指标
        is_custom_selection = bool(selected_indicators)

        essential_fields_names = ['报告日期', '财年']
        for name in essential_fields_names:
            col_name = indicator_mapping[name]['col']
            if col_name in df.columns:
                final_display_items.append((name, indicator_mapping[name]))

        if is_custom_selection:
            # 如果用户自定义了指标，则解析并添加
            requested_friendly_names = [name.strip() for name in selected_indicators.split(',') if name.strip()]
            # 标记是否找到用户请求的非日期指标
            financial_indicators_found_in_request = False
            for name in requested_friendly_names:
                if name in indicator_mapping:
                    col_name = indicator_mapping[name]['col']
                    # 避免重复添加，并确保列存在于数据中
                    if name not in [item[0] for item in final_display_items] and col_name in df.columns:
                        final_display_items.append((name, indicator_mapping[name]))
                        # 如果是财务指标而不是日期/财年
                        if name not in essential_fields_names:
                            financial_indicators_found_in_request = True
                else:
                    output_lines.append(f"警告：未识别的财务指标 '{name}' 将被忽略。")
            # 如果用户指定了指标但没有找到任何有效的财务指标，则返回错误
            if not financial_indicators_found_in_request and len(requested_friendly_names) > 0 and not all(
                    name in essential_fields_names for name in requested_friendly_names):
                output_lines.append("错误：您指定的财务指标均无效或未找到。")
                output_lines.append("可用的财务指标包括：" + ", ".join(
                    [k for k in indicator_mapping.keys() if k not in essential_fields_names]))
                return "\n".join(output_lines)
        else:
            # 如果未自定义指标，则添加默认的关键指标
            for name in default_key_indicators:
                if name in indicator_mapping:
                    col_name = indicator_mapping[name]['col']
                    if name not in [item[0] for item in final_display_items] and col_name in df.columns:
                        final_display_items.append((name, indicator_mapping[name]))

        # 如果最终没有确定任何要显示的指标，则返回错误
        if not final_display_items:
            return f"错误：未能确定要显示的财务指标。请检查您的输入或数据源。"

        # 数据排序
        df_sorted = df.copy()
        if 'REPORT_DATE' in df_sorted.columns:
            # 按 'REPORT_DATE' 降序排列
            df_sorted['REPORT_DATE_PARSED'] = pd.to_datetime(df_sorted['REPORT_DATE'], errors='coerce')
            df_sorted = df_sorted.sort_values(by='REPORT_DATE_PARSED', ascending=False).drop(
                columns='REPORT_DATE_PARSED')
        elif 'FISCAL_YEAR' in df_sorted.columns:
            # 如果没有 'REPORT_DATE'，则按 'FISCAL_YEAR' 降序排列
            df_sorted = df_sorted.sort_values(by='FISCAL_YEAR', ascending=False)

        # 确定要遍历的数据行
        # 如果有日期过滤，则显示所有过滤后的结果；否则，默认显示最近5期
        df_to_display = df_sorted if date_filter_active else df_sorted.head(5)

        if df_to_display.empty:
            output_lines.append("在当前条件下未找到可显示的财务数据记录。")
        else:
            output_lines.append(f"总共找到 {len(df_to_display)} 条记录，将全部显示：")
            output_lines.append("-" * 20)

        # 遍历数据并格式化输出
        for i, row in df_to_display.iterrows():
            report_date_str = row.get('REPORT_DATE', None)
            fiscal_year_str = row.get('FISCAL_YEAR', None)
            report_display_name = '未知报告期'

            # 解析报告日期
            if pd.notna(report_date_str):
                try:
                    report_date_obj = pd.to_datetime(report_date_str, errors='coerce')
                    if pd.notna(report_date_obj):
                        report_display_name = report_date_obj.strftime('%Y年%m月%d日')
                    else:
                        report_display_name = str(report_date_str)
                except Exception:
                    report_display_name = str(report_date_str)
            elif pd.notna(fiscal_year_str):
                report_display_name = f"财年 {fiscal_year_str}"

            output_lines.append(f"--- 报告期: {report_display_name} ---")
            # 标记当前报告期是否找到有效数据
            found_data_for_period = False
            for friendly_name, info in final_display_items:
                col_name = info['col']
                # 检查列是否存在且值不为 NaN
                if col_name in row.index and pd.notna(row[col_name]):
                    formatted_value = info['format'](row[col_name])
                    output_lines.append(f"{friendly_name}: {formatted_value}")
                    found_data_for_period = True
                # 如果是自定义选择的指标，且该指标在该报告期没有数据，则明确提示
                elif is_custom_selection and friendly_name not in essential_fields_names:
                    output_lines.append(f"{friendly_name}: 未找到数据")

            if not found_data_for_period:
                output_lines.append("  (该报告期未能找到任何指定指标的数据或数据不完整。)")
            output_lines.append("-" * 20)

        output_lines.append("=" * 100)

        # 生成最终提示信息
        if date_filter_active:
            summary_period_text = "指定日期范围内的"
        else:
            summary_period_text = f"最近{len(df_to_display)}期"

        if is_custom_selection:
            output_lines.append(
                f"提示：此为港股 '{resolved_name or stock_identifier}' ({resolved_symbol}) {summary_period_text}{indicator}的指定财务指标数据。")
        else:
            output_lines.append(
                f"提示：此为港股 '{resolved_name or stock_identifier}' ({resolved_symbol}) {summary_period_text}{indicator}的关键财务指标数据。")
        output_lines.append("数据源：东方财富网。如需查询特定指标，请明确提出。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股 '{resolved_symbol}' ({resolved_name or stock_identifier}) 财务指标信息时发生错误: {e}"


@tool
def get_us_financial_indicators_em(
        stock_identifier: str,
        indicator: str = "年报",
        selected_indicators: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取东方财富网-美股-财务分析-主要指标中，指定美股的财务指标历史数据，并可按报告日期范围过滤。
    该接口返回包括每股收益、营业收入、净利润、毛利率、净资产收益率、资产负债率等全面财务指标。

    用户必须提供一个美股代码或公司名称来查询其对应的财务指标。
    可选择查询“年报”、“单季报”或“累计季报”数据，默认为“年报”。
    工具默认展示最近5期的所有关键财务指标，但用户可以通过提供 'selected_indicators' 参数筛选，或通过日期参数指定范围。

    Args:
        stock_identifier (str): 必需参数。要查询财务指标的美股代码（如 'TSLA'）或公司名称（如 '特斯拉'）。
        indicator (str): 可选参数。指定查询的财务报表类型。可选值：
                         "年报" (默认值): 查询年度财务指标。
                         "单季报": 查询单个季度财务指标。
                         "累计季报": 查询累计季度（如Q1, Q2, Q3, Q4累计）财务指标。
        selected_indicators (Optional[str]): 可选参数。指定要查询的财务指标列表，多个指标用逗号分隔。
                                             例如："营业收入,归母净利润,毛利率"。
                                             如果提供此参数，则只返回这些指标的信息。
                                             可用的指标包括：
                                             "基本每股收益", "稀释每股收益", "营业收入", "营业收入同比",
                                             "归母净利润", "归母净利润同比", "毛利率", "净利率",
                                             "平均净资产收益率", "总资产报酬率", "资产负债率",
                                             "流动比率", "速动比率"。
                                             报告日期、财报截止日期和货币信息将始终包含在内（如果数据可用）。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYY-MM-DD'。
                                    将过滤“报告日期”(REPORT_DATE)在此日期或之后的数据。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYY-MM-MM'。
                                  将过滤“报告日期”(REPORT_DATE)在此日期或之前的数据。

    Returns:
        str: 一个格式化后的字符串，包含指定美股的财务指标历史数据。
             如果获取失败、未找到股票或找到多个匹配，则返回错误或提示信息。
    """
    # 检查与参数验证
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证 'indicator' 参数是否有效
    if indicator not in ["年报", "单季报", "累计季报"]:
        return f"错误：'indicator' 参数无效。请选择 '年报', '单季报' 或 '累计季报'。"

    # 日期参数解析与验证
    parsed_start_timestamp = None
    parsed_end_timestamp = None
    if start_date:
        try:
            parsed_start_timestamp = pd.to_datetime(start_date)
        except ValueError:
            return "错误：'start_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    if end_date:
        try:
            parsed_end_timestamp = pd.to_datetime(end_date)
        except ValueError:
            return "错误：'end_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"

    # 验证开始日期是否晚于结束日期
    if parsed_start_timestamp and parsed_end_timestamp and parsed_start_timestamp > parsed_end_timestamp:
        return "错误：'start_date' 不能晚于 'end_date'。"

    resolved_symbol = None
    resolved_name = None

    # 股票代码或名称解析
    # 从美股列表中解析用户输入的 stock_identifier
    try:
        # 获取东方财富网美股实时行情数据，用于查找股票
        us_spot_df = ak.stock_us_spot_em()
        if us_spot_df.empty:
            return "错误：无法获取美股列表以解析股票名称。请尝试直接提供股票代码。"

        # 确保 DataFrame 包含 '代码' 和 '名称' 列，并转换为字符串类型
        if '代码' not in us_spot_df.columns or '名称' not in us_spot_df.columns:
            return "错误：获取到的美股列表数据缺少 '代码' 或 '名称' 列，无法解析股票名称。"
        us_spot_df['代码'] = us_spot_df['代码'].astype(str).str.upper()
        us_spot_df['名称'] = us_spot_df['名称'].astype(str)

        # 优先精确匹配股票代码
        matched_by_code = us_spot_df[us_spot_df['代码'] == stock_identifier.upper()]
        if not matched_by_code.empty:
            resolved_symbol = matched_by_code.iloc[0]['代码']
            resolved_name = matched_by_code.iloc[0]['名称']
        else:
            # 如果代码匹配不到，精确匹配公司名称
            matched_by_name = us_spot_df[us_spot_df['名称'] == stock_identifier]
            if not matched_by_name.empty:
                resolved_symbol = matched_by_name.iloc[0]['代码']
                resolved_name = matched_by_name.iloc[0]['名称']
            else:
                # 如果精确名称匹配不到，进行模糊匹配
                fuzzy_matched_by_name = us_spot_df[
                    us_spot_df['名称'].str.contains(stock_identifier, case=False, na=False)]
                if not fuzzy_matched_by_name.empty:
                    if len(fuzzy_matched_by_name) > 1:
                        # 如果找到多个匹配项，提示用户提供更具体的输入
                        output_lines = [
                            f"找到多个与 '{stock_identifier}' 匹配的美股，请提供更具体的股票名称或直接提供股票代码："]
                        output_lines.append("=" * 100)
                        # 只显示前10个匹配项
                        for index, row in fuzzy_matched_by_name.head(10).iterrows():
                            output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                        if len(fuzzy_matched_by_name) > 10:
                            output_lines.append(f"  ... (还有 {len(fuzzy_matched_by_name) - 10} 个更多匹配)")
                        output_lines.append("=" * 100)
                        return "\n".join(output_lines)
                    else:
                        # 成功解析出唯一的股票代码和名称
                        resolved_symbol = fuzzy_matched_by_name.iloc[0]['代码']
                        resolved_name = fuzzy_matched_by_name.iloc[0]['名称']
                else:
                    # 如果所有匹配都失败，则假定用户输入的就是股票代码
                    resolved_symbol = stock_identifier.upper()
                    resolved_name = stock_identifier
    except Exception as e:
        # 如果解析过程中发生错误，为了不中断流程，直接使用用户输入作为股票代码和名称
        resolved_symbol = stock_identifier.upper()
        resolved_name = stock_identifier

    # 如果未能解析出有效的股票代码，则返回错误
    if not resolved_symbol:
        return f"无法为 '{stock_identifier}' 解析出有效的股票代码。请确保输入正确或尝试直接提供股票代码。"

    try:
        # 调用API
        df = ak.stock_financial_us_analysis_indicator_em(symbol=resolved_symbol, indicator=indicator)

        # 检查返回的 DataFrame 是否为空
        if df.empty:
            return f"未能获取美股 '{resolved_name or stock_identifier}' ({resolved_symbol}) 的{indicator}财务指标数据。可能股票代码无效、无数据或数据源暂无数据。"

        # 确保关键日期列存在
        if 'REPORT_DATE' not in df.columns and 'FINANCIAL_DATE' not in df.columns:
            return "错误：获取到的财务指标数据缺少 'REPORT_DATE' 或 'FINANCIAL_DATE' 列，无法进行展示。"

        # 初始化输出列表，添加标题
        output_lines = [
            f"美股 '{resolved_name or stock_identifier}' ({resolved_symbol}) 的{indicator}财务指标历史数据 (数据来源: 东方财富网)。\n"]
        output_lines.append("=" * 100)

        # 日期范围过滤
        date_filter_active = parsed_start_timestamp or parsed_end_timestamp
        if date_filter_active:
            if 'REPORT_DATE' not in df.columns:
                return "错误：数据中缺少 'REPORT_DATE' 列，无法按日期范围进行过滤。"

            # 将 'REPORT_DATE' 列转换为 datetime 类型
            df['REPORT_DATE_dt'] = pd.to_datetime(df['REPORT_DATE'], errors='coerce')
            if parsed_start_timestamp:
                df = df[df['REPORT_DATE_dt'] >= parsed_start_timestamp].copy()
                output_lines.append(f"已过滤，报告日期从 {start_date} 开始。")
            if parsed_end_timestamp:
                df = df[df['REPORT_DATE_dt'] <= parsed_end_timestamp].copy()
                output_lines.append(f"已过滤，报告日期到 {end_date} 结束。")
            # 移除辅助列
            df = df.drop(columns=['REPORT_DATE_dt'])

            # 如果日期过滤后数据为空
            if df.empty:
                date_range_info = ""
                if start_date and end_date:
                    date_range_info = f"在 {start_date} 到 {end_date} 期间"
                elif start_date:
                    date_range_info = f"从 {start_date} 开始"
                elif end_date:
                    date_range_info = f"到 {end_date} 结束"
                return f"在指定日期范围 {date_range_info} 内，未能找到美股 '{resolved_name}' 的财务指标数据。"

        # 定义数值格式化辅助函数
        def format_numeric_value(value, unit="", precision=2, is_percentage=False, is_large_money=False):
            """
            辅助函数，用于格式化数值，处理NaN、百分比和大额金额显示。
            """
            if pd.isna(value):
                return "N/A"
            if is_percentage:
                return f"{value:+.{precision}f}{unit}"
            elif is_large_money:
                # 针对大额金额进行单位转换（亿、万）
                if abs(value) >= 1_0000_0000:
                    return f"{value / 1_0000_0000:,.{precision}f}亿美元"
                elif abs(value) >= 1_0000:
                    return f"{value / 1_0000:,.{precision}f}万美元"
                else:
                    return f"{value:,.{precision}f}{unit}"
            else:
                return f"{value:,.{precision}f}{unit}"

        # 指标映射与选择
        # 指标名称与 DataFrame 实际列名及格式化函数的映射
        indicator_mapping = {
            '报告日期': {'col': 'REPORT_DATE', 'format': lambda x: str(x) if pd.notna(x) else "N/A"},
            '财报截止日期': {'col': 'FINANCIAL_DATE', 'format': lambda x: str(x) if pd.notna(x) else "N/A"},
            '货币': {'col': 'CURRENCY', 'format': lambda x: str(x) if pd.notna(x) else "N/A"},
            '基本每股收益': {'col': 'BASIC_EPS', 'format': lambda x: format_numeric_value(x, precision=4)},
            '稀释每股收益': {'col': 'DILUTED_EPS', 'format': lambda x: format_numeric_value(x, precision=4)},
            '营业收入': {'col': 'OPERATE_INCOME',
                         'format': lambda x: format_numeric_value(x, precision=2, is_large_money=True)},
            '营业收入同比': {'col': 'OPERATE_INCOME_YOY',
                             'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '归母净利润': {'col': 'PARENT_HOLDER_NETPROFIT',
                           'format': lambda x: format_numeric_value(x, precision=2, is_large_money=True)},
            '归母净利润同比': {'col': 'PARENT_HOLDER_NETPROFIT_YOY',
                               'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '毛利率': {'col': 'GROSS_PROFIT_RATIO',
                       'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '净利率': {'col': 'NET_PROFIT_RATIO',
                       'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '平均净资产收益率': {'col': 'ROE_AVG',
                                 'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '总资产报酬率': {'col': 'ROA', 'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '资产负债率': {'col': 'DEBT_ASSET_RATIO',
                           'format': lambda x: format_numeric_value(x, '%', 2, is_percentage=True)},
            '流动比率': {'col': 'CURRENT_RATIO', 'format': lambda x: format_numeric_value(x, precision=2)},
            '速动比率': {'col': 'SPEED_RATIO', 'format': lambda x: format_numeric_value(x, precision=2)},
        }
        # 显示的关键指标列表
        default_key_indicators = [
            '基本每股收益', '营业收入', '营业收入同比', '归母净利润', '归母净利润同比',
            '毛利率', '净利率', '平均净资产收益率', '资产负债率'
        ]
        # 最终要显示的指标列表
        final_display_items: List[tuple] = []
        # 判断用户是否自定义了指标
        is_custom_selection = bool(selected_indicators)

        essential_fields = ['报告日期', '财报截止日期', '货币']
        for name in essential_fields:
            if name in indicator_mapping and indicator_mapping[name]['col'] in df.columns:
                final_display_items.append((name, indicator_mapping[name]))

        if is_custom_selection:
            # 如果用户自定义了指标，则解析并添加
            requested_friendly_names = [name.strip() for name in selected_indicators.split(',') if name.strip()]
            # 标记是否找到用户请求的非日期/货币指标
            financial_indicators_found = False
            for name in requested_friendly_names:
                if name in indicator_mapping and name not in [item[0] for item in final_display_items]:
                    col_name = indicator_mapping[name]['col']
                    # 确保列存在于数据中
                    if col_name in df.columns:
                        final_display_items.append((name, indicator_mapping[name]))
                        if name not in essential_fields:
                            financial_indicators_found = True
                elif name not in indicator_mapping:
                    output_lines.append(f"警告：未识别的指标 '{name}' 将被忽略。")
            # 如果用户指定了指标但没有找到任何有效的财务指标，则返回错误
            if not financial_indicators_found and any(n not in essential_fields for n in requested_friendly_names):
                output_lines.append("错误：您指定的财务指标均无效或未找到。")
                output_lines.append("可用的财务指标包括：" + ", ".join(
                    [k for k in indicator_mapping.keys() if k not in essential_fields]))
                return "\n".join(output_lines)
        else:
            # 如果未自定义指标，则添加默认的关键指标
            for name in default_key_indicators:
                if name in indicator_mapping and name not in [item[0] for item in final_display_items]:
                    col_name = indicator_mapping[name]['col']
                    if col_name in df.columns:
                        final_display_items.append((name, indicator_mapping[name]))

        # 数据排序
        df_sorted = df.copy()
        if 'REPORT_DATE' in df_sorted.columns:
            # 按 'REPORT_DATE' 降序排列
            df_sorted['REPORT_DATE_PARSED'] = pd.to_datetime(df_sorted['REPORT_DATE'], errors='coerce')
            df_sorted = df_sorted.sort_values(by='REPORT_DATE_PARSED', ascending=False).drop(
                columns='REPORT_DATE_PARSED')

        # 确定要遍历的数据行
        # 如果有日期过滤，则显示所有过滤后的结果；否则，默认显示最近5期
        df_to_display = df_sorted if date_filter_active else df_sorted.head(5)

        if df_to_display.empty:
            output_lines.append("在当前条件下未找到可显示的财务数据记录。")
        else:
            if date_filter_active:
                output_lines.append(f"总共找到 {len(df_to_display)} 条记录，将全部显示：")
            output_lines.append("-" * 20)

        # 遍历数据并格式化输出
        for i, row in df_to_display.iterrows():
            report_date_display = row.get('REPORT_DATE', 'N/A')
            if report_date_display != 'N/A':
                try:
                    # 将报告日期格式化为中文日期格式
                    report_date_display = pd.to_datetime(report_date_display).strftime('%Y年%m月%d日')
                except:
                    pass
            output_lines.append(f"--- 报告期: {report_date_display} ---")
            # 标记当前报告期是否找到有效数据
            found_data_for_period = False
            for friendly_name, info in final_display_items:
                col_name = info['col']
                # 检查列是否存在且值不为 NaN
                if col_name in row.index and pd.notna(row[col_name]):
                    formatted_value = info['format'](row[col_name])
                    output_lines.append(f"{friendly_name}: {formatted_value}")
                    found_data_for_period = True
                # 如果是自定义选择的指标，且该指标在该报告期没有数据，则明确提示
                elif is_custom_selection and friendly_name not in essential_fields:
                    output_lines.append(f"{friendly_name}: 未找到数据")

            if not found_data_for_period and len(final_display_items) > len(
                    essential_fields):
                output_lines.append("  (该报告期未能找到任何指定指标的数据或数据不完整。)")
            output_lines.append("-" * 20)

        output_lines.append("=" * 100)

        # 生成最终提示信息
        if date_filter_active:
            summary_period_text = "指定日期范围内的"
        else:
            summary_period_text = f"最近{len(df_to_display)}期"

        if is_custom_selection:
            output_lines.append(
                f"提示：此为美股 '{resolved_name or stock_identifier}' ({resolved_symbol}) {summary_period_text}{indicator}的指定财务指标数据。")
        else:
            output_lines.append(
                f"提示：此为美股 '{resolved_name or stock_identifier}' ({resolved_symbol}) {summary_period_text}{indicator}的关键财务指标数据。")
        output_lines.append("数据源：东方财富网。如需查询特定指标，请明确提出。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取美股 '{resolved_symbol}' ({resolved_name or stock_identifier}) 财务指标信息时发生错误: {e}"


@tool
def get_hk_financial_report_em(
       stock_identifier: str,
       report_type: str,
       indicator: str = "年度",
       balance_sheet_items: Optional[str] = None,
       income_statement_items: Optional[str] = None,
       cash_flow_statement_items: Optional[str] = None,
       start_date: Optional[str] = None,
       end_date: Optional[str] = None
) -> str:
    """
    获取东方财富网-港股-财务报表-三大报表（资产负债表、利润表、现金流量表）的详细数据，并可按报告日期范围过滤。
    该接口返回指定港股在特定报告期（年度或季度）的完整财务报表数据。

    用户必须提供一个港股代码或公司名称，以及要查询的报表类型。
    可选择查询“年度”数据或“报告期”（季度）数据，默认为“年度”。
    工具默认仅展示最近3个报告期的关键科目摘要，但用户可以通过提供特定科目参数来筛选，或通过日期参数指定范围。

    Args:
        stock_identifier (str): 必需参数。要查询财务报表的港股代码（如 '00700'）或公司名称（如 '腾讯控股'）。
        report_type (str): 必需参数。要查询的报表类型。可选值：
                           "资产负债表"
                           "利润表"
                           "现金流量表"
        indicator (str): 可选参数。指定查询的财务报表类型。可选值：
                         "年度" (默认值): 查询年度财务报表。
                         "报告期": 查询报告期（通常为季度）财务报表。
        balance_sheet_items (Optional[str]): 可选参数。仅当 report_type 为 "资产负债表" 时，指定要查询的科目列表。
        income_statement_items (Optional[str]): 可选参数。仅当 report_type 为 "利润表" 时，指定要查询的科目列表。
        cash_flow_statement_items (Optional[str]): 可选参数。仅当 report_type 为 "现金流量表" 时，指定要查询的科目列表。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYY-MM-DD'。
                                    将过滤“报告日期”(REPORT_DATE)在此日期或之后的数据。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYY-MM-DD'。
                                  将过滤“报告日期”(REPORT_DATE)在此日期或之前的数据。

    Returns:
        str: 一个格式化后的字符串，包含指定港股的财务报表历史数据。
             如果获取失败、未找到股票或找到多个匹配，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 验证 report_type 参数是否有效
    if report_type not in ["资产负债表", "利润表", "现金流量表"]:
        return f"错误：'report_type' 参数无效。请选择 '资产负债表', '利润表' 或 '现金流量表'。"
    # 验证 indicator 参数是否有效
    if indicator not in ["年度", "报告期"]:
        return f"错误：'indicator' 参数无效。请选择 '年度' 或 '报告期'。"

    # 日期参数解析与验证
    parsed_start_timestamp = None
    parsed_end_timestamp = None
    if start_date:
        try:
            parsed_start_timestamp = pd.to_datetime(start_date)
        except ValueError:
            return "错误：'start_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    if end_date:
        try:
            parsed_end_timestamp = pd.to_datetime(end_date)
        except ValueError:
            return "错误：'end_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    # 验证 start_date 是否晚于 end_date
    if parsed_start_timestamp and parsed_end_timestamp and parsed_start_timestamp > parsed_end_timestamp:
        return "错误：'start_date' 不能晚于 'end_date'。"

    # 解析股票代码或名称
    # 通过股票名称或代码解析出唯一的港股代码和名称
    resolved_symbol, resolved_name = None, None
    try:
        # 获取港股实时行情数据，用于匹配股票
        hk_spot_df = ak.stock_hk_spot_em()
        if hk_spot_df.empty:
            return "错误：无法获取港股列表以解析股票名称。请尝试直接提供股票代码。"
        if '代码' not in hk_spot_df.columns or '名称' not in hk_spot_df.columns:
            return "错误：获取到的港股列表数据缺少 '代码' 或 '名称' 列。"
        # 确保代码和名称列为字符串类型以便进行匹配
        hk_spot_df['代码'] = hk_spot_df['代码'].astype(str)
        hk_spot_df['名称'] = hk_spot_df['名称'].astype(str)

        # 优先通过精确名称匹配
        matched_stocks = hk_spot_df[hk_spot_df['名称'] == stock_identifier]
        # 如果没有精确匹配，通过包含关系匹配名称
        if matched_stocks.empty:
            matched_stocks = hk_spot_df[hk_spot_df['名称'].str.contains(stock_identifier, case=False, na=False)]
        # 如果仍未匹配，通过代码匹配
        if matched_stocks.empty and stock_identifier.isdigit() and len(stock_identifier) >= 4:
            matched_stocks = hk_spot_df[hk_spot_df['代码'] == stock_identifier]

        # 处理匹配结果
        if matched_stocks.empty:
            return f"未能在港股列表中找到与 '{stock_identifier}' 匹配的股票。"
        elif len(matched_stocks) > 1:
            # 如果找到多个匹配，提示用户提供更具体的信息
            output_lines = [f"找到多个与 '{stock_identifier}' 匹配的港股，请提供更具体的股票名称或直接提供股票代码："]
            output_lines.extend(
                [f"  - 代码: {row['代码']}, 名称: {row['名称']}" for _, row in matched_stocks.head(10).iterrows()])
            if len(matched_stocks) > 10:
                output_lines.append(f"  ... (还有 {len(matched_stocks) - 10} 个更多匹配)")
            return "\n".join(output_lines)
        else:
            # 找到唯一匹配，提取代码和名称
            resolved_symbol = matched_stocks.iloc[0]['代码']
            resolved_name = matched_stocks.iloc[0]['名称']
    except Exception as e:
        return f"解析股票名称 '{stock_identifier}' 时发生错误: {e}。请尝试直接提供股票代码。"
    # 如果最终未能解析出股票代码，则返回错误
    if not resolved_symbol:
        return f"无法为 '{stock_identifier}' 解析出有效的港股代码。"

    try:
        # 调用API
        df = ak.stock_financial_hk_report_em(stock=resolved_symbol, symbol=report_type, indicator=indicator)
        if df.empty:
            return f"未能获取港股 '{resolved_name}' ({resolved_symbol}) 的{indicator}{report_type}数据。"
        # 检查返回数据是否包含必要的列
        if not all(col in df.columns for col in ['REPORT_DATE', 'STD_ITEM_NAME', 'AMOUNT']):
            return "错误：获取到的财务报表数据缺少关键列，无法进行展示。"

        # 初始化输出行列表
        output_lines = [
            f"港股 '{resolved_name}' ({resolved_symbol}) 的{indicator}{report_type}数据 (来源: 东方财富网)。\n"]
        output_lines.append("=" * 100)

        # 日期过滤
        # 检查是否有日期过滤条件
        date_filter_active = parsed_start_timestamp or parsed_end_timestamp
        if date_filter_active:
            # 将 REPORT_DATE 列转换为 datetime 对象
            df['REPORT_DATE_dt'] = pd.to_datetime(df['REPORT_DATE'], errors='coerce')
            if parsed_start_timestamp:
                df = df[df['REPORT_DATE_dt'] >= parsed_start_timestamp].copy()
                output_lines.append(f"已过滤，报告日期从 {start_date} 开始。")
            if parsed_end_timestamp:
                df = df[df['REPORT_DATE_dt'] <= parsed_end_timestamp].copy()
                output_lines.append(f"已过滤，报告日期到 {end_date} 结束。")
            # 移除辅助的 datetime 列
            df = df.drop(columns=['REPORT_DATE_dt'])
            # 如果日期过滤后数据为空
            if df.empty:
                date_range_info = f"在 {start_date or '...'} 到 {end_date or '...'} 期间"
                return f"在指定日期范围 {date_range_info} 内，未能找到港股 '{resolved_name}' 的{report_type}数据。"


        # 格式化金额显示
        def format_amount_value(value, precision=2):
            if pd.isna(value):
                return "N/A"
            if abs(value) >= 1_0000_0000:
                return f"{value / 1_0000_0000:,.{precision}f}亿港元"
            if abs(value) >= 1_0000:
                return f"{value / 1_0000:,.{precision}f}万港元"
            else:
                return f"{value:,.{precision}f}港元"

        # 关键科目映射
        key_items_map = {
            "资产负债表": ["流动资产合计", "非流动资产合计", "资产总计", "流动负债合计", "非流动负债合计", "负债合计",
                           "归属于母公司股东权益合计", "负债和股东权益总计"],
            "利润表": ["营业收入", "营业成本", "营业利润", "利润总额", "归属于母公司股东的净利润", "基本每股收益",
                       "稀释每股收益"],
            "现金流量表": ["经营活动产生的现金流量净额", "投资活动产生的现金流量净额", "筹资活动产生的现金流量净额",
                          "现金及现金等价物净增加额", "期末现金及现金等价物余额"]
        }

        # 确定要显示的科目列表
        selected_key_items, is_custom_selection = [], False
        item_arg_map = {
            "资产负债表": balance_sheet_items,
            "利润表": income_statement_items,
            "现金流量表": cash_flow_statement_items
        }
        custom_items_str = item_arg_map.get(report_type)
        if custom_items_str:
            # 如果用户指定了科目，则使用用户指定的科目
            selected_key_items = [item.strip() for item in custom_items_str.split(',') if item.strip()]
            is_custom_selection = True
        else:
            # 否则使用预定义的关键科目
            selected_key_items = key_items_map.get(report_type, [])
        if not selected_key_items:
            return f"警告：未指定或未能识别 '{report_type}' 的任何有效科目。"

        # 获取所有唯一的报告日期
        report_dates = df['REPORT_DATE'].unique()
        # 将日期转换为 datetime 对象并按降序排序
        report_dates_sorted = sorted([pd.to_datetime(d) for d in report_dates if pd.notna(d)], reverse=True)

        # 如果没有日期过滤，则默认限制为最近3个报告期
        dates_to_display = report_dates_sorted if date_filter_active else report_dates_sorted[:3]

        if not dates_to_display:
            output_lines.append("在当前条件下未找到可显示的财务报告。")
        else:
            if date_filter_active:
                output_lines.append(f"总共找到 {len(dates_to_display)} 份报告，将全部显示：")

        # 遍历每个要显示的报告日期，并提取对应科目的数据
        for date_obj in dates_to_display:
            report_date_display = date_obj.strftime('%Y年%m月%d日')
            # 筛选出当前报告期的数据
            current_report_df = df[df['REPORT_DATE'] == date_obj.strftime('%Y-%m-%d %H:%M:%S')]
            output_lines.append(f"\n--- 报告期: {report_date_display} ---")
            found_items_count = 0
            # 遍历选定的科目，查找并格式化显示金额
            for item_name in selected_key_items:
                item_data = current_report_df[current_report_df['STD_ITEM_NAME'] == item_name]
                if not item_data.empty:
                    amount = item_data.iloc[0]['AMOUNT']
                    output_lines.append(f"{item_name}: {format_amount_value(amount)}")
                    found_items_count += 1
                elif is_custom_selection:
                    # 如果是用户自定义科目且未找到，则明确提示
                    output_lines.append(f"{item_name}: 未找到数据")
            if found_items_count == 0:
                output_lines.append("  (未能找到任何指定科目的数据。)")
            output_lines.append("-" * 20)

        output_lines.append("\n" + "=" * 100)

        # 生成最终提示信息
        # 根据是否有日期过滤，生成不同的总结文本
        if date_filter_active:
            summary_period_text = "指定日期范围内的"
        else:
            summary_period_text = f"最近{len(dates_to_display)}期"

        # 根据是否为自定义科目选择，生成不同的提示信息
        if is_custom_selection:
            output_lines.append(
                f"提示：此为港股 '{resolved_name}' ({resolved_symbol}) {summary_period_text}{indicator}{report_type}的指定科目摘要。")
        else:
            output_lines.append(
                f"提示：此为港股 '{resolved_name}' ({resolved_symbol}) {summary_period_text}{indicator}{report_type}的关键科目摘要。")
        output_lines.append("数据源：东方财富网。如需查询特定科目，请明确提出。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股 '{resolved_symbol}' ({resolved_name}) 的{report_type}信息时发生错误: {e}"

@tool
def get_us_financial_report_em(
        stock_identifier: str,
        report_type: str,
        indicator: str = "年报",
        balance_sheet_items: Optional[List[str]] = None,
        income_statement_items: Optional[List[str]] = None,
        cash_flow_items: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取东方财富网-美股-财务分析-三大报表（资产负债表、综合损益表、现金流量表）的详细数据。
    该接口返回指定美股在特定报告期（年度或季度）的完整财务报表数据，并可按报告期时间范围过滤。

    用户必须提供一个美股代码或公司名称，以及要查询的报表类型。
    可选择查询“年报”、“单季报”或“累计季报”数据，默认为“年报”。
    工具默认仅展示最近3个报告期的关键科目摘要，但用户可以通过可选参数指定要查询的特定科目。

    Args:
        stock_identifier (str): 必需参数。要查询财务报表的美股代码（如 'TSLA'）或公司名称（如 '特斯拉'）。
        report_type (str): 必需参数。要查询的报表类型。可选值：
                           "资产负债表"
                           "综合损益表" (对应利润表)
                           "现金流量表"
        indicator (str): 可选参数。指定查询的财务报表类型。可选值：
                         "年报" (默认值): 查询年度财务报表。
                         "单季报": 查询单个季度财务报表。
                         "累计季报": 查询累计季度（如Q1, Q2, Q3, Q4累计）财务报表。
        balance_sheet_items (Optional[List[str]], optional): 仅当 report_type 为 "资产负债表" 时有效。
                                                                指定要查询的资产负债表科目列表（中文名称）。
                                                                例如: ["总资产", "现金及现金等价物", "应收账款"]。
                                                                如果未提供，则返回默认的关键科目。
        income_statement_items (Optional[List[str]], optional): 仅当 report_type 为 "综合损益表" 时有效。
                                                                 指定要查询的综合损益表科目列表（中文名称）。
                                                                 例如: ["营业总收入", "净利润", "基本每股收益"]。
                                                                 如果未提供，则返回默认的关键科目。
        cash_flow_items (Optional[List[str]], optional): 仅当 report_type 为 "现金流量表" 时有效。
                                                           指定要查询的现金流量表科目列表（中文名称）。
                                                           例如: ["经营活动产生的现金流量净额", "投资活动产生的现金流量净额"]。
                                                           如果未提供，则返回默认的关键科目。
        start_date (Optional[str]): 可选参数。查询的开始日期，格式为 'YYYY-MM-DD'。
                                    将过滤“报告期”在此日期或之后的数据。
        end_date (Optional[str]): 可选参数。查询的结束日期，格式为 'YYYY-MM-DD'。
                                  将过滤“报告期”在此日期或之前的数据。

    Returns:
        str: 一个格式化后的字符串，包含指定美股的财务报表历史数据。
             如果获取失败、未找到股票或找到多个匹配，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if report_type not in ["资产负债表", "综合损益表", "现金流量表"]:
        return f"错误：'report_type' 参数无效。请选择 '资产负债表', '综合损益表' 或 '现金流量表'。"

    if indicator not in ["年报", "单季报", "累计季报"]:
        return f"错误：'indicator' 参数无效。请选择 '年报', '单季报' 或 '累计季报'。"

    # 解析和验证日期参数
    parsed_start_date_obj = None
    parsed_end_date_obj = None
    parsed_start_timestamp = None
    parsed_end_timestamp = None

    if start_date:
        try:
            parsed_start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            parsed_start_timestamp = pd.Timestamp(parsed_start_date_obj)
        except ValueError:
            return "错误：'start_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"
    if end_date:
        try:
            parsed_end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
            parsed_end_timestamp = pd.Timestamp(parsed_end_date_obj)
        except ValueError:
            return "错误：'end_date' 参数格式无效。请使用 'YYYY-MM-DD' 格式。"

    if parsed_start_date_obj and parsed_end_date_obj and parsed_start_date_obj > parsed_end_date_obj:
        return "错误：'start_date' 不能晚于 'end_date'。"

    # 定义不同报表的默认关键科目列表
    key_items_map = {
        "资产负债表": [
            "现金及现金等价物", "总资产", "总负债", "归属于母公司股东权益",
            "流动资产合计", "流动负债合计", "固定资产", "无形资产", "商誉",
            "短期借款", "长期借款", "应收账款", "存货", "应付账款"
        ],
        "综合损益表": [  # 对应利润表
            "营业总收入", "营业总成本", "营业利润", "净利润",
            "归属于母公司股东的净利润", "基本每股收益", "稀释每股收益",
            "销售费用", "管理费用", "研发费用", "财务费用", "所得税费用"
        ],
        "现金流量表": [
            "经营活动产生的现金流量净额", "投资活动产生的现金流量净额",
            "筹资活动产生的现金流量净额", "现金及现金等价物净增加额",
            "期末现金及现金等价物余额", "购建固定资产、无形资产和其他长期资产支付的现金"
        ]
    }

    # 根据 report_type 和用户输入确定要查询的科目列表
    selected_key_items = []
    # 标记是否为用户自定义查询科目
    is_custom_query = False

    if report_type == "资产负债表":
        if balance_sheet_items is not None:
            selected_key_items = balance_sheet_items
            is_custom_query = True
        else:
            selected_key_items = key_items_map["资产负债表"]
    elif report_type == "综合损益表":
        if income_statement_items is not None:
            selected_key_items = income_statement_items
            is_custom_query = True
        else:
            selected_key_items = key_items_map["综合损益表"]
    elif report_type == "现金流量表":
        if cash_flow_items is not None:
            selected_key_items = cash_flow_items
            is_custom_query = True
        else:
            selected_key_items = key_items_map["现金流量表"]

    # 如果经过上述判断，selected_key_items 仍然为空
    if not selected_key_items:
        return f"错误：无法确定 '{report_type}' 的关键科目列表。请检查 'report_type' 参数或联系维护人员。"

    resolved_symbol = None
    resolved_name = None

    # 解析股票代码或名称
    try:
        us_spot_df = ak.stock_us_spot_em()
        if us_spot_df.empty:
            return "错误：无法获取美股列表以解析股票名称。请尝试直接提供股票代码。"

        if '代码' not in us_spot_df.columns or '名称' not in us_spot_df.columns:
            return "错误：获取到的美股列表数据缺少 '代码' 或 '名称' 列，无法解析股票名称。"

        us_spot_df['代码'] = us_spot_df['代码'].astype(str).str.upper()
        us_spot_df['名称'] = us_spot_df['名称'].astype(str)

        # 优先精确匹配代码
        matched_by_code = us_spot_df[us_spot_df['代码'] == stock_identifier.upper()]
        if not matched_by_code.empty:
            resolved_symbol = matched_by_code.iloc[0]['代码']
            resolved_name = matched_by_code.iloc[0]['名称']
        else:
            # 精确匹配名称
            matched_by_name = us_spot_df[us_spot_df['名称'] == stock_identifier]
            if not matched_by_name.empty:
                resolved_symbol = matched_by_name.iloc[0]['代码']
                resolved_name = matched_by_name.iloc[0]['名称']
            else:
                # 模糊匹配名称
                fuzzy_matched_by_name = us_spot_df[
                    us_spot_df['名称'].str.contains(stock_identifier, case=False, na=False)]
                if not fuzzy_matched_by_name.empty:
                    if len(fuzzy_matched_by_name) > 1:
                        output_lines = [
                            f"找到多个与 '{stock_identifier}' 匹配的美股，请提供更具体的股票名称或直接提供股票代码："]
                        output_lines.append("=" * 100)
                        for index, row in fuzzy_matched_by_name.head(10).iterrows():
                            output_lines.append(f"  - 代码: {row['代码']}, 名称: {row['名称']}")
                        if len(fuzzy_matched_by_name) > 10:
                            output_lines.append(f"  ... (还有 {len(fuzzy_matched_by_name) - 10} 个更多匹配)")
                        output_lines.append("=" * 100)
                        return "\n".join(output_lines)
                    else:
                        resolved_symbol = fuzzy_matched_by_name.iloc[0]['代码']
                        resolved_name = fuzzy_matched_by_name.iloc[0]['名称']
                else:
                    # 如果名称和代码都无法匹配，则直接使用用户提供的作为symbol尝试
                    resolved_symbol = stock_identifier.upper()
                    resolved_name = stock_identifier  # 无法解析出名称，暂用identifier

    except Exception as e:
        return f"解析美股名称或代码 '{stock_identifier}' 时发生错误: {e}。请尝试直接提供美股代码。"

    if not resolved_symbol:
        return f"无法为 '{stock_identifier}' 解析出有效的股票代码。请确保输入正确或尝试直接提供股票代码。"

    try:
        # 屌用API
        df = ak.stock_financial_us_report_em(
            stock=resolved_symbol,
            symbol=report_type,
            indicator=indicator
        )

        if df.empty:
            return f"未能获取美股 '{resolved_name or stock_identifier}' ({resolved_symbol}) 的{indicator}{report_type}数据。可能无数据或数据源暂无数据。"

        # 确保关键列存在
        if 'REPORT_DATE' not in df.columns or 'ITEM_NAME' not in df.columns or 'AMOUNT' not in df.columns:
            return "错误：获取到的财务报表数据缺少 'REPORT_DATE', 'ITEM_NAME' 或 'AMOUNT' 列，无法进行展示。"

        # 日期过滤
        # 将 'REPORT_DATE' 列转换为日期时间类型，无效值设为 NaT
        df['REPORT_DATE_DT'] = pd.to_datetime(df['REPORT_DATE'], errors='coerce', format='%Y%m%d')

        filtered_date_info = []

        if parsed_start_timestamp:
            df = df[df['REPORT_DATE_DT'] >= parsed_start_timestamp].copy()
            filtered_date_info.append(f"报告期从 {start_date} 开始")
        if parsed_end_timestamp:
            df = df[df['REPORT_DATE_DT'] <= parsed_end_timestamp].copy()
            filtered_date_info.append(f"报告期到 {end_date} 结束")

        if df.empty:
            date_range_str = " ".join(filtered_date_info) if filtered_date_info else "在指定日期范围"
            return f"在 {date_range_str} 内，未能找到美股 '{resolved_name or stock_identifier}' ({resolved_symbol}) 的{indicator}{report_type}数据。"

        # 定义数值格式化辅助函数
        def format_amount_value(value, precision=2):
            if pd.isna(value):
                return "N/A"
            if value >= 1_000_000_000:
                return f"{value / 1_000_000_000:,.{precision}f}亿美元"
            elif value >= 1_000_000:
                return f"{value / 1_000_000:,.{precision}f}百万美元"
            elif value >= 1_000:
                return f"{value / 1_000:,.{precision}f}千美元"
            else:
                return f"{value:,.{precision}f}美元"

        output_lines = [
            f"美股 '{resolved_name or stock_identifier}' ({resolved_symbol}) 的{indicator}{report_type}数据 (数据来源: 东方财富网)。\n"]
        output_lines.append("=" * 100)

        if filtered_date_info:
            output_lines.append(f"已按时间范围过滤：{'，'.join(filtered_date_info)}。")
            output_lines.append("-" * 100)

        # 获取所有唯一的报告日期，并按降序排列
        report_dates_sorted = sorted(
            [d for d in df['REPORT_DATE_DT'].unique() if pd.notna(d)],
            reverse=True
        )

        # 若有日期限制, 显示限制日期内所有
        if start_date or end_date:
            num_reports_to_show = len(report_dates_sorted)
        # 限制显示最近的3个报告期，但如果过滤后不足3个，则显示所有
        else:
            num_reports_to_show = min(3, len(report_dates_sorted))

        # 用于追踪所有被请求且在数据中找到的科目
        items_actually_found = set()

        for i, date_obj in enumerate(report_dates_sorted):
            if i >= num_reports_to_show:
                break

            # 将 datetime 对象格式化
            report_date_display = date_obj.strftime('%Y年%m月%d日')

            # 过滤出当前报告期的数据
            current_report_df = df[df['REPORT_DATE_DT'] == date_obj]

            output_lines.append(f"--- 报告期: {report_date_display} ---")

            found_items_in_this_report = False
            for item_name in selected_key_items:
                item_data = current_report_df[current_report_df['ITEM_NAME'] == item_name]
                if not item_data.empty:
                    amount = item_data.iloc[0]['AMOUNT']
                    output_lines.append(f"{item_name}: {format_amount_value(amount)}")
                    items_actually_found.add(item_name)
                    found_items_in_this_report = True

            if not found_items_in_this_report:
                output_lines.append("  (未能找到指定科目数据，可能该报告期数据不完整或科目名称不匹配)")

            output_lines.append("-" * 20)

        output_lines.append("=" * 100)

        # 构建最终的提示信息
        summary_message_parts = [
            f"提示：此为美股 '{resolved_name or stock_identifier}' ({resolved_symbol}) 最近{num_reports_to_show}期{indicator}{report_type}"]

        if is_custom_query:
            summary_message_parts.append("的指定科目摘要。")
            # 找出用户请求了但实际未在任何报告期中找到的科目
            not_found_in_any_report = set(selected_key_items) - items_actually_found
            if not_found_in_any_report:
                summary_message_parts.append(
                    f"注意：您请求的科目 '{', '.join(sorted(list(not_found_in_any_report)))}' 未在数据中找到。")
        else:
            summary_message_parts.append("的关键科目摘要。")
            summary_message_parts.append("如需查询特定科目，请明确提出。")

        summary_message_parts.append("数据源：东方财富网。")
        output_lines.append(" ".join(summary_message_parts))

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取美股 '{resolved_name or stock_identifier}' ({resolved_symbol}) 的财务报表数据时发生错误: {e}\n请检查您的参数或稍后再试。"

@tool
def get_a_stock_popularity_rank_em(
    top_n: Optional[int] = 10
) -> str:
    """
    获取东方财富网-股票热度-人气榜中，A股市场当前交易日前100个股票的人气排名数据。
    该接口返回每只股票的当前排名、代码、股票名称、最新价、涨跌额和涨跌幅。

    用户可以通过 'top_n' 参数指定返回前多少条热门股票数据。
    如果未指定 'top_n'，则默认返回前10条数据。
    数据源最多提供前100条数据。

    Args:
        top_n (Optional[int]): 可选参数。指定返回人气榜前多少条数据。
                                默认为 10。最大可获取 100 条。

    Returns:
        str: 一个格式化后的字符串，包含A股市场人气榜指定数量的股票数据。
             如果获取失败或无数据，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not isinstance(top_n, int) or top_n <= 0:
        return "错误：'top_n' 参数必须是大于0的整数。"

    try:
        # 调用API
        df = ak.stock_hot_rank_em()

        if df.empty:
            return "未能获取A股人气榜数据。可能当前无数据或数据源暂无数据。"

        output_lines = ["东方财富网-A股人气榜股票数据。\n"]
        output_lines.append("=" * 100)
        output_lines.append(f"数据反映当前交易日前 {len(df)} 个股票的人气排名。\n")

        # 限制返回的数据量
        if top_n > len(df):
            actual_top_n = len(df)
            output_lines.append(f"提示：您请求返回前 {top_n} 条数据，但数据源最多提供 {len(df)} 条。已返回所有可用数据。\n")
        else:
            actual_top_n = top_n

        # 获取指定数量的数据
        df_to_display = df.head(actual_top_n).copy()

        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['当前排名', '代码', '股票名称', '最新价', '涨跌额', '涨跌幅']
        
        # 过滤掉不存在的列
        available_cols = [col for col in display_cols if col in df_to_display.columns]
        
        # 格式化数值列
        def format_value(value, col_name):
            if pd.isna(value):
                return "N/A"
            if col_name == '最新价':
                return f"{float(value):.2f}"
            elif col_name == '涨跌额':
                return f"{float(value):+.2f}"
            elif col_name == '涨跌幅':
                return f"{float(value):+.2f}%"
            elif col_name == '当前排名':
                return f"{int(value)}"
            return str(value)

        # 应用格式化
        for col in available_cols:
            # 代码和名称不需要特殊格式化
            if col not in ['代码', '股票名称']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))

        # 将DataFrame转换为字符串输出
        output_lines.append(f"以下是人气榜前 {actual_top_n} 名的股票数据：")
        output_lines.append(df_to_display.to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含当前排名、股票代码、股票名称、最新价、涨跌额和涨跌幅。")
        
        # 引导性回复
        if top_n == 10 and len(df) > 10:
            output_lines.append(f"您当前看到的是人气榜前10名股票。如果想查看更多，例如前20名，请将 'top_n' 参数设置为 20。")
            output_lines.append(f"该榜单最多可提供前 {len(df)} 条数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取A股人气榜数据时发生错误: {e}"

@tool
def get_a_stock_surge_rank_em(
    top_n: Optional[int] = 10
) -> str:
    """
    获取东方财富网-个股人气榜-飙升榜中，A股市场当前交易日前100个股票的飙升榜排名数据。
    该接口返回每只股票的排名较昨日变动、当前排名、代码、股票名称、最新价、涨跌额和涨跌幅。

    用户可以通过 'top_n' 参数指定返回前多少条飙升股票数据。
    如果未指定 'top_n'，则默认返回前10条数据。
    数据源最多提供前100条数据。

    Args:
        top_n (Optional[int]): 可选参数。指定返回飙升榜前多少条数据。
                                默认为 10。最大可获取 100 条。

    Returns:
        str: 一个格式化后的字符串，包含A股市场飙升榜指定数量的股票数据。
             如果获取失败或无数据，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not isinstance(top_n, int) or top_n <= 0:
        return "错误：'top_n' 参数必须是大于0的整数。"

    try:
        # 调用API
        df = ak.stock_hot_up_em()

        if df.empty:
            return "未能获取A股飙升榜数据。可能当前无数据或数据源暂无数据。"

        output_lines = ["东方财富网-A股飙升榜股票数据。\n"]
        output_lines.append("=" * 100)
        output_lines.append(f"数据反映当前交易日前 {len(df)} 个股票的人气飙升排名。\n")

        # 限制返回的数据量
        if top_n > len(df):
            actual_top_n = len(df)
            output_lines.append(f"提示：您请求返回前 {top_n} 条数据，但数据源最多提供 {len(df)} 条。已返回所有可用数据。\n")
        else:
            actual_top_n = top_n

        # 获取指定数量的数据
        df_to_display = df.head(actual_top_n).copy()

        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['排名较昨日变动', '当前排名', '代码', '股票名称', '最新价', '涨跌额', '涨跌幅']
        
        # 过滤掉不存在的列
        available_cols = [col for col in display_cols if col in df_to_display.columns]
        
        # 格式化数值列
        def format_value(value, col_name):
            if pd.isna(value):
                return "N/A"
            if col_name == '最新价':
                return f"{float(value):.2f}"
            elif col_name == '涨跌额':
                return f"{float(value):+.2f}"
            elif col_name == '涨跌幅':
                return f"{float(value):+.2f}%"
            elif col_name in ['排名较昨日变动', '当前排名']:
                return f"{int(value)}"
            return str(value)

        # 应用格式化
        for col in available_cols:
            # 代码和名称不需要特殊格式化
            if col not in ['代码', '股票名称']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))

        # 将DataFrame转换为字符串输出
        output_lines.append(f"以下是飙升榜前 {actual_top_n} 名的股票数据：")
        output_lines.append(df_to_display.to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含排名较昨日变动、当前排名、股票代码、股票名称、最新价、涨跌额和涨跌幅。")
        
        # 引导性回复
        if top_n == 10 and len(df) > 10:
            output_lines.append(f"您当前看到的是飙升榜前10名股票。如果想查看更多，例如前20名，请将 'top_n' 参数设置为 20。")
            output_lines.append(f"该榜单最多可提供前 {len(df)} 条数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取A股飙升榜数据时发生错误: {e}"

@tool
def get_hk_stock_popularity_rank_em(
    top_n: Optional[int] = 10
) -> str:
    """
    获取东方财富网-个股人气榜-人气榜中，港股市场当前交易日前100个股票的人气排名数据。
    该接口返回每只股票的当前排名、代码、股票名称、最新价和涨跌幅。

    用户可以通过 'top_n' 参数指定返回前多少条热门港股数据。
    如果未指定 'top_n'，则默认返回前10条数据。
    数据源最多提供前100条数据。

    Args:
        top_n (Optional[int]): 可选参数。指定返回人气榜前多少条数据。
                                默认为 10。最大可获取 100 条。

    Returns:
        str: 一个格式化后的字符串，包含港股市场人气榜指定数量的股票数据。
             如果获取失败或无数据，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not isinstance(top_n, int) or top_n <= 0:
        return "错误：'top_n' 参数必须是大于0的整数。"

    try:
        # 调用API
        df = ak.stock_hk_hot_rank_em()

        if df.empty:
            return "未能获取港股人气榜数据。可能当前无数据或数据源暂无数据。"

        output_lines = ["东方财富网-港股人气榜股票数据。\n"]
        output_lines.append("=" * 100)
        output_lines.append(f"数据反映当前交易日前 {len(df)} 个港股的人气排名。\n")

        # 限制返回的数据量
        if top_n > len(df):
            actual_top_n = len(df)
            output_lines.append(f"提示：您请求返回前 {top_n} 条数据，但数据源最多提供 {len(df)} 条。已返回所有可用数据。\n")
        else:
            actual_top_n = top_n

        # 获取指定数量的数据
        df_to_display = df.head(actual_top_n).copy()

        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['当前排名', '代码', '股票名称', '最新价', '涨跌幅']
        
        # 过滤掉不存在的列
        available_cols = [col for col in display_cols if col in df_to_display.columns]
        
        # 格式化数值列
        def format_value(value, col_name):
            if pd.isna(value):
                return "N/A"
            if col_name == '最新价':
                return f"{float(value):.3f}"
            elif col_name == '涨跌幅':
                return f"{float(value):+.2f}%"
            elif col_name == '当前排名':
                return f"{int(value)}"
            return str(value)

        # 应用格式化
        for col in available_cols:
            # 代码和名称不需要特殊格式化
            if col not in ['代码', '股票名称']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))

        # 将DataFrame转换为字符串输出
        output_lines.append(f"以下是港股人气榜前 {actual_top_n} 名的股票数据：")
        output_lines.append(df_to_display.to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含当前排名、股票代码、股票名称、最新价和涨跌幅。")
        
        # 引导性回复
        if top_n == 10 and len(df) > 10:
            output_lines.append(f"您当前看到的是港股人气榜前10名股票。如果想查看更多，例如前20名，请将 'top_n' 参数设置为 20。")
            output_lines.append(f"该榜单最多可提供前 {len(df)} 条数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股人气榜数据时发生错误: {e}"


@tool
def get_a_stock_hot_rank_detail_em(
        symbol: str,
        date: Optional[str] = None
) -> str:
    """
    获取东方财富网-股票热度-历史趋势及粉丝特征数据。
    该接口返回指定A股股票近期的人气排名、新晋粉丝和铁杆粉丝数据。

    用户必须提供 'symbol' 参数（股票代码，例如 'SZ000665'）。
    可选参数 'date' 用于查询特定日期的数据。如果未提供 'date'，
    则默认返回该股票最近10天的历史数据。

    Args:
        symbol (str): 必填参数。要查询的A股股票代码，例如 "SZ000665"。
                      请确保代码格式正确，包含市场前缀（如 'SH' 或 'SZ'）。
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最近10条数据。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的历史人气排名和粉丝特征数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"
    if not symbol:
        return "错误：'symbol' 参数不能为空。请提供有效的股票代码，例如 'SZ000665'。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        # 根据列名进行特定格式化
        if col_name in ['新晋粉丝', '铁杆粉丝']:
            return f"{float(value):.2%}"
        elif col_name == '排名':
            return f"{int(value)}"
        # 其他类型默认转换为字符串
        return str(value)

    try:
        # 调用API
        df = ak.stock_hot_rank_detail_em(symbol=symbol)

        # 未能获取到数据
        if df.empty:
            return f"未能获取股票 '{symbol}' 的历史热度数据。可能该股票无相关数据或代码不正确。"

        # 确保 '时间' 列是 datetime 类型以便于日期过滤和排序
        if '时间' in df.columns:
            df['时间'] = pd.to_datetime(df['时间'])
            df = df.sort_values(by='时间', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '时间' 列，无法处理。"

        # 初始化输出行列表，添加标题
        output_lines = [f"东方财富网-A股股票 '{symbol}' 历史热度及粉丝特征数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最近数据
        if date:
            try:
                # 将 date 参数转换为 datetime 对象
                query_date = pd.to_datetime(date)
            except ValueError:
                # 如果日期格式不正确，返回错误信息
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2025-07-24'。"

            # 过滤出指定日期的数据
            filtered_df = df[df['时间'].dt.date == query_date.date()]

            # 如果指定日期没有数据
            if filtered_df.empty:
                output_lines.append(f"未找到股票 '{symbol}' 在 '{date}' 的历史热度数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最近数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是股票 '{symbol}' 在 '{date}' 的热度数据：")
                df_to_display = filtered_df.head(1).copy()

        else:
            # 如果未指定日期，则返回最近 MAX_ROWS_TO_DISPLAY天的数据
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是股票 '{symbol}' 最近 {len(df_to_display)} 天的热度数据：")

        # 定义要显示和格式化的列
        display_cols = ['时间', '排名', '证券代码', '新晋粉丝', '铁杆粉丝']
        # 筛选出 DataFrame 中实际存在的列
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 对 DataFrame 中需要格式化的列应用格式化函数
        for col in available_cols:
            if col not in ['时间', '证券代码']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))

        # 将 '时间' 列格式化
        if '时间' in df_to_display.columns:
            df_to_display['时间'] = df_to_display['时间'].dt.strftime('%Y-%m-%d')

        # 将格式化后的 DataFrame 转换为字符串并添加到输出列表
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含时间、排名、证券代码、新晋粉丝比例和铁杆粉丝比例。")

        # 为用户提供进一步查询的指导
        if not date:
            output_lines.append(f"您当前看到的是股票 '{symbol}' 最近 {len(df_to_display)} 天的热度数据。")
            output_lines.append(
                f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2025-07-20'。")
            if len(df) > MAX_ROWS_TO_DISPLAY:
                output_lines.append(f"该股票共有 {len(df)} 条历史热度数据可供查询。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票 '{symbol}' 历史热度数据时发生错误: {e}"


@tool
def get_hk_stock_hot_rank_detail_em(
        symbol: str,
        date: Optional[str] = None
) -> str:
    """
    获取东方财富网-股票热度-历史趋势数据。
    该接口返回指定港股股票近期的人气排名数据。

    用户必须提供 'symbol' 参数（港股代码，例如 '00700'）。
    可选参数 'date' 用于查询特定日期的数据。如果未提供 'date'，
    则默认返回该股票最近10天的历史数据。

    Args:
        symbol (str): 必填参数。要查询的港股代码，例如 "00700"。
                      请确保代码格式正确。
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最近10条数据。

    Returns:
        str: 一个格式化后的字符串，包含指定港股的历史人气排名数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"
    if not symbol:
        return "错误：'symbol' 参数不能为空。请提供有效的港股代码，例如 '00700'。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        if col_name == '排名':
            return f"{int(value)}"
        return str(value)

    try:
        # 调用API
        df = ak.stock_hk_hot_rank_detail_em(symbol=symbol)

        # 如果返回的 DataFrame 为空，表示未能获取到数据
        if df.empty:
            return f"未能获取港股 '{symbol}' 的历史热度数据。可能该股票无相关数据或代码不正确。"

        # 确保 '时间' 列是 datetime 类型
        if '时间' in df.columns:
            df['时间'] = pd.to_datetime(df['时间'])
            df = df.sort_values(by='时间', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '时间' 列，无法处理。"

        # 初始化输出行列表，添加标题
        output_lines = [f"东方财富网-港股股票 '{symbol}' 历史热度排名数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最近数据

        if date:
            try:
                # 将 date 参数转换为 datetime 对象的日期部分
                query_date = pd.to_datetime(date).date()
            except ValueError:
                # 如果日期格式不正确，返回错误信息
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2025-07-24'。"

            # 过滤出指定日期的数据
            filtered_df = df[df['时间'].dt.date == query_date]

            # 如果指定日期没有数据
            if filtered_df.empty:
                output_lines.append(f"未找到港股 '{symbol}' 在 '{date}' 的历史热度数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最近数据。")
                return "\n".join(output_lines)
            else:
                # 显示特定日期的数据
                output_lines.append(f"以下是港股 '{symbol}' 在 '{date}' 的热度数据：")
                df_to_display = filtered_df.head(1).copy()

        else:
            # 如果未指定日期，则返回最近 MAX_ROWS_TO_DISPLAY 天的数据
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是港股 '{symbol}' 最近 {len(df_to_display)} 天的热度数据：")

        # 定义要显示和格式化的列
        display_cols = ['时间', '排名', '证券代码']
        # 筛选出 DataFrame 中实际存在的列
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 对 DataFrame 中需要格式化的列应用格式化函数
        for col in available_cols:
            if col not in ['时间', '证券代码']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))

        # 将 '时间' 列格式化
        if '时间' in df_to_display.columns:
            df_to_display['时间'] = df_to_display['时间'].dt.strftime('%Y-%m-%d')

        # 将格式化后的 DataFrame 转换为字符串并添加到输出列表
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含时间、排名和证券代码。")

        # 为用户提供进一步查询的指导
        if not date:
            output_lines.append(f"您当前看到的是港股 '{symbol}' 最近 {len(df_to_display)} 天的热度数据。")
            output_lines.append(
                f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2025-07-20'。")
            if len(df) > MAX_ROWS_TO_DISPLAY:
                output_lines.append(f"该股票共有 {len(df)} 条历史热度数据可供查询。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股 '{symbol}' 历史热度数据时发生错误: {e}"


@tool
def get_stock_hot_keyword_em(
    symbol: str,
    top_n: Optional[int] = 10
) -> str:
    """
    获取东方财富网-个股人气榜-热门关键词数据。
    该接口返回指定A股股票在最近交易日的热门关键词及其热度。

    用户必须提供 'symbol' 参数（股票代码，例如 'SZ000665'）。
    可选参数 'top_n' 用于指定返回前多少条热门关键词数据。
    如果未指定 'top_n'，则默认返回前10条数据。

    Args:
        symbol (str): 必填参数。要查询的A股股票代码，例如 "SZ000665"。
                      请确保代码格式正确，包含市场前缀（如 'SH' 或 'SZ'）。
        top_n (Optional[int]): 可选参数。指定返回热门关键词的前多少条数据。
                                默认为 10。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的最新热门关键词数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：'symbol' 参数不能为空。请提供有效的股票代码，例如 'SZ000665'。"
    
    if not isinstance(top_n, int) or top_n <= 0:
        return "错误：'top_n' 参数必须是大于0的整数。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        if col_name == '热度':
            return f"{int(value)}"
        return str(value)

    try:
        # 调用API
        df = ak.stock_hot_keyword_em(symbol=symbol)

        if df.empty:
            return f"未能获取股票 '{symbol}' 的热门关键词数据。可能该股票无相关数据或代码不正确。"

        output_lines = [f"东方财富网-A股股票 '{symbol}' 热门关键词数据。\n"]
        output_lines.append("=" * 100)

        # 确保 '时间' 列是 datetime 类型并获取最新时间
        if '时间' in df.columns:
            df['时间'] = pd.to_datetime(df['时间'])
            latest_time = df['时间'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
            output_lines.append(f"数据时间点：{latest_time}\n")
        else:
            output_lines.append(f"警告：获取到的数据中缺少 '时间' 列。\n")

        # 限制返回的数据量
        if top_n > len(df):
            actual_top_n = len(df)
            output_lines.append(f"提示：您请求返回前 {top_n} 条关键词，但数据源只提供 {len(df)} 条。已返回所有可用数据。\n")
        else:
            actual_top_n = top_n

        # 获取指定数量的数据
        df_to_display = df.head(actual_top_n).copy()

        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['时间', '股票代码', '概念名称', '概念代码', '热度']
        
        # 过滤掉不存在的列
        available_cols = [col for col in display_cols if col in df_to_display.columns]
        
        # 应用格式化
        for col in available_cols:
            if col not in ['时间', '股票代码', '概念名称', '概念代码']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '时间' 列为字符串，去除秒数，保持简洁
        if '时间' in df_to_display.columns:
            df_to_display['时间'] = df_to_display['时间'].dt.strftime('%Y-%m-%d %H:%M')


        # 将DataFrame转换为字符串输出
        output_lines.append(f"以下是股票 '{symbol}' 最热门的 {actual_top_n} 个关键词：")
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含时间、股票代码、概念名称、概念代码和热度。")
        
        # 引导性回复
        if top_n == 10 and len(df) > 10:
            output_lines.append(f"您当前看到的是股票 '{symbol}' 最热门的10个关键词。如果想查看更多，例如前20个，请将 'top_n' 参数设置为 20。")
            output_lines.append(f"该股票共有 {len(df)} 个热门关键词可供查询。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票 '{symbol}' 热门关键词数据时发生错误: {e}"

@tool
def get_a_stock_hot_rank_latest_em(
    symbol: str
) -> str:
    """
    获取东方财富网-个股人气榜-最新排名数据。
    该接口返回指定A股股票在最新交易日的详细人气指标，包括市场类型、
    市场总股票数、计算时间、当前排名、排名较昨日变动、历史排名变动等。

    用户必须提供 'symbol' 参数（股票代码，例如 'SZ000665'）。

    Args:
        symbol (str): 必填参数。要查询的A股股票代码，例如 "SZ000665"。
                      请确保代码格式正确，包含市场前缀（如 'SH' 或 'SZ'）。

    Returns:
        str: 一个格式化后的字符串，包含指定股票的最新人气排名详细数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：'symbol' 参数不能为空。请提供有效的股票代码，例如 'SZ000665'。"

    try:
        # 调用API
        df = ak.stock_hot_rank_latest_em(symbol=symbol)

        if df.empty:
            return f"未能获取股票 '{symbol}' 的最新人气排名详细数据。可能该股票无相关数据或代码不正确。"

        output_lines = [f"东方财富网-A股股票 '{symbol}' 最新人气排名详细数据。\n"]
        output_lines.append("=" * 100)

        # 将DataFrame转换为字典，方便按键值对处理
        if 'item' not in df.columns or 'value' not in df.columns:
            return f"错误：获取到的数据格式不正确，缺少 'item' 或 'value' 列。"
            
        data_dict = df.set_index('item')['value'].to_dict()

        # 定义一个映射字典
        display_names = {
            'marketType': '市场类型',
            'marketAllCount': '市场总股票数',
            'calcTime': '数据计算时间',
            'innerCode': '内部代码',
            'srcSecurityCode': '股票代码',
            'rank': '当前人气排名',
            'rankChange': '排名较昨日变动',
            'hisRankChange': '历史排名变动',
            'hisRankChange_rank': '历史排名变动排名',
            'flag': '标志'
        }

        # 格式化并输出数据
        for item_key, item_value in data_dict.items():
            display_name = display_names.get(item_key, item_key)
            # 转换为字符串
            formatted_value = str(item_value)

            if item_key == 'calcTime':
                try:
                    # 将时间字符串转换为更友好的格式
                    formatted_value = pd.to_datetime(item_value).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    pass # 如果转换失败，保留原始字符串
            elif item_key in ['marketAllCount', 'rank', 'rankChange', 'hisRankChange', 'hisRankChange_rank', 'flag']:
                try:
                    # 格式化为整数
                    formatted_value = f"{int(item_value)}"
                except (ValueError, TypeError):
                    pass
            
            output_lines.append(f"{display_name}: {formatted_value}")

        output_lines.append("=" * 100)
        output_lines.append(f"提示：以上是股票 '{symbol}' 在东方财富股吧的最新人气详细数据。")
        output_lines.append("这些数据反映了该股票在当前时点的人气状态和变化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取股票 '{symbol}' 最新人气排名详细数据时发生错误: {e}"

@tool
def get_hk_stock_hot_rank_latest_em(
    symbol: str
) -> str:
    """
    获取东方财富网-个股人气榜-最新排名数据。
    该接口返回指定港股股票在最新交易日的详细人气指标，包括市场类型、
    市场总股票数、计算时间、当前排名、排名较昨日变动、历史排名变动等。

    用户必须提供 'symbol' 参数（港股代码，例如 '00700'）。

    Args:
        symbol (str): 必填参数。要查询的港股代码，例如 "00700"。
                      请确保代码格式正确。

    Returns:
        str: 一个格式化后的字符串，包含指定港股的最新人气排名详细数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    if not symbol:
        return "错误：'symbol' 参数不能为空。请提供有效的港股代码，例如 '00700'。"

    try:
        # 调用API
        df = ak.stock_hk_hot_rank_latest_em(symbol=symbol)

        if df.empty:
            return f"未能获取港股 '{symbol}' 的最新人气排名详细数据。可能该股票无相关数据或代码不正确。"

        output_lines = [f"东方财富网-港股股票 '{symbol}' 最新人气排名详细数据。\n"]
        output_lines.append("=" * 100)

        # 将DataFrame转换为字典，方便按键值对处理
        if 'item' not in df.columns or 'value' not in df.columns:
            return f"错误：获取到的数据格式不正确，缺少 'item' 或 'value' 列。"
            
        data_dict = df.set_index('item')['value'].to_dict()

        # 定义一个映射字典
        display_names = {
            'marketType': '市场类型',
            'marketAllCount': '市场总股票数',
            'calcTime': '数据计算时间',
            'innerCode': '内部代码',
            'srcSecurityCode': '股票代码',
            'rank': '当前人气排名',
            'rankChange': '排名较昨日变动',
            'hisRankChange': '历史排名变动',
            'hisRankChange_rank': '历史排名变动排名',
            'flag': '标志'
        }

        # 格式化并输出数据
        for item_key, item_value in data_dict.items():
            display_name = display_names.get(item_key, item_key)
            # 转换为字符串
            formatted_value = str(item_value)

            if item_key == 'calcTime':
                try:
                    # 将时间字符串转换为更友好的格式
                    formatted_value = pd.to_datetime(item_value).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    # 如果转换失败，保留原始字符串
                    pass
            elif item_key in ['marketAllCount', 'rank', 'rankChange', 'hisRankChange', 'hisRankChange_rank', 'flag']:
                try:
                    # 格式化为整数
                    formatted_value = f"{int(item_value)}"
                except (ValueError, TypeError):
                    # 如果转换失败，保留原始字符串
                    pass
            
            output_lines.append(f"{display_name}: {formatted_value}")

        output_lines.append("=" * 100)
        output_lines.append(f"提示：以上是港股 '{symbol}' 在东方财富股吧的最新人气详细数据。")
        output_lines.append("这些数据反映了该股票在当前时点的人气状态和变化。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取港股 '{symbol}' 最新人气排名详细数据时发生错误: {e}"
    
@tool
def get_us_fed_interest_rate_report(
    date: Optional[str] = None
) -> str:
    """
    获取美联储利率决议的历史报告数据。
    该接口提供自1982年9月27日至今的美联储利率决议报告，
    包含商品名称、日期、今值、预测值和前值。

    用户可以通过 'date' 参数指定查询特定日期的数据，格式为 "YYYY-MM-DD"。
    如果未指定 'date'，则默认返回最新的10条利率决议数据。

    Args:
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               例如 "2024-09-19"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最新的10条数据。

    Returns:
        str: 一个格式化后的字符串，包含美联储利率决议报告数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        if col_name in ['今值', '预测值', '前值']:
            return f"{float(value):.2f}%"
        return str(value)

    try:
        # 调用API
        df = ak.macro_bank_usa_interest_rate()

        if df.empty:
            return "未能获取美联储利率决议报告数据。可能当前无数据或数据源暂无数据。"

        # 确保 '日期' 列是 datetime 类型以便于日期过滤和排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            # 按照日期降序排列，以便获取最新数据
            df = df.sort_values(by='日期', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '日期' 列，无法处理。"

        output_lines = ["美联储利率决议报告数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最新数据
        if date:
            try:
                query_date = pd.to_datetime(date).date()
            except ValueError:
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2024-09-19'。"
            
            # 过滤指定日期的数据
            filtered_df = df[df['日期'].dt.date == query_date]

            if filtered_df.empty:
                output_lines.append(f"未找到美联储在 '{date}' 的利率决议报告数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最新数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是美联储在 '{date}' 的利率决议报告：")
                # 显示所有匹配该日期的数据
                df_to_display = filtered_df.copy()
                
        else:
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是美联储最近 {len(df_to_display)} 条利率决议报告：")
            
        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['商品', '日期', '今值', '预测值', '前值']
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 应用格式化
        for col in available_cols:
            if col not in ['商品', '日期']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '日期' 列为字符串
        if '日期' in df_to_display.columns:
            df_to_display['日期'] = df_to_display['日期'].dt.strftime('%Y-%m-%d')

        # 将DataFrame转换为字符串输出
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含商品名称、日期、今值、预测值和前值。所有利率值单位为百分比(%)。")
        
        # 引导性回复
        if not date:
            output_lines.append(f"您当前看到的是美联储最近 {len(df_to_display)} 条利率决议报告。")
            output_lines.append(f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2024-09-19'。")
            output_lines.append(f"该工具提供自 {df['日期'].min().strftime('%Y-%m-%d')} 至今的所有历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取美联储利率决议报告数据时发生错误: {e}"

@tool
def get_euro_ecb_interest_rate_report(
    date: Optional[str] = None
) -> str:
    """
    获取欧洲央行利率决议的历史报告数据。
    该接口提供自1999年1月1日至今的欧洲央行利率决议报告，
    包含商品名称、日期、今值、预测值和前值。

    用户可以通过 'date' 参数指定查询特定日期的数据，格式为 "YYYY-MM-DD"。
    如果未指定 'date'，则默认返回最新的10条利率决议数据。

    Args:
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               例如 "2024-09-12"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最新的10条数据。

    Returns:
        str: 一个格式化后的字符串，包含欧洲央行利率决议报告数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        # 利率值通常显示为百分比，保留两位小数
        if col_name in ['今值', '预测值', '前值']:
            return f"{float(value):.2f}%"
        return str(value)

    try:
        # 调用API
        df = ak.macro_bank_euro_interest_rate()

        if df.empty:
            return "未能获取欧洲央行利率决议报告数据。可能当前无数据或数据源暂无数据。"

        # 确保 '日期' 列是 datetime 类型以便于日期过滤和排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            # 按照日期降序排列，以便获取最新数据
            df = df.sort_values(by='日期', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '日期' 列，无法处理。"

        output_lines = ["欧洲央行利率决议报告数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最新数据
        if date:
            try:
                query_date = pd.to_datetime(date).date()
            except ValueError:
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2024-09-12'。"
            
            # 过滤指定日期的数据
            filtered_df = df[df['日期'].dt.date == query_date]

            if filtered_df.empty:
                output_lines.append(f"未找到欧洲央行在 '{date}' 的利率决议报告数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最新数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是欧洲央行在 '{date}' 的利率决议报告：")
                # 显示所有匹配该日期的数据
                df_to_display = filtered_df.copy()
                
        else:
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是欧洲央行最近 {len(df_to_display)} 条利率决议报告：")
            
        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['商品', '日期', '今值', '预测值', '前值']
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 应用格式化
        for col in available_cols:
            if col not in ['商品', '日期']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '日期' 列为字符串
        if '日期' in df_to_display.columns:
            df_to_display['日期'] = df_to_display['日期'].dt.strftime('%Y-%m-%d')

        # 将DataFrame转换为字符串输出
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含商品名称、日期、今值、预测值和前值。所有利率值单位为百分比(%)。")
        
        # 引导性回复
        if not date:
            output_lines.append(f"您当前看到的是欧洲央行最近 {len(df_to_display)} 条利率决议报告。")
            output_lines.append(f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2024-09-12'。")
            output_lines.append(f"该工具提供自 {df['日期'].min().strftime('%Y-%m-%d')} 至今的所有历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取欧洲央行利率决议报告数据时发生错误: {e}"

@tool
def get_newzealand_rba_interest_rate_report(
    date: Optional[str] = None
) -> str:
    """
    获取新西兰联储利率决议的历史报告数据。
    该接口提供自1999年4月1日至今的新西兰联储利率决议报告，
    包含商品名称、日期、今值、预测值和前值。

    用户可以通过 'date' 参数指定查询特定日期的数据，格式为 "YYYY-MM-DD"。
    如果未指定 'date'，则默认返回最新的10条利率决议数据。

    Args:
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               例如 "2024-08-14"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最新的10条数据。

    Returns:
        str: 一个格式化后的字符串，包含新西兰联储利率决议报告数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        # 利率值通常显示为百分比，保留两位小数
        if col_name in ['今值', '预测值', '前值']:
            return f"{float(value):.2f}%"
        return str(value)

    try:
        # 调用API
        df = ak.macro_bank_newzealand_interest_rate()

        if df.empty:
            return "未能获取新西兰联储利率决议报告数据。可能当前无数据或数据源暂无数据。"

        # 确保 '日期' 列是 datetime 类型以便于日期过滤和排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            # 按照日期降序排列，以便获取最新数据
            df = df.sort_values(by='日期', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '日期' 列，无法处理。"

        output_lines = ["新西兰联储利率决议报告数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最新数据
        if date:
            try:
                query_date = pd.to_datetime(date).date()
            except ValueError:
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2024-08-14'。"
            
            # 过滤指定日期的数据
            filtered_df = df[df['日期'].dt.date == query_date]

            if filtered_df.empty:
                output_lines.append(f"未找到新西兰联储在 '{date}' 的利率决议报告数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最新数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是新西兰联储在 '{date}' 的利率决议报告：")
                # 显示所有匹配该日期的数据
                df_to_display = filtered_df.copy()
                
        else:
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是新西兰联储最近 {len(df_to_display)} 条利率决议报告：")
            
        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['商品', '日期', '今值', '预测值', '前值']
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 应用格式化
        for col in available_cols:
            if col not in ['商品', '日期']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '日期' 列为字符串
        if '日期' in df_to_display.columns:
            df_to_display['日期'] = df_to_display['日期'].dt.strftime('%Y-%m-%d')

        # 将DataFrame转换为字符串输出
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含商品名称、日期、今值、预测值和前值。所有利率值单位为百分比(%)。")
        
        # 引导性回复
        if not date:
            output_lines.append(f"您当前看到的是新西兰联储最近 {len(df_to_display)} 条利率决议报告。")
            output_lines.append(f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2024-08-14'。")
            output_lines.append(f"该工具提供自 {df['日期'].min().strftime('%Y-%m-%d')} 至今的所有历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取新西兰联储利率决议报告数据时发生错误: {e}"

@tool
def get_china_pboc_interest_rate_report(
    date: Optional[str] = None
) -> str:
    """
    获取中国央行利率决议的历史报告数据。
    该接口提供自1991年1月5日至今的中国央行利率决议报告，
    包含商品名称、日期、今值、预测值和前值。

    用户可以通过 'date' 参数指定查询特定日期的数据，格式为 "YYYY-MM-DD"。
    如果未指定 'date'，则默认返回最新的10条利率决议数据。

    Args:
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               例如 "2019-11-20"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最新的10条数据。

    Returns:
        str: 一个格式化后的字符串，包含中国央行利率决议报告数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        # 利率值通常显示为百分比，保留两位小数
        if col_name in ['今值', '预测值', '前值']:
            return f"{float(value):.2f}%"
        return str(value)

    try:
        # 调用API
        df = ak.macro_bank_china_interest_rate()

        if df.empty:
            return "未能获取中国央行利率决议报告数据。可能当前无数据或数据源暂无数据。"

        # 确保 '日期' 列是 datetime 类型
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            # 按照日期降序排列，以便获取最新数据
            df = df.sort_values(by='日期', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '日期' 列，无法处理。"

        output_lines = ["中国央行利率决议报告数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最新数据
        if date:
            try:
                query_date = pd.to_datetime(date).date()
            except ValueError:
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2019-11-20'。"
            
            # 过滤指定日期的数据
            filtered_df = df[df['日期'].dt.date == query_date]

            if filtered_df.empty:
                output_lines.append(f"未找到中国央行在 '{date}' 的利率决议报告数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最新数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是中国央行在 '{date}' 的利率决议报告：")
                # 显示所有匹配该日期的数据
                df_to_display = filtered_df.copy()
                
        else:
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是中国央行最近 {len(df_to_display)} 条利率决议报告：")
            
        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['商品', '日期', '今值', '预测值', '前值']
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 应用格式化
        for col in available_cols:
            if col not in ['商品', '日期']: # 这些列不需要特殊格式化
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '日期' 列为字符串
        if '日期' in df_to_display.columns:
            df_to_display['日期'] = df_to_display['日期'].dt.strftime('%Y-%m-%d')

        # 将DataFrame转换为字符串输出
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含商品名称、日期、今值、预测值和前值。所有利率值单位为百分比(%)。")
        
        # 引导性回复
        if not date:
            output_lines.append(f"您当前看到的是中国央行最近 {len(df_to_display)} 条利率决议报告。")
            output_lines.append(f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2019-11-20'。")
            output_lines.append(f"该工具提供自 {df['日期'].min().strftime('%Y-%m-%d')} 至今的所有历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取中国央行利率决议报告数据时发生错误: {e}"

@tool
def get_switzerland_snb_interest_rate_report(
    date: Optional[str] = None
) -> str:
    """
    获取瑞士央行利率决议的历史报告数据。
    该接口提供自2008年3月13日至今的瑞士央行利率决议报告，
    包含商品名称、日期、今值、预测值和前值。

    用户可以通过 'date' 参数指定查询特定日期的数据，格式为 "YYYY-MM-DD"。
    如果未指定 'date'，则默认返回最新的10条利率决议数据。

    Args:
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               例如 "2024-09-26"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最新的10条数据。

    Returns:
        str: 一个格式化后的字符串，包含瑞士央行利率决议报告数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        # 利率值通常显示为百分比，保留两位小数
        if col_name in ['今值', '预测值', '前值']:
            return f"{float(value):.2f}%"
        return str(value)

    try:
        # 调用API
        df = ak.macro_bank_switzerland_interest_rate()

        if df.empty:
            return "未能获取瑞士央行利率决议报告数据。可能当前无数据或数据源暂无数据。"

        # 确保 '日期' 列是 datetime 类型以便于日期过滤和排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            # 按照日期降序排列，以便获取最新数据
            df = df.sort_values(by='日期', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '日期' 列，无法处理。"

        output_lines = ["瑞士央行利率决议报告数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最新数据
        if date:
            try:
                query_date = pd.to_datetime(date).date()
            except ValueError:
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2024-09-26'。"
            
            # 过滤指定日期的数据
            filtered_df = df[df['日期'].dt.date == query_date]

            if filtered_df.empty:
                output_lines.append(f"未找到瑞士央行在 '{date}' 的利率决议报告数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最新数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是瑞士央行在 '{date}' 的利率决议报告：")
                # 显示所有匹配该日期的数据
                df_to_display = filtered_df.copy()
                
        else:
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是瑞士央行最近 {len(df_to_display)} 条利率决议报告：")
            
        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['商品', '日期', '今值', '预测值', '前值']
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 应用格式化
        for col in available_cols:
            if col not in ['商品', '日期']: # 这些列不需要特殊格式化
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '日期' 列为字符串
        if '日期' in df_to_display.columns:
            df_to_display['日期'] = df_to_display['日期'].dt.strftime('%Y-%m-%d')

        # 将DataFrame转换为字符串输出
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含商品名称、日期、今值、预测值和前值。所有利率值单位为百分比(%)。")
        
        # 引导性回复
        if not date:
            output_lines.append(f"您当前看到的是瑞士央行最近 {len(df_to_display)} 条利率决议报告。")
            output_lines.append(f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2024-09-26'。")
            output_lines.append(f"该工具提供自 {df['日期'].min().strftime('%Y-%m-%d')} 至今的所有历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取瑞士央行利率决议报告数据时发生错误: {e}"

@tool
def get_uk_boe_interest_rate_report(
    date: Optional[str] = None
) -> str:
    """
    获取英国央行利率决议的历史报告数据。
    该接口提供自1970年1月1日至今的英国央行利率决议报告，
    包含商品名称、日期、今值、预测值和前值。

    用户可以通过 'date' 参数指定查询特定日期的数据，格式为 "YYYY-MM-DD"。
    如果未指定 'date'，则默认返回最新的10条利率决议数据。

    Args:
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               例如 "2024-08-01"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最新的10条数据。

    Returns:
        str: 一个格式化后的字符串，包含英国央行利率决议报告数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        # 利率值通常显示为百分比，保留两位小数
        if col_name in ['今值', '预测值', '前值']:
            return f"{float(value):.2f}%"
        return str(value)

    try:
        # 调用API
        df = ak.macro_bank_english_interest_rate()

        if df.empty:
            return "未能获取英国央行利率决议报告数据。可能当前无数据或数据源暂无数据。"

        # 确保 '日期' 列是 datetime 类型以便于日期过滤和排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            # 按照日期降序排列，以便获取最新数据
            df = df.sort_values(by='日期', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '日期' 列，无法处理。"

        output_lines = ["英国央行利率决议报告数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最新数据
        if date:
            try:
                query_date = pd.to_datetime(date).date()
            except ValueError:
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2024-08-01'。"
            
            # 过滤指定日期的数据
            filtered_df = df[df['日期'].dt.date == query_date]

            if filtered_df.empty:
                output_lines.append(f"未找到英国央行在 '{date}' 的利率决议报告数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最新数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是英国央行在 '{date}' 的利率决议报告：")
                # 显示所有匹配该日期的数据
                df_to_display = filtered_df.copy()
                
        else:
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是英国央行最近 {len(df_to_display)} 条利率决议报告：")
            
        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['商品', '日期', '今值', '预测值', '前值']
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 应用格式化
        for col in available_cols:
            if col not in ['商品', '日期']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '日期' 列为字符串
        if '日期' in df_to_display.columns:
            df_to_display['日期'] = df_to_display['日期'].dt.strftime('%Y-%m-%d')

        # 将DataFrame转换为字符串输出
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含商品名称、日期、今值、预测值和前值。所有利率值单位为百分比(%)。")
        
        # 引导性回复
        if not date:
            output_lines.append(f"您当前看到的是英国央行最近 {len(df_to_display)} 条利率决议报告。")
            output_lines.append(f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2024-08-01'。")
            output_lines.append(f"该工具提供自 {df['日期'].min().strftime('%Y-%m-%d')} 至今的所有历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取英国央行利率决议报告数据时发生错误: {e}"

@tool
def get_australia_rba_interest_rate_report(
    date: Optional[str] = None
) -> str:
    """
    获取澳洲联储利率决议的历史报告数据。
    该接口提供自1980年2月1日至今的澳洲联储利率决议报告，
    包含商品名称、日期、今值、预测值和前值。

    用户可以通过 'date' 参数指定查询特定日期的数据，格式为 "YYYY-MM-DD"。
    如果未指定 'date'，则默认返回最新的10条利率决议数据。

    Args:
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               例如 "2024-08-06"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最新的10条数据。

    Returns:
        str: 一个格式化后的字符串，包含澳洲联储利率决议报告数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        # 利率值通常显示为百分比，保留两位小数
        if col_name in ['今值', '预测值', '前值']:
            return f"{float(value):.2f}%"
        return str(value)

    try:
        # 调用API
        df = ak.macro_bank_australia_interest_rate()

        if df.empty:
            return "未能获取澳洲联储利率决议报告数据。可能当前无数据或数据源暂无数据。"

        # 确保 '日期' 列是 datetime 类型以便于日期过滤和排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            # 按照日期降序排列，以便获取最新数据
            df = df.sort_values(by='日期', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '日期' 列，无法处理。"

        output_lines = ["澳洲联储利率决议报告数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最新数据
        if date:
            try:
                query_date = pd.to_datetime(date).date()
            except ValueError:
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2024-08-06'。"
            
            # 过滤指定日期的数据
            filtered_df = df[df['日期'].dt.date == query_date]

            if filtered_df.empty:
                output_lines.append(f"未找到澳洲联储在 '{date}' 的利率决议报告数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最新数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是澳洲联储在 '{date}' 的利率决议报告：")
                # 显示所有匹配该日期的数据
                df_to_display = filtered_df.copy()
                
        else:
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是澳洲联储最近 {len(df_to_display)} 条利率决议报告：")
            
        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['商品', '日期', '今值', '预测值', '前值']
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 应用格式化
        for col in available_cols:
            if col not in ['商品', '日期']:
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '日期' 列为字符串
        if '日期' in df_to_display.columns:
            df_to_display['日期'] = df_to_display['日期'].dt.strftime('%Y-%m-%d')

        # 将DataFrame转换为字符串输出
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含商品名称、日期、今值、预测值和前值。所有利率值单位为百分比(%)。")
        
        # 引导性回复
        if not date:
            output_lines.append(f"您当前看到的是澳洲联储最近 {len(df_to_display)} 条利率决议报告。")
            output_lines.append(f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2024-08-06'。")
            output_lines.append(f"该工具提供自 {df['日期'].min().strftime('%Y-%m-%d')} 至今的所有历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取澳洲联储利率决议报告数据时发生错误: {e}"

@tool
def get_japan_boj_interest_rate_report(
    date: Optional[str] = None
) -> str:
    """
    获取日本央行利率决议的历史报告数据。
    该接口提供自2008年2月14日至今的日本央行利率决议报告，
    包含商品名称、日期、今值、预测值和前值。

    用户可以通过 'date' 参数指定查询特定日期的数据，格式为 "YYYY-MM-DD"。
    如果未指定 'date'，则默认返回最新的10条利率决议数据。

    Args:
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               例如 "2024-07-31"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最新的10条数据。

    Returns:
        str: 一个格式化后的字符串，包含日本央行利率决议报告数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        # 利率值通常显示为百分比，保留两位小数
        if col_name in ['今值', '预测值', '前值']:
            return f"{float(value):.2f}%"
        return str(value)

    try:
        # 调用API
        df = ak.macro_bank_japan_interest_rate()

        if df.empty:
            return "未能获取日本央行利率决议报告数据。可能当前无数据或数据源暂无数据。"

        # 确保 '日期' 列是 datetime 类型以便于日期过滤和排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            # 按照日期降序排列，以便获取最新数据
            df = df.sort_values(by='日期', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '日期' 列，无法处理。"

        output_lines = ["日本央行利率决议报告数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最新数据
        if date:
            try:
                query_date = pd.to_datetime(date).date()
            except ValueError:
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2024-07-31'。"
            
            # 过滤指定日期的数据
            filtered_df = df[df['日期'].dt.date == query_date]

            if filtered_df.empty:
                output_lines.append(f"未找到日本央行在 '{date}' 的利率决议报告数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最新数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是日本央行在 '{date}' 的利率决议报告：")
                # 显示所有匹配该日期的数据
                df_to_display = filtered_df.copy()
                
        else:
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是日本央行最近 {len(df_to_display)} 条利率决议报告：")
            
        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['商品', '日期', '今值', '预测值', '前值']
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 应用格式化
        for col in available_cols:
            if col not in ['商品', '日期']: # 这些列不需要特殊格式化
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '日期' 列为字符串
        if '日期' in df_to_display.columns:
            df_to_display['日期'] = df_to_display['日期'].dt.strftime('%Y-%m-%d')

        # 将DataFrame转换为字符串输出
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含商品名称、日期、今值、预测值和前值。所有利率值单位为百分比(%)。")
        
        # 引导性回复
        if not date:
            output_lines.append(f"您当前看到的是日本央行最近 {len(df_to_display)} 条利率决议报告。")
            output_lines.append(f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2024-07-31'。")
            output_lines.append(f"该工具提供自 {df['日期'].min().strftime('%Y-%m-%d')} 至今的所有历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取日本央行利率决议报告数据时发生错误: {e}"

@tool
def get_russia_cbr_interest_rate_report(
    date: Optional[str] = None
) -> str:
    """
    获取俄罗斯央行利率决议的历史报告数据。
    该接口提供自2003年6月1日至今的俄罗斯央行利率决议报告，
    包含商品名称、日期、今值、预测值和前值。

    用户可以通过 'date' 参数指定查询特定日期的数据，格式为 "YYYY-MM-DD"。
    如果未指定 'date'，则默认返回最新的10条利率决议数据。

    Args:
        date (Optional[str]): 可选参数。指定查询的日期，格式为 "YYYY-MM-DD"。
                               例如 "2024-07-26"。
                               如果提供，则返回该日期的数据。
                               如果未提供，则返回最新的10条数据。

    Returns:
        str: 一个格式化后的字符串，包含俄罗斯央行利率决议报告数据。
             如果获取失败、未找到数据或参数格式不正确，则返回错误或提示信息。
    """
    if ak is None:
        return "错误：akshare 库未安装。请先安装该库才能使用此工具。"

    # 定义数值格式化辅助函数
    def format_value(value, col_name):
        if pd.isna(value):
            return "N/A"
        # 利率值通常显示为百分比，保留两位小数
        if col_name in ['今值', '预测值', '前值']:
            return f"{float(value):.2f}%"
        return str(value)

    try:
        # 调用APi
        df = ak.macro_bank_russia_interest_rate()

        if df.empty:
            return "未能获取俄罗斯央行利率决议报告数据。可能当前无数据或数据源暂无数据。"

        # 确保 '日期' 列是 datetime 类型以便于日期过滤和排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            # 按照日期降序排列，以便获取最新数据
            df = df.sort_values(by='日期', ascending=False).reset_index(drop=True)
        else:
            return f"错误：获取到的数据中缺少 '日期' 列，无法处理。"

        output_lines = ["俄罗斯央行利率决议报告数据。\n"]
        output_lines.append("=" * 100)

        # 根据 date 参数进行过滤或返回最新数据
        if date:
            try:
                query_date = pd.to_datetime(date).date()
            except ValueError:
                return f"错误：'date' 参数格式不正确。请使用 'YYYY-MM-DD' 格式，例如 '2024-07-26'。"
            
            # 过滤指定日期的数据
            filtered_df = df[df['日期'].dt.date == query_date]

            if filtered_df.empty:
                output_lines.append(f"未找到俄罗斯央行在 '{date}' 的利率决议报告数据。")
                output_lines.append("=" * 100)
                output_lines.append("提示：您可以尝试查询其他日期，或不指定日期以查看最新数据。")
                return "\n".join(output_lines)
            else:
                output_lines.append(f"以下是俄罗斯央行在 '{date}' 的利率决议报告：")
                # 显示所有匹配该日期的数据
                df_to_display = filtered_df.copy()
                
        else:
            df_to_display = df.head(MAX_ROWS_TO_DISPLAY).copy()
            output_lines.append(f"以下是俄罗斯央行最近 {len(df_to_display)} 条利率决议报告：")
            
        # 定义要显示的列，并确保它们在DataFrame中存在
        display_cols = ['商品', '日期', '今值', '预测值', '前值']
        available_cols = [col for col in display_cols if col in df_to_display.columns]

        # 应用格式化
        for col in available_cols:
            if col not in ['商品', '日期']: # 这些列不需要特殊格式化
                df_to_display[col] = df_to_display[col].apply(lambda x: format_value(x, col))
        
        # 格式化 '日期' 列为字符串
        if '日期' in df_to_display.columns:
            df_to_display['日期'] = df_to_display['日期'].dt.strftime('%Y-%m-%d')

        # 将DataFrame转换为字符串输出
        output_lines.append(df_to_display[available_cols].to_string(index=False))

        output_lines.append("=" * 100)
        output_lines.append("提示：数据包含商品名称、日期、今值、预测值和前值。所有利率值单位为百分比(%)。")
        
        # 引导性回复
        if not date:
            output_lines.append(f"您当前看到的是俄罗斯央行最近 {len(df_to_display)} 条利率决议报告。")
            output_lines.append(f"如果想查询特定日期的数据，请将 'date' 参数设置为 'YYYY-MM-DD' 格式，例如：date='2024-07-26'。")
            output_lines.append(f"该工具提供自 {df['日期'].min().strftime('%Y-%m-%d')} 至今的所有历史数据。")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取俄罗斯央行利率决议报告数据时发生错误: {e}"

@tool
def get_china_national_tax_receipts(num_rows: Optional[int] = 10) -> str:
    """
    获取中国全国税收收入数据。
    该接口返回自 2005 年第一季度至今的所有历史数据。
    数据包括季度（日期）、税收收入合计（单位：亿元）、较上年同期增长率（单位：%）和季度环比增长率。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  如果指定的条数大于可用数据总量，将返回所有数据。

    Returns:
        str: 中国全国税收收入数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要。
    """
    try:
        # 调用API
        df = ak.macro_china_national_tax_receipts()

        if df.empty:
            return "未能获取到中国全国税收收入数据，返回数据为空。"

        # 根据 num_rows 参数返回数据
        if num_rows is None or num_rows <= 0:
            # 返回所有数据
            return f"中国全国税收收入数据 (所有历史数据):\n{df.to_string()}"
        else:
            # 返回指定条数的最新数据
            return f"中国全国税收收入数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取中国全国税收收入数据时发生错误: {e}"

@tool
def get_china_enterprise_boom_index(num_rows: Optional[int] = 10) -> str:
    """
    获取中国企业景气及企业家信心指数数据。
    该接口返回自 2005 年第一季度至今的所有历史数据。
    数据包括季度（日期）、企业景气指数（指数、同比、环比）和企业家信心指数（指数、同比、环比）。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  如果指定的条数大于可用数据总量，将返回所有数据。

    Returns:
        str: 中国企业景气及企业家信心指数数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要。
    """
    try:
        # 调用API
        df = ak.macro_china_enterprise_boom_index()

        if df.empty:
            return "未能获取到中国企业景气及企业家信心指数数据，返回数据为空。"

        # 根据 num_rows 参数返回数据
        if num_rows is None or num_rows <= 0:
            # 返回所有数据
            return f"中国企业景气及企业家信心指数数据 (所有历史数据):\n{df.to_string()}"
        else:
            # 返回指定条数的最新数据
            return f"中国企业景气及企业家信心指数数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取中国企业景气及企业家信心指数数据时发生错误: {e}"

@tool
def get_china_qyspjg(
   num_rows: Optional[int] = 10,
   date: Optional[str] = None,
) -> str:
    """
    获取中国企业商品价格指数数据。
    该接口返回自 2005 年 1 月至今的所有历史数据。
    数据包括月份、总指数（指数值、同比增长、环比增长）、农产品指数（指数值、同比增长、环比增长）、
    矿产品指数（指数值、同比增长、环比增长）以及煤油电指数（指数值、同比增长、环比增长）。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个月份数据，格式为 'YYYY-MM' 或 'YYYYMM'。
                              例如：'2022-09' 或 '202209'。
                              如果设置此参数，将返回该月份的详细数据，并忽略 `num_rows` 参数。

    Returns:
        str: 中国企业商品价格指数数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定月份的详细数据。
    """
    try:
        # 调用API
        df = ak.macro_china_qyspjg()

        if df.empty:
            return "未能获取到中国企业商品价格指数数据，返回数据为空。"

        # 处理 date 参数
        if date is not None:
            try:
                # 解析日期
                if '-' in date:
                    parsed_date = datetime.strptime(date, '%Y-%m')
                else:
                    parsed_date = datetime.strptime(date, '%Y%m')

                # 将解析后的日期格式化为与 '月份' 列匹配的字符串
                formatted_date_str = parsed_date.strftime('%Y年%m月份')

                # 过滤 DataFrame，查找匹配的日期
                monthly_data = df[df['月份'] == formatted_date_str]

                if not monthly_data.empty:
                    return f"中国企业商品价格指数在 {formatted_date_str} 的数据:\n{monthly_data.to_string(index=False)}"
                else:
                    return f"无法找到中国企业商品价格指数在 {formatted_date_str} 的数据。请检查日期是否正确或该月份是否有数据。"
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM' 或 'YYYYMM' 格式，例如 '2022-09' 或 '202209'。"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"中国企业商品价格指数数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"中国企业商品价格指数数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取中国企业商品价格指数数据时发生错误: {e}"

@tool
def get_china_stock_market_cap(
   num_rows: Optional[int] = 10,
   date: Optional[str] = None,
) -> str:
    """
    获取全国股票交易统计表数据。
    该接口返回自 2008 年 1 月至今的月度数据。
    数据包括数据日期、上海和深圳的发行总股本、市价总值、成交金额、成交量，
    以及上海和深圳的A股最高和最低综合股价指数。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个月份数据，格式为 'YYYY-MM' 或 'YYYYMM'。
                              例如：'2022-11' 或 '202211'。
                              如果设置此参数，将返回该月份的详细数据，并忽略 `num_rows` 参数。

    Returns:
        str: 全国股票交易统计表数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定月份的详细数据。
    """
    try:
        # 调用API
        df = ak.macro_china_stock_market_cap()

        if df.empty:
            return "未能获取到全国股票交易统计表数据，返回数据为空。"

        # 处理 date 参数
        if date is not None:
            try:
                # 解析日期
                if '-' in date:
                    parsed_date = datetime.strptime(date, '%Y-%m')
                else:
                    parsed_date = datetime.strptime(date, '%Y%m')

                # 将解析后的日期格式化为与 '数据日期' 列匹配的字符串
                formatted_date_str = parsed_date.strftime('%Y年%m月份')

                # 过滤 DataFrame，查找匹配的日期
                monthly_data = df[df['数据日期'] == formatted_date_str]

                if not monthly_data.empty:
                    return f"全国股票交易统计表在 {formatted_date_str} 的数据:\n{monthly_data.to_string(index=False)}"
                else:
                    return f"无法找到全国股票交易统计表在 {formatted_date_str} 的数据。请检查日期是否正确或该月份是否有数据。"
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM' 或 'YYYYMM' 格式，例如 '2022-11' 或 '202211'。"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"全国股票交易统计表数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"全国股票交易统计表数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取全国股票交易统计表数据时发生错误: {e}"

@tool
def get_rmb_loan_data(
   num_rows: Optional[int] = 10,
   date: Optional[str] = None,
) -> str:
    """
    获取中国新增人民币贷款数据。
    该接口返回同花顺-数据中心-宏观数据中，自 2008 年 1 月至今的月度新增人民币贷款及累计人民币贷款数据。
    数据包括月份、新增人民币贷款（总额、同比、环比）和累计人民币贷款（总额、同比）。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个月份数据，格式为 'YYYY-MM'。
                              例如：'2024-06'。
                              如果设置此参数，将返回该月份的详细数据，并忽略 `num_rows` 参数。

    Returns:
        str: 新增人民币贷款数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定月份的详细数据。
    """
    try:
        # 调用API
        df = ak.macro_rmb_loan()

        if df.empty:
            return "未能获取到新增人民币贷款数据，返回数据为空。"

        # 处理 date 参数
        if date is not None:
            try:
                # 验证日期格式
                # 将输入日期解析为 datetime 对象，确保格式正确
                datetime.strptime(date, '%Y-%m')

                # 过滤 DataFrame，查找匹配的日期
                monthly_data = df[df['月份'] == date]

                if not monthly_data.empty:
                    return f"新增人民币贷款在 {date} 的数据:\n{monthly_data.to_string(index=False)}"
                else:
                    return f"无法找到新增人民币贷款在 {date} 的数据。请检查日期是否正确或该月份是否有数据。"
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM' 格式，例如 '2024-06'。"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"新增人民币贷款数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"新增人民币贷款数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取新增人民币贷款数据时发生错误: {e}"

@tool
def get_rmb_deposit_data(
   num_rows: Optional[int] = 10,
   date: Optional[str] = None,
) -> str:
    """
    获取中国人民币存款余额数据。
    该接口返回同花顺-数据中心-宏观数据中，人民币存款余额的月度数据。
    数据包括月份、新增存款（数量、同比、环比）、新增企业存款（数量、同比、环比）、
    新增储蓄存款（数量、同比、环比）以及新增其他存款（数量、同比、环比）。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个月份数据，格式为 'YYYY-MM'。
                              例如：'2024-06'。
                              如果设置此参数，将返回该月份的详细数据，并忽略 `num_rows` 参数。

    Returns:
        str: 人民币存款余额数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定月份的详细数据。
    """
    try:
        # 调用API
        df = ak.macro_rmb_deposit()

        if df.empty:
            return "未能获取到人民币存款余额数据，返回数据为空。"

        # 处理 date 参数
        if date is not None:
            try:
                # 验证日期格式
                # 将输入日期解析为 datetime 对象，确保格式正确
                datetime.strptime(date, '%Y-%m')

                # 过滤 DataFrame，查找匹配的日期
                monthly_data = df[df['月份'] == date]

                if not monthly_data.empty:
                    return f"人民币存款余额在 {date} 的数据:\n{monthly_data.to_string(index=False)}"
                else:
                    return f"无法找到人民币存款余额在 {date} 的数据。请检查日期是否正确或该月份是否有数据。"
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM' 格式，例如 '2024-06'。"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"人民币存款余额数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"人民币存款余额数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取人民币存款余额数据时发生错误: {e}"
@tool
def get_china_fx_gold_reserves(
   num_rows: Optional[int] = 10,
   date: Optional[str] = None,
) -> str:
    """
    获取中国外汇和黄金储备数据。
    该接口返回自 2008 年 1 月至今的月度数据。
    数据包括月份、黄金储备（数值、同比、环比）和国家外汇储备（数值、同比、环比）。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个月份数据，格式为 'YYYY-MM' 或 'YYYYMM'。
                              例如：'2022-07' 或 '202207'。
                              如果设置此参数，将返回该月份的详细数据，并忽略 `num_rows` 参数。

    Returns:
        str: 中国外汇和黄金储备数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定月份的详细数据。
    """
    try:
        # 调用API
        df = ak.macro_china_fx_gold()

        if df.empty:
            return "未能获取到中国外汇和黄金储备数据，返回数据为空。"

        # 处理 date 参数
        if date is not None:
            try:
                # 解析日期
                if '-' in date:
                    parsed_date = datetime.strptime(date, '%Y-%m')
                else:
                    parsed_date = datetime.strptime(date, '%Y%m')

                # 将解析后的日期格式化为与 '月份' 列匹配的字符串
                formatted_date_str = parsed_date.strftime('%Y年%m月份')

                # 过滤 DataFrame，查找匹配的日期
                monthly_data = df[df['月份'] == formatted_date_str]

                if not monthly_data.empty:
                    return f"中国外汇和黄金储备在 {formatted_date_str} 的数据:\n{monthly_data.to_string(index=False)}"
                else:
                    return f"无法找到中国外汇和黄金储备在 {formatted_date_str} 的数据。请检查日期是否正确或该月份是否有数据。"
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM' 或 'YYYYMM' 格式，例如 '2022-07' 或 '202207'。"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"中国外汇和黄金储备数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"中国外汇和黄金储备数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取中国外汇和黄金储备数据时发生错误: {e}"


@tool
def get_china_money_supply_data(
    num_rows: Optional[int] = 10,
    date: Optional[str] = None,
) -> str:
    """
    获取中国货币供应量数据。
    该接口返回东方财富-经济数据-中国宏观中，自 2008 年 1 月至今的月度货币供应量数据。
    数据包括月份、货币和准货币(M2)（数量、同比增长、环比增长）、
    货币(M1)（数量、同比增长、环比增长）以及流通中的现金(M0)（数量、同比增长、环比增长）。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个月份数据，格式为 'YYYY-MM' 或 'YYYYMM'。
                              例如：'2022-10' 或 '202210'。
                              如果设置此参数，将返回该月份的详细数据，并忽略 `num_rows` 参数。

    Returns:
        str: 中国货币供应量数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定月份的详细数据。
    """
    try:
        # 调用API
        df = ak.macro_china_money_supply()

        if df.empty:
            return "未能获取到中国货币供应量数据，返回数据为空。"

        # 处理 date 参数
        if date is not None:
            try:
                # 解析日期
                if '-' in date:
                    parsed_date = datetime.strptime(date, '%Y-%m')
                else:
                    parsed_date = datetime.strptime(date, '%Y%m')

                # 将解析后的日期格式化为与 '月份' 列匹配的字符串
                formatted_date_str = parsed_date.strftime('%Y年%m月份')

                # 过滤 DataFrame，查找匹配的日期
                monthly_data = df[df['月份'] == formatted_date_str]

                if not monthly_data.empty:
                    return f"中国货币供应量在 {formatted_date_str} 的数据:\n{monthly_data.to_string(index=False)}"
                else:
                    return f"无法找到中国货币供应量在 {formatted_date_str} 的数据。请检查日期是否正确或该月份是否有数据。"
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY-MM' 或 'YYYYMM' 格式，例如 '2022-10' 或 '202210'。"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"中国货币供应量数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"中国货币供应量数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取中国货币供应量数据时发生错误: {e}"


@tool
def get_china_urban_unemployment_data(
        num_rows: Optional[int] = 10,
        date: Optional[str] = None,
        item_type: Optional[str] = None,
) -> str:
    """
    获取中国城镇调查失业率数据。
    该接口返回国家统计局的月度城镇调查失业率数据，数据包含多种失业率分类。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个月份数据，格式为 'YYYYMM'。
                              例如：'202304'。
                              如果设置此参数，将返回该月份的详细数据，并忽略 `num_rows` 参数。
        item_type (Optional[str]): 指定要查询的失业率项目类型。
                                   例如：'全国城镇调查失业率', '全国城镇16—24岁劳动力失业率',
                                   '全国城镇25—59岁劳动力失业率', '全国城镇本地户籍劳动力失业率' 等。
                                   如果指定此参数，结果将只包含该项目类型的数据。

    Returns:
        str: 城镇调查失业率数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定月份和/或项目类型的详细数据。
    """
    try:
        # 调用API
        df = ak.macro_china_urban_unemployment()

        if df.empty:
            return "未能获取到城镇调查失业率数据，返回数据为空。"

        # 确保 'date' 列是字符串类型，以便进行字符串比较
        df['date'] = df['date'].astype(str)

        # 过滤数据
        filtered_df = df.copy()

        if date is not None:
            # 验证日期格式
            if not (isinstance(date, str) and len(date) == 6 and date.isdigit()):
                return f"日期格式 '{date}' 无效，请使用 'YYYYMM' 格式，例如 '202304'。"

            # 过滤指定日期的数据
            filtered_df = filtered_df[filtered_df['date'] == date]
            if filtered_df.empty:
                return f"无法找到城镇调查失业率在 {date} 的数据。请检查日期是否正确或该月份是否有数据。"

            # 如果指定了日期，则忽略 num_rows，直接返回该日期的所有或指定item_type数据
            if item_type is not None:
                final_df = filtered_df[filtered_df['item'] == item_type]
                if final_df.empty:
                    return f"在 {date} 无法找到 '{item_type}' 的城镇调查失业率数据。请检查项目类型是否正确。"
                return f"城镇调查失业率在 {date} 的 '{item_type}' 数据:\n{final_df.to_string(index=False)}"
            else:
                return f"城镇调查失业率在 {date} 的所有项目数据:\n{filtered_df.to_string(index=False)}"
        else:
            # 如果没有指定 date，则根据 item_type 和 num_rows 返回数据
            if item_type is not None:
                filtered_df = filtered_df[filtered_df['item'] == item_type]
                if filtered_df.empty:
                    return f"无法找到 '{item_type}' 的城镇调查失业率数据。请检查项目类型是否正确。"

                # 对过滤后的数据按日期降序排序，以便取最新的N条
                filtered_df = filtered_df.sort_values(by='date', ascending=False)

                if num_rows is None or num_rows <= 0:
                    return f"'{item_type}' 的所有历史城镇调查失业率数据:\n{filtered_df.to_string(index=False)}"
                else:
                    return f"'{item_type}' 的最新 {num_rows} 条城镇调查失业率数据:\n{filtered_df.head(num_rows).to_string(index=False)}"
            else:
                # 既没有指定 date 也没有指定 item_type
                # 对原始数据按日期降序排序，以便取最新的N条
                filtered_df = filtered_df.sort_values(by='date', ascending=False)

                if num_rows is None or num_rows <= 0:
                    return f"城镇调查失业率数据 (所有历史数据):\n{filtered_df.to_string(index=False)}"
                else:
                    return f"城镇调查失业率数据 (最新 {num_rows} 条):\n{filtered_df.head(num_rows).to_string(index=False)}"

    except Exception as e:
        return f"获取城镇调查失业率数据时发生错误: {e}"

@tool
def get_hk_unemployment_rate(
   num_rows: Optional[int] = 10,
   date: Optional[str] = None,
) -> str:
    """
    获取中国香港失业率数据。
    该接口返回东方财富-经济数据一览-中国香港的失业率月度数据。
    数据包括时间、前值、现值和发布日期。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个月份数据。支持以下格式：
                              - 'YYYY年MM月' (例如：'2024年03月')
                              - 'YYYY-MM' (例如：'2024-03')
                              - 'YYYYMM' (例如：'202403')
                              函数内部会将 'YYYY-MM' 或 'YYYYMM' 格式转换为 DataFrame 中实际的
                              'YYYY年MM月' 格式进行匹配。
                              如果设置此参数，将返回该月份的详细数据，并忽略 `num_rows` 参数。

    Returns:
        str: 中国香港失业率数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定月份的详细数据。
    """
    try:
        # 调用API
        df = ak.macro_china_hk_rate_of_unemployment()

        if df.empty:
            return "未能获取到中国香港失业率数据，返回数据为空。"

        # 确保 '时间' 列是字符串类型，以便进行字符串比较
        df['时间'] = df['时间'].astype(str)

        # 处理 date 参数
        if date is not None:
            formatted_date_str = None
            try:
                # 直接匹配 'YYYY年MM月' 格式
                if '年' in date and '月' in date:
                    formatted_date_str = date
                else:
                    # 解析 'YYYY-MM' 或 'YYYYMM' 格式
                    if '-' in date:
                        parsed_date = datetime.strptime(date, '%Y-%m')
                    else:
                        parsed_date = datetime.strptime(date, '%Y%m')
                    # 将解析后的日期转换为 DataFrame '时间' 列的实际格式 'YYYY年MM月'
                    formatted_date_str = parsed_date.strftime('%Y年%m月')

                # 过滤 DataFrame，查找匹配的日期
                monthly_data = df[df['时间'] == formatted_date_str]

                if not monthly_data.empty:
                    return f"中国香港失业率在 {formatted_date_str} 的数据:\n{monthly_data.to_string(index=False)}"
                else:
                    return f"无法找到中国香港失业率在 {formatted_date_str} 的数据。请检查日期是否正确或该月份是否有数据。"
            except ValueError:
                return f"日期格式 '{date}' 无效，请使用 'YYYY年MM月'、'YYYY-MM' 或 'YYYYMM' 格式，例如 '2024年03月' 或 '2024-03'。"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"中国香港失业率数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"中国香港失业率数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取中国香港失业率数据时发生错误: {e}"


@tool
def get_hk_gdp_data(
   num_rows: Optional[int] = 10,
   date: Optional[str] = None,
) -> str:
    """
    获取中国香港 GDP 数据。
    该接口返回东方财富-经济数据一览-中国香港的季度 GDP 数据。
    数据包括时间、前值、现值和发布日期。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个季度数据。支持以下格式：
                              - 'YYYY第Q季度' (例如：'2024第1季度') - 直接匹配 DataFrame 中的格式
                              - 'YYYY-Q' (例如：'2024-1') - Q为季度数(1-4)
                              - 'YYYYQ' (例如：'20241') - Q为季度数(1-4)
                              - 'YYYY-Q季度' (例如：'2024-1季度')
                              函数内部会将其他格式转换为 DataFrame 中实际的 'YYYY第Q季度' 格式进行匹配。
                              如果设置此参数，将返回该季度的数据，并忽略 `num_rows` 参数。

    Returns:
        str: 中国香港 GDP 数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定季度的数据。
    """
    try:
        # 调用API
        df = ak.macro_china_hk_gbp()

        if df.empty:
            return "未能获取到中国香港 GDP 数据，返回数据为空。"

        # 确保 '时间' 列是字符串类型，以便进行字符串比较
        df['时间'] = df['时间'].astype(str)

        # 处理 date 参数
        if date is not None:
            formatted_date_str = None
            try:
                # 直接匹配 'YYYY第Q季度' 格式
                if '第' in date and '季度' in date:
                    formatted_date_str = date
                else:
                    # 解析 'YYYY-Q', 'YYYYQ', 'YYYY-Q季度' 等格式
                    match = re.match(r'(\d{4})[ -]?[Qq]?(\d)[季度]?', date, re.IGNORECASE)
                    if match:
                        year = match.group(1)
                        quarter = match.group(2)
                        if quarter in ['1', '2', '3', '4']:
                            formatted_date_str = f"{year}第{quarter}季度"
                        else:
                            raise ValueError("季度数无效，请使用 1 到 4。")
                    else:
                        raise ValueError("日期格式无法识别。")

                # 过滤 DataFrame，查找匹配的日期
                quarterly_data = df[df['时间'] == formatted_date_str]

                if not quarterly_data.empty:
                    return f"中国香港 GDP 在 {formatted_date_str} 的数据:\n{quarterly_data.to_string(index=False)}"
                else:
                    return f"无法找到中国香港 GDP 在 {formatted_date_str} 的数据。请检查日期是否正确或该季度是否有数据。"
            except ValueError as ve:
                return f"日期格式 '{date}' 无效。请使用 'YYYY第Q季度'、'YYYY-Q' 或 'YYYYQ' 等格式，例如 '2024第1季度' 或 '2024-1'。错误详情: {ve}"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"中国香港 GDP 数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"中国香港 GDP 数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取中国香港 GDP 数据时发生错误: {e}"


@tool
def get_hk_gdp_yoy_data(
   num_rows: Optional[int] = 10,
   date: Optional[str] = None,
) -> str:
    """
    获取中国香港 GDP 同比增长数据。
    该接口返回东方财富-经济数据一览-中国香港的季度 GDP 同比增长数据。
    数据包括时间、前值、现值和发布日期。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个季度数据。支持以下格式：
                              - 'YYYY第Q季度' (例如：'2024第1季度') - 直接匹配 DataFrame 中的格式
                              - 'YYYY-Q' (例如：'2024-1') - Q为季度数(1-4)
                              - 'YYYYQ' (例如：'20241') - Q为季度数(1-4)
                              - 'YYYY-Q季度' (例如：'2024-1季度')
                              函数内部会将其他格式转换为 DataFrame 中实际的 'YYYY第Q季度' 格式进行匹配。
                              如果设置此参数，将返回该季度的数据，并忽略 `num_rows` 参数。

    Returns:
        str: 中国香港 GDP 同比增长数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定季度的数据。
    """
    try:
        # 调用API
        df = ak.macro_china_hk_gbp_ratio()

        if df.empty:
            return "未能获取到中国香港 GDP 同比增长数据，返回数据为空。"

        # 确保 '时间' 列是字符串类型，以便进行字符串比较
        df['时间'] = df['时间'].astype(str)

        # 处理 date 参数
        if date is not None:
            formatted_date_str = None
            try:
                # 直接匹配 'YYYY第Q季度' 格式
                if '第' in date and '季度' in date:
                   formatted_date_str = date
                else:
                    # 解析 'YYYY-Q', 'YYYYQ', 'YYYY-Q季度' 等格式
                    match = re.match(r'(\d{4})[ -]?[Qq]?(\d)[季度]?', date, re.IGNORECASE)
                    if match:
                        year = match.group(1)
                        quarter = match.group(2)
                        if quarter in ['1', '2', '3', '4']:
                            formatted_date_str = f"{year}第{quarter}季度"
                        else:
                            raise ValueError("季度数无效，请使用 1 到 4。")
                    else:
                        raise ValueError("日期格式无法识别。")

                # 过滤 DataFrame，查找匹配的日期
                quarterly_data = df[df['时间'] == formatted_date_str]

                if not quarterly_data.empty:
                    return f"中国香港 GDP 同比增长在 {formatted_date_str} 的数据:\n{quarterly_data.to_string(index=False)}"
                else:
                    return f"无法找到中国香港 GDP 同比增长在 {formatted_date_str} 的数据。请检查日期是否正确或该季度是否有数据。"
            except ValueError as ve:
                return f"日期格式 '{date}' 无效。请使用 'YYYY第Q季度'、'YYYY-Q' 或 'YYYYQ' 等格式，例如 '2024第1季度' 或 '2024-1'。错误详情: {ve}"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"中国香港 GDP 同比增长数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"中国香港 GDP 同比增长数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取中国香港 GDP 同比增长数据时发生错误: {e}"


@tool
def get_usa_gdp_data(
   num_rows: Optional[int] = 10,
   date: Optional[str] = None,
) -> str:
    """
    获取美国国内生产总值(GDP)报告数据。
    该接口返回金十数据提供的美国国内生产总值(GDP)报告，数据区间从 2008 年 2 月 28 日至今，为月度数据。
    数据包括商品类型（固定为“美国国内生产总值(GDP)”）、日期、今值、预测值和前值。

    Args:
        num_rows (Optional[int]): 指定返回数据的条数。默认为 10 条最新数据。
                                  如果设置为 None 或 0，则返回所有历史数据。
                                  此参数在 `date` 参数未指定时生效。
        date (Optional[str]): 指定要查询的单个报告日期。支持以下格式：
                              - 'YYYY-MM-DD' (例如：'2024-04-25') - 直接匹配 DataFrame 中的格式
                              - 'YYYY/MM/DD' (例如：'2024/04/25')
                              - 'YYYYMMDD' (例如：'20240425')
                              函数内部会将其他格式转换为 DataFrame 中实际的 'YYYY-MM-DD' 格式进行匹配。
                              如果设置此参数，将返回该日期的详细数据，并忽略 `num_rows` 参数。

    Returns:
        str: 美国国内生产总值(GDP)报告数据的字符串表示。如果获取失败，则返回错误信息。
             通常返回指定条数的最新数据摘要，或指定日期的详细数据。
    """
    try:
        # 调用API
        df = ak.macro_usa_gdp_monthly()

        if df.empty:
            return "未能获取到美国国内生产总值(GDP)数据，返回数据为空。"

        # 确保 '日期' 列是字符串类型，以便进行字符串比较
        df['日期'] = df['日期'].astype(str)

        # 处理 date 参数
        if date is not None:
            formatted_date_str = None
            try:
                # 解析多种日期格式并转换为 'YYYY-MM-DD'
                if '-' in date:
                    parsed_date = datetime.strptime(date, '%Y-%m-%d')
                elif '/' in date:
                    parsed_date = datetime.strptime(date, '%Y/%m/%d')
                elif len(date) == 8 and date.isdigit():
                    parsed_date = datetime.strptime(date, '%Y%m%d')
                else:
                    raise ValueError("日期格式无法识别。")

                # 将解析后的日期转换为 DataFrame '日期' 列的实际格式 'YYYY-MM-DD'
                formatted_date_str = parsed_date.strftime('%Y-%m-%d')

                # 过滤 DataFrame，查找匹配的日期
                daily_data = df[df['日期'] == formatted_date_str]

                if not daily_data.empty:
                    return f"美国国内生产总值(GDP)在 {formatted_date_str} 的数据:\n{daily_data.to_string(index=False)}"
                else:
                    return f"无法找到美国国内生产总值(GDP)在 {formatted_date_str} 的数据。请检查日期是否正确或该日期是否有数据。"
            except ValueError as ve:
                return f"日期格式 '{date}' 无效。请使用 'YYYY-MM-DD'、'YYYY/MM/DD' 或 'YYYYMMDD' 格式，例如 '2024-04-25'。错误详情: {ve}"
            except Exception as e:
                return f"处理日期参数时发生错误: {e}"
        else:
            # 如果没有指定 date，则根据 num_rows 返回数据
            if num_rows is None or num_rows <= 0:
                # 返回所有数据
                return f"美国国内生产总值(GDP)数据 (所有历史数据):\n{df.to_string()}"
            else:
                # 返回指定条数的最新数据
                return f"美国国内生产总值()数据 (最新 {num_rows} 条):\n{df.tail(num_rows).to_string()}"

    except Exception as e:
        return f"获取美国国内生产总值(GDP)数据时发生错误: {e}"


@tool
def get_global_macro_calendar(
   date: str,
) -> str:
    """
    获取华尔街见闻-日历-宏观数据（全球宏观经济事件日历）。
    该接口返回指定日期的全球宏观经济事件，包括事件时间、地区、事件描述、重要性、
    今值、预期值、前值以及相关链接。

    Args:
        date (str): 指定要查询的日期。支持以下格式：
                    - 'YYYYMMDD' (例如：'20240514') - AkShare 接口要求的格式
                    - 'YYYY-MM-DD' (例如：'2024-05-14') - 函数内部会转换为 YYYYMMDD 格式

    Returns:
        str: 指定日期的全球宏观经济日历数据的字符串表示。如果获取失败或指定日期无数据，则返回错误信息。
    """
    try:
        # 验证并格式化日期
        formatted_date = None
        try:
            if '-' in date:
                # 解析 'YYYY-MM-DD' 格式
                parsed_date = datetime.strptime(date, '%Y-%m-%d')
                formatted_date = parsed_date.strftime('%Y%m%d')
            elif len(date) == 8 and date.isdigit():
                # 已经是 'YYYYMMDD' 格式，额外验证日期是否合法
                datetime.strptime(date, '%Y%m%d')
                formatted_date = date
            else:
                raise ValueError("日期格式无效。")
        except ValueError as ve:
            return f"日期格式 '{date}' 无效。请使用 'YYYYMMDD' 或 'YYYY-MM-DD' 格式，例如 '20240514' 或 '2024-05-14'。错误详情: {ve}"

        # 调用API
        df = ak.macro_info_ws(date=formatted_date)

        if df.empty:
            return f"未能获取到 {formatted_date} 的全球宏观经济日历数据，返回数据为空或该日期无事件。"

        # 格式化输出
        # 确保所有列都可见，并去除索引
        return f"{formatted_date} 的全球宏观经济日历数据:\n{df.to_string(index=False)}"

    except Exception as e:
        return f"获取全球宏观经济日历数据时发生错误: {e}"


@tool
def get_global_macro_events(
   date: str,
) -> str:
    """
    获取全球宏观指标重大事件数据。
    该接口返回百度股市通提供的指定日期的全球宏观经济事件，包括日期、时间、地区、事件描述、
    公布值、预期值、前值以及重要性。

    Args:
        date (str): 指定要查询的日期。支持以下格式：
                    - 'YYYYMMDD' (例如：'20241107') - AkShare 接口要求的格式
                    - 'YYYY-MM-DD' (例如：'2024-11-07') - 函数内部会转换为 YYYYMMDD 格式

    Returns:
        str: 指定日期的全球宏观经济事件数据的字符串表示。如果获取失败或指定日期无数据，则返回错误信息。
    """
    try:
        # 验证并格式化日期
        formatted_date = None
        try:
            if '-' in date:
                # 尝试解析 'YYYY-MM-DD' 格式
                parsed_date = datetime.strptime(date, '%Y-%m-%d')
                formatted_date = parsed_date.strftime('%Y%m%d')
            elif len(date) == 8 and date.isdigit():
                # 已经是 'YYYYMMDD' 格式，额外验证日期是否合法
                datetime.strptime(date, '%Y%m%d')
                formatted_date = date
            else:
                raise ValueError("日期格式无效。")
        except ValueError as ve:
            return f"日期格式 '{date}' 无效。请使用 'YYYYMMDD' 或 'YYYY-MM-DD' 格式，例如 '20241107' 或 '2024-11-07'。错误详情: {ve}"

        # 调用API
        df = ak.news_economic_baidu(date=formatted_date)

        if df.empty:
            return f"未能获取到 {formatted_date} 的全球宏观经济事件数据，返回数据为空或该日期无事件。"

        # 格式化输出
        # 确保所有列都可见，并去除索引
        return f"{formatted_date} 的全球宏观经济事件数据:\n{df.to_string(index=False)}"

    except Exception as e:
        return f"获取全球宏观经济事件数据时发生错误: {e}"


@tool
def get_crypto_spot_prices(
        trade_pair: Optional[str] = None,
        market: Optional[str] = None
) -> str:
    """
    获取主流加密货币的实时行情数据。
    此工具将返回包括比特币 (BTC)、莱特币 (LTC)、比特币现金 (BCH) 等在不同交易所的最新报价、
    涨跌额、涨跌幅、24小时最高价、24小时最低价和24小时成交量等信息，并提供数据更新时间。
    数据来源于金十数据中心，提供当前时点的加密货币市场概览。

    Args:
        trade_pair (Optional[str]): 可选参数，指定要查询的交易品种，例如 'BTCUSD', 'LTCUSD', 'BTCEUR'。
                                     如果不指定，则返回所有可用交易品种的数据。
        market (Optional[str]): 可选参数，指定要查询的交易所市场，例如 'Bitfinex(香港)', 'Kraken(美国)', 'Bitstamp(美国)'， ‘CEX.IO(伦敦)’。
                                 如果不指定，则返回所有可用市场的数据。

    Returns:
        str: 包含主流加密货币实时行情数据的字符串表示。
             为了简洁，通常只展示部分关键列和前几条数据。
             如果获取失败或没有匹配的数据，将返回相应的提示信息。
    """
    try:
        # 调用API
        df = ak.crypto_js_spot()

        if df.empty:
            return "未能获取到加密货币实时行情数据。请稍后再试或检查网络连接。"

        # 检查DataFrame是否包含预期的关键列
        required_columns = ["交易品种", "市场", "最近报价", "涨跌幅", "24小时最高", "24小时最低", "更新时间"]
        if not all(col in df.columns for col in required_columns):
            return f"成功获取加密货币实时行情数据，但数据结构可能与预期不符。原始数据前5条:\n{df.head().to_string()}"

        # 根据 trade_pair 参数进行过滤
        if trade_pair:
            df = df[df["交易品种"].str.upper() == trade_pair.upper()]
            if df.empty:
                return f"未找到交易品种 '{trade_pair}' 的实时行情数据。"

        # 根据 market 参数进行过滤
        if market:
            df = df[df["市场"] == market]
            if df.empty:
                return f"未找到市场 '{market}' 的实时行情数据。"

        if df.empty:
            return "根据您提供的筛选条件，未能找到匹配的加密货币实时行情数据。"

        # 选取关键列并限制输出条数
        # 如果过滤后数据量较少，则全部展示；否则展示前10条
        display_df = df[required_columns].head(10) if len(df) > 10 else df[required_columns]

        filter_info = []
        if trade_pair:
            filter_info.append(f"交易品种: {trade_pair}")
        if market:
            filter_info.append(f"市场: {market}")

        if filter_info:
            return_message = f"成功获取加密货币实时行情数据 ({', '.join(filter_info)}):\n"
        else:
            return_message = "成功获取加密货币实时行情数据 (前10条):\n"

        return return_message + display_df.to_string()

    except Exception as e:
        return f"获取加密货币实时行情数据时发生错误: {e}"


@tool
def get_bitcoin_hold_report(
        category: Optional[str] = None,
        company_name_chinese: Optional[str] = None,
        country: Optional[str] = None
) -> str:
    """
    获取全球主要机构（如上市公司、政府机构、私营企业、ETF等）当前时点的比特币持仓报告。
    此工具将返回各机构的比特币持仓量、持仓市值、持仓成本、占总市值比重等详细信息，
    以及机构的分类和所在国家/地区。数据来源于金十数据中心。

    Args:
        category (Optional[str]): 可选参数，用于筛选特定分类的机构持仓报告。
                                  支持的分类包括：'上市公司', '政府机构', '私营企业', 'ETF'。
                                  例如：'上市公司'。
                                  如果不指定，则返回所有分类的机构数据。
        company_name_chinese (Optional[str]): 可选参数，用于筛选特定公司（中文名称）的持仓报告。
                                              例如：'特斯拉', 'MicroStrategy'。
                                              此参数支持模糊匹配。
                                              如果不指定，则返回所有公司数据。
        country (Optional[str]): 可选参数，用于筛选特定国家/地区的机构持仓报告。
                                 例如：'美国', '中国', '日本'。
                                 此参数支持精确匹配。
                                 如果不指定，则返回所有国家/地区的数据。

    Returns:
        str: 包含比特币持仓报告数据的字符串表示。
             为了简洁，通常只展示部分关键列和前几条数据。
             如果获取失败或没有匹配的数据，将返回相应的提示信息。
    """
    try:
        # 调用API
        df = ak.crypto_bitcoin_hold_report()

        if df.empty:
            return "未能获取到比特币持仓报告数据。请稍后再试或检查网络连接。"

        # 检查DataFrame是否包含预期的关键列
        required_columns = [
            "公司名称-中文", "分类", "国家/地区", "持仓量", "市值",
            "比特币占市值比重", "持仓占比", "持仓成本", "当日持仓市值", "查询日期"
        ]
        if not all(col in df.columns for col in required_columns):
            return f"成功获取比特币持仓报告数据，但数据结构可能与预期不符。原始数据前5条:\n{df.head().to_string()}"

        # 根据 category 参数进行过滤
        if category:
            df = df[df["分类"] == category]
            if df.empty:
                return f"未找到分类为 '{category}' 的比特币持仓报告数据。"

        # 根据 company_name_chinese 参数进行模糊匹配过滤
        if company_name_chinese:
            df = df[df["公司名称-中文"].str.contains(company_name_chinese, case=False, na=False)]
            if df.empty:
                return f"未找到公司名称包含 '{company_name_chinese}' 的比特币持仓报告数据。"

        # 根据 country 参数进行精确匹配过滤
        if country:
            if "国家/地区" in df.columns and not df["国家/地区"].empty:
                df = df[df["国家/地区"] == country]
            else:
                return "数据中不包含国家/地区信息，无法按国家筛选。"
            if df.empty:
                return f"未找到国家/地区为 '{country}' 的比特币持仓报告数据。"

        if df.empty:
            return "根据您提供的筛选条件，未能找到匹配的比特币持仓报告数据。"

        # 选取关键列并限制输出条数
        # 如果过滤后数据量较少，则全部展示；否则展示前10条
        display_columns = [
            "公司名称-中文", "分类", "国家/地区", "持仓量", "市值",
            "比特币占市值比重", "持仓占比", "查询日期"
        ]
        # 确保 display_columns 中的所有列都存在于 df 中
        existing_display_columns = [col for col in display_columns if col in df.columns]

        display_df = df[existing_display_columns].head(10) if len(df) > 10 else df[existing_display_columns]

        filter_info = []
        if category:
            filter_info.append(f"分类: {category}")
        if company_name_chinese:
            filter_info.append(f"公司名称: '{company_name_chinese}'")
        if country:
            filter_info.append(f"国家/地区: {country}")

        if filter_info:
            return_message = f"成功获取比特币持仓报告数据 ({', '.join(filter_info)}):\n"
        else:
            return_message = "成功获取比特币持仓报告数据 (前10条):\n"

        return return_message + display_df.to_string()

    except Exception as e:
        return f"获取比特币持仓报告数据时发生错误: {e}"


@tool
def get_stock_market_pe(
        symbol: Literal["上证", "深证", "创业板", "科创版"],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取中国A股主要板块（上证、深证、创业板、科创板）的历史市盈率数据。
    此工具将返回指定板块在不同日期的指数和平均市盈率，对于科创板则提供总市值和市盈率。
    支持按日期范围进行查询。

    Args:
        symbol (Literal["上证", "深证", "创业板", "科创板"]): 必选参数，指定要查询的A股市场板块。
                                                         可选值包括："上证", "深证", "创业板", "科创板"。
        start_date (Optional[str]): 可选参数，查询的起始日期，格式为 'YYYY-MM-DD'，例如 '2023-01-01'。
                                    如果指定，将返回从该日期（含）之后的数据。
        end_date (Optional[str]): 可选参数，查询的结束日期，格式为 'YYYY-MM-DD'，例如 '2023-12-31'。
                                  如果指定，将返回到该日期（含）之前的数据。
                                  如果只提供 start_date 而没有 end_date，则返回从 start_date 到最新数据。
                                  如果只提供 end_date 而没有 start_date，则返回从最早数据到 end_date。

    Returns:
        str: 包含指定板块历史市盈率数据的字符串表示。
             为了简洁，通常只展示部分关键列和前几条数据（最近的10条）。
             如果获取失败、参数无效或没有匹配的数据，将返回相应的提示信息。
    """
    try:
        # 验证 symbol 参数
        valid_symbols = {"上证", "深证", "创业板", "科创版"}
        if symbol not in valid_symbols:
            return f"无效的股票市场板块参数 '{symbol}'。请选择 '上证', '深证', '创业板' 或 '科创版'。"

        # 调用API
        df = ak.stock_market_pe_lg(symbol=symbol)

        if df.empty:
            return f"未能获取到 {symbol} 板块的市盈率数据。请稍后再试或检查网络连接。"

        # 根据 symbol 类型选择要展示的列
        if symbol in ["上证", "深证", "创业板"]:
            required_columns = ["日期", "指数", "平均市盈率"]
        # 注意这里是 "科创版" 而不是 "科创板"
        elif symbol == "科创版":
            required_columns = ["日期", "总市值", "市盈率"]
        else:
            return f"内部错误：未知的 symbol 类型 '{symbol}'，无法确定显示列。"

        # 检查DataFrame是否包含预期的关键列
        if not all(col in df.columns for col in required_columns):
            return f"成功获取 {symbol} 板块的市盈率数据，但数据结构可能与预期不符。原始数据前5条:\n{df.head().to_string()}"

        # 将日期列转换为 datetime 对象以便过滤
        df["日期"] = pd.to_datetime(df["日期"])

        # 应用日期过滤
        filtered_info = []
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                df = df[df["日期"] >= start_dt]
                filtered_info.append(f"从 {start_date}")
            except ValueError:
                return f"无效的起始日期格式 '{start_date}'。请使用 'YYYY-MM-DD' 格式。"

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df["日期"] <= end_dt]
                filtered_info.append(f"到 {end_date}")
            except ValueError:
                return f"无效的结束日期格式 '{end_date}'。请使用 'YYYY-MM-DD' 格式。"

        if df.empty:
            if filtered_info:
                return f"在 {' '.join(filtered_info)} 范围内，未能找到 {symbol} 板块的市盈率数据。"
            else:
                return f"未能获取到 {symbol} 板块的市盈率数据。请稍后再试或检查网络连接。"

        # 选取关键列并限制输出条数
        # 如果过滤后数据量较少，则全部展示；否则展示最新的10条
        if start_date or end_date:
            display_df = df
        else:
            display_df = df[required_columns].tail(10) if len(df) > 10 else df[required_columns]

        return_message = f"成功获取 {symbol} 板块的历史市盈率数据"
        if filtered_info:
            return_message += f" ({' '.join(filtered_info)})"
        return_message += f" (共 {len(df)} 条数据，显示最近 {len(display_df)} 条):\n{display_df.to_string()}"

        return return_message

    except Exception as e:
        return f"获取 {symbol} 板块市盈率数据时发生错误: {e}"

@tool
def get_stock_index_pe_lg(
        symbol: Literal[
            "上证50", "沪深300", "上证380", "创业板50", "中证500", "上证180",
            "深证红利", "深证100", "中证1000", "上证红利", "中证100", "中证800"
        ],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取中国A股主要指数的历史市盈率数据。
    此工具将返回指定指数在不同日期的指数点位以及多种市盈率指标，
    包括等权静态市盈率、静态市盈率、静态市盈率中位数、等权滚动市盈率、
    滚动市盈率和滚动市盈率中位数。支持按日期范围进行查询。
    如果指定了日期范围，将返回该范围内的所有数据。

    Args:
        symbol (Literal[...]): 必选参数，指定要查询的A股指数。
                               可选值包括："上证50", "沪深300", "上证380", "创业板50",
                               "中证500", "上证180", "深证红利", "深证100", "中证1000",
                               "上证红利", "中证100", "中证800"。
        start_date (Optional[str]): 可选参数，查询的起始日期，格式为 'YYYY-MM-DD'，例如 '2023-01-01'。
                                    如果指定，将返回从该日期（含）之后的数据。
        end_date (Optional[str]): 可选参数，查询的结束日期，格式为 'YYYY-MM-DD'，例如 '2023-12-31'。
                                  如果指定，将返回到该日期（含）之前的数据。
                                  如果只提供 start_date 而没有 end_date，则返回从 start_date 到最新数据。
                                  如果只提供 end_date 而没有 start_date，则返回从最早数据到 end_date。

    Returns:
        str: 包含指定指数历史市盈率数据的字符串表示。
             如果指定了日期范围，将返回所有匹配的数据。否则，为了简洁，通常只展示最近的10条数据。
             如果获取失败、参数无效或没有匹配的数据，将返回相应的提示信息。
    """
    try:
        # 验证 symbol 参数
        valid_symbols = {
            "上证50", "沪深300", "上证380", "创业板50", "中证500", "上证180",
            "深证红利", "深证100", "中证1000", "上证红利", "中证100", "中证800"
        }
        if symbol not in valid_symbols:
            return f"无效的指数参数 '{symbol}'。请从以下选项中选择：{', '.join(valid_symbols)}。"

        # 调用API
        df = ak.stock_index_pe_lg(symbol=symbol)

        if df.empty:
            return f"未能获取到 {symbol} 指数的市盈率数据。请稍后再试或检查网络连接。"

        # 定义所有可能的输出列
        all_possible_columns = [
            "日期", "指数", "等权静态市盈率", "静态市盈率", "静态市盈率中位数",
            "等权滚动市盈率", "滚动市盈率", "滚动市盈率中位数"
        ]

        # 检查DataFrame是否包含预期的关键列
        # 确保至少包含日期和指数，以及至少一种市盈率
        if "日期" not in df.columns or "指数" not in df.columns or \
                not any(col in df.columns for col in ["静态市盈率", "滚动市盈率"]):
            return f"成功获取 {symbol} 指数的市盈率数据，但数据结构可能与预期不符。原始数据前5条:\n{df.head().to_string()}"

        # 将日期列转换为 datetime 对象以便过滤
        df["日期"] = pd.to_datetime(df["日期"])

        # 表示是否进行了日期过滤
        date_filtered = False
        filtered_info = []

        # 应用日期过滤
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                df = df[df["日期"] >= start_dt]
                filtered_info.append(f"从 {start_date}")
                date_filtered = True
            except ValueError:
                return f"无效的起始日期格式 '{start_date}'。请使用 'YYYY-MM-DD' 格式。"

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df["日期"] <= end_dt]
                filtered_info.append(f"到 {end_date}")
                date_filtered = True
            except ValueError:
                return f"无效的结束日期格式 '{end_date}'。请使用 'YYYY-MM-DD' 格式。"

        if df.empty:
            if filtered_info:
                return f"在 {' '.join(filtered_info)} 范围内，未能找到 {symbol} 指数的市盈率数据。"
            else:
                return f"未能获取到 {symbol} 指数的市盈率数据。请稍后再试或检查网络连接。"

        # 选取所有存在的关键列
        display_columns = [col for col in all_possible_columns if col in df.columns]

        # 根据是否进行了日期过滤来决定返回的数据量
        if date_filtered:
            # 如果有日期过滤，返回所有匹配的数据
            display_df = df[display_columns]
            return_message = f"成功获取 {symbol} 指数的历史市盈率数据"
            if filtered_info:
                return_message += f" ({' '.join(filtered_info)})"
            return_message += f" (共 {len(display_df)} 条数据):\n{display_df.to_string()}"
        else:
            # 如果没有日期过滤，为了简洁，返回最近的10条数据
            display_df = df[display_columns].tail(10)
            return_message = f"成功获取 {symbol} 指数的历史市盈率数据 (最近 {len(display_df)} 条):\n{display_df.to_string()}"

        return return_message

    except Exception as e:
        return f"获取 {symbol} 指数市盈率数据时发生错误: {e}"

@tool
def get_stock_market_pb(
        symbol: Literal["上证", "深证", "创业板", "科创版"],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取中国A股主要板块（上证、深证、创业板、科创板）的历史市净率数据。
    此工具将返回指定板块在不同日期的指数点位以及多种市净率指标，
    包括市净率、等权市净率和市净率中位数。支持按日期范围进行查询。
    如果指定了日期范围，将返回该范围内的所有数据。
    数据来源于乐咕乐股。

    Args:
        symbol (Literal["上证", "深证", "创业板", "科创版"]): 必选参数，指定要查询的A股市场板块。
                                                         可选值包括："上证", "深证", "创业板", "科创版"。
        start_date (Optional[str]): 可选参数，查询的起始日期，格式为 'YYYY-MM-DD'，例如 '2023-01-01'。
                                    如果指定，将返回从该日期（含）之后的数据。
        end_date (Optional[str]): 可选参数，查询的结束日期，格式为 'YYYY-MM-DD'，例如 '2023-12-31'。
                                  如果指定，将返回到该日期（含）之前的数据。
                                  如果只提供 start_date 而没有 end_date，则返回从 start_date 到最新数据。
                                  如果只提供 end_date 而没有 start_date，则返回从最早数据到 end_date。

    Returns:
        str: 包含指定板块历史市净率数据的字符串表示。
             如果指定了日期范围，将返回所有匹配的数据。否则，为了简洁，通常只展示最近的10条数据。
             如果获取失败、参数无效或没有匹配的数据，将返回相应的提示信息。
    """
    try:
        # 验证 symbol 参数
        valid_symbols = {"上证", "深证", "创业板", "科创版"}
        if symbol not in valid_symbols:
            return f"无效的股票市场板块参数 '{symbol}'。请选择 '上证', '深证', '创业板' 或 '科创版'。"

        # 调用API
        df = ak.stock_market_pb_lg(symbol=symbol)

        if df.empty:
            return f"未能获取到 {symbol} 板块的市净率数据。请稍后再试或检查网络连接。"

        # 定义所有可能的输出列
        required_columns = ["日期", "指数", "市净率", "等权市净率", "市净率中位数"]

        # 检查DataFrame是否包含预期的关键列
        if not all(col in df.columns for col in required_columns):
            return f"成功获取 {symbol} 板块的市净率数据，但数据结构可能与预期不符。原始数据前5条:\n{df.head().to_string()}"

        # 将日期列转换为 datetime 对象以便过滤
        df["日期"] = pd.to_datetime(df["日期"])

        # 表示是否进行了日期过滤
        date_filtered = False
        filtered_info = []

        # 应用日期过滤
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                df = df[df["日期"] >= start_dt]
                filtered_info.append(f"从 {start_date}")
                date_filtered = True
            except ValueError:
                return f"无效的起始日期格式 '{start_date}'。请使用 'YYYY-MM-DD' 格式。"

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df["日期"] <= end_dt]
                filtered_info.append(f"到 {end_date}")
                date_filtered = True
            except ValueError:
                return f"无效的结束日期格式 '{end_date}'。请使用 'YYYY-MM-DD' 格式。"

        if df.empty:
            if filtered_info:
                return f"在 {' '.join(filtered_info)} 范围内，未能找到 {symbol} 板块的市净率数据。"
            else:
                return f"未能获取到 {symbol} 板块的市净率数据。请稍后再试或检查网络连接。"

        # 选取所有存在的关键列
        display_columns = [col for col in required_columns if col in df.columns]

        # 根据是否进行了日期过滤来决定返回的数据量
        if date_filtered:
            # 如果有日期过滤，返回所有匹配的数据
            display_df = df[display_columns]
            return_message = f"成功获取 {symbol} 板块的历史市净率数据"
            if filtered_info:
                return_message += f" ({' '.join(filtered_info)})"
            return_message += f" (共 {len(display_df)} 条数据):\n{display_df.to_string()}"
        else:
            # 如果没有日期过滤，为了简洁，返回最近的10条数据
            display_df = df[display_columns].tail(10)
            return_message = f"成功获取 {symbol} 板块的历史市净率数据 (最近 {len(display_df)} 条):\n{display_df.to_string()}"

        return return_message

    except Exception as e:
        return f"获取 {symbol} 板块市净率数据时发生错误: {e}"


@tool
def get_stock_index_pb_lg(
        symbol: Literal[
            "上证50", "沪深300", "上证380", "创业板50", "中证500", "上证180",
            "深证红利", "深证100", "中证1000", "上证红利", "中证100", "中证800"
        ],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取中国A股主要指数的历史市净率数据。
    此工具将返回指定指数在不同日期的指数点位以及多种市净率指标，
    包括市净率、等权市净率和市净率中位数。支持按日期范围进行查询。
    如果指定了日期范围，将返回该范围内的所有数据。
    数据来源于乐咕乐股。

    Args:
        symbol (Literal[...]): 必选参数，指定要查询的A股指数。
                               可选值包括："上证50", "沪深300", "上证380", "创业板50",
                               "中证500", "上证180", "深证红利", "深证100", "中证1000",
                               "上证红利", "中证100", "中证800"。
        start_date (Optional[str]): 可选参数，查询的起始日期，格式为 'YYYY-MM-DD'，例如 '2023-01-01'。
                                    如果指定，将返回从该日期（含）之后的数据。
        end_date (Optional[str]): 可选参数，查询的结束日期，格式为 'YYYY-MM-DD'，例如 '2023-12-31'。
                                  如果指定，将返回到该日期（含）之前的数据。
                                  如果只提供 start_date 而没有 end_date，则返回从 start_date 到最新数据。
                                  如果只提供 end_date 而没有 start_date，则返回从最早数据到 end_date。

    Returns:
        str: 包含指定指数历史市净率数据的字符串表示。
             如果指定了日期范围，将返回所有匹配的数据。否则，为了简洁，通常只展示最近的10条数据。
             如果获取失败、参数无效或没有匹配的数据，将返回相应的提示信息。
    """
    try:
        # 验证 symbol 参数
        valid_symbols = {
            "上证50", "沪深300", "上证380", "创业板50", "中证500", "上证180",
            "深证红利", "深证100", "中证1000", "上证红利", "中证100", "中证800"
        }
        if symbol not in valid_symbols:
            return f"无效的指数参数 '{symbol}'。请从以下选项中选择：{', '.join(valid_symbols)}。"

        # 调用API
        df = ak.stock_index_pb_lg(symbol=symbol)

        if df.empty:
            return f"未能获取到 {symbol} 指数的市净率数据。请稍后再试或检查网络连接。"

        # 定义所有可能的输出列
        required_columns = ["日期", "指数", "市净率", "等权市净率", "市净率中位数"]

        # 检查DataFrame是否包含预期的关键列
        if not all(col in df.columns for col in required_columns):
            return f"成功获取 {symbol} 指数的市净率数据，但数据结构可能与预期不符。原始数据前5条:\n{df.head().to_string()}"

        # 将日期列转换为 datetime 对象以便过滤
        df["日期"] = pd.to_datetime(df["日期"])

        # 表示是否进行了日期过滤
        date_filtered = False
        filtered_info = []

        # 应用日期过滤
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                df = df[df["日期"] >= start_dt]
                filtered_info.append(f"从 {start_date}")
                date_filtered = True
            except ValueError:
                return f"无效的起始日期格式 '{start_date}'。请使用 'YYYY-MM-DD' 格式。"

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df["日期"] <= end_dt]
                filtered_info.append(f"到 {end_date}")
                date_filtered = True
            except ValueError:
                return f"无效的结束日期格式 '{end_date}'。请使用 'YYYY-MM-DD' 格式。"

        if df.empty:
            if filtered_info:
                return f"在 {' '.join(filtered_info)} 范围内，未能找到 {symbol} 指数的市净率数据。"
            else:
                return f"未能获取到 {symbol} 指数的市净率数据。请稍后再试或检查网络连接。"

        # 选取所有存在的关键列
        display_columns = [col for col in required_columns if col in df.columns]

        # 根据是否进行了日期过滤来决定返回的数据量
        if date_filtered:
            # 如果有日期过滤，返回所有匹配的数据
            display_df = df[display_columns]
            return_message = f"成功获取 {symbol} 指数的历史市净率数据"
            if filtered_info:
                return_message += f" ({' '.join(filtered_info)})"
            return_message += f" (共 {len(display_df)} 条数据):\n{display_df.to_string()}"
        else:
            # 如果没有日期过滤，为了简洁，返回最近的10条数据
            display_df = df[display_columns].tail(10)
            return_message = f"成功获取 {symbol} 指数的历史市净率数据 (最近 {len(display_df)} 条):\n{display_df.to_string()}"

        return return_message

    except Exception as e:
        return f"获取 {symbol} 指数市净率数据时发生错误: {e}"


@tool
def get_stock_valuation_baidu(
        symbol: str,
        indicator: Literal["总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"],
        period: Literal["近一年", "近三年", "近五年", "近十年", "全部"],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取指定A股股票在百度股市通上的历史估值数据。
    此工具将返回指定股票在不同日期的特定估值指标（如总市值、市盈率等）的历史数据。
    用户可以选择查询的时间周期，或指定精确的日期范围。
    如果指定了日期范围，将返回该范围内的所有数据。

    Args:
        symbol (str): 必选参数，A股股票代码，例如 "002044"。
        indicator (Literal["总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"]):
                    必选参数，指定要查询的估值指标。
                    可选值包括："总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"。
        period (Literal["近一年", "近三年", "近五年", "近十年", "全部"]):
                必选参数，指定查询的相对时间周期。
                可选值包括："近一年", "近三年", "近五年", "近十年", "全部"。
                注意：如果同时提供了 start_date 或 end_date，此参数将被内部设置为 "全部"，
                以确保获取完整数据后再进行精确日期过滤。
        start_date (Optional[str]): 可选参数，查询的起始日期，格式为 'YYYY-MM-DD'，例如 '2023-01-01'。
                                    如果指定，将返回从该日期（含）之后的数据。
        end_date (Optional[str]): 可选参数，查询的结束日期，格式为 'YYYY-MM-DD'，例如 '2023-12-31'。
                                  如果指定，将返回到该日期（含）之前的数据。
                                  如果只提供 start_date 而没有 end_date，则返回从 start_date 到最新数据。
                                  如果只提供 end_date 而没有 start_date，则返回从最早数据到 end_date。

    Returns:
        str: 包含指定股票历史估值数据的字符串表示。
             将返回所选周期或日期范围内的所有数据。
             如果获取失败、参数无效或没有匹配的数据，将返回相应的提示信息。
    """
    try:
        # 验证 indicator 参数
        valid_indicators = {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"}
        if indicator not in valid_indicators:
            return f"无效的估值指标参数 '{indicator}'。请从以下选项中选择：{', '.join(valid_indicators)}。"

        # 验证 period 参数
        valid_periods = {"近一年", "近三年", "近五年", "近十年", "全部"}
        if period not in valid_periods:
            return f"无效的时间周期参数 '{period}'。请从以下选项中选择：{', '.join(valid_periods)}。"

        # 根据 start_date 和 end_date 来决定实际传递给 AKShare 的 period 参数
        akshare_period = period
        date_filtered = False
        filtered_info = []

        if start_date or end_date:
            # 如果有日期过滤，则获取全部数据再过滤
            akshare_period = "全部"
            date_filtered = True

        # 调用API
        df = ak.stock_zh_valuation_baidu(symbol=symbol, indicator=indicator, period=akshare_period)

        if df.empty:
            return f"未能获取到股票代码 '{symbol}' 在 '{akshare_period}' 周期内 '{indicator}' 的估值数据。请检查股票代码或稍后再试。"

        # 检查DataFrame是否包含预期的关键列
        required_columns = ["date", "value"]
        if not all(col in df.columns for col in required_columns):
            return f"成功获取股票 '{symbol}' 的估值数据，但数据结构可能与预期不符。原始数据前5条:\n{df.head().to_string()}"

        # 将日期列转换为 datetime 对象以便过滤
        df["date"] = pd.to_datetime(df["date"])

        # 应用日期过滤
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                df = df[df["date"] >= start_dt]
                filtered_info.append(f"从 {start_date}")
            except ValueError:
                return f"无效的起始日期格式 '{start_date}'。请使用 'YYYY-MM-DD' 格式。"

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df["date"] <= end_dt]
                filtered_info.append(f"到 {end_date}")
            except ValueError:
                return f"无效的结束日期格式 '{end_date}'。请使用 'YYYY-MM-DD' 格式。"

        if df.empty:
            if filtered_info:
                return f"在 {' '.join(filtered_info)} 范围内，未能找到股票 '{symbol}' 的 '{indicator}' 估值数据。"
            else:
                return f"未能获取到股票 '{symbol}' 的 '{indicator}' 估值数据。请稍后再试或检查网络连接。"

        # 选取关键列并格式化日期
        display_df = df[required_columns].copy()
        display_df["date"] = display_df["date"].dt.strftime('%Y-%m-%d')

        return_message = f"成功获取股票 '{symbol}' 的 '{indicator}' 历史估值数据"
        if filtered_info:
            return_message += f" ({' '.join(filtered_info)})"
        else:
            return_message += f" (周期: '{period}')"

        return_message += f" (共 {len(display_df)} 条数据):\n{display_df.to_string()}"

        return return_message

    except Exception as e:
        return f"获取股票 '{symbol}' 估值数据时发生错误: {e}"

@tool
def get_stock_value_em(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns_to_return: Optional[List[str]] = None
) -> str:
    """
    获取指定A股股票在东方财富网上的历史估值分析数据。
    此工具将返回指定股票在不同日期的当日收盘价、涨跌幅、总市值、流通市值、
    总股本、流通股本，以及PE(TTM)、PE(静)、市净率、PEG值、市现率和市销率等估值指标。
    支持按日期范围进行查询，并可选择返回特定的数据列。
    如果指定了日期范围，将返回该范围内的所有数据。

    Args:
        symbol (str): 必选参数，A股股票代码，例如 "002044" 或 "300766"。
        start_date (Optional[str]): 可选参数，查询的起始日期，格式为 'YYYY-MM-DD'，例如 '2023-01-01'。
                                    如果指定，将返回从该日期（含）之后的数据。
        end_date (Optional[str]): 可选参数，查询的结束日期，格式为 'YYYY-MM-DD'，例如 '2023-12-31'。
                                  如果指定，将返回到该日期（含）之前的数据。
                                  如果只提供 start_date 而没有 end_date，则返回从 start_date 到最新数据。
                                  如果只提供 end_date 而没有 start_date，则返回从最早数据到 end_date。
        columns_to_return (Optional[List[str]]): 可选参数，指定需要返回的列名列表。
                                                可选值包括：'数据日期', '当日收盘价', '当日涨跌幅', '总市值',
                                                '流通市值', '总股本', '流通股本', 'PE(TTM)', 'PE(静)',
                                                '市净率', 'PEG值', '市现率', '市销率'。
                                                如果未指定，将返回所有可用列。

    Returns:
        str: 包含指定股票历史估值数据的字符串表示。
             如果指定了日期范围，将返回所有匹配的数据。否则，为了简洁，通常只展示最近的10条数据。
             如果获取失败、参数无效或没有匹配的数据，将返回相应的提示信息。
    """
    try:
        # 定义所有可能的输出列
        all_possible_columns = [
            "数据日期", "当日收盘价", "当日涨跌幅", "总市值", "流通市值",
            "总股本", "流通股本", "PE(TTM)", "PE(静)", "市净率",
            "PEG值", "市现率", "市销率"
        ]

        # 调用API
        df = ak.stock_value_em(symbol=symbol)

        if df.empty:
            return f"未能获取到股票代码 '{symbol}' 的估值数据。请检查股票代码或稍后再试。"

        # 检查DataFrame是否包含预期的关键列
        # 至少包含日期和一些核心估值指标
        if "数据日期" not in df.columns or not any(col in df.columns for col in ["PE(TTM)", "市净率", "总市值"]):
            return f"成功获取股票 '{symbol}' 的估值数据，但数据结构可能与预期不符。原始数据前5条:\n{df.head().to_string()}"

        # 将日期列转换为 datetime 对象以便过滤
        df["数据日期"] = pd.to_datetime(df["数据日期"])

        # 表示是否进行了日期过滤
        date_filtered = False
        filtered_info = []

        # 应用日期过滤
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                df = df[df["数据日期"] >= start_dt]
                filtered_info.append(f"从 {start_date}")
                date_filtered = True
            except ValueError:
                return f"无效的起始日期格式 '{start_date}'。请使用 'YYYY-MM-DD' 格式。"

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df["数据日期"] <= end_dt]
                filtered_info.append(f"到 {end_date}")
                date_filtered = True
            except ValueError:
                return f"无效的结束日期格式 '{end_date}'。请使用 'YYYY-MM-DD' 格式。"

        if df.empty:
            if filtered_info:
                return f"在 {' '.join(filtered_info)} 范围内，未能找到股票 '{symbol}' 的估值数据。"
            else:
                return f"未能获取到股票 '{symbol}' 的估值数据。请稍后再试或检查网络连接。"

        # 确定最终要显示的列
        if columns_to_return:
            # 过滤掉用户请求但实际DataFrame中不存在的列
            requested_and_available_columns = [
                col for col in columns_to_return if col in df.columns and col in all_possible_columns
            ]
            # 确保日期列在最前面
            if "数据日期" not in requested_and_available_columns and "数据日期" in df.columns:
                requested_and_available_columns.insert(0, "数据日期")

            if not requested_and_available_columns:
                return f"请求的列 '{', '.join(columns_to_return)}' 均无效或不存在于数据中。"

            display_columns = requested_and_available_columns

            # 检查是否有用户请求的列被忽略
            ignored_columns = [col for col in columns_to_return if col not in display_columns]
            if ignored_columns:
                return_message_prefix = f"注意：请求的列 '{', '.join(ignored_columns)}' 不可用。已返回以下列：{', '.join(display_columns)}。\n"
            else:
                return_message_prefix = ""

        else:
            # 如果没有指定列，则返回所有有效的列
            display_columns = [col for col in all_possible_columns if col in df.columns]
            return_message_prefix = ""

        # 格式化日期为 YYYY-MM-DD 字符串
        df_to_display = df[display_columns].copy()
        if "数据日期" in df_to_display.columns:
            df_to_display["数据日期"] = df_to_display["数据日期"].dt.strftime('%Y-%m-%d')

        # 根据是否进行了日期过滤来决定返回的数据量
        if date_filtered:
            # 如果有日期过滤，返回所有匹配的数据
            display_df = df_to_display
            return_message = f"成功获取股票 '{symbol}' 的历史估值数据"
            if filtered_info:
                return_message += f" ({' '.join(filtered_info)})"
            return_message += f" (共 {len(display_df)} 条数据):\n{display_df.to_string()}"
        else:
            # 如果没有日期过滤，返回最近的10条数据
            display_df = df_to_display.tail(10)
            return_message = f"成功获取股票 '{symbol}' 的历史估值数据 (最近 {len(display_df)} 条):\n{display_df.to_string()}"

        return return_message_prefix + return_message

    except Exception as e:
        return f"获取股票 '{symbol}' 估值数据时发生错误: {e}"


@tool
def get_hk_valuation_baidu(
        symbol: str,
        indicator: Literal["总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"],
        period: Literal["近一年", "近三年", "全部"],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> str:
    """
    获取指定香港股票在百度股市通上的历史估值数据。
    此工具将返回指定股票在不同日期的特定估值指标（如总市值、市盈率等）的历史数据。
    用户可以选择查询的时间周期，或指定精确的日期范围。
    如果指定了日期范围，将返回该范围内的所有数据。返回的数据固定包含 'date' 和 'value' 两列。

    Args:
        symbol (str): 必选参数，香港股票代码，例如 "02358"。
        indicator (Literal["总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"]):
                    必选参数，指定要查询的估值指标。
                    可选值包括："总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"。
        period (Literal["近一年", "近三年", "全部"]):
                必选参数，指定查询的相对时间周期。
                可选值包括："近一年", "近三年", "全部"。
                注意：如果同时提供了 start_date 或 end_date，此参数将被内部设置为 "全部"，
                以确保获取完整数据后再进行精确日期过滤。
        start_date (Optional[str]): 可选参数，查询的起始日期，格式为 'YYYY-MM-DD'，例如 '2023-01-01'。
                                    如果指定，将返回从该日期（含）之后的数据。
        end_date (Optional[str]): 可选参数，查询的结束日期，格式为 'YYYY-MM-DD'，例如 '2023-12-31'。
                                  如果指定，将返回到该日期（含）之前的数据。
                                  如果只提供 start_date 而没有 end_date，则返回从 start_date 到最新数据。
                                  如果只提供 end_date 而没有 start_date，则返回从最早数据到 end_date。

    Returns:
        str: 包含指定股票历史估值数据的字符串表示。
             将返回所选周期或日期范围内的所有数据。
             如果获取失败、参数无效或没有匹配的数据，将返回相应的提示信息。
    """
    try:
        # 验证 indicator 参数
        valid_indicators = {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"}
        if indicator not in valid_indicators:
            return f"无效的估值指标参数 '{indicator}'。请从以下选项中选择：{', '.join(valid_indicators)}。"

        # 验证 period 参数
        valid_periods = {"近一年", "近三年", "全部"}
        if period not in valid_periods:
            return f"无效的时间周期参数 '{period}'。请从以下选项中选择：{', '.join(valid_periods)}。"

        # 根据 start_date 和 end_date 来决定实际传递给 AKShare 的 period 参数
        akshare_period = period
        date_filtered = False
        filtered_info = []

        if start_date or end_date:
            # 如果有日期过滤，则获取全部数据再过滤
            akshare_period = "全部"
            date_filtered = True

        # 调用API
        df = ak.stock_hk_valuation_baidu(symbol=symbol, indicator=indicator, period=akshare_period)

        if df.empty:
            return f"未能获取到香港股票 '{symbol}' 在 '{akshare_period}' 周期内 '{indicator}' 的估值数据。请检查股票代码或稍后再试。"

        # 检查DataFrame是否包含预期的关键列
        required_columns_check = ["date", "value"]
        if not all(col in df.columns for col in required_columns_check):
            return f"成功获取香港股票 '{symbol}' 的估值数据，但数据结构可能与预期不符。原始数据前5条:\n{df.head().to_string()}"

        # 将日期列转换为 datetime 对象以便过滤
        df["date"] = pd.to_datetime(df["date"])

        # 应用日期过滤
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                df = df[df["date"] >= start_dt]
                filtered_info.append(f"从 {start_date}")
            except ValueError:
                return f"无效的起始日期格式 '{start_date}'。请使用 'YYYY-MM-DD' 格式。"

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df["date"] <= end_dt]
                filtered_info.append(f"到 {end_date}")
            except ValueError:
                return f"无效的结束日期格式 '{end_date}'。请使用 'YYYY-MM-DD' 格式。"

        if df.empty:
            if filtered_info:
                return f"在 {' '.join(filtered_info)} 范围内，未能找到香港股票 '{symbol}' 的 '{indicator}' 估值数据。"
            else:
                return f"未能获取到香港股票 '{symbol}' 的 '{indicator}' 估值数据。请稍后再试或检查网络连接。"

        # 固定返回 'date' 和 'value' 两列
        display_columns = ["date", "value"]

        # 格式化日期
        df_to_display = df[display_columns].copy()
        df_to_display["date"] = df_to_display["date"].dt.strftime('%Y-%m-%d')

        # 根据是否进行了日期过滤来决定返回的数据量
        if date_filtered:
            # 如果有日期过滤，返回所有匹配的数据
            display_df = df_to_display
            return_message = f"成功获取香港股票 '{symbol}' 的 '{indicator}' 历史估值数据"
            if filtered_info:
                return_message += f" ({' '.join(filtered_info)})"
            return_message += f" (共 {len(display_df)} 条数据):\n{display_df.to_string()}"
        else:
            # 如果没有日期过滤，返回最近的10条数据
            display_df = df_to_display.tail(10)
            return_message = f"成功获取香港股票 '{symbol}' 的 '{indicator}' 历史估值数据 (周期: '{period}', 最近 {len(display_df)} 条):\n{display_df.to_string()}"

        return return_message

    except Exception as e:
        return f"获取香港股票 '{symbol}' 估值数据时发生错误: {e}"

# 包含所有 tool 函数的列表, 其顺序要求与API_SPECS中的顺序一致
all_tools = [get_intraday_time_series, get_daily_time_series, get_daily_adjusted_time_series, get_weekly_time_series, get_weekly_adjusted_time_series, get_monthly_time_series,
             get_monthly_adjusted_time_series, get_quote_endpoint,  get_market_status, get_historical_options_chain, get_sse_market_summary,  get_szse_security_category_summary,
             get_szse_regional_trading_summary, get_szse_sector_trading_summary, get_sse_daily_deal_summary, get_eastmoney_individual_stock_info, get_eastmoney_stock_bid_ask_quote,
             get_eastmoney_all_a_shares_realtime_quotes, get_sh_a_shares_realtime_quotes, get_sz_a_shares_realtime_quotes, get_bj_a_shares_realtime_quotes,
             get_eastmoney_new_a_shares_realtime_quotes, get_cy_a_shares_realtime_quotes, get_kc_a_shares_realtime_quotes, get_ab_shares_comparison_realtime_quotes,
             get_stock_historical_data, get_stock_minute_data_em, get_stock_intraday_data_em, get_b_shares_realtime_quotes, get_b_shares_historical_data, get_b_shares_minute_data,
             get_risk_warning_stocks_quotes, get_new_a_shares_quotes, get_ah_shares_realtime_quotes, get_ah_shares_historical_data, get_ah_shares_names, get_all_a_shares_code_name,
             get_sh_stock_info, get_sz_stock_info, get_bj_stock_info, get_sz_stock_name_change, get_main_stock_holders, get_hk_stock_spot_em, get_hk_stock_hist,
             get_hk_company_profile_em, get_hk_security_profile_em, get_hk_famous_spot_em, overview, income_statement, balance_sheet, cash_flow, earnings, dividends, splits,
             earnings_calendar, get_eastmoney_stock_trading_halt_info, get_cninfo_stock_profile, get_eastmoney_institutional_research_stats, get_a_share_goodwill_market_overview ,
             get_baidu_stock_trading_halt_info, get_baidu_dividend_info, get_baidu_financial_report_release_info, get_eastmoney_stock_news, get_eastmoney_stock_performance_report,
             get_eastmoney_stock_performance_express_report, get_eastmoney_stock_performance_forecast, get_cninfo_ipo_summary, get_eastmoney_bj_balance_sheet, get_eastmoney_profit_statement,
             get_eastmoney_cash_flow_statement, get_eastmoney_executive_shareholding_changes, get_tonghuashun_stock_fund_flow, get_eastmoney_individual_fund_flow, get_eastmoney_shareholder_meetings,
             get_sina_stock_history_dividend, get_cninfo_stock_dividend_history, get_eastmoney_limit_up_stocks_pool, get_eastmoney_limit_down_stocks_pool, get_eastmoney_strong_stocks_pool,
             get_eastmoney_sub_new_stocks_pool, get_eastmoney_broken_limit_up_stocks_pool, get_news_or_sentiment, get_exchange_rate, get_digital_daily, get_digital_weekly,
             get_digital_monthly, get_chinamoney_bond_info, get_sse_bond_deal_summary, get_chinamoney_bond_spot_quotes, get_chinabond_yield_curve, get_hs_bond_realtime_quotes,
             get_hs_bond_daily_historical_data, get_sina_convertible_bond_profile, get_sina_convertible_bond_summary, get_hs_convertible_bond_realtime_quotes,
             get_eastmoney_convertible_bond_list, get_eastmoney_convertible_bond_details, get_eastmoney_convertible_bond_value_analysis, get_china_us_bond_rates,
             get_cninfo_treasury_bond_issues, get_cninfo_local_government_bond_issues, get_cninfo_corporate_bond_issues, get_cninfo_convertible_bond_issues, get_company_news_cn,
             get_company_news_en, get_foreign_company_news_en, get_eastmoney_sh_sz_balance_sheet, get_ths_balance_sheet, get_stock_jgdy_detail_em, get_news_report_time_baidu,
             get_hk_financial_indicators_em, get_us_financial_indicators_em, get_hk_financial_report_em, get_us_financial_report_em, get_a_stock_pre_market_min_data_em,
             get_b_stock_spot_data_sina, get_a_stock_popularity_rank_em, get_a_stock_surge_rank_em, get_hk_stock_popularity_rank_em, get_a_stock_hot_rank_detail_em,
             get_hk_stock_hot_rank_detail_em, get_stock_hot_keyword_em, get_a_stock_hot_rank_latest_em, get_hk_stock_hot_rank_latest_em, get_us_fed_interest_rate_report,
             get_euro_ecb_interest_rate_report, get_newzealand_rba_interest_rate_report, get_china_pboc_interest_rate_report, get_switzerland_snb_interest_rate_report,
             get_uk_boe_interest_rate_report, get_australia_rba_interest_rate_report, get_japan_boj_interest_rate_report, get_russia_cbr_interest_rate_report, get_china_national_tax_receipts,
             get_china_enterprise_boom_index, get_china_qyspjg, get_china_stock_market_cap, get_rmb_loan_data, get_rmb_deposit_data, get_china_fx_gold_reserves, get_china_money_supply_data,
             get_china_urban_unemployment_data, get_hk_unemployment_rate, get_hk_gdp_data, get_hk_gdp_yoy_data, get_usa_gdp_data, get_global_macro_calendar, get_global_macro_events,
             get_crypto_spot_prices, get_bitcoin_hold_report, get_stock_market_pe, get_stock_index_pe_lg, get_stock_market_pb, get_stock_index_pb_lg, get_stock_valuation_baidu,
             get_stock_value_em, get_hk_valuation_baidu]

from all_api_section import API_SPECS


def save_dict_to_json(data, filename):
    """
    存储字典到本地文件

    Args: data: 要保存的字典数据
          filename: 保存文件路径
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"字典已成功保存到 {filename}")
    except Exception as e:
        print(f"保存字典失败: {e}")

def load_dict_from_json(filename):
    """
    从本地文件加载字典

    Args: filename: 文件路径
    """
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在。")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"字典已成功从 {filename} 加载")
        return data
    except Exception as e:
        print(f"加载字典失败: {e}")
        return None
    
if __name__ == "__main__":

    # 创建API.name和Tool.name的字典
    API_TOOL_dic = {}
    for api, tool in zip(API_SPECS, all_tools):
        API_TOOL_dic[api["name"]] = tool.name

    API_TOOL_dic_path = "./api_dic.json"

    # 保存字典
    save_dict_to_json(API_TOOL_dic, API_TOOL_dic_path)

    print("\n--- 模拟程序重新启动或在另一个文件使用 ---")

    # 加载字典
    loaded_dict = load_dict_from_json(API_TOOL_dic_path)

    if loaded_dict:
        print("\n字典的前3个键值对:")
        count = 0
        num_to_print = 3
        for key, value in loaded_dict.items():
            if count < num_to_print:
                print(f"  {key}: {value}")
                count += 1
            else:
                break

    print("API_SPECS:", len(API_SPECS))
    print("all_tools:", len(all_tools))
    print("API_TOOL_dic:", len(API_TOOL_dic))
    
    # 检查 API_SPECS 中是否有重复的 name
    api_names = [api["name"] for api in API_SPECS]
    duplicates = {name: count for name, count in zip(api_names, [api_names.count(name) for name in api_names]) if count > 1}

    if duplicates:
        print("发现重复的 api['name']:", duplicates)
    else:
        print("API_SPECS 中没有重复的 name")
