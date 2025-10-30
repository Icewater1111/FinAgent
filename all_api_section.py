import os
import sys

from langchain_community.embeddings import DashScopeEmbeddings
from typing import List, Dict, Any, Optional, Type
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

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
    # 如果内部路径找不到，作为备用，尝试从当前工作目录加载 (开发环境可能需要)
    print(f"内部 .env 文件未找到: {dotenv_path}，尝试从当前工作目录加载...")
    # 尝试默认加载方式
    load_dotenv(override=True)


#全部API的文档列表，用于匹配API文档与其tool
API_SPECS = [
    {
        "name": "TIME_SERIES_INTRADAY",
        "tool_name": "get_intraday_time_series"
    },
    {
        "name": "TIME_SERIES_DAILY",
        "tool_name": "get_daily_time_series"
    },
    {
        "name": "TIME_SERIES_DAILY_ADJUSTED",
        "tool_name": "get_daily_adjusted_time_series"
    },
    {
        "name": "TIME_SERIES_WEEKLY",
        "tool_name": "get_weekly_time_series"
    },
    {
        "name": "TIME_SERIES_WEEKLY_ADJUSTED",
        "tool_name": "get_weekly_adjusted_time_series"
    },
    {
        "name": "TIME_SERIES_MONTHLY",
        "tool_name": "get_monthly_time_series"
    },
    {
        "name": "TIME_SERIES_MONTHLY_ADJUSTED",
        "tool_name": "get_monthly_adjusted_time_series"
    },
    {
        "name": "Quote Endpoint Trending",
        "tool_name": "get_quote_endpoint"
    },
    {
        "name": "MARKET_STATUS",
        "tool_name": "get_market_status"
    },
    {
        "name": "HISTORICAL_OPTIONS",
        "tool_name": "get_historical_options_chain"
    },
    {
        "name": "AKShare Shanghai Stock Exchange Summary",
        "tool_name": "get_sse_market_summary"
    },
    {
        "name": "AKShare Shenzhen Stock Exchange Security Category Statistics",
        "tool_name": "get_szse_security_category_summary"
    },
    {
        "name": "AKShare Shenzhen Stock Exchange Regional Trading Summary",
        "tool_name": "get_szse_regional_trading_summary"
    },
    {
        "name": "AKShare Shenzhen Stock Exchange Stock Industry Trading Data",
        "tool_name": "get_szse_sector_trading_summary"
    },
    {
        "name": "AKShare Shanghai Stock Exchange Daily Deal Summary",
        "tool_name": "get_sse_daily_deal_summary"
    },
    {
        "name": "AKShare Eastmoney Individual Stock Information Query",
        "tool_name": "get_eastmoney_individual_stock_info"
    },
    {
        "name": "AKShare Eastmoney Stock Bid/Ask Quote",
        "tool_name": "get_eastmoney_stock_bid_ask_quote"
    },
    {
        "name": "AKShare Eastmoney All A-Shares Real-time Market Data",
        "tool_name": "get_eastmoney_all_a_shares_realtime_quotes"
    },
    {
        "name": "AKShare Eastmoney Shanghai A-Shares Real-time Market Data",
        "tool_name": "get_sh_a_shares_realtime_quotes"
    },
    {
        "name": "AKShare Eastmoney Shenzhen A-Shares Real-time Market Data",
        "tool_name": "get_sz_a_shares_realtime_quotes"
    },
    {
        "name": "AKShare Eastmoney Beijing A-Shares Real-time Market Data",
        "tool_name": "get_bj_a_shares_realtime_quotes"
    },
    {
        "name": "AKShare Eastmoney New Shares Real-time Market Data",
        "tool_name": "get_eastmoney_new_a_shares_realtime_quotes"
    },
    {
        "name": "AKShare ChiNext A-Shares Real-time Quotes (Eastmoney)",
        "tool_name": "get_cy_a_shares_realtime_quotes"
    },
    {
        "name": "AKShare Eastmoney Sci-Tech Innovation Board A-Shares Real-time Market Data",
        "tool_name": "get_kc_a_shares_realtime_quotes"
    },
    {
        "name": "AKShare Eastmoney AB-Shares Comparison Real-time Market Data",
        "tool_name": "get_ab_shares_comparison_realtime_quotes"
    },
    {
        "name": "AKShare Eastmoney A-Shares Historical Market Data",
        "tool_name": "get_stock_historical_data"
    },
    {
        "name": "AKShare Eastmoney A-Shares Minute-level Historical Market Data",
        "tool_name": "get_stock_minute_data_em"
    },
    {
        "name": "AKShare Eastmoney Stock Intraday Data",
        "tool_name": "get_stock_intraday_data_em"
    },
    {
        "name": "AKShare Eastmoney B-Shares Real-time Market Data",
        "tool_name": "get_b_shares_realtime_quotes"
    },
    {
        "name": "AKShare B-Shares Historical Market Data",
        "tool_name": "get_b_shares_historical_data"
    },
    {
        "name": "AKShare Sina B-Shares Minute-level Historical Market Data",
        "tool_name": "get_b_shares_minute_data"
    },
    {
        "name": "AKShare Eastmoney Risk Warning A-Shares Real-time Market Data",
        "tool_name": "get_risk_warning_stocks_quotes"
    },
    {
        "name": "AKShare Eastmoney New A-Shares Real-time Market Data",
        "tool_name": "get_new_a_shares_quotes"
    },
    {
        "name": "AKShare Eastmoney AH-Shares Real-time Market Data",
        "tool_name": "get_ah_shares_realtime_quotes"
    },
    {
        "name": "AKShare Tencent AH-Shares Historical Market Data",
        "tool_name": "get_ah_shares_historical_data"
    },
    {
        "name": "AKShare Tencent AH-Shares Name Dictionary",
        "tool_name": "get_ah_shares_names"
    },
    {
        "name": "AKShare All A-Shares Code and Name",
        "tool_name": "get_all_a_shares_code_name"
    },
    {
        "name": "AKShare Shanghai Stock Exchange Stock Info",
        "tool_name": "get_sh_stock_info"
    },
    {
        "name": "AKShare Shenzhen Stock Exchange Stock Info",
        "tool_name": "get_sz_stock_info"
    },
    {
        "name": "AKShare Beijing Stock Exchange Stock Info",
        "tool_name": "get_bj_stock_info"
    },
    {
        "name": "AKShare Shenzhen Stock Name Change History",
        "tool_name": "get_sz_stock_name_change"
    },
    {
        "name": "AKShare Stock Main Stock Holder",
        "tool_name": "get_main_stock_holders"
    },
    {
        "name": "AKShare Hong Kong Stock Spot Market Data",
        "tool_name": "get_hk_stock_spot_em"
    },
    {
        "name": "AKShare Hong Kong Stock Historical Market Data",
        "tool_name": "get_hk_stock_hist"
    },
    {
        "name": "AKShare Hong Kong Stock Company Profile",
        "tool_name": "get_hk_company_profile_em"
    },
    {
        "name": "AKShare Hong Kong Stock Security Profile",
        "tool_name": "get_hk_security_profile_em"
    },
    {
        "name": "AKShare Hong Kong Famous Stock Real-time Spot Data",
        "tool_name": "get_hk_famous_spot_em"
    },
    {
        "name": "OVERVIEW",
        "tool_name": "overview"
    },
    {
        "name": "INCOME_STATEMENT",
        "tool_name": "income_statement"
    },
    {
        "name": "BALANCE_SHEET",
        "tool_name": "balance_sheet"
    },
    {
        "name": "CASH_FLOW",
        "tool_name": "cash_flow"
    },
    {
        "name": "EARNINGS",
        "tool_name": "earnings"
    },
    {
        "name": "DIVIDENDS",
        "tool_name": "dividends"
    },
    {
        "name": "SPLITS",
        "tool_name": "splits"
    },
    {
        "name": "EARNINGS_CALENDAR",
        "tool_name": "earnings_calendar"
    },
    {
        "name": "AKShare Eastmoney Stock Trading Halt and Resumption Information Query",
        "tool_name": "get_eastmoney_stock_trading_halt_info"
    },
    {
        "name": "AKShare Cninfo Stock Company Profile Query",
        "tool_name": "get_cninfo_stock_profile"
    },
    {
        "name": "AKShare Eastmoney Institutional Research Statistics Query",
        "tool_name": "get_eastmoney_institutional_research_stats"
    },
    {
        "name": "AKShare Eastmoney A-Share Goodwill Market Overview",
        "tool_name": "get_a_share_goodwill_market_overview"
    },
    {
        "name": "AKShare Baidu Stock Trading Halt and Resumption Information (HK/Other Markets)",
        "tool_name": "get_baidu_stock_trading_halt_info"
    },
    {
        "name": "AKShare Baidu Stock Trading Dividend Information (HK/Other Markets)",
        "tool_name": "get_baidu_dividend_info"
    },
    {
        "name": "AKShare Baidu Stock Trading Financial Report Release Information (HK/Other Markets)",
        "tool_name": "get_baidu_financial_report_release_info"
    },
    {
        "name": "AKShare Eastmoney Stock News Query",
        "tool_name": "get_eastmoney_stock_news"
    },
    {
        "name": "AKShare Eastmoney Stock Performance Report Query",
        "tool_name": "get_eastmoney_stock_performance_report"
    },
    {
        "name": "AKShare Eastmoney Stock Performance Express Report Query",
        "tool_name": "get_eastmoney_stock_performance_express_report"
    },
    {
        "name": "AKShare Eastmoney Stock Performance Forecast Query",
        "tool_name": "get_eastmoney_stock_performance_forecast"
    },
    {
        "name": "AKShare Cninfo Stock IPO Summary Query",
        "tool_name": "get_cninfo_ipo_summary"
    },
    {
        "name": "AKShare Eastmoney Beijing Stock Exchange Balance Sheet Query",
        "tool_name": "get_eastmoney_bj_balance_sheet"
    },
    {
        "name": "AKShare Eastmoney Stock Profit Statement Query",
        "tool_name": "get_eastmoney_profit_statement"
    },
    {
        "name": "AKShare Eastmoney Stock Cash Flow Statement Query",
        "tool_name": "get_eastmoney_cash_flow_statement"
    },
    {
        "name": "AKShare Eastmoney Executive Shareholding Changes Query",
        "tool_name": "get_eastmoney_executive_shareholding_changes"
    },
    {
        "name": "AKShare Tonghuashun Stock Fund Flow Query",
        "tool_name": "get_tonghuashun_stock_fund_flow"
    },
    {
        "name": "AKShare Eastmoney Individual Stock Fund Flow Query",
        "tool_name": "get_eastmoney_individual_fund_flow"
    },
    {
        "name": "AKShare Eastmoney Shareholder Meeting Query",
        "tool_name": "get_eastmoney_shareholder_meetings"
    },
    {
        "name": "AKShare Sina Stock History Dividend Query",
        "tool_name": "get_sina_stock_history_dividend"
    },
    {
        "name": "AKShare Cninfo Stock Dividend History Query",
        "tool_name": "get_cninfo_stock_dividend_history"
    },
    {
        "name": "AKShare Eastmoney Limit Up Stocks Pool Query",
        "tool_name": "get_eastmoney_limit_up_stocks_pool"
    },
    {
        "name": "AKShare Eastmoney Limit Down Stocks Pool Query",
        "tool_name": "get_eastmoney_limit_down_stocks_pool"
    },
    {
        "name": "AKShare Eastmoney Strong Stocks Pool Query",
        "tool_name": "get_eastmoney_strong_stocks_pool"
    },
    {
        "name": "AKShare Eastmoney Sub New Stocks Pool Query",
        "tool_name": "get_eastmoney_sub_new_stocks_pool"
    },
    {
        "name": "AKShare Eastmoney Broken Limit Up Stocks Pool Query",
        "tool_name": "get_eastmoney_broken_limit_up_stocks_pool"
    },
    {
        "name": "NEWS_SENTIMENT",
        "tool_name": "get_news_or_sentiment"
    },
    {
        "name": "CURRENCY_EXCHANGE_RATE",
        "tool_name": "get_exchange_rate"
    },
    {
        "name": "DIGITAL_CURRENCY_DAILY",
        "tool_name": "get_digital_daily"
    },
    {
        "name": "DIGITAL_CURRENCY_WEEKLY",
        "tool_name": "get_digital_weekly"
    },
    {
        "name": "DIGITAL_CURRENCY_MONTHLY",
        "tool_name": "get_digital_monthly"
    },
    {
        "name": "AKShare ChinaMoney Bond Information Query",
        "tool_name": "get_chinamoney_bond_info"
    },
    {
        "name": "AKShare Shanghai Stock Exchange Bond Deal Summary",
        "tool_name": "get_sse_bond_deal_summary"
    },
    {
        "name": "AKShare ChinaMoney Bond Spot Market Maker Quotes",
        "tool_name": "get_chinamoney_bond_spot_quotes"
    },
    {
        "name": "AKShare ChinaBond Yield Curve Query",
        "tool_name": "get_chinabond_yield_curve"
    },
    {
        "name": "AKShare Shanghai and Shenzhen Bond Real-time Quotes",
        "tool_name": "get_hs_bond_realtime_quotes"
    },
    {
        "name": "AKShare Shanghai and Shenzhen Bond Daily Historical Data",
        "tool_name": "get_hs_bond_daily_historical_data"
    },
    {
        "name": "AKShare Sina Convertible Bond Profile",
        "tool_name": "get_sina_convertible_bond_profile"
    },
    {
        "name": "AKShare Sina Convertible Bond Summary",
         "tool_name": "get_sina_convertible_bond_summary"
    },
    {
        "name": "AKShare Shanghai and Shenzhen Convertible Bond Real-time Quotes",
        "tool_name": "get_hs_convertible_bond_realtime_quotes"
    },
    {
        "name": "AKShare Eastmoney Convertible Bond List",
        "tool_name": "get_eastmoney_convertible_bond_list"
    },
    {
        "name": "AKShare Eastmoney Convertible Bond Details",
        "tool_name": "get_eastmoney_convertible_bond_details"
    },
    {
        "name": "AKShare Eastmoney Convertible Bond Value Analysis",
        "tool_name": "get_eastmoney_convertible_bond_value_analysis"
    },
    {
        "name": "AKShare China-US Bond Yield Rates",
        "tool_name": "get_china_us_bond_rates"
    },
    {
        "name": "AKShare Cninfo Treasury Bond Issues",
        "tool_name": "get_cninfo_treasury_bond_issues"
    },
    {
        "name": "AKShare Cninfo Local Government Bond Issues",
        "tool_name": "get_cninfo_local_government_bond_issues"
    },
    {
        "name": "AKShare Cninfo Corporate Bond Issues",
        "tool_name": "get_cninfo_corporate_bond_issues"
    },
    {
        "name": "AKShare Cninfo Convertible Bond Issues",
        "tool_name": "get_cninfo_convertible_bond_issues"
    },
    {
        "name": "NewsAPI Company News Search (Chinese)",
        "tool_name": "get_company_news_cn"
    },
    {
        "name": "NewsAPI Company News Search (English)",
        "tool_name": "get_company_news_en"
    },
    {
        "name": "NewsAPI Foreign Company News Search (English)",
        "tool_name": "get_foreign_company_news_en"
    },
    {
       "name": "AKShare Eastmoney Shanghai & Shenzhen Stock Exchange Balance Sheet Query",
       "tool_name": "get_eastmoney_sh_sz_balance_sheet"
    },
    {
        "name": "AKShare Tonghuashun Financial Debt Balance Sheet Query",
        "tool_name": "get_ths_balance_sheet"
    },
    {
        "name": "AKShare Stock Institutional Research Detail (Eastmoney)",
        "tool_name": "get_stock_jgdy_detail_em"
    },
    {
        "name": "AKShare Baidu Financial Report Release Time",
        "tool_name": "get_news_report_time_baidu"
    },
    {
        "name": "AKShare Hong Kong Stock Financial Analysis Indicators (Eastmoney)",
        "tool_name": "get_hk_financial_indicators_em"
    },
    {
        "name": "AKShare US Stock Financial Analysis Indicators (Eastmoney)",
        "tool_name": "get_us_financial_indicators_em"
    },
    {
        "name": "AKShare Hong Kong Stock Financial Statements (Eastmoney)",
        "tool_name": "get_hk_financial_report_em"
    },
    {
        "name": "AKShare US Stock Financial Statements (Eastmoney)",
        "tool_name": "get_us_financial_report_em"
    },
    {
        "name": "AKShare Eastmoney A-Share Pre-Market Minute Data",
        "tool_name": "get_a_stock_pre_market_min_data_em"
    },
    {
        "name": "AKShare Sina B-Share Spot Quote",
        "tool_name": "get_b_stock_spot_data_sina"
    },
    {
        "name": "AKShare Eastmoney A-Share Popularity Rank",
        "tool_name": "get_a_stock_popularity_rank_em"
    },
    {
        "name": "AKShare Eastmoney A-Share Surge Rank",
        "tool_name": "get_a_stock_surge_rank_em"
    },
    {
        "name": "AKShare Eastmoney Hong Kong Stock Popularity Rank",
        "tool_name": "get_hk_stock_popularity_rank_em"
    },
    {
        "name": "AKShare Eastmoney A-Share Hot Rank Detail",
        "tool_name": "get_a_stock_hot_rank_detail_em"
    },
    {
        "name": "AKShare Eastmoney Hong Kong Stock Hot Rank Detail",
        "tool_name": "get_hk_stock_hot_rank_detail_em"
    },
    {
        "name": "AKShare Eastmoney Stock Hot Keywords",
        "tool_name": "get_stock_hot_keyword_em"
    },
    {
        "name": "AKShare Eastmoney A-Share Latest Hot Rank Details",
        "tool_name": "get_a_stock_hot_rank_latest_em"
    },
    {
        "name": "AKShare Eastmoney Hong Kong Stock Latest Hot Rank Details",
        "tool_name": "get_hk_stock_hot_rank_latest_em"
    },
    {
        "name": "AKShare US Federal Reserve Interest Rate Report",
        "tool_name": "get_us_fed_interest_rate_report"
    },
    {
        "name": "AKShare European Central Bank (ECB) Interest Rate Report",
        "tool_name": "get_euro_ecb_interest_rate_report"
    },
    {
        "name": "AKShare New Zealand Reserve Bank (RBNZ) Interest Rate Report",
        "tool_name": "get_newzealand_rba_interest_rate_report"
    },
    {
        "name": "AKShare China People's Bank of China (PBOC) Interest Rate Report",
        "tool_name": "get_china_pboc_interest_rate_report"
    },
    {
        "name": "AKShare Swiss National Bank (SNB) Interest Rate Report",
        "tool_name": "get_switzerland_snb_interest_rate_report"
    },
    {
        "name": "AKShare UK Bank of England (BoE) Interest Rate Report",
        "tool_name": "get_uk_boe_interest_rate_report"
    },
    {
        "name": "AKShare Australia Reserve Bank (RBA) Interest Rate Report",
        "tool_name": "get_australia_rba_interest_rate_report"
    },
    {
        "name": "AKShare Japan Bank of Japan (BoJ) Interest Rate Report",
        "tool_name": "get_japan_boj_interest_rate_report"
    },
    {
        "name": "AKShare Russia Central Bank (CBR) Interest Rate Report",
        "tool_name": "get_russia_cbr_interest_rate_report"
    },
    {
       "name": "AKSHARE_MACRO_CHINA_NATIONAL_TAX_RECEIPTS",
       "tool_name": "get_china_national_tax_receipts"
    },
    {
       "name":"AKSHARE_MACRO_CHINA_ENTERPRISE_BOOM_INDEX",
       "tool_name": "get_china_enterprise_boom_index"
    },
    {
       "name": "AKSHARE_MACRO_CHINA_QYSPJG",
       "tool_name": "get_china_qyspjg"
    },
    {
       "name": "AKSHARE_MACRO_CHINA_STOCK_MARKET_CAP",
       "tool_name": "get_china_stock_market_cap"
    },
    {
       "name": "AKSHARE_MACRO_RMB_LOAN",
       "tool_name": "get_rmb_loan_data"
    },
    {
        "name": "AKSHARE_MACRO_RMB_DEPOSIT",
        "tool_name": "get_rmb_deposit_data"
    },
    {
       "name": "AKSHARE_MACRO_CHINA_FX_GOLD",
       "tool_name": "get_china_fx_gold_reserves"
    },
    {
       "name": "AKSHARE_MACRO_CHINA_MONEY_SUPPLY",
       "tool_name": "get_china_money_supply_data"
    },
    {
       "name": "AKSHARE_MACRO_CHINA_URBAN_UNEMPLOYMENT",
       "tool_name": "get_china_urban_unemployment_data"
    },
    {
       "name": "AKSHARE_MACRO_CHINA_HK_RATE_OF_UNEMPLOYMENT",
       "tool_name": "get_hk_unemployment_rate"
    },
    {
       "name": "AKSHARE_MACRO_CHINA_HK_GDP",
       "tool_name": "get_hk_gdp_data"
    },
    {
       "name": "AKSHARE_MACRO_CHINA_HK_GDP_RATIO",
       "tool_name": "get_hk_gdp_yoy_data"
    },
    {
       "name": "AKSHARE_MACRO_USA_GDP_MONTHLY",
       "tool_name": "get_usa_gdp_data"
    },
    {
       "name": "AKSHARE_MACRO_INFO_WS",
       "tool_name": "get_global_macro_calendar"
    },
    {
       "name": "AKSHARE_NEWS_ECONOMIC_BAIDU",
       "tool_name": "get_global_macro_events"
    },
    {
       "name": "AKSHARE_CRYPTO_JS_SPOT",
       "tool_name": "get_crypto_spot_prices"
    },
    {
        "name": "AKSHARE_CRYPTO_BITCOIN_HOLD_REPORT",
        "tool_name": "get_bitcoin_hold_report"
    },
    {
        "name": "AKSHARE_STOCK_MARKET_PE_LG",
        "tool_name": "get_stock_market_pe"
    },
    {
        "name": "AKSHARE_STOCK_INDEX_PE_LG",
        "tool_name": "get_stock_index_pe_lg"
    },
    {
        "name": "AKSHARE_STOCK_MARKET_PB_LG",
        "tool_name": "get_stock_market_pb"
    },
    {
        "name": "AKSHARE_STOCK_INDEX_PB_LG",
        "tool_name": "get_stock_index_pb_lg"
    },
    {
        "name": "AKSHARE_STOCK_ZH_VALUATION_BAIDU",
        "tool_name": "get_stock_valuation_baidu"
    },
    {
        "name": "AKSHARE_STOCK_VALUE_EM",
        "tool_name": "get_stock_value_em"
    },
    {
        "name": "AKSHARE_STOCK_HK_VALUATION_BAIDU",
        "tool_name": "get_hk_valuation_baidu"
    },
]

# 把所有的API大致分为 美股市场 中国大陆市场 港股市场 和 其他市场 四个部分，可能同一个API会出现在多个部分

# 美股市场API
AM_API_SPECS = [
    {
        "name": "TIME_SERIES_INTRADAY",
        "description": {
            "功能描述": "获取指定股票在交易日内的盘中（Intraday）开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据。此接口支持获取未经股票拆分和股息派发调整的原始交易数据，以及经过调整后的数据，以便于进行更准确的历史分析。",
            "数据类型": "历史行情数据 (OHLCV)",
            "数据粒度": "盘中（支持1分钟、5分钟、15分钟、30分钟、60分钟等多种时间间隔）",
            "应用场景": "股票技术分析（如绘制K线图、计算均线等）、量化交易策略回测、市场趋势研究、高频数据分析.",
            "API提供商": "Alpha Vantage"
        },
        "keywords": [
            "股票", "行情", "盘中", "Intraday", "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量",
            "K线图", "蜡烛图", "技术分析", "量化交易", "策略回测", "历史数据", "分钟数据", "实时数据",
            "盘前交易", "盘后交易", "美股", "欧洲股市", "全球市场", "股票拆分", "股息调整", "原始数据",
            "调整后数据", "市场趋势", "高频数据", "均线", "交易数据", "证券数据", "时间序列"
        ]
    },
    {
        "name": "TIME_SERIES_DAILY",
         "description": {
            "功能描述": "获取指定全球股票的日频（Daily）开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据。此接口提供超过20年的历史数据，支持获取未经股票拆分和股息派发调整的原始交易数据，以及经过调整后的数据（复权），以便于进行更准确的历史分析。",
            "数据类型": "历史行情数据 (OHLCV)",
            "数据粒度": "日频（Daily）",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "日频", "Daily", "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量",
            "K线图", "蜡烛图", "技术分析", "量化交易", "策略回测", "历史数据", "日线", "中长期回测",
            "复权", "除权除息", "调整数据", "原始数据", "美股", "欧洲股市", "全球市场", "市场趋势",
            "证券数据", "时间序列", "收盘数据"
        ]
    },
    {
        "name": "TIME_SERIES_DAILY_ADJUSTED",
        "description": {
            "功能描述": "获取指定全球股票的日频（Daily）开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据。此接口的核心特点是提供了经过股票拆分和股息派发事件调整后的收盘价（adjusted close values），这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。同时，它也包含了原始（未经调整）的OHLCV值。",
            "数据类型": "历史行情数据 (OHLCV，包含复权数据)",
            "数据粒度": "日频（Daily）",
            "应用场景": "股票长期表现分析、真实回报计算、量化交易策略回测（尤其是需要考虑复权因素的策略）、财务报表分析、K线图绘制（复权后更准确）。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "日频", "Daily", "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量",
            "复权", "调整后数据", "除权除息", "股票拆分", "股息派发", "股息调整", "复权数据", "调整后收盘价",
            "K线图", "蜡烛图", "技术分析", "量化交易", "策略回测", "长期表现", "真实回报", "财务分析",
            "历史数据", "日线", "美股", "欧洲股市", "全球市场", "市场趋势", "证券数据", "时间序列", "收盘数据"
        ]
    },
    {
        "name": "TIME_SERIES_WEEKLY",
        "description": {
            "功能描述": "获取指定全球股票的周频（Weekly）时间序列数据，包括每周最后一个交易日的开盘价、最高价、最低价、收盘价和成交量（OHLCV）。此接口提供超过20年的历史数据，适用于对股票进行中长期趋势分析和策略回测。",

            "数据类型": "历史行情数据 (OHLCV)",
            "数据粒度": "周频（Weekly）",
            "应用场景": "股票中长期趋势分析、宏观市场研究、周线级别量化交易策略回测、长期投资决策辅助、绘制周K线图。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "周频", "Weekly", "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量",
            "K线图", "蜡烛图", "技术分析", "量化交易", "策略回测", "历史数据", "周线", "中长期分析",
            "趋势分析", "宏观研究", "长期投资", "原始数据", "未经调整", "美股", "欧洲股市", "全球市场",
            "证券数据", "时间序列", "收盘数据"
        ]
    },
    {
        "name": "TIME_SERIES_WEEKLY_ADJUSTED",
         "description": {
            "功能描述": "获取指定全球股票的周频（Weekly）时间序列数据。此接口的核心特点是提供了经过股票拆分和股息派发事件调整后的数据，包括每周最后一个交易日的开盘价、最高价、最低价、收盘价、周调整收盘价（weekly adjusted close）、周成交量以及周股息（weekly dividend）。这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。它涵盖了20多年的历史数据。",

            "数据类型": "历史行情数据 (OHLCV，包含复权数据及股息信息)",
            "数据粒度": "周频（Weekly）",
            "应用场景": "股票长期表现分析、真实回报计算、量化交易策略回测（尤其是需要考虑复权和股息因素的策略）、财务报表分析、绘制周K线图（复权后更准确）、股息收益分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "周频", "Weekly", "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量",
            "复权", "调整后数据", "除权除息", "股票拆分", "股息派发", "股息调整", "复权数据", "调整后收盘价",
            "周股息", "股息信息", "K线图", "蜡烛图", "技术分析", "量化交易", "策略回测", "长期表现",
            "真实回报", "财务分析", "股息收益", "历史数据", "周线", "美股", "欧洲股市", "全球市场",
            "市场趋势", "证券数据", "时间序列", "收盘数据"
        ]
    },
    {
        "name": "TIME_SERIES_MONTHLY",
        "description": {
            "功能描述": "获取指定全球股票的月频（Monthly）时间序列数据，包括每月最后一个交易日的开盘价、最高价、最低价、收盘价和成交量（OHLCV）。此接口提供超过20年的历史数据，适用于对股票进行长期趋势分析和策略回测。",

            "数据类型": "历史行情数据 (OHLCV)",
            "数据粒度": "月频（Monthly）",
            "应用场景": "股票长期趋势分析、宏观市场研究、月线级别量化交易策略回测、长期投资决策辅助、绘制月K线图。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "月频", "Monthly", "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量",
            "K线图", "蜡烛图", "技术分析", "量化交易", "策略回测", "历史数据", "月线", "长期趋势",
            "宏观研究", "长期投资", "原始数据", "未经调整", "美股", "欧洲股市", "全球市场",
            "证券数据", "时间序列", "收盘数据", "复权数据"
        ]
    },
    {
        "name": "TIME_SERIES_MONTHLY_ADJUSTED",
        "description": {
            "功能描述": "获取指定全球股票的月频（Monthly）时间序列数据。此接口的核心特点是提供了经过股票拆分和股息派发事件调整后的数据，包括每月最后一个交易日的开盘价、最高价、最低价、收盘价、月调整收盘价（monthly adjusted close）、月成交量以及月股息（monthly dividend）。这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。它涵盖了20多年的历史数据。",

            "数据类型": "历史行情数据 (OHLCV，包含复权数据及股息信息)",
            "数据粒度": "月频（Monthly）",
            "应用场景": "股票长期表现分析、真实回报计算、量化交易策略回测（尤其是需要考虑复权和股息因素的策略）、财务报表分析、绘制月K线图（复权后更准确）、股息收益分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "月频", "Monthly", "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量",
            "复权", "调整后数据", "除权除息", "股票拆分", "股息派发", "股息调整", "复权数据", "调整后收盘价",
            "月股息", "股息信息", "K线图", "蜡烛图", "技术分析", "量化交易", "策略回测", "长期表现",
            "真实回报", "财务分析", "股息收益", "历史数据", "月线", "美股", "欧洲股市", "全球市场",
            "市场趋势", "证券数据", "时间序列", "收盘数据"
        ]
    },
    {
        "name": "AKSHARE_MACRO_USA_GDP_MONTHLY",
        "description": {
           "功能描述": "获取美国国内生产总值（GDP）报告的历史数据。此接口提供自2008年2月28日至今的每月GDP数据，包括今值、预测值和前值。",

           "数据类型": "GDP，宏观经济指标",
           "数据粒度": "月度（Monthly）",
           "应用场景": "宏观经济分析、美国经济增长趋势研究、经济周期判断、市场预期与实际数据对比分析、投资决策参考、历史数据回溯。",
           "API提供商": "Akshare",
       },
        "keywords": [
            "美国GDP", "GDP报告", "美国经济", "宏观经济", "经济指标", "月度GDP", "Monthly GDP",
            "国内生产总值", "金十数据", "历史数据", "经济增长", "经济周期", "预测值", "前值", "今值",
            "实际值", "市场预期", "投资决策", "数据回溯", "美国宏观", "经济数据", "USA GDP"
        ]
    },
    {
        "name": "Quote Endpoint Trending",
        "description": {
            "功能描述": "获取指定单一股票代码的最新实时或准实时价格（如最新成交价、买卖价）和成交量信息。此接口设计用于查询单个股票的即时行情数据，不适用于批量查询。",

            "数据类型": "实时行情数据 / 报价数据",
            "数据粒度": "最新（Current / Latest）",
            "应用场景": "实时股票监控、交易决策辅助、个人投资组合跟踪、股票行情展示、单一股票信息查询。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "报价", "实时", "准实时", "最新价格", "即时行情", "成交量", "买卖价", "最新成交价",
            "单只股票", "单一股票", "实时报价", "交易量", "高频", "市场动态", "股票监控", "交易决策",
            "投资组合", "行情展示", "美股", "欧洲股市", "全球市场", "证券报价", 
        ]
    },
    {
        "name": "HISTORICAL_OPTIONS",
        "description": {
            "功能描述": "获取指定股票在特定日期的完整历史期权链数据。此接口提供期权的隐含波动率（Implied Volatility, IV）以及常见的希腊字母（如 Delta, Gamma, Theta, Vega, Rho）。期权链数据会按到期日按时间顺序排序，同一到期日内的合约则按行权价从低到高排序。",

            "数据类型": "历史期权链数据（包括看涨/看跌期权、行权价、到期日、成交量、未平仓量、隐含波动率、希腊字母）",
            "数据粒度": "特定日期（每日快照），提供超过15年的历史数据",
            "应用场景": "期权策略回测、波动率分析、期权定价模型开发与验证、风险管理（通过希腊字母）、量化交易策略开发（基于期权数据）、历史期权市场行为研究、教育与学术研究。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "期权", "历史数据", "历史期权数据", "期权链",
            "隐含波动率", "IV", "希腊字母", "Delta", "Gamma", "Theta", "Vega", "Rho",
            "看涨期权", "看跌期权", "行权价", "到期日", "成交量", "未平仓量",
            "期权策略", "量化交易", "波动率分析", "期权定价", "期权定价模型",
            "风险管理", "期权回测", "市场行为", "衍生品", "金融数据",
            "美股", "欧洲股市", "全球市场", "Alpha Vantage",
            "特定日期", "每日快照", "证券"
        ]
    },
    {
        "name": "OVERVIEW",
        "description": {
            "功能描述": "获取指定股票的公司信息、财务比率以及其他关键指标。",
            "数据类型": "公司基本面数据、财务数据、关键指标",
            "数据粒度": "按公司/财报周期更新",
            "应用场景": "公司基本面分析、财务健康度评估、投资决策支持。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "公司信息", "基本面分析", "欧美市场", "全球股票", "财务比率", "关键指标",
            "公司概览", "企业数据", "财报数据", "财务健康", "投资决策", "估值", "行业分析",
            "股票基本面", "公司资料", "财务报表", "盈利能力", "偿债能力", "运营能力", "成长性",
            "Alpha Vantage", "非中国市场", "股票分析"
        ]
    },
    {
        "name": "INCOME_STATEMENT",
         "description": {
            "功能描述": "获取目标公司的年度和季度损益表数据，包含映射到美国证券交易委员会（SEC）所采用的 GAAP 和 IFRS 分类体系的标准化字段。",
            "数据类型": "财务报表数据 (损益表)",
            "数据粒度": "年度和季度",
            "应用场景": "公司财务分析、盈利能力评估、投资研究、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "损益表", "利润表", "Income Statement", "财务报表", "财务数据", "年度报表", "季度报表",
            "GAAP", "IFRS", "SEC", "标准化", "会计准则", "公司财务", "盈利能力", "营收", "收入",
            "利润", "净利润", "毛利", "费用", "基本面分析", "投资研究", "财务模型", "财务健康",
            "全球股票", "欧美市场", "非中国公司", "Alpha Vantage", "企业分析", "财报数据",
            "财务趋势", "公司业绩"
        ]
    },
    {
        "name": "BALANCE_SHEET",
        "description": {
            "功能描述": "获取目标公司的年度和季度资产负债表数据，包含映射到美国证券交易委员会（SEC）所采用的 GAAP 和 IFRS 分类体系的标准化字段。",
            "数据类型": "财务报表数据 (资产负债表)",
            "数据粒度": "年度和季度",
            "应用场景": "公司财务分析、资产负债结构评估、偿债能力分析、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "资产负债表", "资产负债状况表", "Balance Sheet", "财务报表", "财务数据", "年度报表", "季度报表",
            "GAAP", "IFRS", "SEC", "标准化", "会计准则", "公司财务", "资产", "负债", "所有者权益",
            "流动资产", "非流动资产", "流动负债", "非流动负债", "股东权益", "基本面分析", "投资研究",
            "财务模型", "财务健康", "偿债能力", "流动性", "资本结构", "全球股票", "欧美市场",
            "非中国公司", "Alpha Vantage", "企业分析", "财报数据", "财务状况", "公司业绩"
        ]
    },
    {
        "name": "CASH_FLOW",
        "description": {
            "功能描述": "获取目标公司的年度和季度现金流量表数据，包含映射至美国证券交易委员会（SEC）所采纳的 GAAP 和 IFRS 分类体系的标准化字段。",
            "数据类型": "财务报表数据 (现金流量表)",
            "数据粒度": "年度和季度",
            "应用场景": "公司财务分析、现金流分析、经营活动评估、投资研究、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "现金流量表", "现金流", "Cash Flow Statement", "Cash Flow", "财务报表", "财务数据",
            "年度报表", "季度报表", "GAAP", "IFRS", "SEC", "标准化", "会计准则", "公司财务",
            "经营活动现金流", "投资活动现金流", "筹资活动现金流", "自由现金流", "净现金流",
            "基本面分析", "投资研究", "财务模型", "财务健康", "偿债能力", "流动性", "债务偿还",
            "全球股票", "欧美市场", "非中国公司", "Alpha Vantage", "企业分析", "财报数据",
            "公司业绩", "现金流分析"
        ]
    },
    {
        "name": "EARNINGS",
        "description": {
            "功能描述": "获取目标公司的年度和季度每股收益（EPS）数据。季度数据还包括分析师预测和盈余意外指标。",
            "数据类型": "财务数据 (每股收益、分析师预测、盈余意外)",
            "数据粒度": "年度和季度",
            "应用场景": "公司盈利能力分析、投资决策、分析师预期与实际业绩对比、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "每股收益", "EPS", "Earnings Per Share", "盈利", "公司盈利", "业绩", "分析师预测",
            "盈余意外", "Earnings Surprise", "超预期", "不及预期", "共识预测", "实际EPS",
            "年度EPS", "季度EPS", "年报", "季报", "财务数据", "基本面分析", "投资决策",
            "估值", "市场情绪", "量化策略", "业绩分析", "财务分析", "预期管理", "全球股票",
            "欧美市场", "非中国公司", "Alpha Vantage", "公司业绩", "盈利能力"
        ]
    },
    {
        "name": "DIVIDENDS",
        "description": {
            "功能描述": "获取历史和未来（已宣布的）分红分配信息。",
            "数据类型": "分红数据",
            "数据粒度": "按分红事件",
            "应用场景": "股息投资策略、现金流预测、股票估值、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "分红", "股息", "派息", "股利", "现金分红", "股息分配", "分红记录", "股息历史",
            "未来分红", "已宣布分红", "Dividends", "Dividend History", "Ex-Dividend Date",
            "Payout", "Dividend Yield", "Dividend Policy", "股息投资", "收益率", "现金流预测",
            "股票估值", "长期投资", "股息再投资", "基本面分析", "投资策略", "全球股票", "欧美市场",
            "非中国公司", "Alpha Vantage", "公司分红", "派息政策", "股息增长"
        ]
    },
    {
        "name": "SPLITS",
        "description": {
            "功能描述": "获取历史拆股（Stock Split）事件信息。",
            "数据类型": "公司事件数据 (拆股)",
            "数据粒度": "按事件",
            "应用场景": "股票历史数据调整（例如，调整历史股价和成交量以反映拆股影响）、财务分析、投资组合管理、技术分析数据校准。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "拆股", "Stock Split", "股票拆分", "股票分割", "拆股事件", "历史拆股", "数据调整",
            "股价调整", "成交量调整", "复权", "前复权", "后复权", "历史数据校准", "技术分析",
            "回溯测试", "投资组合分析", "财务分析", "公司事件", "资本结构变动", "股票数据",
            "全球股票", "欧美市场", "非中国公司", "Alpha Vantage", "估值", "交易量"
        ]
    },
    {
        "name": "EARNINGS_CALENDAR",
        "description": {
            "功能描述": "返回未来 3、6 或 12 个月内预计发布财报的公司列表。",
            "数据类型": "财报发布日程/日历",
            "数据粒度": "未来3、6或12个月",
            "应用场景": "市场事件跟踪、投资策略规划（如财报季交易）、风险管理、公司基本面研究。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "财报日历", "财报发布", "Earnings Calendar", "Earnings Release", "Earnings Announcement",
            "业绩报告", "财务报告", "财报季", "业绩发布", "报告日期", "Reporting Date", "未来财报",
            "近期财报", "市场事件", "投资策略", "风险管理", "基本面分析", "量化交易", "新闻交易",
            "事件驱动投资", "投资者关系", "公司事件", "全球股票", "欧美市场", "非中国公司",
            "Alpha Vantage", "上市公司财报", "财报预告", "发布日程"
        ]
    },
    {
        "name": "NEWS_SENTIMENT",
         "description": {
            "功能描述": "提供来自全球顶级新闻媒体的实时及历史市场金融新闻和情绪数据，涵盖股票、加密货币、外汇等多种资产类别，以及财政政策、并购、IPO等广泛主题。此API可用于训练LLM模型或增强交易策略，结合其他金融数据API可提供全面的市场洞察。",
            "数据类型": "市场新闻、情绪数据",
            "数据粒度": "实时及历史新闻事件",
            "应用场景": "LLM模型训练、量化交易策略增强、市场情绪分析、宏观经济研究、特定资产新闻追踪。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "市场新闻", "金融新闻", "财经新闻", "情绪数据", "新闻情绪", "舆情数据", "新闻分析",
            "股票新闻", "加密货币新闻", "外汇新闻", "实时新闻", "历史新闻", "市场动态", "突发新闻",
            "新闻事件", "媒体数据", "Sentiment Analysis", "Market News", "Financial News",
            "News Feed", "Real-time News", "Historical News", "Stock News", "Crypto News",
            "Forex News", "LLM训练", "量化交易", "算法交易", "宏观经济", "并购", "IPO",
            "财政政策", "货币政策", "公司新闻", "行业新闻", "事件驱动", "风险管理", "投资研究",
            "Alpha Vantage", "全球市场", "市场情绪", "新闻源", "数据筛选", "数据排序"
        ]
    },
    {
        "name": "AKShare China-US Bond Yield Rates",
        "description": {
            "功能描述": "获取中国和美国国债收益率的历史数据。此接口提供两国不同期限（2年、5年、10年、30年）的国债收益率以及10年-2年期限利差和GDP年增率的历史时间序列数据。",
            "数据类型": "国债收益率历史数据，宏观经济数据",
            "数据粒度": "每日（Daily）",
            "应用场景": "宏观经济分析、利率走势研究、中美利差分析、债券市场分析、资产配置决策、量化投资策略开发。",
            "API提供商": "Akshare",
        },
        "keywords": [
            "国债收益率", "中国国债", "美国国债", "中美利差", "中美GDP增率", "债券收益率", "利率",
            "无风险利率", "Treasury Yield", "Bond Yield", "2年期国债", "5年期国债", "10年期国债",
            "30年期国债", "Yield Curve", "收益率曲线", "GDP", "经济增长", "宏观经济指标", "通胀",
            "经济周期", "利差分析", "宏观分析", "利率分析", "债券分析", "资产配置", "量化投资",
            "投资策略", "中国债券市场", "美国债券市场", "中美市场", "Akshare", "历史数据", "时间序列数据"
        ]
    },
    {
        "name": "CURRENCY_EXCHANGE_RATE",
        "description": {
            "功能描述": "获取任意一对数字货币或实体货币的实时汇率。此API提供即时的货币兑换比率，适用于多种金融场景。",
            "数据类型": "实时汇率数据",
            "数据粒度": "实时",
            "应用场景": "货币兑换、外汇交易、加密货币交易、金融应用开发、汇率监控、投资组合管理。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "实时汇率", "货币兑换", "外汇", "数字货币", "加密货币", "法定货币", "实体货币",
            "汇率查询", "外汇牌价", "即时汇率", "汇率监控","投资组合管理", "金融应用", "Alpha Vantage"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_DAILY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每日历史时间序列数据。此API提供每日的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每日午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",
            "数据类型": "每日历史行情数据 (OHLCV)",
            "数据粒度": "每日（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "历史数据", "每日数据", "交易市场", "OHLCV", "开盘价",
            "最高价", "最低价", "收盘价", "成交量", "时间序列", "投资组合分析","Alpha Vantage"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_WEEKLY",
         "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每周历史时间序列数据。此API提供每周的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每周午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",
            "数据类型": "每周历史行情数据 (OHLCV)",
            "数据粒度": "每周（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "历史数据", "每周数据", "交易市场", "OHLCV", "开盘价",
            "最高价", "最低价", "收盘价", "成交量", "时间序列", "数字资产", "区块链",
            "历史价格", "交易数据", "K线数据", "行情数据", "市场数据", "回溯测试",
            "技术分析", "量化交易", "风险管理", "套利", "价格分析", "市场趋势", "美元报价",
            "基础货币", "投资组合分析", "中长期趋势", "Alpha Vantage"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_MONTHLY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每月历史时间序列数据。此API提供每月的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每月午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",
            "数据类型": "每月历史行情数据 (OHLCV)",
            "数据粒度": "每月（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "历史数据", "每月数据", "交易市场", "OHLCV", "开盘价",
            "最高价", "最低价", "收盘价", "成交量", "时间序列", "数字资产", "区块链",
            "历史价格", "交易数据", "K线数据", "行情数据", "市场数据", "回溯测试",
            "技术分析", "量化交易", "风险管理", "套利", "价格分析", "市场趋势", "美元报价",
            "基础货币", "投资组合分析", "超长期趋势", "宏观市场分析", "Alpha Vantage"
        ]
    },
    {
        "name": "NewsAPI Foreign Company News Search (English)",
        "description": {
            "功能描述": "查询指定国外公司的相关英文新闻资讯。",
            "数据类型": "新闻资讯 / 文章列表。",
            "数据粒度": "文章级别。",
            "应用场景": "国际公司舆情监控、全球市场趋势分析、国际投资决策、跨国企业研究。",
        },
        "keywords": [
            "国外公司", "外国公司", "海外公司", "国际公司", "英文新闻", "英语新闻", "国际新闻",
            "新闻资讯", "新闻报道", "文章", "媒体报道", "舆情监控", "企业舆情", "品牌声誉",
            "全球市场", "国际投资", "跨国企业", "竞品分析", "风险预警", "新闻搜索", "新闻查询",
            "公司新闻", "财经新闻", "商业新闻", "非中国大陆公司", "国际媒体", "NewsAPI"
        ]
    },
    {
        "name": "AKShare US Stock Financial Analysis Indicators (Eastmoney)",
        "description": {
            "功能描述": "获取美股-财务分析-主要指标中，指定美股的财务指标历史数据。",
            "数据类型": "财务指标历史数据",
            "数据粒度": "年报、单季报或累计季报",
            "应用场景": "美股公司财务状况分析、业绩趋势研究、投资价值评估、财务模型构建。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "美股", "美国股票", "美国股市", "财务指标", "财务数据", "财务分析", "财报", "业绩",
            "每股收益", "营业收入", "营收", "净利润", "利润", "毛利率", "净资产收益率",
            "资产负债率", "基本面", "公司分析", "投资价值", "估值", "财务模型", "股票筛选",
            "AKShare", "历史数据", "年报", "季报", "累计季报", "东方财富", "美股财报"
        ]
    },
    {
        "name": "AKShare US Stock Financial Statements (Eastmoney)",
        "description": {
            "功能描述": "获取美股-财务分析-三大报表（资产负债表、综合损益表、现金流量表）的详细数据。",
            "数据类型": "财务报表详细数据",
            "数据粒度": "年报、单季报或累计季报",
            "应用场景": "美股公司财务状况深度分析、业绩构成研究、现金流健康度评估、财务模型构建。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "美股", "美国股票", "美国股市", "财务报表", "三大报表", "资产负债表", "综合损益表",
            "利润表", "现金流量表", "财务数据", "财报数据", "历史财务数据", "公司财务", "资产",
            "负债", "所有者权益", "收入", "成本", "利润", "现金流", "财务状况", "财务健康",
            "基本面分析", "公司估值", "投资分析", "业绩构成", "现金流分析", "AKShare", "东方财富",
            "年报", "季报", "累计季报", "美股财报"
        ]
    },
    {
        "name": "AKShare Baidu Financial Report Release Time",
        "description": {
            "功能描述": "获取财报发行数据，主要包括港股和部分美股的财报发布时间。",
            "数据类型": "财报发行时间表",
            "数据粒度": "每日 (按指定日期查询)",
            "应用场景": "追踪公司财报发布日程、投资决策参考、市场事件预警。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "财报发行", "财报时间", "财报发布", "业绩发布", "发布时间", "百度股市通", "港股",
            "香港股票", "美股", "美国股票", "财报日程", "业绩预告", "公司公告", "股票代码",
            "交易所", "财报期", "市场事件", "投资参考", "投资决策", "时间表", "AKShare"
        ]
    },
    {
        "name": "AKShare US Federal Reserve Interest Rate Report",
        "description": {
            "功能描述": "获取美联储利率决议的历史报告数据。该接口提供自1982年9月27日至今的美联储利率决议报告，包含商品名称、日期、今值、预测值和前值。",
            "数据类型": "历史利率决议数据",
            "数据粒度": "每次决议会议",
            "应用场景": "美联储货币政策分析、历史利率走势研究、经济数据回溯、市场预期与实际结果对比。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "美联储", "联邦储备", "利率", "利率决议", "联邦基金利率", "货币政策", "央行",
            "央行政策", "宏观经济", "经济数据", "历史数据", "历史利率", "利率报告", "今值",
            "预测值", "前值", "利率走势", "市场预期", "经济周期", "通胀", "债券市场", "汇率影响",
            "政策分析", "AKShare"
        ]
    },
        {
        "name": "AKSHARE_CRYPTO_JS_SPOT",
        "description": {
            "功能描述": "获取全球主要加密货币交易所的实时交易数据，包括最近报价、涨跌额、涨跌幅、24小时最高价、24小时最低价、24小时成交量以及更新时间。",
            "数据类型": "实时行情数据",
            "数据粒度": "当前时点（实时快照）",
            "应用场景": "加密货币实时价格监控、市场动态分析、交易决策辅助、数据看板展示、自动化交易系统的数据输入。",
            "API提供商": "AKShare (数据源为金十数据)",
            "限制与注意事项": "该接口提供的是实时快照数据，不提供历史数据查询功能。数据更新频率取决于金十数据源的实时推送。请注意，24小时最高、最低和成交量数据是针对特定交易品种和货币对的，使用时需关注其对应的货币币种。",
        },
        "keywords": [
            "加密货币", "数字货币", "比特币", "以太坊", "莱特币", "比特币现金", "实时行情",
            "实时报价", "交易数据", "市场数据", "行情数据", "最新报价", "涨跌幅", "成交量",
            "交易所", "市场动态", "交易决策", "自动化交易", "价格监控", "数据看板", "套利",
            "AKShare"
        ]
    },
    {
        "name": "AKSHARE_CRYPTO_BITCOIN_HOLD_REPORT",
        "description": {
            "功能描述": "获取全球主要机构实时的比特币持仓报告数据。此报告提供了各机构的比特币持仓量、持仓市值、持仓成本、占总市值比重等详细信息，以及机构的分类和所在国家/地区。",
            "数据类型": "实时报告数据（快照）",
            "数据粒度": "当前时点",
            "应用场景": "了解全球机构对比特币的投资和配置情况、分析大型机构的比特币持仓趋势、评估机构资金对比特币市场的影响、进行宏观加密货币市场分析、研究比特币的机构采用率，还可用于监控市场的比特币资金流向和投资者行为分析。",
            "API提供商": "AKShare (数据源为金十数据)",
            "限制与注意事项": "该接口提供的是当前时点的比特币持仓快照数据，不提供历史持仓变化趋势。数据更新频率取决于金十数据源的实时推送。请注意，'比特币占市值比重' 和 '持仓占比' 的单位是百分比。如果指定了筛选条件但没有匹配的数据，将返回空结果提示。",
        },
        "keywords": [
            "比特币", "持仓", "机构持仓", "加密货币", "比特币持仓", "数字货币持仓", "机构投资",
            "持仓量", "持仓市值", "持仓成本", "市值比重", "机构分类", "国家地区", "资金流向",
            "投资者行为", "宏观分析", "市场影响", "采用率", "投资报告", "实时报告", "AKShare"
        ]
    },
    {
        "name": "MARKET_STATUS",
        "description": {
            "功能描述": "获取全球主要股票、外汇和加密货币交易场所的当前市场状态（开放或关闭）。此接口提供了一个实时的快照，显示哪些市场正在交易，哪些已经休市。",
            "数据类型": "市场状态 / 实时状态",
            "数据粒度": "实时快照（Current Snapshot）",
            "应用场景": "交易日历查询、自动化交易系统的前置检查（判断市场是否开盘）、跨市场交易策略的协调、用户界面中显示市场开放状态、全球市场概览。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "市场状态", "交易时间", "市场开放", "市场关闭", "开盘", "休市", "实时状态", "当前状态",
            "股票市场", "外汇市场", "加密货币市场", "全球市场", "交易场所", "股市", "汇市", "数字货币",
            "实时数据", "即时快照", "金融市场", "交易日历", "市场概览", "市场查询", "交易判断",
            "Alpha Vantage", "交易时间表", "开放状态", "关闭状态"
        ]
    },
]

# 中国市场API
CN_API_SPECS = [
    {
        "name": "MARKET_STATUS",
        "description": {
            "功能描述": "获取全球主要股票、外汇和加密货币交易场所的当前市场状态（开放或关闭）。此接口提供了一个实时的快照，显示哪些市场正在交易，哪些已经休市。",
            "数据类型": "市场状态 / 实时状态",
            "数据粒度": "实时快照（Current Snapshot）",
            "应用场景": "交易日历查询、自动化交易系统的前置检查（判断市场是否开盘）、跨市场交易策略的协调、用户界面中显示市场开放状态、全球市场概览。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "市场状态", "交易时间", "市场开放", "市场关闭", "股票市场", "外汇市场", "加密货币市场",
            "全球市场", "交易场所", "实时市场", "市场概览", "交易判断", "休市", "开盘", "闭市",
            "交易日历", "自动化交易", "资产类别", "金融市场", "Alpha Vantage"
        ]
    },
    {
        "name": "AKShare Shanghai Stock Exchange Summary",
        "description": {
            "功能描述": "获取上海证券交易所 (SSE) 每日市场概况统计数据。",
            "数据类型": "市场统计数据",
            "数据粒度": "日频（每日更新，反映最近一个交易日的数据）",
            "应用场景": "宏观市场分析、市场情绪判断、投资策略参考、学术研究、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "上海证券交易所", "上交所", "SSE", "市场概况", "市场统计", "日频数据", "流通股本",
            "总市值", "平均市盈率", "市盈率", "上市公司数量", "宏观指标", "市场情绪", "A股",
            "沪市", "中国股市", "股票市场", "交易数据", "金融数据", "AKShare"
        ]
    },
    {
        "name": "AKShare Shenzhen Stock Exchange Security Category Statistics",
        "description": {
            "功能描述": "获取深圳证券交易所 (SZSE) 指定日期的市场总貌中，按证券类别（如股票、基金、债券等）划分的统计数据。",
            "数据类型": "市场统计数据 / 证券类别统计",
            "数据粒度": "日频（可指定历史日期）",
            "应用场景": "深圳市场结构分析、各类证券市场活跃度与规模研究、宏观市场研究、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "深圳证券交易所", "深交所", "SZSE", "证券类别", "市场统计", "日频数据", "股票",
            "基金", "债券", "市场总貌", "证券数量", "成交金额", "总市值", "流通市值", "市场结构",
            "活跃度", "规模", "金融数据", "中国股市", "A股市场", "AKShare"
        ]
    },
    {
        "name": "AKShare Shenzhen Stock Exchange Regional Trading Summary",
        "description": {
            "功能描述": "获取深圳证券交易所 (SZSE) 指定月份的按地区划分的市场交易数据。",
            "数据类型": "市场统计数据 / 地区交易数据",
            "数据粒度": "月频（可指定历史月份）",
            "应用场景": "区域经济分析、投资者行为分析、市场活跃度区域分布研究、宏观经济研究。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "深圳证券交易所", "深交所", "SZSE", "地区交易", "交易额", "月频数据", "区域分析",
            "区域经济", "投资者行为", "市场活跃度", "市场份额", "股票交易", "基金交易", "债券交易",
            "宏观经济", "中国股市", "A股市场", "AKShare"
        ]
    },
    {
        "name": "AKShare Shenzhen Stock Exchange Stock Industry Trading Data",
         "description": {
            "功能描述": "获取深圳证券交易所 (SZSE) 指定时间段（月/年）的股票行业成交统计数据。",
            "数据类型": "行业交易统计数据",
            "数据粒度": "月频或年频（可查询历史数据）",
            "应用场景": "行业表现分析、市场热点识别、行业轮动策略、投资组合行业配置、宏观经济研究、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "深圳证券交易所", "深交所", "SZSE", "行业交易", "月频数据", "年频数据", "市场份额",
            "股票行业", "行业成交", "成交统计", "交易活跃度", "交易天数", "成交金额", "成交股数",
            "成交笔数", "行业占比", "行业分析", "市场热点", "行业轮动", "投资组合", "行业配置",
            "宏观经济", "金融数据", "中国股市", "A股市场", "AKShare"
        ]
    },
        {
        "name": "AKShare Shanghai Stock Exchange Daily Deal Summary",
         "description": {
            "功能描述": "获取上海证券交易所 (SSE) 指定日期的每日股票成交概况数据，提供挂牌数、市值、成交额、成交量、市盈率、换手率等，并按股票总计、主板A/B股、科创板、股票回购等类别细分。",
            "数据类型": "每日市场概况 / 交易统计数据",
            "数据粒度": "日频（可查询指定历史日期的数据）",
            "应用场景": "每日市场回顾、板块表现分析、市场估值与流动性分析、宏观市场趋势研究、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "上海证券交易所", "上交所", "SSE", "成交概况", "每日成交", "市场概况", "交易统计",
            "市值", "成交量", "成交额", "市盈率", "换手率", "日频数据", "A股", "沪市",
            "主板", "科创板", "股票回购", "挂牌数", "流通市值", "市场估值", "流动性", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Individual Stock Information Query",
        "description": {
            "功能描述": "获取指定股票代码的个股详细信息，包括基本面和市场表现数据。",
            "数据类型": "个股基本信息 / 股票概况",
            "数据粒度": "实时/快照（最新价），静态（基本面数据）",
            "应用场景": "个股基本面分析、投资决策参考、股票筛选、市场结构研究、金融数据可视化。",
            "API提供商": "AKShare ",
        },
        "keywords": [
            "个股信息", "股票信息", "基本面", "实时价", "最新价", "市值", "股本", "行业",
            "A股", "东方财富", "股票概况", "公司信息", "流通股本", "总股本", "上市时间",
            "股票筛选", "投资决策", "市场结构", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock Bid/Ask Quote",
        "description": {
            "功能描述": "获取指定股票代码的实时行情报价数据，包括买卖盘五档、最新价、涨跌幅、成交量、成交金额等。",
            "数据类型": "实时行情数据 / 盘口数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时交易监控、量化交易策略、市场深度分析、盘中决策辅助、技术分析、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "实时行情", "买卖盘", "五档报价", "盘口数据", "成交量", "成交金额", "A股",
            "东方财富", "股票报价", "市场深度", "最新价", "涨跌幅", "换手率", "量比",
            "今开", "昨收", "涨跌停", "内外盘", "交易监控", "量化交易", "技术分析", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney All A-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取所有沪深京A股上市公司的实时行情数据。",
            "数据类型": "实时行情数据 / 市场概况数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时市场监控、全市场股票筛选、量化交易策略、市场情绪分析、投资组合管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "沪深京A股", "A股", "全A股", "实时行情", "市场概况", "股票行情", "最新价", "涨跌幅",
            "成交量", "成交额", "市值", "总市值", "流通市值", "市盈率", "股票筛选", "量化交易",
            "市场监控", "东方财富", "AKShare", "中国股市", "实时数据", "盘中数据"
        ]
    },
    {
        "name": "AKShare Eastmoney Shanghai A-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取所有沪A股上市公司的实时行情数据，包括股票代码、名称、最新价、涨跌幅、成交量、成交额、市盈率、市值等。",
            "数据类型": "实时行情数据 / 市场概况数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时市场监控、沪A股筛选、量化交易策略、市场情绪分析、投资组合管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "沪A股", "上海A股", "实时行情", "股票行情", "市场概况", "最新价", "涨跌幅", "成交量",
            "成交额", "换手率", "市值", "总市值", "流通市值", "市盈率", "市净率", "振幅",
            "最高价", "最低价", "今开", "昨收", "量比", "涨速", "股票筛选", "量化交易",
            "市场监控", "东方财富", "AKShare", "中国股市", "实时数据", "盘中数据"
        ]
    },
    {
        "name": "AKShare Eastmoney Shenzhen A-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取所有深A股上市公司实时行情数据，包括代码、名称、最新价、涨跌幅、成交量、成交额、市盈率、市值等全面市场指标。",
            "数据类型": "实时行情数据 / 市场概况数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时市场监控、深A股筛选、量化交易策略、市场情绪分析、投资组合管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "深A股", "深圳A股", "实时行情", "股票行情", "市场概况", "最新价", "涨跌幅", "成交量",
            "成交额", "换手率", "市值", "总市值", "流通市值", "市盈率", "市净率", "振幅",
            "最高价", "最低价", "今开", "昨收", "量比", "涨速", "股票筛选", "量化交易",
            "市场监控", "东方财富", "AKShare", "中国股市", "实时数据", "盘中数据"
        ]
    },
    {
        "name": "AKShare Eastmoney Beijing A-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取所有京A股上市公司的实时行情数据，包括股票代码、名称、最新价、涨跌幅、成交量、成交额、市盈率、市值等全面市场指标。",
            "数据类型": "实时行情数据 / 市场概况数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时市场监控、京A股筛选、量化交易策略、市场情绪分析、投资组合管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "京A股", "北京A股", "北京证券交易所", "北交所", "实时行情", "股票行情", "市场概况",
            "最新价", "涨跌幅", "成交量", "成交额", "市值", "总市值", "流通市值", "市盈率",
            "市净率", "振幅", "最高价", "最低价", "今开", "昨收", "量比", "换手率", "涨速",
            "股票代码", "股票名称", "股票筛选", "量化交易", "市场监控", "东方财富", "AKShare",
            "中国股市", "实时数据", "盘中数据", "投资组合管理", "金融数据可视化", "市场情绪分析",
            "投资决策"
        ]
    },
    {
        "name": "AKShare Eastmoney New Shares Real-time Market Data",
        "description": {
            "功能描述": "获取所有新股上市公司的实时行情数据，包括股票代码、名称、最新价、涨跌幅、成交量、成交额、市盈率、市值等全面市场指标，并特别包含上市时间信息。",
            "数据类型": "实时行情数据 / 市场概况数据 / 新股信息",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时新股市场监控、新股投资机会发现、新股表现分析、打新策略辅助、市场情绪分析、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "新股", "A股", "新股市场", "实时行情", "股票行情", "市场概况", "最新价", "涨跌幅",
            "成交量", "成交额", "市值", "总市值", "流通市值", "市盈率", "市净率", "上市时间",
            "振幅", "最高价", "最低价", "今开", "昨收", "量比", "换手率", "涨速", "股票代码",
            "股票名称", "新股监控", "投资机会", "打新", "市场情绪", "金融数据", "东方财富", "AKShare",
            "中国股市", "实时数据", "盘中数据", "投资决策", "新股表现"
        ]
    },
    {
        "name": "AKShare ChiNext A-Shares Real-time Quotes (Eastmoney)",
         "description": {
            "功能描述": "获取创业板上市公司的实时行情数据，包括股票代码、名称、最新价、涨跌幅、成交量、成交额、市盈率、市值等。",
            "数据类型": "实时行情数据 / 市场概况数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时市场监控、创业板股票筛选、量化交易策略、市场情绪分析、投资组合管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "创业板", "A股", "创业板A股", "实时行情", "股票行情", "市场概况", "最新价", "涨跌幅",
            "成交量", "成交额", "市值", "总市值", "流通市值", "市盈率", "市净率", "振幅",
            "最高价", "最低价", "今开", "昨收", "量比", "换手率", "涨速", "股票代码", "股票名称",
            "股票筛选", "量化交易", "市场监控", "东方财富", "AKShare", "中国股市", "实时数据",
            "盘中数据", "投资组合管理", "金融数据可视化", "市场情绪分析", "投资决策"
        ]
    },
    {
        "name": "AKShare Eastmoney Sci-Tech Innovation Board A-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取所有科创板上市公司的实时行情数据，包括股票代码、名称、最新价、涨跌幅、成交量、成交额、市盈率、市值等。",
            "数据类型": "实时行情数据 / 市场概况数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时市场监控、科创板股票筛选、量化交易策略、市场情绪分析、投资组合管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "科创板", "A股", "科创板A股", "实时行情", "股票行情", "市场概况", "最新价", "涨跌幅",
            "成交量", "成交额", "市值", "总市值", "流通市值", "市盈率", "市净率", "振幅",
            "最高价", "最低价", "今开", "昨收", "量比", "换手率", "涨速", "股票代码", "股票名称",
            "股票筛选", "量化交易", "市场监控", "东方财富", "AKShare", "中国股市", "实时数据",
            "盘中数据", "投资组合管理", "金融数据可视化", "市场情绪分析", "投资决策"
        ]
    },
    {
        "name": "AKShare Eastmoney AB-Shares Comparison Real-time Market Data",
        "description": {
            "功能描述": "获取所有A/B股比价的实时行情数据，包括A股和B股各自的实时行情指标，以及A股与B股的比价信息。",
            "数据类型": "实时行情数据 / 市场概况数据 / A/B股比价数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "A/B股套利分析、市场效率研究、投资者情绪对比、跨市场联动分析、投资组合管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "B股", "AB股", "比价", "实时行情", "最新价", "涨跌幅", "套利",
            "中国A/B股市场", "市场概况", "盘中数据", "股票代码", "股票名称", "套利分析", "市场效率",
            "投资者情绪", "跨市场联动", "投资组合", "金融数据", "投资决策", "AKShare", "东方财富",
            "实时数据", "比价数据", "市场分析"
        ]
    },
    {
        "name": "AKShare Eastmoney A-Shares Historical Market Data",
        "description": {
            "功能描述": "获取指定沪深京A股上市公司的历史行情数据。用户可根据股票代码、查询周期（日、周、月）、日期范围以及复权方式进行查询。",
            "数据类型": "历史行情数据",
            "数据粒度": "日、周、月",
            "应用场景": "历史股价走势分析、技术指标计算、量化交易策略回测、股票估值、除权除息影响分析、长期投资收益率计算、市场周期研究。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "沪深京", "A股", "历史数据", "历史行情", "复权", "日线", "周线", "月线",
            "中国沪深京A股市场", "股票代码", "查询周期", "日期范围", "不复权", "前复权", "后复权",
            "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额", "股价走势", "技术指标",
            "量化交易", "策略回测", "股票估值", "除权除息", "投资收益", "市场周期", "金融数据",
            "AKShare", "东方财富", "历史股价", "数据分析"
        ]
    },
    {
        "name": "AKShare Eastmoney A-Shares Minute-level Historical Market Data",
        "description": {
            "功能描述": "获取指定沪深京A股上市公司的分时（分钟级）历史行情数据。用户可根据股票代码、日期时间范围、数据频率以及复权方式进行查询。",
            "数据类型": "历史行情数据",
            "数据粒度": "分钟级 (1, 5, 15, 30, 60 分钟)",
            "应用场景": "高频交易策略回测、日内交易分析、短线技术指标计算、市场微观结构研究、盘中异动监控。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "沪深京A股", "A股", "分钟级数据", "分时数据", "历史行情", "历史数据", "高频数据",
            "交易数据", "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额", "复权",
            "回测", "日内交易", "短线交易", "市场微观结构", "盘中异动", "东方财富", "AKShare",
            "股票数据", "时间序列", "技术分析"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock Intraday Data",
        "description": {
            "功能描述": "获取指定股票在最近一个交易日内的分时（日内）交易数据，包括盘前数据。",
            "数据类型": "日内行情数据 / 分时数据 / 逐笔数据 (近似)",
            "数据粒度": "分时 (通常为分钟级或更细的成交明细)",
            "应用场景": "日内交易策略分析、盘口分析、主力资金流向判断、高频交易行为研究、短线交易信号生成、市场微观结构分析、盘中异动监控。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "日内数据", "分时数据", "盘中数据", "逐笔数据", "成交明细", "盘口数据",
            "买卖盘", "成交价", "成交量", "主力资金", "高频交易", "短线交易", "市场微观结构",
            "异动监控", "东方财富", "AKShare", "中国股市", "实时数据", "盘前数据", "交易明细"
        ]
    },
    {
        "name": "AKShare Eastmoney B-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取所有B股上市公司的实时行情数据，包括股票代码、名称、最新价、涨跌幅、成交量、成交额、市盈率、市值等。",
            "数据类型": "实时行情数据 / 市场概况数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时市场监控、B股股票筛选、量化交易策略、市场情绪分析、投资组合管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "B股", "B股市场", "实时行情", "股票行情", "市场概况", "最新价", "涨跌幅", "成交量",
            "成交额", "市值", "总市值", "流通市值", "市盈率", "市净率", "振幅", "最高价",
            "最低价", "今开", "昨收", "量比", "换手率", "涨速", "股票代码", "股票名称", "股票筛选",
            "量化交易", "市场监控", "东方财富", "AKShare", "中国股市", "实时数据", "盘中数据",
            "投资组合管理", "金融数据可视化", "市场情绪分析", "投资决策", "上海B股", "深圳B股"
        ]
    },
    {
        "name": "AKShare B-Shares Historical Market Data",
        "description": {
            "功能描述": "获取指定B股上市公司的历史行情日频率数据或复权因子。用户可指定股票代码、日期范围以及复权方式或复权因子类型。",
            "数据类型": "历史行情数据 / 复权因子数据",
            "数据粒度": "日频率 (历史行情数据)",
            "应用场景": "B股历史股价走势分析、技术指标计算、量化交易策略回测、股票估值、除权除息影响分析、自定义复权计算。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "B股", "历史行情", "历史数据", "复权", "前复权", "后复权", "复权因子",
            "中国B股", "日频率", "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额",
            "新浪财经", "数据分析", "股价走势", "技术分析", "策略回测", "股票估值", "除权除息",
            "自定义复权", "金融数据", "历史股价", "上海B股", "深圳B股", "AKShare"
        ]
    },
    {
        "name": "AKShare Sina B-Shares Minute-level Historical Market Data",
        "description": {
            "功能描述": "获取指定B股股票或指数在最近一个交易日内的历史分时（分钟级）行情数据。用户可根据股票代码、数据频率和复权方式进行查询。",
            "数据类型": "历史行情数据",
            "数据粒度": "分钟级 (1, 5, 15, 30, 60 分钟)",
            "应用场景": "日内交易策略回测（针对最近交易日）、盘中行为分析、短线技术指标计算、市场微观结构研究、盘中异动监控、特定交易日的股价走势分析。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "B股", "分时行情", "分钟数据", "历史数据", "盘中数据", "最近交易日",
            "中国B股", "新浪财经", "分钟级", "高频数据", "日内交易", "开盘价", "收盘价",
            "最高价", "最低价", "成交量", "成交额", "B股指数", "股票指数", "复权", "前复权",
            "后复权", "盘中异动", "短线交易", "市场微观结构", "技术指标", "策略回测", "金融数据",
            "股价走势", "AKShare", "上海B股", "深圳B股"
        ]
    },
    {
        "name": "AKShare Eastmoney Risk Warning A-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取当前交易日风险警示板（ST/*ST 股票）所有A股上市公司的实时行情数据，包括股票代码、名称、最新价、涨跌幅、成交量、成交额、市盈率、市值等。",
            "数据类型": "实时行情数据 / 市场概况数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时监控ST/*ST股票行情、风险警示股票筛选、特定风险股票交易策略分析、市场风险情绪评估、投资组合中风险股票管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "风险警示板", "ST股票", "*ST股票", "实时行情", "最新价", "涨跌幅", "成交额",
            "换手率", "市盈率", "市净率", "中国A股", "风险警示", "ST股", "*ST股", "退市风险",
            "实时数据", "盘中数据", "市场概况", "股票行情", "交易数据", "成交量", "市值", "总市值",
            "流通市值", "振幅", "最高价", "最低价", "今开", "昨收", "量比", "涨速", "股票筛选",
            "交易策略", "风险管理", "投资组合", "市场情绪", "东方财富", "AKShare", "中国股市",
            "金融数据可视化", "合规监控"
        ]
    },
    {
        "name": "AKShare Eastmoney New A-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取当前交易日新股板块所有A股上市公司的实时行情数据，包括股票代码、名称、最新价、涨跌幅、成交量、成交额、市盈率、市净率等。",
            "数据类型": "实时行情数据 / 市场概况数据",
            "数据粒度": "实时（盘中数据）",
            "应用场景": "实时监控新股行情、新股上市表现分析、打新策略辅助、新股市场情绪评估、投资组合中新股管理、金融数据可视化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "新股", "实时行情", "最新价", "涨跌幅", "成交额", "换手率", "市盈率", "市净率",
            "东方财富", "AKShare", "盘中数据", "市场概况", "股票代码", "股票名称", "成交量", "振幅",
            "最高价", "最低价", "今开", "昨收", "量比", "总市值", "流通市值", "新股上市", "打新",
            "市场情绪", "投资组合", "金融数据可视化", "股票筛选", "投资决策", "中国股市", "实时数据"
        ]
    },
    {
        "name": "AKShare Eastmoney AH-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取所有A+H上市公司实时行情数据，包括A股和H股的最新价、涨跌幅、代码，以及A/H股的比价和溢价。",
            "数据类型": "实时行情数据 / 跨市场对比数据",
            "数据粒度": "实时（盘中数据，约 15 分钟延迟）",
            "应用场景": "A+H股套利分析、跨市场估值比较、投资组合配置、市场效率研究、AH股溢价/折价趋势分析、特定A+H股票实时监控。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "H股", "A+H股", "实时行情", "最新价", "涨跌幅", "比价", "溢价", "套利",
            "跨市场", "东方财富", "AKShare", "盘中数据", "市场概况", "股票代码", "股票名称", "估值分析",
            "投资组合配置", "市场效率", "溢价折价", "实时监控", "中国股市", "香港股市", "沪港通",
            "深港通", "港股", "金融数据可视化", "投资决策"
        ]
    },
    {
        "name": "AKShare Tencent AH-Shares Historical Market Data",
        "description": {
            "功能描述": "获取指定A+H上市公司（通过港股代码查询）的历史行情日频率数据。用户可指定港股股票代码、查询年份范围以及复权方式。",
            "数据类型": "历史行情数据",
            "数据粒度": "日频率",
            "应用场景": "A+H股历史股价走势分析、跨市场股票估值研究、技术指标计算、量化交易策略回测、除权除息影响分析、长期投资收益率计算、市场周期研究。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A+H股", "历史行情", "复权", "前复权", "后复权", "腾讯财经", "AKShare", "港股代码",
            "日频率", "历史数据", "股价走势", "跨市场估值", "技术指标", "量化交易", "策略回测",
            "除权除息", "投资收益", "市场周期", "开盘价", "收盘价", "最高价", "最低价", "成交量",
            "成交额", "金融数据可视化", "投资决策", "中国股市", "香港股市"
        ]
    },
    {
        "name": "AKShare Tencent AH-Shares Name Dictionary",
        "description": {
            "功能描述": "获取腾讯财经提供的所有A+H上市公司的股票代码和名称映射关系，用于辅助查询。",
            "数据类型": "股票代码与名称字典 / 映射关系",
            "数据粒度": "非时间序列数据 (静态映射)",
            "应用场景": "根据公司名称查找A+H股港股代码、作为其他A+H股接口的前置查询、数据验证与清洗、构建A+H股列表、自动化脚本中动态查找股票代码。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "股票代码", "股票名称", "字典", "上市公司", "A+H股", "腾讯财经", "AKShare", "映射关系",
            "静态数据", "辅助查询", "港股代码", "模糊匹配", "数据验证", "数据清洗", "股票列表",
            "自动化脚本", "中国股市", "香港股市", "数据整合"
        ]
    },
    {
        "name": "AKShare All A-Shares Code and Name",
        "description": {
            "功能描述": "获取沪深京三个交易所所有A股的股票代码和股票简称数据（代码与名称的映射关系）。",
            "数据类型": "股票代码与名称字典 / 映射关系",
            "数据粒度": "非时间序列数据 (静态映射)",
            "应用场景": "股票代码查询、作为其他数据接口的前置查询、数据验证与清洗、构建股票列表、自动化脚本中动态查找股票信息。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "股票代码", "股票名称", "沪深京", "上市公司", "字典", "AKShare", "映射关系",
            "静态数据", "沪市", "深市", "京市", "上海证券交易所", "深圳证券交易所", "北京证券交易所",
            "辅助查询", "数据验证", "数据清洗", "股票列表", "自动化脚本", "中国股市", "全市场",
            "市场概况", "投资组合管理"
        ]
    },
    {
        "name": "AKShare Shanghai Stock Exchange Stock Info",
        "description": {
            "功能描述": "获取上海证券交易所上市公司的股票代码、简称、公司全称和上市日期等详细信息。",
            "数据类型": "上市公司基本信息 / 股票列表",
            "数据粒度": "非时间序列数据 (静态列表)",
            "应用场景": "构建股票池、公司信息查询、数据接口前置查询、市场研究、自动化脚本。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "上交所", "上海证券交易所", "A股", "B股", "科创板", "主板A股", "主板B股",
            "公司全称", "上市日期", "上市公司", "股票信息", "证券代码", "股票简称", "基本信息",
            "股票列表", "静态数据", "股票池", "市场研究", "自动化脚本", "中国股市", "公司概况",
            "股票筛选"
        ]
    },
    {
        "name": "AKShare Shenzhen Stock Exchange Stock Info",
        "description": {
            "功能描述": "获取深圳证券交易所上市公司的股票代码、简称、所属行业、股本结构、上市日期等详细基本信息。",
            "数据类型": "上市公司基本信息 / 股票列表",
            "数据粒度": "非时间序列数据 (静态列表)",
            "应用场景": "构建股票池、公司信息查询、数据接口前置查询、市场研究、自动化脚本。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "深交所", "深圳证券交易所", "A股", "B股", "CDR", "AB股", "上市公司", "所属行业",
            "股本", "总股本", "流通股本", "上市日期", "股票信息", "证券代码", "股票简称", "基本信息",
            "股票列表", "静态数据", "股票池", "市场研究", "自动化脚本", "中国股市", "公司概况",
            "股票筛选"
        ]
    },
    {
        "name": "AKShare Beijing Stock Exchange Stock Info",
        "description": {
            "功能描述": "获取北京证券交易所所有上市公司的股票代码、简称、总股本、流通股本、上市日期、所属行业和地区等详细信息。",
            "数据类型": "上市公司基本信息 / 股票列表",
            "数据粒度": "非时间序列数据 (静态列表)",
            "应用场景": "构建股票池、公司信息查询、数据接口前置查询、市场研究、自动化脚本。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "北交所", "北京证券交易所", "上市公司", "总股本", "流通股本", "上市日期", "所属行业",
            "地区", "股票信息", "京A股", "证券代码", "股票简称", "基本信息", "股票列表", "静态数据",
            "股票池", "市场研究", "自动化脚本", "中国股市", "公司概况", "股票筛选"
        ]
    },
    {
        "name": "AKShare Shenzhen Stock Name Change History",
        "description": {
            "功能描述": "获取深圳证券交易所股票的名称变更历史数据，包括变更日期、证券代码、证券简称以及变更前后的名称信息。",
            "数据类型": "股票名称变更历史数据",
            "数据粒度": "事件级别 (按名称变更事件发生)",
            "应用场景": "历史数据追溯、公司事件分析、数据清洗与整合、合规与审计、自动化脚本处理历史名称变化。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "深交所", "深圳证券交易所", "股票名称变更", "名称变更历史", "历史数据", "证券简称",
            "公司名称", "名称变更", "历史名称", "数据追溯", "数据清洗", "合规", "审计", "事件数据",
            "股票代码", "公司事件", "金融研究", "股票数据管理"
        ]
    },
    {
        "name": "AKShare Stock Main Stock Holder",
        "description": {
            "功能描述": "获取指定A股股票的所有历史主要股东持股数据，包括持股数量、持股比例、股本性质等详细信息。",
            "数据类型": "上市公司主要股东持股历史数据",
            "数据粒度": "按报告期/公告日期 (历史快照数据)",
            "应用场景": "分析股东结构、辅助投资决策、评估市场情绪、金融研究、数据清洗与整合。",
            "API提供商": "AKShare ",
        },
        "keywords": [
            "主要股东", "持股历史", "A股", "持股详细信息", "股东结构", "股权结构", "大股东",
            "持股变动", "新浪财经", "公司治理", "投资决策", "金融研究", "风险评估", "报告期",
            "公告日期", "持股比例", "持股数量", "股本性质", "中国A股市场", "上市公司", "AKShare",
            "股权集中度", "资金流向", "量化策略"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock Trading Halt and Resumption Information Query",
        "description": {
            "功能描述": "获取股票停复牌详细信息。",
            "数据类型": "股票交易状态信息 (停牌、复牌)",
            "数据粒度": "指定日期",
            "应用场景": "股票交易策略制定、风险管理、市场事件跟踪、投资组合调整。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "停牌", "复牌", "停复牌信息", "停牌原因", "交易状态", "风险管理",
            "市场事件", "投资组合", "东方财富", "AKShare", "预计复牌", "所属市场", "停牌时间",
            "停牌截止时间", "停牌期限", "信息披露", "合规性监控", "股票代码", "股票名称", "盘中异动",
            "量化交易"
        ]
    },
    {
        "name": "AKShare Cninfo Stock Company Profile Query",
        "description": {
            "功能描述": "获取指定股票代码的公司概况详细信息。",
            "数据类型": "公司基本信息、公司概况",
            "数据粒度": "按股票代码查询",
            "应用场景": "公司基本面研究、投资决策、行业分析、背景调查、数据分析。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国市场", "公司概况", "公司信息", "公司简介", "上市公司", "基本信息", "所属行业",
            "注册资金", "成立日期", "上市日期", "主营业务", "经营范围", "巨潮资讯", "AKShare",
            "A股", "B股", "H股", "企业信息", "静态数据", "背景调查", "行业分析", "投资决策",
            "证券信息", "公司档案", "曾用简称", "法人代表", "注册地址", "办公地址"
        ]
    },
    {
        "name": "AKShare Eastmoney Institutional Research Statistics Query",
        "description": {
            "功能描述": "获取机构调研统计历史数据，包括最新价、涨跌幅、接待机构数量、接待方式、接待人员、接待地点、接待日期和公告日期等详细信息",
            "数据类型": "机构调研活动数据",
            "数据粒度": "指定日期，返回所有股票的机构调研详细信息",
            "应用场景": "机构行为分析、市场情绪判断、投资策略参考、股票基本面研究、追踪机构热点。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "机构调研", "机构统计", "机构行为", "市场情绪", "投资策略", "基本面研究",
            "机构热点", "东方财富", "AKShare", "股票代码", "股票名称", "接待机构", "接待方式",
            "接待日期", "公告日期", "调研内容", "量化交易", "风险预警", "上市公司", "投资机构",
            "历史数据", "最新价", "涨跌幅"
        ]
    },
    {
        "name": "AKShare Eastmoney A-Share Goodwill Market Overview",
        "description": {
            "功能描述": "获取东A股市场整体商誉（Goodwill）概况的历史数据，包括每个报告期的商誉、商誉减值、净资产、商誉占净资产比例、商誉减值占净资产比例、净利润规模以及商誉减值占净利润比例等关键财务指标",
            "数据类型": "财务指标、商誉数据",
            "数据粒度": "按报告期（历史数据）",
            "应用场景": "A股市场整体风险评估、财务分析、宏观经济研究、投资策略制定（特别是针对商誉风险的策略）、公司基本面分析辅助。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "商誉", "Goodwill", "风险评估", "财务指标", "商誉减值", "净资产",
            "净利润", "市场概况", "历史数据", "报告期", "东方财富", "AKShare", "宏观经济",
            "投资策略", "公司基本面", "市场风险", "金融研究", "中国股市", "商誉风险"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock News Query",
        "description": {
            "功能描述": "获取指定个股的新闻资讯数据，包括关键词、新闻标题、新闻内容、发布时间、文章来源和新闻链接等详细信息",
            "数据类型": "股票新闻、新闻资讯",
            "数据粒度": "指定股票代码或关键词，返回当日最近100条新闻",
            "应用场景": "股票动态跟踪、事件驱动投资、舆情分析、市场情绪监测、投资决策辅助、公司新闻回顾。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "股票新闻", "个股新闻", "舆情", "新闻舆情", "新闻资讯", "东方财富",
            "AKShare", "实时新闻", "事件驱动", "市场情绪", "投资决策", "公司新闻", "风险预警",
            "媒体分析", "新闻链接", "发布时间", "文章来源", "新闻标题", "关键词"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock Performance Report Query",
        "description": {
            "功能描述": "获取A股上市公司指定日期的业绩报告数据，包括每股收益、营业总收入（含同比和环比）、净利润（含同比和环比）、每股净资产、净资产收益率等详细财务指标。",
            "数据类型": "公司财务业绩报告",
            "数据粒度": "指定日期（通常为报告期末），覆盖A股市场",
            "应用场景": "上市公司业绩分析、基本面研究、价值投资、财务健康度评估、行业比较、投资组合构建、选股策略。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "业绩报告", "财务报告", "财务业绩", "基本面", "价值投资", "每股收益",
            "营业收入", "净利润", "每股净资产", "净资产收益率", "销售毛利率", "东方财富",
            "AKShare", "上市公司", "财务分析", "行业比较", "选股策略", "报告期", "公告日期",
            "盈利能力", "成长性", "中国股市"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock Performance Express Report Query",
        "description": {
            "功能描述": "获取A股上市公司指定日期的业绩快报数据，包括每股收益、营业收入（含去年同期、同比和环比）、净利润（含去年同期、同比和环比）、每股净资产、净资产收益率等详细财务指标。",
            "数据类型": "公司财务业绩快报",
            "数据粒度": "指定日期（通常为报告期末），覆盖A股市场",
            "应用场景": "上市公司业绩初步分析、基本面快速评估、投资决策辅助、市场热点追踪、业绩预增预减筛选。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "业绩快报", "财务报告", "财务业绩", "基本面", "每股收益", "营业收入",
            "净利润", "每股净资产", "净资产收益率", "东方财富", "AKShare", "上市公司", "财务分析",
            "行业", "公告日期", "市场板块", "证券类型", "盈利能力", "成长性", "中国股市", "业绩预览"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock Performance Forecast Query",
        "description": {
            "功能描述": "获取A股上市公司指定日期的业绩预告数据，包括预测指标（如预增、预减、扭亏等）、业绩变动、预测数值（净利润或营收）、业绩变动幅度等详细信息",
            "数据类型": "公司财务业绩预告",
            "数据粒度": "指定日期，覆盖A股市场",
            "应用场景": "上市公司业绩预期分析、投资决策辅助、风险评估、事件驱动投资、选股策略（如业绩预增股筛选）、市场情绪判断。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "业绩预告", "财务预测", "业绩预期", "基本面", "预增", "预减", "扭亏",
            "首亏", "续盈", "续亏", "不确定", "业绩变动", "净利润", "营业收入", "东方财富",
            "AKShare", "上市公司", "风险评估", "事件驱动", "选股策略", "市场情绪", "公告日期",
            "预告类型", "业绩变动原因", "中国股市"
        ]
    },
    {
        "name": "AKShare Cninfo Stock IPO Summary Query",
        "description": {
            "功能描述": "获取指定股票代码的上市相关数据，包括招股公告日期、中签率公告日、每股面值、总发行数量、发行前/后每股净资产、摊薄发行市盈率、募集资金净额、上网发行日期、上市日期、发行价格、发行费用总额、上网发行中签率和主承销商等详细信息。",
            "数据类型": "首次公开发行 (IPO) 详细数据，上市信息",
            "数据粒度": "指定股票代码",
            "应用场景": "新股研究、IPO分析、新股申购策略制定、上市公司首次发行信息查询、投资决策辅助。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "IPO", "新股", "上市", "巨潮资讯", "招股书", "发行价格", "中签率",
            "募集资金", "主承销商", "发行市盈率", "上市日期", "新股申购", "打新", "证券发行",
            "首次公开发行", "股票代码", "公司信息", "投资决策", "风险评估", "金融研究", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Beijing Stock Exchange Balance Sheet Query",
        "description": {
            "功能描述": "获取指定日期的北交所（北京证券交易所）上市公司资产负债表数据，包括货币资金、应收账款、存货、总资产（含同比）、应付账款、预收账款、总负债（含同比）、资产负债率、股东权益合计和公告日期等详细财务指标",
            "数据类型": "公司财务报表、资产负债表",
            "数据粒度": "指定日期（通常为报告期末），覆盖北交所市场",
            "应用场景": "北交所上市公司财务分析、资产结构分析、偿债能力评估、基本面研究、投资决策辅助、风险管理。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "北交所", "北京证券交易所", "资产负债表", "偿债能力", "东方财富", "财务报表",
            "资产结构", "负债结构", "股东权益", "货币资金", "应收账款", "存货", "总资产",
            "总负债", "资产负债率", "公告日期", "基本面分析", "风险管理", "财务健康", "AKShare",
            "上市公司", "财务分析", "报告期", "历史数据", "中国股市"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock Profit Statement Query",
        "description": {
            "功能描述": "获取指定日期的A股上市公司利润表数据，包括净利润（含同比）、营业总收入（含同比）、营业总支出（含销售、管理、财务费用）、营业利润、利润总额和公告日期等详细财务指标。",
            "数据类型": "公司财务报表、利润表",
            "数据粒度": "指定日期（通常为报告期末），覆盖A股市场",
            "应用场景": "A股上市公司财务分析、盈利能力评估、费用结构分析、基本面研究、投资决策辅助、经营效率分析。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "利润表", "净利润", "盈利能力", "东方财富", "财务报表", "营业收入",
            "营业支出", "销售费用", "管理费用", "财务费用", "营业利润", "利润总额", "公告日期",
            "基本面分析", "经营效率", "成长性", "价值投资", "AKShare", "上市公司", "财务分析",
            "报告期", "历史数据", "中国股市"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock Cash Flow Statement Query",
        "description": {
            "功能描述": "获取指定日期的A股上市公司现金流量表数据，包括净现金流（含同比）、经营性现金流（含净现金流占比）、投资性现金流（含净现金流占比）、融资性现金流（含净现金流占比）和公告日期等详细财务指标。",
            "数据类型": "公司财务报表、现金流量表",
            "数据粒度": "指定日期（通常为报告期末），覆盖A股市场",
            "应用场景": "A股上市公司财务分析、现金流分析、运营质量评估、偿债能力评估、基本面研究、投资决策辅助、风险管理。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "现金流量表", "现金流分析", "东方财富", "财务报表", "净现金流",
            "经营性现金流", "投资性现金流", "融资性现金流", "现金流占比", "公告日期",
            "基本面分析", "运营质量", "偿债能力", "风险管理", "盈利质量", "自由现金流",
            "AKShare", "上市公司", "财务分析", "报告期", "历史数据", "中国股市"
        ]
    },
    {
        "name": "AKShare Eastmoney Executive Shareholding Changes Query",
        "description": {
            "功能描述": "获取高管及股东持股变动数据，包括增持和减持记录。",
            "数据类型": "公司内部人士持股变动（增持/减持）",
            "数据粒度": "可查询所有变动类型，或按增持/减持筛选。覆盖A股上市公司。",
            "应用场景": "公司内部人士行为分析、股票投资信心评估、市场情绪判断、潜在利好/利空信号识别、投资决策辅助、跟踪大股东/高管持股动向。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "高管持股", "股东持股", "持股变动", "增持", "减持", "东方财富",
            "内部人士", "高管", "股东", "股权变动", "持股比例", "总股本", "流通股", "变动数量",
            "变动日期", "公告日期", "投资信心", "市场情绪", "利好", "利空", "大宗交易",
            "公司治理", "AKShare", "中国股市", "内部交易", "风险预警", "上市公司", "实时数据"
        ]
    },
    {
        "name": "AKShare Tonghuashun Stock Fund Flow Query",
        "description": {
            "description": {
            "功能描述": "获取个股资金流数据，包括最新价、涨跌幅、换手率、流入资金、流出资金、净额、成交额等详细信息。",
            "数据类型": "个股资金流向、市场交易数据",
            "数据粒度": "个股层面，支持即时资金流以及3日、5日、10日、20日等不同时间周期的资金流排行数据。",
            "应用场景": "个股资金流向分析、市场热度评估、主力资金跟踪、短线交易策略制定、量化投资模型构建、投资决策辅助。",
            "API提供商": "AKShare",
        },
        },
        "keywords": [
            "中国A股", "A股", "资金流", "资金流向", "个股资金", "流入资金", "流出资金",
            "净流入", "主力资金", "散户资金", "资金排行", "短线交易", "量化投资", "同花顺",
            "股票资金", "交易数据", "市场热度", "换手率", "成交额", "股票代码", "股票简称",
            "市场情绪", "股票异动", "投资决策", "数据中心", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Individual Stock Fund Flow Query",
        "description": {
            "功能描述": "获取指定股票和市场的近100个交易日的个股资金流数据。",
            "数据类型": "个股资金流向、市场交易数据",
            "数据粒度": "指定股票，每日数据，覆盖近100个交易日。",
            "应用场景": "特定股票资金流向分析、主力资金行为跟踪、市场热度评估、交易策略制定、量化投资模型构建、投资决策辅助、风险管理。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "A股", "个股资金流", "资金流向", "主力资金", "超大单", "大单", "中单",
            "小单", "净流入", "净占比", "资金博弈", "每日数据", "历史资金流", "东方财富",
            "股票资金", "交易策略", "量化模型", "投资决策", "风险管理", "短期趋势", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Shareholder Meeting Query",
        "description": {
            "功能描述": "获取所有A股上市公司的股东大会信息。",
            "数据类型": "公司治理信息、股东大会公告、提案内容",
            "数据粒度": "所有A股上市公司，按股东大会事件维度。",
            "应用场景": "公司治理研究、股东大会议题跟踪、投票权行使、重大决策分析、投资者关系管理、公司基本面研究、风险管理。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "A股", "股东大会", "公司治理", "重大决策", "提案内容", "股权登记",
            "网络投票", "决议公告", "公司公告", "投资者关系", "基本面研究", "风险管理",
            "合规性", "市场事件", "东方财富", "上市公司", "投票权", "AKShare"
        ]
    },
    {
        "name": "AKShare Sina Stock History Dividend Query",
        "description": {
            "功能描述": "获取所有A股上市公司的历史分红数据。",
            "数据类型": "公司财务数据、分红派息数据、融资历史",
            "数据粒度": "所有A股上市公司，历史分红及融资概况。",
            "应用场景": "公司分红政策分析、股东回报能力评估、融资历史研究、长期投资价值评估、基本面分析、量化投资策略构建。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "A股", "历史分红", "分红数据", "股息", "分红政策", "派息", "融资历史",
            "融资概况", "股东回报", "长期投资", "基本面分析", "量化投资", "股票筛选", "新浪财经",
            "公司财务", "分红次数", "融资总额", "融资次数", "上市日期", "AKShare"
        ]
    },
    {
        "name": "AKShare Cninfo Stock Dividend History Query",
        "description": {
            "功能描述": "获取指定股票的历史分红数据。",
            "数据类型": "公司财务数据、分红派息数据",
            "数据粒度": "指定股票，历史分红记录。",
            "应用场景": "特定股票分红政策分析、历史派息情况研究、股东回报评估、除权除息日跟踪、投资决策辅助、基本面分析。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "A股", "历史分红", "分红数据", "分红派息", "股息", "送股", "转增",
            "派息比例", "股权登记日", "除权日", "派息日", "股份到账日", "分红类型", "巨潮资讯",
            "公司财务", "股东回报", "投资决策", "基本面分析", "税务规划", "股权变动", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Limit Up Stocks Pool Query",
        "description": {
            "功能描述": "获取指定日期的涨停股池数据，包括涨跌幅、最新价、成交额、流通市值、总市值、换手率、封板资金、首次封板时间、最后封板时间、炸板次数、等详细信息。",
            "数据类型": "市场热点数据、涨停股票列表、资金动向指标",
            "数据粒度": "指定日期，每日涨停股票列表及其详细特征。",
            "应用场景": "市场热点分析、涨停板策略研究、短线交易机会识别、主力资金行为跟踪、量化投资模型构建、市场情绪判断。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "A股", "涨停股", "涨停板", "涨停池", "涨停股票", "市场热点", "资金动向",
            "封板资金", "连板数", "所属行业", "涨跌幅", "成交额", "流通市值", "总市值", "换手率",
            "首次封板", "最后封板", "炸板次数", "涨停统计", "短线交易", "主力资金", "市场情绪",
            "东方财富", "股票池", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Limit Down Stocks Pool Query",
        "description": {
            "功能描述": "获取指定日期的跌停股池数据，包括涨跌幅、最新价、成交额、流通市值、总市值、动态市盈率、换手率、封单资金、最后封板时间、板上成交额、连续跌停次数、开板次数和所属行业等详细信息。",
            "数据类型": "市场风险数据、跌停股票列表、资金流出指标",
            "数据粒度": "指定日期，每日跌停股票列表及其详细特征。",
            "应用场景": "市场风险评估、弱势股票识别、跌停板策略研究、资金流出跟踪、量化投资模型构建、风险预警。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "A股", "跌停股", "跌停板", "跌停池", "跌停股票", "市场风险", "资金流出",
            "封单资金", "连续跌停", "开板次数", "所属行业", "涨跌幅", "成交额", "流通市值",
            "总市值", "动态市盈率", "换手率", "最后封板", "板上成交额", "弱势股票", "风险预警",
            "东方财富", "股票池", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Strong Stocks Pool Query",
        "description": {
            "功能描述": "获取指定日期的强势股池数据，包括涨跌幅、最新价、涨停价、成交额、流通市值、总市值、换手率、涨速、是否新高、量比、涨停统计、入选理由和所属行业等详细信息",
            "数据类型": "市场活跃股票数据、强势股列表、股票特征指标",
            "数据粒度": "指定日期，每日强势股票列表及其详细特征。",
            "应用场景": "市场强势股识别、上涨动因分析、短线交易机会发现、量化投资策略构建、市场情绪判断、热点板块追踪。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "A股", "强势股", "强势股池", "市场活跃", "股票特征", "涨速", "新高",
            "量比", "涨停统计", "入选理由", "所属行业", "涨跌幅", "成交额", "流通市值", "总市值",
            "换手率", "上涨动因", "短线交易", "量化投资", "市场情绪", "热点板块", "东方财富",
            "股票池", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Sub New Stocks Pool Query",
        "description": {
            "功能描述": "获取指定日期的次新股池数据，包括涨跌幅、最新价、涨停价、成交额、流通市值、总市值、换手率、开板几日、开板日期、上市日期、是否新高、涨停统计和所属行业等详细信息。",
            "数据类型": "市场活跃股票数据、次新股列表、新股表现指标",
            "数据粒度": "指定日期，每日次新股票列表及其详细特征。",
            "应用场景": "次新股投资策略研究、新股上市表现分析、开板情况跟踪、市场热点识别、量化投资模型构建、风险评估。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "A股", "次新股", "次新股池", "新股", "新股上市", "开板", "开板日期",
            "上市日期", "市场热点", "投资策略", "风险评估", "东方财富", "股票池", "涨跌幅",
            "成交额", "流通市值", "总市值", "换手率", "涨停统计", "所属行业", "AKShare"
        ]
    },
    {
        "name": "AKShare Eastmoney Broken Limit Up Stocks Pool Query",
        "description": {
            "功能描述": "获取东的指定日期的炸板股池数据。炸板股是指盘中曾触及涨停但最终未能封住涨停的股票。",
            "数据类型": "市场活跃股票数据、涨停失败股票列表、市场情绪指标",
            "数据粒度": "指定日期，每日炸板股票列表及其详细特征。",
            "应用场景": "市场情绪分析、资金博弈研究、短线交易风险识别、追高风险评估、量化投资策略构建、市场异动监控。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国A股", "A股", "炸板股", "涨停失败", "涨停打开", "市场情绪", "资金博弈", "短线交易",
            "追高风险", "市场异动", "东方财富", "股票池", "涨跌幅", "成交额", "流通市值", "总市值",
            "换手率", "涨速", "首次封板", "炸板次数", "涨停统计", "振幅", "所属行业", "AKShare"
        ]
    },
    {
        "name": "CURRENCY_EXCHANGE_RATE",
        "description": {
            "功能描述": "获取任意一对数字货币或实体货币的实时汇率。此API提供即时的货币兑换比率，适用于多种金融场景。",
            "数据类型": "实时汇率数据",
            "数据粒度": "实时",
            "应用场景": "货币兑换、外汇交易、加密货币交易、金融应用开发、汇率监控、投资组合管理。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "实时汇率", "货币兑换", "外汇", "数字货币汇率", "加密货币汇率", "实体货币汇率",
            "汇率监控", "金融应用", "投资组合", "国际贸易", "跨境支付", "比特币", "美元",
            "货币比率", "Alpha Vantage"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_DAILY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每日历史时间序列数据。此API提供每日的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每日午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",
            "数据类型": "每日历史行情数据 (OHLCV)",
            "数据粒度": "每日（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "历史数据", "每日数据", "交易市场", "OHLCV", "开盘价",
            "最高价", "最低价", "收盘价", "成交量", "时间序列", "历史行情", "每日行情", "交易数据",
            "市场趋势", "回测", "投资组合", "技术分析", "量化模型", "风险管理", "美元报价",
            "Alpha Vantage"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_WEEKLY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每周历史时间序列数据。此API提供每周的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每周午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",
            "数据类型": "每周历史行情数据 (OHLCV)",
            "数据粒度": "每周（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "历史数据", "每周数据", "交易市场", "OHLCV", "开盘价",
            "最高价", "最低价", "收盘价", "成交量", "时间序列", "历史行情", "每周行情", "交易数据",
            "市场趋势", "回测", "投资组合", "技术分析", "量化模型", "中长期趋势", "美元报价",
            "Alpha Vantage"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_MONTHLY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每月历史时间序列数据。此API提供每月的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每月午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",
            "数据类型": "每月历史行情数据 (OHLCV)",
            "数据粒度": "每月（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "历史数据", "每月数据", "交易市场", "OHLCV", "开盘价",
            "最高价", "最低价", "收盘价", "成交量", "时间序列", "历史行情", "每月行情", "交易数据",
            "市场趋势", "回测", "投资组合", "技术分析", "量化模型", "超长期趋势", "宏观市场",
            "美元报价", "Alpha Vantage"
        ]
    },
    {
        "name": "AKShare ChinaMoney Bond Information Query",
        "description": {
           "功能描述": "从中国外汇交易中心暨全国银行间同业拆借中心（ChinaMoney）获取债券信息查询数据。此接口提供债券的基本信息，包括债券简称、代码、发行人、债券类型、发行日期、最新债项评级等。",
           "数据类型": "债券基本信息",
           "数据粒度": "单条或多条债券信息（根据查询条件）",
           "应用场景": "债券信息查询、市场研究、投资组合分析、信用风险评估、债券数据整理与分析。",
           "API提供商": "Akshare",
        },
        "keywords": [
           "债券", "债券基本信息", "债券概况", "发行人", "债券类型", "债项评级", "中国银行间债券市场",
           "债券代码", "发行日期", "付息方式", "主承销商", "信用风险", "债券筛选", "Akshare"
        ]
    },
    {
        "name": "AKShare Shanghai Stock Exchange Bond Deal Summary",
        "description": {
           "功能描述": "获取上海证券交易所（上登债券信息网）提供的指定交易日的债券现券市场概览数据。此接口提供包括债券现货数量、托管只数、托管市值和托管面值等关键统计信息。",
           "数据类型": "市场概览统计数据",
           "数据粒度": "每日",
           "应用场景": "债券市场每日概况分析、市场规模统计、流动性评估、宏观经济研究、监管数据报告。",
           "API提供商": "Akshare",
        },
       "keywords": [
           "债券", "现券", "市场概览", "上海证券交易所", "债券市场", "托管只数", "托管市值",
           "托管面值", "每日数据", "市场规模", "流动性评估", "宏观经济", "Akshare"
       ]
    },
    {
        "name": "AKShare ChinaMoney Bond Spot Market Maker Quotes",
        "description": {
            "功能描述": "从中国外汇交易中心暨全国银行间同业拆借中心（ChinaMoney）获取现券市场的做市报价数据。此接口提供当前所有债券的做市商报价信息，包括报价机构、债券简称、买入/卖出净价以及买入/卖出收益率。",
            "数据类型": "债券做市报价（行情）数据",
            "数据粒度": "实时快照（单次返回所有当前数据）",
            "应用场景": "债券交易决策、市场流动性分析、收益率曲线构建、做市商报价跟踪、债券估值。",
            "API提供商": "Akshare",
        },
        "keywords": [
           "债券", "现券", "做市报价", "行情数据", "中国银行间债券市场", "银行间市场", "做市商",
           "报价机构", "债券简称", "买入净价", "卖出净价", "买入收益率", "卖出收益率", "实时数据",
           "市场数据", "交易决策", "流动性分析", "收益率曲线", "债券估值", "套利", "Akshare"
        ]
    },
    {
        "name": "AKShare ChinaBond Yield Curve Query",
        "description": {
            "功能描述": "获取中国债券信息网提供的国债及其他债券的收益率曲线数据。此接口允许用户查询指定日期范围内不同期限（如3个月、6个月、1年、3年、5年、7年、10年、30年）的债券收益率。",
            "数据类型": "债券收益率曲线数据",
            "数据粒度": "每日",
            "应用场景": "债券市场分析、收益率曲线构建与分析、利率风险管理、宏观经济分析、固定收益投资策略研究、债券估值。",
            "API提供商": "Akshare",
        },
        "keywords": [
           "中国国债", "收益率曲线", "国债", "债券收益率", "历史数据", "中国债券市场", "利率",
           "期限", "宏观经济", "固定收益", "投资策略", "债券估值", "资产配置", "市场预期", "Akshare"
        ]
    },
    {
        "name": "AKShare Shanghai and Shenzhen Bond Real-time Quotes",
        "description": {
          "功能描述": "获取沪深两市债券的实时行情数据。此接口提供所有沪深债券的最新交易信息，包括价格、涨跌幅、买卖盘、成交量和成交额等。",
          "数据类型": "实时行情数据",
          "数据粒度": "实时",
          "应用场景": "债券实时行情监控、交易决策、市场分析、投资组合管理、数据分析。",
          "API提供商": "Akshare",
        },
        "keywords": [
          "沪深债券", "实时行情", "债券交易", "可转债", "新浪财经", "债券代码", "债券名称",
          "最新价", "涨跌幅", "成交量", "成交额", "买卖盘", "市场分析", "投资组合", "量化交易",
          "Akshare"
        ]
    },
    {
        "name": "AKShare Shanghai and Shenzhen Bond Daily Historical Data",
        "description": {
           "功能描述": "获取沪深两市特定债券（包括可转债）的每日历史行情数据。此接口提供指定债券的每日开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据。",
           "数据类型": "历史行情数据 (OHLCV)",
           "数据粒度": "每日（Daily）",
           "应用场景": "沪深债券（尤其是可转债）的历史走势分析、交易策略回测、市场趋势研究、投资组合表现分析。",
           "API提供商": "Akshare",
        },
        "keywords": [
           "沪深债券", "历史行情", "每日数据", "可转债", "OHLCV", "K线数据", "历史走势",
           "交易策略回测", "市场趋势", "投资组合", "技术分析", "量化模型", "新浪财经", "Akshare"
        ]
    },
    {
        "name": "AKShare Sina Convertible Bond Profile",
        "description": {
            "功能描述": "获取沪深可转债的详细资料信息。此接口提供指定可转债的各项基本信息和关键参数，以键值对的形式返回，包括但不限于发行人信息、发行条款、转股条款、到期条款等",
            "数据类型": "可转债详情资料",
            "数据粒度": "单条可转债",
            "应用场景": "可转债基本面分析、投资决策、风险评估、条款研究、数据整理。",
            "API提供商": "Akshare",
        },
        "keywords": [
          "可转债", "沪深可转债", "详情资料", "基本信息", "发行条款", "转股条款", "到期条款",
          "基本面分析", "投资决策", "风险评估", "条款研究", "发行人信息", "评级信息", "新浪财经", "Akshare"
        ]
    },
    {
        "name": "AKShare Sina Convertible Bond Summary",
         "description": {
             "功能描述": "获取沪深可转债的债券概况数据。此接口提供指定可转债的概览性信息，通常包括一些核心的交易和估值指标，以键值对的形式返回，包含最新价格、溢价率、转股价值、到期收益率等关键指标",
             "数据类型": "可转债概况数据",
             "数据粒度": "单条可转债",
             "应用场景": "可转债快速评估、投资筛选、市场表现概览、数据对比分析。",
             "API提供商": "Akshare",
         },
         "keywords": [
             "可转债", "沪深可转债", "债券概况", "核心指标", "估值指标", "溢价率", "转股价值",
             "到期收益率", "投资筛选", "市场表现", "数据对比", "新浪财经", "Akshare"
         ]
    },
    {
        "name": "AKShare Shanghai and Shenzhen Convertible Bond Real-time Quotes",
        "description": {
           "功能描述": "获取沪深两市可转债的实时行情数据。此接口提供所有沪深可转债的最新交易信息，包括价格、涨跌幅、成交量和成交额等。",
           "数据类型": "实时行情数据",
           "数据粒度": "实时",
           "应用场景": "可转债实时行情监控、交易决策、市场分析、投资组合管理。",
           "API提供商": "Akshare",
       },
        "keywords": [
           "沪深可转债", "可转债", "实时行情", "实时交易", "市场分析", "投资组合", "交易决策",
           "量化交易", "套利", "新浪财经", "债券代码", "涨跌幅", "成交量", "成交额", "Akshare"
        ]
    },
    {
        "name": "AKShare Eastmoney Convertible Bond List",
         "description": {
           "功能描述": "获取当前交易时刻的所有沪深可转债的综合数据一览表。此接口提供可转债的发行、申购、上市以及实时交易和估值相关的详细信息。",
           "数据类型": "可转债综合数据",
           "数据粒度": "单条可转债，当前交易时刻",
           "应用场景": "可转债申购分析、可转债投资筛选、估值分析、市场概览、数据研究、风险评估。",
           "API提供商": "Akshare",
         },
        "keywords": [
           "可转债", "沪深可转债", "综合数据", "全面数据", "申购分析", "投资筛选", "估值分析",
           "市场概览", "风险评估", "东方财富", "债券代码", "转股溢价率", "信用评级", "发行规模",
           "上市时间", "中签率", "转股价值", "Akshare"
        ]
    },
    {
        "name": "AKShare Eastmoney Convertible Bond Details",
        "description": {
           "功能描述": "获取指定沪深可转债的详细资料。此接口根据 `indicator` 参数的不同，提供可转债的基本信息、中签号、筹资用途或重要日期等多种维度的详细数据。",
           "数据类型": "可转债详细资料",
           "数据粒度": "单条可转债",
           "应用场景": "可转债深度分析、发行条款研究、申购信息查询、历史事件追溯、筹资项目了解。",
           "API提供商": "Akshare",
        },
        "keywords": [
           "沪深可转债", "可转债详情", "深度详细信息", "基本信息", "中签号", "筹资用途", "重要日期",
           "发行条款", "申购信息", "历史事件", "基本面研究", "投资决策", "风险管理", "东方财富", "Akshare"
        ]
    },
    {
        "name": "AKShare Eastmoney Convertible Bond Value Analysis",
        "description": {
           "功能描述": "获取指定沪深可转债的历史价值分析数据。此接口提供特定可转债在不同日期的收盘价、纯债价值、转股价值、纯债溢价率和转股溢价率等关键指标。",
           "数据类型": "可转债价值分析数据 (时间序列)",
           "数据粒度": "每日（Daily），针对单只可转债",
           "应用场景": "可转债估值分析、历史表现回溯、投资策略制定、溢价率趋势研究、套利机会识别。",
           "API提供商": "Akshare",
        },
        "keywords": [
           "可转债", "沪深可转债", "价值分析", "估值分析", "历史估值", "溢价率", "纯债价值",
           "转股价值", "纯债溢价率", "转股溢价率", "历史数据", "时间序列", "投资策略", "套利机会",
           "东方财富", "Akshare"
        ]
    },
    {
        "name": "AKShare China-US Bond Yield Rates",
        "description": {
           "功能描述": "获取中国和美国国债收益率的历史数据。此接口提供两国不同期限（2年、5年、10年、30年）的国债收益率以及10年-2年期限利差和GDP年增率的历史时间序列数据。",
           "数据类型": "国债收益率历史数据，宏观经济数据",
           "数据粒度": "每日（Daily）",
           "应用场景": "宏观经济分析、利率走势研究、中美利差分析、债券市场分析、资产配置决策、量化投资策略开发。",
           "API提供商": "Akshare",
        },
           "keywords": [
               "国债收益率", "中国国债", "美国国债", "中美利差", "中美GDP增率", "利率", "期限利差",
               "GDP年增率", "宏观经济", "国际市场", "债券市场分析", "资产配置", "量化投资", "东方财富", "Akshare"
           ]
    },
    {
        "name": "AKShare Cninfo Treasury Bond Issues",
         "description": {
           "功能描述": "获取中国国债的发行数据。此接口提供在指定日期范围内发行的国债的详细信息，包括发行量、价格、日期、对象等。",
           "数据类型": "国债发行数据",
           "数据粒度": "单条国债发行记录，可按日期范围查询",
           "应用场景": "国债市场研究、债券发行趋势分析、固定收益投资分析、历史发行数据回溯、市场容量评估。",
           "API提供商": "Akshare",
        },
        "keywords": [
           "国债", "债券发行", "中国债券市场", "国债发行数据", "发行量", "发行价格", "发行日期",
           "发行对象", "交易市场", "发行方式", "公告日期", "巨潮资讯", "固定收益", "政府融资", "Akshare"
        ]
    },
    {
        "name": "AKShare Cninfo Local Government Bond Issues",
         "description": {
          "功能描述": "获取中国地方政府债券的发行数据。此接口提供在指定日期范围内发行的所有地方政府债券的详细信息，包括发行量、价格、日期、对象等。",
          "数据类型": "地方政府债券发行数据",
          "数据粒度": "单条地方债发行记录，可按日期范围查询",
          "应用场景": "地方政府债务分析、地方债市场研究、固定收益投资分析、区域经济研究、历史发行数据回溯。",
          "API提供商": "Akshare",
        },
        "keywords": [
          "地方债", "地方政府债券", "债券发行", "中国债券市场", "地方债发行数据", "发行量", "发行价格",
          "发行日期", "发行对象", "交易市场", "发行方式", "公告日期", "巨潮资讯", "区域经济", "地方财政",
          "政府债务", "Akshare"
        ]
    },
    {
        "name": "AKShare Cninfo Corporate Bond Issues",
        "description": {
          "功能描述": "获取中国企业债券的发行数据。此接口提供在指定日期范围内发行的所有企业债券的详细信息，包括发行量、价格、日期、对象、承销方式和募资用途等。",
          "数据类型": "企业债券发行数据",
          "数据粒度": "单条企业债发行记录，可按日期范围查询",
          "应用场景": "企业债券市场研究、公司融资分析、固定收益投资分析、信用风险评估、历史发行数据回溯。",
          "API提供商": "Akshare",
        },
        "keywords": [
           "企业债", "公司债", "债券发行", "中国债券市场", "企业债发行数据", "发行量", "发行价格",
           "发行日期", "发行对象", "承销方式", "募资用途", "巨潮资讯", "公司融资", "信用分析",
           "行业融资", "Akshare"
        ]
    },
    {
        "name": "AKShare Cninfo Convertible Bond Issues",
         "description": {
           "功能描述": "获取中国可转债的发行数据。此接口提供在指定日期范围内发行的所有可转债的详细信息，包括发行量、价格、转股条款、申购信息等。",
           "数据类型": "可转债发行数据",
           "数据粒度": "单条可转债发行记录，可按日期范围查询",
           "应用场景": "可转债市场研究、新债申购分析、转股条款分析、历史发行数据回溯、固定收益投资策略制定。",
           "API提供商": "Akshare",
       },
        "keywords": [
           "可转债", "债券发行", "中国债券市场", "可转债发行", "新债申购", "转股条款", "发行数据",
           "申购信息", "优先配售", "历史发行", "固定收益", "巨潮资讯", "Akshare"
        ]
    },
    {
        "name": "NewsAPI Company News Search (Chinese)",
        "description": {
            "功能描述": "查询指定国内公司的相关中文新闻资讯。主要获取语言为中文的新闻，返回新闻的标题、来源、发布日期和链接等核心信息。",
            "数据类型": "新闻资讯 / 文章列表。",
            "数据粒度": "文章级别。",
            "应用场景": "公司舆情监控、市场事件分析、行业动态跟踪、金融研究与报告。",
            "API提供商": "NewsAPI。",
        },
        "keywords": [
            "公司新闻", "中文新闻", "国内公司", "新闻资讯", "舆情监控", "媒体报道", "金融新闻",
            "企业新闻", "公司动态", "市场事件", "行业动态", "NewsAPI"
        ]
    },
    {
        "name": "NewsAPI Company News Search (English)",
        "description": {
            "功能描述": "查询指定国内公司的相关英文新闻资讯。主要获取语言为英文的新闻，返回新闻的标题、来源、发布日期和链接等核心信息。",
            "数据类型": "新闻资讯 / 文章列表。",
            "数据粒度": "文章级别。",
            "应用场景": "国际舆情分析、海外市场风险评估、跨文化研究、投资决策。",
            "API提供商": "NewsAPI。",
        },
        "keywords": [
            "公司新闻", "英文新闻", "国内公司", "国际媒体", "新闻资讯", "舆情监控", "媒体报道",
            "金融新闻", "企业新闻", "国际舆情", "海外市场", "NewsAPI"
        ]
    },
    {
       "name": "AKShare Eastmoney Shanghai & Shenzhen Stock Exchange Balance Sheet Query",
        "description": {
            "功能描述": "获取指定日期的沪深（上海和深圳证券交易所）上市公司资产负债表数据。返回股票代码、股票简称、货币资金、应收账款、存货、总资产（含同比）、应付账款、预收账款、总负债（含同比）、资产负债率、股东权益合计和公告日期等详细财务指标。",
            "数据类型": "公司财务报表、资产负债表",
            "数据粒度": "指定日期（通常为报告期末），覆盖沪深市场所有上市公司。",
            "应用场景": "沪深上市公司财务分析、资产结构分析、偿债能力评估、基本面研究、投资决策辅助、风险管理。",
            "API提供商": "AKShare",
        },
       "keywords": [
           "A股", "沪深", "上海证券交易所", "深圳证券交易所", "资产负债表", "偿债能力", "公司财务",
           "财务报表", "资产结构", "基本面研究", "投资决策", "风险管理", "东方财富", "Akshare"
       ]
    },
    {
        "name": "AKShare Tonghuashun Financial Debt Balance Sheet Query",
        "description": {
            "功能描述": "获取指定股票的资产负债表所有历史数据。",
            "数据类型": "公司财务报表、资产负债表、历史财务数据",
            "数据粒度": "所有历史数据，可选择按报告期、按年度或按单季度查询。",
            "应用场景": "股票历史财务分析、资产结构演变分析、偿债能力长期趋势评估、基本面深入研究、投资价值判断。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "沪深", "上海证券交易所", "深圳证券交易所", "同花顺", "资产负债表", "偿债能力",
            "历史财务数据", "公司财务", "财务报表", "资产结构", "基本面分析", "投资价值", "风险预警", "Akshare"
        ]
    },
    {
        "name": "AKShare Stock Institutional Research Detail (Eastmoney)",
        "description": {
            "功能描述": "提供特定日期内所有或指定股票的机构调研详细信息，包括最新价、涨跌幅、调研机构、机构类型、调研人员、接待方式、接待人员、接待地点、调研日期和公告日期等。",
            "数据类型": "历史机构调研详细数据",
            "数据粒度": "每日 (按指定日期查询)",
            "应用场景": "机构调研活动分析、特定公司调研情况追踪、市场情绪洞察、投资策略制定。",
            "API提供商": "AKShare (数据源：东方财富网)",
        },
        "keywords": [
            "机构调研", "详细数据", "股票调研", "调研日期", "公告日期", "机构活动", "市场活动",
            "机构类型", "调研人员", "市场情绪", "投资策略", "主力资金", "东方财富", "Akshare"
        ]
    },
    {
        "name": "AKShare Eastmoney A-Share Pre-Market Minute Data",
        "description": {
            "功能描述": "获取指定A股股票最近一个交易日的分钟级数据，包含盘前分钟数据。提供指定股票在最近一个交易日的每分钟开盘价、收盘价、最高价、最低价、成交量（手）、成交额和最新价。数据覆盖盘前交易时段",
            "数据类型": "分钟级K线数据 (OHLCV)",
            "数据粒度": "分钟",
            "应用场景": "A股盘前行情分析、日内交易策略回测、高频数据研究、市场异动监控。",
            "API提供商": "AKShare (数据源：东方财富网)",
        },
        "keywords": [
            "A股", "盘前数据", "分钟数据", "实时行情", "股票行情", "K线数据", "分钟级数据",
            "日内交易", "高频数据", "市场异动", "量化交易", "东方财富", "Akshare"
        ]
    },
    {
        "name": "AKShare Sina B-Share Spot Quote",
        "description": {
            "功能描述": "获取B股市场的实时行情数据。可以查询所有B股公司的实时行情，也可以通过股票代码或名称查询特定B股的行情。提供B股市场所有或指定股票的最新价格、涨跌额、涨跌幅、买入价、卖出价、昨日收盘价、今日开盘价、最高价、最低价、成交量（股）和成交额（元）。",
            "数据类型": "实时行情数据",
            "数据粒度": "实时",
            "应用场景": "B股市场整体行情监控、B股股票价格实时查询、市场活跃度分析、特定B股投资决策参考。",
            "API提供商": "AKShare (数据源：新浪财经)",
        },
        "keywords": [
            "B股", "实时行情", "股票", "市场概览", "个股查询", "行情数据", "报价", "B股市场",
            "新浪财经", "实时价格", "最新价格", "盘中数据", "涨跌", "成交", "开盘价", "收盘价",
            "最高价", "最低价", "买卖价", "成交量", "成交额", "B股报价", "B股交易", "B股数据",
            "当日行情", "股票价格", "市场监控", "投资决策"
        ]
    },
    {
        "name": "AKShare Eastmoney A-Share Popularity Rank",
        "description": {
            "功能描述": "获取A股市场当前交易日前100个股票的人气排名数据。",
            "数据类型": "人气排名及实时行情数据",
            "数据粒度": "每日 (当前交易日)",
            "应用场景": "A股市场热点追踪、人气股票发现、市场情绪分析、短线交易机会识别。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "股票热度", "人气榜", "排名", "市场情绪", "热点股票", "热门股", "东方财富",
            "人气排名", "关注度", "热门排行", "股票排行", "情绪指标", "当日热点", "短线交易",
            "A股人气", "股票关注度", "实时行情", "最新价", "涨跌幅", 
        ]
    },
    {
        "name": "AKShare Eastmoney A-Share Surge Rank",
        "description": {
            "功能描述": "获取A股市场当前交易日前100个股票的飙升榜排名数据。用户可以指定返回前多少条飙升股票数据。",
            "数据类型": "飙升榜排名及实时行情数据",
            "数据粒度": "每日 (当前交易日)",
            "应用场景": "A股市场热点追踪、潜力股发现、市场情绪变化分析、短线交易机会识别。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "股票热度", "飙升榜", "排名", "市场情绪", "热门股票", "涨幅榜", "潜力股", "飙升股", "东方财富",
            "飙升", "涨幅", "涨速", "涨幅排名", "异动", "快速上涨", "趋势变化", "短线机会", "当日飙升",
            "热门涨幅", "人气飙升", "股票异动", "实时行情", "最新价", "涨跌幅",
        ]
    },
    {
        "name": "AKShare Eastmoney A-Share Hot Rank Detail",
        "description": {
            "功能描述": "通过指定股票代码，可以查询该股票近期的人气排名历史数据，包括排名、新晋粉丝比例和铁杆粉丝比例。",
            "数据类型": "历史人气排名及粉丝特征数据",
            "数据粒度": "每日",
            "应用场景": "A股股票人气趋势分析、粉丝结构研究、市场关注度变化追踪、投资情绪分析。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "股票热度", "历史趋势", "粉丝特征", "排名", "人气", "关注度", "股吧",
            "东方财富", "历史数据", "趋势分析", "粉丝比例", "新晋粉丝", "铁杆粉丝", "股吧数据",
            "情绪分析", "关注度变化", "粉丝构成", "股票排名历史", "个股人气", "股民情绪", "历史人气"
        ]
    },
    {
        "name": "AKShare Eastmoney Stock Hot Keywords",
        "description": {
            "功能描述": "通过指定股票代码，可以查询该股票在最近交易日的热门关键词及其热度。",
            "数据类型": "热门关键词及热度数据",
            "数据粒度": "最新交易日时点数据",
            "应用场景": "A股股票热点概念追踪、市场情绪分析、特定股票关注点识别、题材投资参考。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "股票热度", "热门关键词", "概念", "股吧", "市场情绪", "东方财富", "关键词",
            "热度", "概念股", "题材股", "行业热点", "事件驱动", "讨论焦点", "股民关注", "股票概念",
            "热点追踪", "最新数据", "A股热点", "舆情分析", "股吧热词", "股票话题"
        ]
    },
    {
        "name": "AKShare Eastmoney A-Share Latest Hot Rank Details",
        "description": {
            "功能描述": "通过指定股票代码，可以查询该股票在最新交易日的详细人气指标，包括市场类型、市场总股票数、计算时间、当前排名、排名较昨日变动、历史排名变动等。",
            "数据类型": "最新人气指标详情",
            "数据粒度": "最新交易日时点数据",
            "应用场景": "A股股票实时人气查询、市场关注度分析、个股热度变化追踪。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "股票热度", "最新排名", "关注度", "股吧", "实时数据", "东方财富", "人气指标",
            "详细数据", "当前排名", "排名变动", "股吧排名", "实时人气", "A股排名", "股票人气",
            "市场关注", "最新行情", "人气详情", "个股人气榜", "股吧人气", "排名变化"
        ]
    },
    {
        "name": "AKShare Eastmoney Hong Kong Stock Latest Hot Rank Details",
        "description": {
            "功能描述": "通过指定港股代码，可以查询该股票在最新交易日的详细人气指标，包括市场类型、市场总股票数、计算时间、当前排名、排名较昨日变动、历史排名变动等。",
            "数据类型": "最新人气指标详情",
            "数据粒度": "最新交易日时点数据",
            "应用场景": "港股实时人气查询、市场关注度分析、个股热度变化追踪。",
            "API提供商": "AKShare (数据源：东方财富网)",
        },
        "keywords": [
            "港股", "股票热度", "最新排名", "关注度", "股吧", "实时数据", "东方财富", "人气指标",
            "详细数据", "当前排名", "排名变动", "股吧排名", "实时人气", "港股排名", "股票人气",
            "市场关注", "最新行情", "香港股市", "港股市场", "人气详情", "个股人气榜", "股吧人气", "排名变化"
        ]
    },
    {
        "name": "AKShare China People's Bank of China (PBOC) Interest Rate Report",
        "description": {
            "功能描述": "获取中国央行利率决议的历史报告数据。该接口提供自1991年1月5日至今的中国央行利率决议报告，包含商品名称、日期、今值、预测值和前值。",
            "数据类型": "历史利率决议数据",
            "数据粒度": "每次决议会议",
            "应用场景": "中国央行货币政策分析、历史利率走势研究、经济数据回溯、市场预期与实际结果对比。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国央行", "PBOC", "利率", "央行", "宏观经济", "历史数据", "货币政策", "利率决议",
            "利率报告", "今值", "预测值", "前值", "利率走势", "经济数据", "市场预期", "央行政策",
            "中国经济", "金融政策", "基准利率", "利率调整", "货币政策报告", "中国人民银行"
        ]
    },
    {
       "name": "AKSHARE_MACRO_CHINA_NATIONAL_TAX_RECEIPTS",
       "description": {
           "功能描述": "获取中国全国税收收入的历史数据。此接口提供自2005年第一季度至今的税收收入总额、同比增速以及季度环比增速。",
           "数据类型": "税收收入数据，宏观经济指标",
           "数据粒度": "季度（Quarterly），包括单季度和累计季度数据",
           "应用场景": "宏观经济分析、财政政策研究、税收趋势分析、经济周期判断、历史数据回溯。",
           "API提供商": "Akshare",
       },
       "keywords": [
           "全国税收收入", "中国税收", "宏观经济", "经济指标", "财政收入", "税收数据", "历史税收",
           "同比增速", "环比增速", "季度数据", "税收总额", "财政政策", "税收趋势", "经济周期",
           "东方财富", "国家税收", "税收增长", "财政数据"
       ]
    },
    {
       "name":"AKSHARE_MACRO_CHINA_ENTERPRISE_BOOM_INDEX",
       "description": {
           "功能描述": "获取中国企业景气指数和企业家信心指数的历史数据。此接口提供自2005年第一季度至今的这两个指数的指数值、同比增速以及环比增速。",
           "数据类型": "景气指数，信心指数，宏观经济指标",
           "数据粒度": "季度（Quarterly）",
           "应用场景": "宏观经济分析、企业经营状况评估、市场情绪分析、经济周期判断、历史数据回溯。",
           "API提供商": "Akshare",
       },
       "keywords": [
           "企业景气指数", "企业家信心指数", "经济指标", "宏观经济", "景气度", "信心指数", "历史数据",
           "同比增速", "环比增速", "季度数据", "企业经营", "市场情绪", "经济周期", "东方财富",
           "中国经济", "商业景气", "企业信心", "景气指数"
       ]
    },
    {
       "name": "AKSHARE_MACRO_CHINA_QYSPJG",
        "description": {
           "功能描述": "获取中国企业商品价格指数的历史数据。此接口提供自2005年1月至今的总体企业商品价格指数以及农产品、矿产品、煤油电等主要分类指数的指数值、同比增长和环比增长数据。",
           "数据类型": "企业商品价格指数，通货膨胀指标，宏观经济指标",
           "数据粒度": "月度（Monthly）",
           "应用场景": "宏观经济分析、通货膨胀研究、商品价格趋势分析、产业链成本分析、历史数据回溯、经济周期判断。",
           "API提供商": "Akshare",
       },
       "keywords": [
           "企业商品价格指数", "农产品价格", "矿产品价格", "煤油电价格", "商品价格", "通货膨胀",
           "宏观经济", "物价指数", "历史数据", "同比增长", "环比增长", "月度数据", "分类指数",
           "产业链成本", "东方财富", "价格指数", "商品市场", "中国物价", "通胀指标", "生产价格"
       ]
    },
    {
       "name": "AKSHARE_MACRO_CHINA_STOCK_MARKET_CAP",
        "description": {
            "功能描述": "获取中国全国股票交易统计数据。此接口提供自2008年1月至今的上海和深圳证券交易所的月度股票交易统计信息，包括发行总股本、市价总值、成交金额、成交量以及A股最高和最低综合股价指数。",
            "数据类型": "股票交易统计数据，市场概览数据，宏观经济指标",
            "数据粒度": "月度（Monthly）",
            "应用场景": "股票市场分析、宏观经济分析、市场流动性研究、投资策略制定、历史数据回溯、交易所对比分析。",
            "API提供商": "Akshare",
        },
       "keywords": [
           "全国股票交易", "股票市场", "市场统计", "宏观经济", "股票数据", "交易量", "成交额",
           "总股本", "市值", "市价总值", "A股指数", "上海证券交易所", "深圳证券交易所", "交易所数据",
           "市场流动性", "投资策略", "历史数据", "月度数据", "东方财富", "股市统计", "股票交易量",
           "股票市值", "综合指数", "中国股市", "证券市场"
       ]
    },
    {
       "name":  "AKSHARE_MACRO_RMB_LOAN",
       "description": {
           "功能描述": "获取中国新增人民币贷款的历史数据。此接口提供每月新增人民币贷款总额、同比和环比增速，以及累计人民币贷款总额和同比增速。",
           "数据类型": "新增人民币贷款，累计人民币贷款，宏观经济指标，金融数据",
           "数据粒度": "月度（Monthly）",
           "应用场景": "宏观经济分析、货币政策研究、信贷增长分析、金融市场流动性评估、历史数据回溯。",
           "API提供商": "Akshare",
       },
       "keywords": [
           "新增人民币贷款", "累计人民币贷款", "人民币贷款", "贷款数据", "宏观经济", "金融市场",
           "信贷数据", "货币政策", "信贷增长", "流动性", "历史数据", "月度数据", "同花顺",
           "贷款总额", "同比增速", "环比增速", "中国信贷", "银行信贷", "社会融资"
       ]
    },
    {
        "name": "AKSHARE_MACRO_RMB_DEPOSIT",
         "description": {
            "功能描述": "获取中国人民币存款余额的历史数据。此接口提供每月新增存款总额、新增企业存款、新增储蓄存款以及新增其他存款的数量、同比和环比增速。",
            "数据类型": "人民币存款余额，宏观经济指标，金融数据",
            "数据粒度": "月度（Monthly）",
            "应用场景": "宏观经济分析、货币政策研究、存款结构分析、金融市场流动性评估、居民和企业资金流向研究、历史数据回溯。",
            "API提供商": "Akshare",
        },
        "keywords": [
            "人民币存款", "存款余额", "新增存款", "企业存款", "储蓄存款", "宏观经济", "金融市场",
            "存款结构", "资金流向", "流动性", "历史数据", "月度数据", "同花顺", "存款总额",
            "新增企业存款", "新增储蓄存款", "新增其他存款", "同比增速", "环比增速", "中国存款", "银行存款"
        ]
    },
    {
       "name": "AKSHARE_MACRO_CHINA_FX_GOLD",
       "description": {
           "功能描述": "获取中国外汇和黄金储备的历史数据。此接口提供自2008年1月至今的每月黄金储备和国家外汇储备的数值、同比增速以及环比增速。",
           "数据类型": "外汇储备，黄金储备，宏观经济指标",
           "数据粒度": "月度（Monthly）",
           "应用场景": "宏观经济分析、国际收支平衡研究、货币政策分析、黄金市场分析、历史数据回溯。",
           "API提供商": "Akshare",
       },
       "keywords": [
           "外汇储备", "黄金储备", "外汇市场", "黄金市场", "宏观经济", "国家储备", "储备资产",
           "历史数据", "月度数据", "东方财富", "同比增速", "环比增速", "中国外储", "央行储备",
           "国际收支", "货币政策", "黄金价格", "外汇管理"
       ]
    },
    {
       "name": "AKSHARE_MACRO_CHINA_MONEY_SUPPLY",
       "description": {
           "功能描述": "获取中国货币供应量（M0（流通中的现金）、M1（狭义货币）和M2（广义货币））的历史数据。此接口提供自2008年1月至今的每月M0、M1、M2的数量、同比增长和环比增长数据。",
           "数据类型": "货币供应量，宏观经济指标，金融数据",
           "数据粒度": "月度（Monthly）",
           "应用场景": "宏观经济分析、货币政策研究、通货膨胀预测、金融市场流动性分析、历史数据回溯。",
           "API提供商": "Akshare",
       },
       "keywords": [
           "中国货币供应量", "货币供应量", "M0", "M1", "M2", "宏观经济", "金融市场", "货币政策",
           "通货膨胀", "流动性分析", "历史数据", "月度数据", "东方财富", "广义货币", "狭义货币",
           "流通现金", "同比增速", "环比增速", "通胀预测", "央行数据", "货币政策工具"
       ]
    },
    {
       "name": "AKSHARE_MACRO_CHINA_URBAN_UNEMPLOYMENT",
       "description": {
           "功能描述": "从国家统计局获取中国城镇调查失业率的历史数据。此接口提供每月全国城镇调查失业率、不同年龄段（16-24岁、25-59岁）、不同户籍（本地户籍、外来户籍）以及企业就业人员周平均工作时间等详细数据。",
           "数据类型": "失业率，就业数据，宏观经济指标",
           "数据粒度": "月度（Monthly）",
           "应用场景": "宏观经济分析、劳动力市场研究、就业形势评估、社会经济政策制定、历史数据回溯。",
           "API提供商": "Akshare",
       },
       "keywords": [
           "城镇调查失业率", "失业率", "就业", "宏观经济", "劳动力市场", "就业数据", "历史数据",
           "月度数据", "国家统计局", "就业形势", "社会经济", "政策制定", "年龄段失业率",
           "户籍失业率", "平均工作时间", "就业状况", "中国失业率", "劳动力数据", "就业指标"
       ]
    },
    {
        "name": "AKSHARE_CRYPTO_JS_SPOT",
        "description": {
            "功能描述": "获取全球主要加密货币交易所的实时交易数据，包括最近报价、涨跌额、涨跌幅、24小时最高价、24小时最低价、24小时成交量以及更新时间。",
            "数据类型": "实时行情数据",
            "数据粒度": "当前时点（实时快照）",
            "应用场景": "加密货币实时价格监控、市场动态分析、交易决策辅助、数据看板展示、自动化交易系统的数据输入。",
            "API提供商": "AKShare (数据源为金十数据)",
        },
        "keywords": [
            "加密货币", "数字货币", "比特币", "以太坊", "实时行情", "交易数据", "全球交易所",
            "实时报价", "涨跌额", "涨跌幅", "最高价", "最低价", "成交量", "BTC", "ETH", "LTC",
            "BCH", "虚拟货币", "币价", "市场动态", "交易决策", "数据看板", "自动化交易", "加密资产",
            "数字资产", "行情监控", "瞬时数据", "币圈", "交易所数据"
        ]
    },
    {
        "name": "AKSHARE_CRYPTO_BITCOIN_HOLD_REPORT",
        "description": {
            "功能描述": "获取全球主要机构实时的比特币持仓报告数据。此报告提供了各机构的比特币持仓量、持仓市值、持仓成本、占总市值比重等详细信息，以及机构的分类和所在国家/地区。",
            "数据类型": "实时报告数据（快照）",
            "数据粒度": "当前时点",
            "应用场景": "了解全球机构对比特币的投资和配置情况、分析大型机构的比特币持仓趋势、评估机构资金对比特币市场的影响、进行宏观加密货币市场分析、研究比特币的机构采用率，还可用于监控市场的比特币资金流向和投资者行为分析。",
            "API提供商": "AKShare (数据源为金十数据)",
        },
        "keywords": [
            "比特币", "持仓", "机构持仓", "加密货币", "比特币持仓", "数字货币", "虚拟货币",
            "机构投资", "持仓报告", "实时数据", "全球机构", "持仓量", "持仓市值", "持仓成本",
            "占总市值比重", "资金流向", "投资者行为", "市场影响", "宏观分析", "比特币市场",
            "机构配置", "投资趋势", "链上数据", "加密资产"
        ]
    },
    {
        "name": "AKSHARE_STOCK_MARKET_PE_LG",
        "description": {
            "功能描述": "获取中国A股主要板块（包括上证、深证、创业板、科创板）的历史市盈率数据。此接口提供指定板块在不同日期的指数和平均市盈率，对于科创板则提供总市值和市盈率。",
            "数据类型": "历史估值数据（市盈率）",
            "数据粒度": "日期（数据频率可能因板块而异，例如部分板块可能提供月末数据，科创板提供每日数据）",
            "应用场景": "市场估值分析、投资策略制定、历史数据回溯、风险评估、市场趋势研究。",
            "API提供商": "AKShare (数据源为乐咕乐股)",
        },
        "keywords": [
            "市盈率", "上证", "深证", "创业板", "科创板", "A股", "板块估值", "历史市盈率",
            "估值数据", "市场估值", "盈利估值", "市盈率历史", "板块数据", "估值水平", "长期数据",
            "投资策略", "风险评估", "市场趋势", "A股估值", "PE", "PE_TTM", "静态市盈率", "动态市盈率"
        ]
    },
    {
        "name": "AKSHARE_STOCK_INDEX_PE_LG",
        "description": {
            "功能描述": "获取中国A股主要指数（如上证50、沪深300、中证500等）的历史市盈率数据。此接口提供指定指数在不同日期的指数点位以及多种市盈率指标，包括等权静态市盈率、静态市盈率、静态市盈率中位数、等权滚动市盈率、滚动市盈率和滚动市盈率中位数。",
            "数据类型": "历史估值数据（多种市盈率指标）",
            "数据粒度": "日期（通常为交易日数据）",
            "应用场景": "指数估值分析、市场宏观研究、投资组合管理、量化策略开发、历史数据回溯分析、评估市场泡沫或低估区域。",
            "API提供商": "AKShare (数据源为乐咕乐股)"
        },
        "keywords": [
            "市盈率", "上证50", "沪深300", "中证500", "指数估值", "A股指数", "历史市盈率",
            "估值数据", "指数PE", "静态市盈率", "滚动市盈率", "TTM市盈率", "市盈率中位数",
            "等权市盈率", "盈利估值", "市场泡沫", "低估区域", "量化策略", "指数分析", "估值水平",
            "估值趋势", "PE", "PE_TTM", "指数点位", "多维度估值"
        ]
    },
    {
        "name": "AKSHARE_STOCK_MARKET_PB_LG",
        "description": {
            "功能描述": "获取中国A股主要板块（包括上证、深证、创业板、科创版）的历史市净率数据。此接口提供指定板块在不同日期的指数点位以及多种市净率指标，包括市净率、等权市净率和市净率中位数。",
            "数据类型": "历史估值数据（市净率）",
            "数据粒度": "日期（通常为交易日数据）",
            "应用场景": "市场估值分析、投资策略制定、历史数据回溯、风险评估、市场趋势研究，特别是关注资产估值而非盈利估值时。",
            "API提供商": "AKShare (数据源为乐咕乐股)",
        },
        "keywords": [
            "市净率", "上证", "深证", "创业板", "科创版", "A股", "板块估值", "历史市净率",
            "估值数据", "市场估值", "资产估值", "市净率历史", "板块数据", "估值水平", "长期数据",
            "投资策略", "风险评估", "市场趋势", "A股估值", "PB", "等权市净率", "市净率中位数"
        ]
    },
    {
        "name": "AKSHARE_STOCK_INDEX_PB_LG",
        "description": {
            "功能描述": "获取中国A股主要指数（如上证50、沪深300、中证500等）的历史市净率数据。此接口提供指定指数在不同日期的指数点位以及多种市净率指标，包括市净率、等权市净率和市净率中位数。",
            "数据类型": "历史估值数据（多种市净率指标）",
            "数据粒度": "日期（通常为交易日数据）",
            "应用场景": "指数估值分析、市场宏观研究、投资组合管理、量化策略开发、历史数据回溯分析、评估市场泡沫或低估区域，特别是关注资产估值时。",
            "API提供商": "AKShare (数据源为乐咕乐股)",
        },
        "keywords": [
            "市净率", "上证50", "沪深300", "中证500", "指数估值", "A股指数", "历史市净率",
            "估值数据", "指数PB", "资产估值", "市场泡沫", "低估区域", "量化策略", "指数分析",
            "估值水平", "估值趋势", "PB", "等权市净率", "市净率中位数", "指数点位", "多维度估值"
        ]
    },
    {
        "name": "AKSHARE_STOCK_ZH_VALUATION_BAIDU",
        "description": {
            "功能描述": "获取指定A股股票的历史估值数据，包括总市值、市盈率（TTM）、市盈率（静态）、市净率和市现率。",
            "数据类型": "历史估值数据（总市值、市盈率、市净率、市现率）",
            "数据粒度": "日期（通常为交易日数据）",
            "应用场景": "个股估值分析、投资决策辅助、历史数据回溯、量化策略开发、股票基本面研究，特别是需要查询特定日期范围内估值数据时。",
            "API提供商": "AKShare (数据源为百度股市通)",
        },
        "keywords": [
            "估值", "总市值", "市盈率", "市净率", "市现率", "个股", "A股", "历史估值",
            "股票估值", "百度股市通", "PE", "PB", "PS", "PC", "TTM市盈率", "静态市盈率",
            "估值指标", "基本面研究", "投资决策", "量化策略", "历史数据", "股票代码查询"
        ]
    },
    {
        "name": "AKSHARE_STOCK_VALUE_EM",
        "description": {
            "功能描述": "获取指定A股股票历史估值分析数据。此接口提供股票在不同日期的当日收盘价、涨跌幅、总市值、流通市值、总股本、流通股本，以及多种估值指标，包括PE(TTM)、PE(静)、市净率、PEG值、市现率和市销率。",
            "数据类型": "历史估值与交易数据",
            "数据粒度": "日期（通常为交易日数据）",
            "应用场景": "个股估值分析、基本面研究、投资决策辅助、历史数据回溯、量化策略开发、股票估值模型构建。",
            "API提供商": "AKShare (数据源为东方财富网)",
        },
        "keywords": [
            "估值", "市值", "市盈率", "市净率", "市现率", "市销率", "PEG", "个股", "A股",
            "历史估值", "股票估值", "东方财富", "PE", "PB", "PS", "PC", "PE_TTM", "静态市盈率",
            "流通市值", "总股本", "流通股本", "收盘价", "涨跌幅", "估值指标", "基本面研究",
            "投资决策", "量化策略", "历史数据", "股票代码查询", "估值模型"
        ]
    }
]

# 港股市场API
HK_API_SPECS = [
    {
        "name": "MARKET_STATUS",
        "description": {
            "功能描述": "获取全球主要股票、外汇和加密货币交易场所的当前市场状态（开放或关闭）。此接口提供了一个实时的快照，显示哪些市场正在交易，哪些已经休市。",
            "数据类型": "市场状态 / 实时状态",
            "数据粒度": "实时快照（Current Snapshot）",
            "应用场景": "交易日历查询、自动化交易系统的前置检查（判断市场是否开盘）、跨市场交易策略的协调、用户界面中显示市场开放状态、全球市场概览。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "市场状态", "交易时间", "市场开放", "市场关闭", "开盘", "休市", "实时状态", "当前状态",
            "股票市场", "外汇市场", "加密货币市场", "全球市场", "交易场所", "股市", "汇市", "数字货币",
            "实时数据", "即时快照", "金融市场", "交易日历", "市场概览", "市场查询", "交易判断",
            "Alpha Vantage", "交易时间表", "开放状态", "关闭状态"
        ]
    },
    {
        "name": "AKShare Eastmoney AH-Shares Real-time Market Data",
        "description": {
            "功能描述": "获取所有A+H上市公司实时行情数据，包括A股和H股的最新价、涨跌幅、代码，以及A/H股的比价和溢价。",
            "数据类型": "实时行情数据 / 跨市场对比数据",
            "数据粒度": "实时（盘中数据，约 15 分钟延迟）",
            "应用场景": "A+H股套利分析、跨市场估值比较、投资组合配置、市场效率研究、AH股溢价/折价趋势分析、特定A+H股票实时监控。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A股", "H股", "A+H股", "A+H股行情", "实时行情", "即时行情", "动态行情", "盘中行情",
            "最新价", "涨跌幅", "股票代码", "比价", "溢价", "折价", "AH比价", "AH溢价", "AH折价",
            "套利", "套利分析", "跨市场", "跨市场估值", "估值比较", "投资组合", "市场效率",
            "东方财富", "上市公司", "盘中数据", "股票监控", "中国股市", "沪深港", "金融数据"
        ]
    },
    {
        "name": "AKShare Tencent AH-Shares Historical Market Data",
        "description": {
            "功能描述": "获取指定A+H上市公司（通过港股代码查询）的历史行情日频率数据。用户可指定港股股票代码、查询年份范围以及复权方式。",
            "数据类型": "历史行情数据",
            "数据粒度": "日频率",
            "应用场景": "A+H股历史股价走势分析、跨市场股票估值研究、技术指标计算、量化交易策略回测、除权除息影响分析、长期投资收益率计算、市场周期研究。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "A+H股", "历史行情", "历史股价", "过往数据", "历史数据", "历史K线",
            "复权", "前复权", "后复权", "不复权", "除权除息", "复权数据",
            "日频率", "日线数据", "K线", "K线数据", "K线图", "时间序列",
            "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额", "交易量", "交易额",
            "腾讯财经", "港股代码", "股票代码查询", "指定年份", "年份范围", "数据查询",
            "技术分析", "量化回测", "策略回测", "股价走势分析", "跨市场估值研究", "长期投资",
            "市场周期", "金融数据"
        ]
    },
    {
        "name": "AKShare Tencent AH-Shares Name Dictionary",
        "description": {
            "功能描述": "获取所有A+H上市公司的股票代码和名称映射关系，用于辅助查询。",
            "数据类型": "股票代码与名称字典 / 映射关系",
            "数据粒度": "非时间序列数据 (静态映射)",
            "应用场景": "根据公司名称查找A+H股港股代码、作为其他A+H股接口的前置查询、数据验证与清洗、构建A+H股列表、自动化脚本中动态查找股票代码。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "股票代码", "股票名称", "代码名称", "名称代码", "字典", "映射", "对照表", "对应关系",
            "上市公司", "A+H股", "A股代码", "H股代码", "港股代码", "股票列表",
            "查询", "辅助查询", "名称查询", "代码查询", "数据验证", "数据清洗", "动态查找",
            "腾讯财经", "静态数据", "非时间序列数据", "公司信息", "股票信息"
        ]
    },
    {
        "name": "AKShare Hong Kong Stock Spot Market Data",
        "description": {
            "功能描述": "获取所有香港股票的实时行情数据。包括最新价、涨跌幅、成交量等全面行情指标。",
            "数据类型": "实时行情数据 / 盘中数据",
            "数据粒度": "实时 (但有15分钟延时)",
            "应用场景": "港股行情查询、市场概览、投资监控（有延时）、数据分析、自动化脚本。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "港股", "香港股票", "香港股市", "港交所", "香港联交所",
            "实时行情", "即时行情", "现货行情", "盘中数据", "行情查询", "最新行情",
            "最新价", "涨跌幅", "成交量", "成交额", "开盘价", "收盘价", "最高价", "最低价",
            "东方财富", "延时数据", "交易数据", "市场概览", "投资监控", "股票数据", "金融数据"
        ]
    },
    {
        "name": "AKShare Hong Kong Stock Historical Market Data",
        "description": {
            "功能描述": "获取指定香港股票（港股）的历史行情数据，包括开盘价、收盘价、最高价、最低价、成交量、成交额等K线数据。",
            "数据类型": "历史行情数据 / K线数据",
            "数据粒度": "日线、周线、月线",
            "应用场景": "技术分析、量化回测、市场研究、数据可视化、投资组合管理。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "港股", "香港股票", "历史行情", "历史数据", "历史股价", "过往数据",
            "K线数据", "K线图", "日线", "周线", "月线", "时间周期", "历史走势",
            "复权", "前复权", "后复权", "不复权", "除权除息", "复权数据",
            "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额", "交易量", "交易额",
            "股票代码查询", "公司名称查询", "特定股票", "数据查询",
            "技术分析", "量化回测", "策略回测", "市场研究", "数据可视化", "投资组合管理",
            "金融数据"
        ]
    },
    {
        "name": "AKShare Hong Kong Stock Company Profile",
        "description": {
            "功能描述": "获取指定香港股票（港股）的详细公司资料和基本信息。包括公司名称、注册地、成立日期、所属行业、董事长、联系方式、公司介绍等全面的公司基本资料",
            "数据类型": "上市公司基本资料 / 公司简介",
            "数据粒度": "非时间序列数据 (静态信息)",
            "应用场景": "公司基本面研究、尽职调查、数据清洗与整合、合规与审计、自动化脚本获取公司基本信息。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "港股", "香港股票", "公司资料", "公司信息", "企业信息", "公司简介", "公司概况", "上市公司信息",
            "基本资料", "基本信息", "公司名称", "注册地", "成立日期", "所属行业", "董事长", "联系方式",
            "公司介绍", "基本面", "基本面研究", "尽职调查", "合规", "审计", "静态信息", "股票信息"
        ]
    },
    {
        "name": "AKShare Hong Kong Stock Security Profile",
        "description": {
            "功能描述": "获取指定香港股票（港股）的详细证券资料，包括证券代码、上市日期、发行信息、交易单位、所属交易所及互联互通资格等。",
            "数据类型": "证券基本资料 / 上市信息",
            "数据粒度": "非时间序列数据 (静态信息)",
            "应用场景": "证券信息查询、投资策略筛选、数据清洗与整合、量化交易、合规与风控。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "港股", "香港股票", "证券资料", "证券信息", "股票资料", "股票信息", "上市信息", "上市资料",
            "证券代码", "股票代码", "股票简称", "上市日期", "发行信息", "发行价", "交易单位", "每手股数",
            "交易所", "ISIN", "沪深港通", "互联互通", "交易规则", "静态信息",
            "投资策略", "量化交易", "合规", "风控", "金融数据"
        ]
    },
    {
        "name": "AKShare Hong Kong Famous Stock Real-time Spot Data",
        "description": {
            "功能描述": "获取行情中心-港股市场-知名港股中，指定股票的实时行情数据。包括最新价、涨跌幅、成交量等全面指标",
            "数据类型": "实时行情数据 / 盘中数据",
            "数据粒度": "实时 (取决于源数据提供商的更新频率)",
            "应用场景": "知名港股行情查询、投资监控、数据分析、自动化脚本。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "知名港股", "热门港股", "港股", "香港股票", "实时行情", "即时行情", "盘中数据", "动态行情",
            "最新价", "涨跌幅", "成交量", "行情数据", "实时数据", "股票行情",
            "东方财富", "行情中心", "投资监控", "数据分析", "自动化"
        ]
    },
    {
        "name": "AKShare Cninfo Stock Company Profile Query",
        "description": {
            "功能描述": "获取指定股票代码的公司概况详细信息。返回公司名称、英文名称、曾用简称、股票代码（A/B/H股）、所属市场、所属行业、法人代表、注册资金、成立日期、上市日期、官方网站、联系方式、注册地址、办公地址、主营业务、经营范围和机构简介等详细信息。",
            "数据类型": "公司基本信息、公司概况",
            "数据粒度": "按股票代码查询",
            "应用场景": "公司基本面研究、投资决策、行业分析、背景调查、数据分析。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国市场", "A股", "B股", "H股", "中国A股", "中国B股", "中国H股", "沪深市场",
            "公司概况", "公司信息", "公司简介", "企业信息", "公司基本信息", "公司档案", "股票概况",
            "巨潮资讯", "股票代码查询", "公司名称", "英文名称", "曾用简称", "所属市场", "所属行业",
            "法人代表", "注册资金", "成立日期", "上市日期", "官方网站", "联系方式", "注册地址",
            "办公地址", "主营业务", "经营范围", "机构简介", "基本面研究", "投资决策", "行业分析",
            "背景调查", "金融数据"
        ]
    },
    {
        "name": "AKShare Baidu Stock Trading Halt and Resumption Information (HK/Other Markets)",
        "description": {
            "功能描述": "获取指定日期的股票停复牌数据。提供股票代码、股票简称、交易所、停牌时间、复牌时间以及停牌事项说明等详细信息。",
            "数据类型": "股票交易状态信息 (停牌、复牌)",
            "数据粒度": "指定日期",
            "应用场景": "港股及其他非A股市场的交易策略制定、风险管理、市场事件跟踪、投资组合调整。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国港股", "港股", "香港市场", "非A股市场", "境外市场", "停牌", "复牌", "停复牌",
            "停复牌信息", "交易状态", "股票状态", "交易提醒", "百度股市通", "指定日期",
            "股票代码", "股票简称", "交易所", "停牌时间", "复牌时间", "停牌事项",
            "交易策略", "风险管理", "市场事件", "投资组合调整", "金融数据"
        ]
    },
    {
        "name": "AKShare Baidu Stock Trading Dividend Information (HK/Other Markets)",
        "description": {
            "功能描述": "获取指定日期的股票分红派息数据。提供股票代码、除权日、分红、送股、转增、实物派发、交易所和报告期等详细信息。",
            "数据类型": "股票分红派息信息",
            "数据粒度": "指定日期",
            "应用场景": "港股及其他非A股市场的投资收益分析、股息策略制定、除权除息日跟踪、投资组合管理。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国港股", "港股", "香港市场", "非A股市场", "境外市场", "分红", "派息", "股息",
            "分红派息", "利润分配", "权益分派", "除权除息", "除权日", "送股", "转增", "实物派发",
            "报告期", "股票代码", "交易所", "交易提醒", "百度股市通", "指定日期",
            "投资收益", "股息策略", "投资组合管理", "金融数据"
        ]
    },
    {
        "name": "AKShare Baidu Stock Trading Financial Report Release Information (HK/Other Markets)",
        "description": {
            "功能描述": "获取指定日期的股票财报发行数据。提供股票代码、交易所、股票简称和财报期等信息。",
            "数据类型": "公司财报发布时间信息",
            "数据粒度": "指定日期，主要针对港股及其他非A股市场",
            "应用场景": "港股及其他非A股市场的投资决策、财报季关注、市场事件跟踪、基本面分析辅助。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "中国港股", "港股", "香港市场", "非A股市场", "境外市场", "国际市场",
            "财报发布", "财务报告", "财报披露", "业绩报告", "公司业绩", "发布时间", "披露时间",
            "交易提醒", "百度股市通", "指定日期", "股票代码", "交易所", "股票简称", "财报期",
            "投资决策", "财报季", "市场事件跟踪", "基本面分析", "金融数据"
        ]
    },
    {
        "name": "NEWS_SENTIMENT",
        "description": {
            "功能描述": "提供来自全球顶级新闻媒体的实时及历史市场金融新闻和情绪数据，涵盖股票、加密货币、外汇等多种资产类别，以及财政政策、并购、IPO等广泛主题。此API可用于训练LLM模型或增强交易策略，结合其他金融数据API可提供全面的市场洞察。",
            "数据类型": "市场新闻、情绪数据",
            "数据粒度": "实时及历史新闻事件",
            "应用场景": "LLM模型训练、量化交易策略增强、市场情绪分析、宏观经济研究、特定资产新闻追踪。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "市场新闻", "金融新闻", "全球新闻", "实时新闻", "历史新闻", "新闻事件", "新闻数据",
            "情绪数据", "新闻情绪", "市场情绪", "情绪分析", "舆情分析", "舆情数据", "情感分析",
            "股票新闻", "加密货币新闻", "外汇新闻", "资产类别新闻", "宏观新闻",
            "LLM模型训练", "量化交易策略", "交易策略增强", "宏观经济研究", "特定资产追踪",
            "财政政策", "并购", "IPO", "市场主题", "新闻筛选", "时间范围", "排序",
            "Alpha Vantage", "市场洞察", "数据源", "媒体新闻", "金融市场"
        ]
    },
    {
        "name": "AKShare China-US Bond Yield Rates",
        "description": {
           "功能描述": "获取中国和美国国债收益率的历史数据。此接口提供两国不同期限（2年、5年、10年、30年）的国债收益率以及10年-2年期限利差和GDP年增率的历史时间序列数据。",
           "数据类型": "国债收益率历史数据，宏观经济数据",
           "数据粒度": "每日（Daily）",
           "应用场景": "宏观经济分析、利率走势研究、中美利差分析、债券市场分析、资产配置决策、量化投资策略开发。",
           "API提供商": "Akshare",
        },
        "keywords": [
            "国债收益率", "债券收益率", "国债利率", "利率数据", "收益率曲线",
            "中国国债", "美国国债", "中美利差", "利差", "期限利差", "中美关系", "中美经济",
            "GDP年增率", "宏观经济数据", "经济指标", "国民生产总值", "经济增长率",
            "历史数据", "每日数据", "时间序列数据", "长期数据",
            "2年期国债", "5年期国债", "10年期国债", "30年期国债", "国债期限",
            "宏观经济分析", "利率走势研究", "债券市场分析", "资产配置决策", "量化投资策略",
            "东方财富", "金融数据", "债券市场"
        ]
    },
    {
        "name": "AKSHARE_MACRO_CHINA_HK_RATE_OF_UNEMPLOYMENT",
        "description": {
           "功能描述": "获取中国香港地区的失业率历史数据。此接口提供每月失业率的前值、现值以及发布日期。",
           "数据类型": "失业率，就业数据，宏观经济指标",
           "数据粒度": "月度（Monthly）",
           "应用场景": "宏观经济分析、香港劳动力市场研究、就业形势评估、经济周期判断、历史数据回溯。",
           "API提供商": "Akshare",
       },
        "keywords": [
            "香港失业率", "失业率", "中国香港", "香港宏观经济", "劳动力市场", "就业市场",
            "就业数据", "就业率", "宏观经济指标", "经济数据", "社会经济",
            "月度数据", "历史数据", "前值", "现值", "发布日期", "时间序列",
            "东方财富", "宏观经济分析", "就业形势评估", "经济周期判断", "历史数据回溯",
            "金融数据", "香港经济"
        ]
    },
    {
        "name": "AKSHARE_MACRO_CHINA_HK_GBP",
        "description": {
           "功能描述": "获取中国香港地区的本地生产总值（GDP）历史数据。此接口提供每季度GDP的前值、现值以及发布日期。",
           "数据类型": "GDP，宏观经济指标",
           "数据粒度": "季度（Quarterly）",
           "应用场景": "宏观经济分析、香港经济增长研究、经济周期判断、投资决策参考、历史数据回溯。",
           "API提供商": "Akshare",
       },
        "keywords": [
            "香港GDP", "中国香港", "本地生产总值", "国民生产总值", "GDP数据",
            "宏观经济数据", "宏观经济指标", "经济数据", "经济增长", "经济总量",
            "季度数据", "历史数据", "前值", "现值", "发布日期", "时间序列",
            "东方财富", "宏观经济分析", "香港经济增长研究", "经济周期判断", "投资决策参考", "历史数据回溯",
            "金融数据", "香港经济"
        ]
    },
    {
        "name": "AKSHARE_MACRO_CHINA_HK_GBP_RATIO",
        "description": {
           "功能描述": "获取中国香港地区的本地生产总值（GDP）同比增速历史数据。此接口提供每季度GDP同比增速的前值、现值以及发布日期。",
           "数据类型": "GDP同比增速，宏观经济指标",
           "数据粒度": "季度（Quarterly）",
           "应用场景": "宏观经济分析、香港经济增长趋势研究、经济周期判断、投资决策参考、历史数据回溯。",
           "API提供商": "Akshare",
       },
        "keywords": [
            "香港GDP同比", "GDP同比增速", "GDP增速", "本地生产总值增速", "经济增长率",
            "中国香港", "香港宏观经济", "宏观经济数据", "宏观经济指标", "经济数据", "经济增长趋势",
            "季度数据", "历史数据", "前值", "现值", "发布日期", "时间序列",
            "东方财富", "宏观经济分析", "经济周期判断", "投资决策参考", "历史数据回溯",
            "金融数据", "香港经济"
        ]
    },
    {
        "name": "CURRENCY_EXCHANGE_RATE",
         "description": {
            "功能描述": "获取任意一对数字货币或实体货币的实时汇率。此API提供即时的货币兑换比率，适用于多种金融场景。",
            "数据类型": "实时汇率数据",
            "数据粒度": "实时",
            "应用场景": "货币兑换、外汇交易、加密货币交易、金融应用开发、汇率监控、投资组合管理。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "实时汇率", "即时汇率", "当前汇率", "货币兑换", "汇率查询", "兑换比率",
            "外汇", "外汇市场", "外汇交易", "实体货币", "法币",
            "数字货币", "加密货币", "虚拟货币", "数字资产", "区块链货币", "比特币", "以太坊",
            "汇率监控", "投资组合管理", "金融应用", "Alpha Vantage", "货币对", "牌价"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_DAILY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每日历史时间序列数据。此API提供每日的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每日午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",
            "数据类型": "每日历史行情数据 (OHLCV)",
            "数据粒度": "每日（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "虚拟货币", "数字资产", "区块链货币",
            "历史数据", "每日数据", "日行情", "K线数据", "OHLCV", "时间序列",
            "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "交易市场", "数字货币市场", "加密货币交易", "UTC刷新", "美元报价", "市场货币报价",
            "历史价格分析", "交易策略回测", "市场趋势研究", "跨市场价格比较", "投资组合分析",
            "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_WEEKLY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每周历史时间序列数据。此API提供每周的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每周午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",
            "数据类型": "每周历史行情数据 (OHLCV)",
            "数据粒度": "每周（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "虚拟货币", "数字资产", "区块链货币",
            "历史数据", "每周数据", "周行情", "K线数据", "OHLCV", "时间序列",
            "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "交易市场", "数字货币市场", "加密货币交易", "UTC刷新", "美元报价", "市场货币报价",
            "历史价格分析", "交易策略回测", "市场趋势研究", "跨市场价格比较", "投资组合分析",
            "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_MONTHLY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每月历史时间序列数据。此API提供每月的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每月午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",
            "数据类型": "每月历史行情数据 (OHLCV)",
            "数据粒度": "每月（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "虚拟货币", "数字资产", "区块链货币",
            "历史数据", "每月数据", "月行情", "K线数据", "OHLCV", "时间序列",
            "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "交易市场", "数字货币市场", "加密货币交易", "UTC刷新", "美元报价", "市场货币报价",
            "历史价格分析", "交易策略回测", "市场趋势研究", "跨市场价格比较", "投资组合分析",
            "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "NewsAPI Company News Search (Chinese)",
        "description": {
            "功能描述": "查询指定国内公司的相关中文新闻资讯。主要获取语言为中文的新闻，返回新闻的标题、来源、发布日期和链接等核心信息。",
            "数据类型": "新闻资讯 / 文章列表。",
            "数据粒度": "文章级别。",
            "应用场景": "公司舆情监控、市场事件分析、行业动态跟踪、金融研究与报告。",
            "API提供商": "NewsAPI。",
        },
        "keywords": [
            "公司新闻", "企业新闻", "中文新闻", "中文资讯", "新闻资讯", "新闻文章", "文章列表",
            "国内公司", "中国公司", "大陆公司", "上市公司新闻",
            "舆情监控", "公司舆情", "市场事件分析", "行业动态", "金融研究", "媒体报道",
            "新闻标题", "新闻来源", "发布日期", "新闻链接", "NewsAPI", "金融新闻"
        ]
    },
    {
        "name": "NewsAPI Company News Search (English)",
        "description": {
            "功能描述": "查询指定国内公司的相关英文新闻资讯。主要获取语言为英文的新闻，返回新闻的标题、来源、发布日期和链接等核心信息。",
            "数据类型": "新闻资讯 / 文章列表。",
            "数据粒度": "文章级别。",
            "应用场景": "国际舆情分析、海外市场风险评估、跨文化研究、投资决策。",
            "API提供商": "NewsAPI。",
        },
        "keywords": [
            "公司新闻", "企业新闻", "英文新闻", "英文资讯", "新闻资讯", "新闻文章", "文章列表",
            "国内公司", "中国公司", "大陆公司", "上市公司新闻",
            "国际媒体", "海外媒体", "国际舆情", "海外市场", "风险评估", "跨文化研究", "投资决策",
            "新闻标题", "新闻来源", "发布日期", "新闻链接", "NewsAPI", "金融新闻"
        ]
    },
    {
        "name": "AKShare Hong Kong Stock Financial Analysis Indicators (Eastmoney)",
        "description": {
            "功能描述": "获取港股-财务分析-主要指标中，指定港股的财务指标历史数据。包括每股收益、营业收入、净利润、毛利率、净资产收益率、资产负债率等。",
            "数据类型": "财务指标历史数据",
            "数据粒度": "年度或报告期 (季度)",
            "应用场景": "港股公司财务状况分析、业绩趋势研究、投资价值评估、财务模型构建。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "港股", "香港股票", "财务指标", "财务分析", "财务数据", "基本面数据", "公司财务",
            "主要指标", "历史财务", "年度数据", "季度数据", "报告期数据", "财务报表",
            "每股收益", "EPS", "营业收入", "营收", "净利润", "利润", "毛利率", "净资产收益率", "ROE",
            "资产负债率", "负债率", "现金流", "盈利能力", "偿债能力", "运营能力", "成长能力",
            "东方财富", "投资价值评估", "财务模型", "业绩趋势", "公司分析", "基本面分析", "金融数据"
        ]
    },
    {
        "name": "AKShare Hong Kong Stock Financial Statements (Eastmoney)",
        "description": {
            "功能描述": "获取财务报表-三大报表（资产负债表、利润表、现金流量表）的详细数据，包括各项资产、负债、所有者权益、收入、成本、利润、现金流量等详细科目",
            "数据类型": "财务报表详细数据",
            "数据粒度": "年度或报告期 (季度)",
            "应用场景": "港股公司财务状况深度分析、业绩构成研究、现金流健康度评估、财务模型构建。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "港股", "香港股票", "财务报表", "三大报表", "财务数据", "公司财务", "基本面数据",
            "资产负债表", "利润表", "现金流量表", "资产", "负债", "所有者权益", "收入", "成本", "利润", "现金流量", "科目", "财报科目",
            "年度数据", "季度数据", "报告期数据", "历史财务", "详细数据", "完整报表",
            "东方财富", "财务分析", "深度分析", "业绩构成", "现金流分析", "财务健康度", "财务模型", "金融数据"
        ]
    },
    {
        "name": "AKShare Eastmoney Hong Kong Stock Popularity Rank",
        "description": {
            "功能描述": "获取个股人气榜中，港股市场当前交易日前100个股票的人气排名数据。",
            "数据类型": "人气排名及实时行情数据",
            "数据粒度": "每日 (当前交易日)",
            "应用场景": "港股市场热点追踪、人气港股发现、市场情绪分析、投资机会识别。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "港股", "香港股票", "股票热度", "人气榜", "人气排名", "个股人气", "热门股票", "热门榜单", "关注度",
            "实时行情", "最新价", "涨跌幅", "股票代码", "股票名称", "排名数据", "前100",
            "每日数据", "当前交易日", "实时数据",
            "东方财富", "市场情绪", "热点追踪", "投资机会", "市场分析", "金融数据"
        ]
    },
    {
        "name": "AKShare Eastmoney Hong Kong Stock Hot Rank Detail",
        "description": {
            "功能描述": "通过指定港股代码，可以查询该股票近期的人气排名历史数据.",
            "数据类型": "历史人气排名数据",
            "数据粒度": "每日",
            "应用场景": "港股人气趋势分析、市场关注度变化追踪、投资情绪分析。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "港股", "香港股票", "股票热度", "人气排名", "历史趋势", "历史数据", "过往数据",
            "排名", "人气", "关注度", "股吧", "股吧关注度", "关注度变化",
            "每日数据", "时间序列", "近期数据",
            "东方财富", "人气趋势分析", "市场关注度追踪", "投资情绪分析", "指定港股", "金融数据"
        ]
    },
    {
        "name": "AKShare Eastmoney Hong Kong Stock Latest Hot Rank Details",
        "description": {
            "功能描述": "通过指定港股代码，可以查询该股票在最新交易日的详细人气指标，包括市场类型、市场总股票数、计算时间、当前排名、排名较昨日变动、历史排名变动等。",
            "数据类型": "最新人气指标详情",
            "数据粒度": "最新交易日时点数据",
            "应用场景": "港股实时人气查询、市场关注度分析、个股热度变化追踪。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "港股", "香港股票", "股票热度", "人气榜", "最新排名", "最新人气", "人气指标", "人气详情",
            "关注度", "股吧", "股吧人气", "实时数据", "最新数据", "时点数据", "最新交易日",
            "当前排名", "排名变动", "昨日变动", "历史变动", "市场类型", "市场总股票数", "计算时间",
            "东方财富", "市场关注度分析", "个股热度追踪", "实时查询", "金融数据"
        ]
    },
    {
        "name": "AKSHARE_CRYPTO_JS_SPOT",
         "description": {
            "功能描述": "获取全球主要加密货币交易所的实时交易数据，包括最近报价、涨跌额、涨跌幅、24小时最高价、24小时最低价、24小时成交量以及更新时间。",
            "数据类型": "实时行情数据",
            "数据粒度": "当前时点（实时快照）",
            "应用场景": "加密货币实时价格监控、市场动态分析、交易决策辅助、数据看板展示、自动化交易系统的数据输入。",
            "API提供商": "AKShare (数据源为金十数据)",
        },
        "keywords": [
            "加密货币", "数字货币", "虚拟货币", "加密资产", "区块链资产",
            "比特币", "BTC", "以太坊", "ETH", "莱特币", "LTC", "比特现金", "BCH", "主流币",
            "实时行情", "即时行情", "实时交易", "实时价格", "最新报价", "实时快照", "盘中数据",
            "涨跌额", "涨跌幅", "24小时最高价", "24小时最低价", "24小时成交量", "交易量", "价格", "报价",
            "全球交易所", "加密货币交易所", "交易平台", "市场动态", "价格监控", "交易决策",
            "数据看板", "自动化交易", "AKShare", "金融数据"
        ]
    },
    {
        "name": "AKSHARE_CRYPTO_BITCOIN_HOLD_REPORT",
        "description": {
            "功能描述": "获取全球主要机构实时的比特币持仓报告数据。此报告提供了各机构的比特币持仓量、持仓市值、持仓成本、占总市值比重等详细信息，以及机构的分类和所在国家/地区。",
            "数据类型": "实时报告数据（快照）",
            "数据粒度": "当前时点",
            "应用场景": "了解全球机构对比特币的投资和配置情况、分析大型机构的比特币持仓趋势、评估机构资金对比特币市场的影响、进行宏观加密货币市场分析、研究比特币的机构采用率，还可用于监控市场的比特币资金流向和投资者行为分析。",
            "API提供商": "AKShare (数据源为金十数据)",
        },
        "keywords": [
            "比特币", "BTC", "加密货币", "数字货币", "虚拟货币",
            "持仓", "机构持仓", "大户持仓", "巨鲸持仓", "持仓报告", "持仓量", "持仓市值",
            "持仓成本", "持仓占比", "机构", "公司", "企业", "上市公司", "投资机构",
            "全球机构", "市场影响", "资金流向", "投资者行为", "宏观分析", "投资配置",
            "持仓趋势", "市场影响评估", "机构采用率", "实时报告", "快照数据", "统计数据",
            "AKShare", "金融数据", "加密市场分析"
        ]
    },
    {
        "name": "AKSHARE_STOCK_HK_VALUATION_BAIDU",
         "description": {
            "功能描述": "获取指定香港股票在百度股市通上的历史估值数据，包括总市值、市盈率（TTM）、市盈率（静态）、市净率和市现率。用户可以选择查询的时间周期，或指定精确的日期范围。",
            "数据类型": "历史估值数据（总市值、市盈率、市净率、市现率）",
            "数据粒度": "日期（通常为交易日数据）",
            "应用场景": "港股个股估值分析、投资决策辅助、历史数据回溯、量化策略开发、股票基本面研究，特别是需要查询特定日期范围内或特定估值指标数据时。",
            "API提供商": "AKShare (数据源为百度股市通)",
        },
        "keywords": [
            "估值", "股票估值", "历史估值", "估值指标", "估值分析", "价值评估",
            "总市值", "市值", "市盈率", "PE", "TTM市盈率", "静态市盈率", "市净率", "PB", "市现率", "PS",
            "港股", "香港股票", "香港市场", "个股", "单只股票",
            "历史数据", "历史回溯", "时间序列", "日期范围", "交易日数据", "过往数据",
            "百度股市通", "投资决策", "量化策略", "策略开发", "基本面研究",
            "AKShare", "金融数据", "股票分析"
        ]
    }
]

# 其他市场API
OTHERS_API_SPECS = [
    {
        "name": "TIME_SERIES_INTRADAY",
        "description": {
            "功能描述": "获取指定股票在交易日内的盘中（Intraday）开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据。此接口支持获取未经股票拆分和股息派发调整的原始交易数据，以及经过调整后的数据，以便于进行更准确的历史分析。在金融领域，这种OHLCV数据常被称为“蜡烛图”数据，是技术分析的基础。",
            "数据类型": "历史行情数据 (OHLCV)",
            "数据粒度": "盘中（支持1分钟、5分钟、15分钟、30分钟、60分钟等多种时间间隔）",
            "应用场景": "股票技术分析（如绘制K线图、计算均线等）、量化交易策略回测、市场趋势研究、高频数据分析.",
            "API提供商": "Alpha Vantage"
        },
        "keywords": [
            "股票", "行情", "盘中", "历史行情", "实时盘中", "OHLCV", "K线数据", "蜡烛图",
            "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "股票拆分", "股息调整", "复权", "原始数据", "调整数据", "未经调整",
            "美股", "欧洲股票", "全球市场", "盘前交易", "盘后交易", "交易时段",
            "分钟线", "1分钟", "5分钟", "15分钟", "30分钟", "60分钟", "高频数据",
            "技术分析", "量化交易", "策略回测", "市场趋势", "均线计算", "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "TIME_SERIES_DAILY",
        "description": {
            "功能描述": "获取指定全球股票的日频（Daily）开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据。此接口提供超过20年的历史数据，支持获取未经股票拆分和股息派发调整的原始交易数据，以及经过调整后的数据（复权），以便于进行更准确的历史分析。在金融领域，这种OHLCV数据常被称为“蜡烛图”数据，是技术分析和中长期策略回测的基础。",
            "数据类型": "历史行情数据 (OHLCV)",
            "数据粒度": "日频（Daily）",
            "应用场景": "股票技术分析（如绘制K线图、计算均线等）、量化交易策略回测（尤其是中长期策略）、市场趋势研究、复权处理。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "日频", "日线", "历史行情", "OHLCV", "K线数据", "蜡烛图",
            "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "股票拆分", "股息派发", "复权", "原始数据", "调整数据", "未经调整", "除权除息",
            "美股", "欧洲股票", "全球市场", "交易日数据",
            "中长期回测", "量化交易", "策略回测", "技术分析", "市场趋势研究", "均线计算", "历史分析",
            "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "TIME_SERIES_DAILY_ADJUSTED",
        "description": {
            "功能描述": "获取指定全球股票的日频（Daily）开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据。此接口的核心特点是提供了经过股票拆分和股息派发事件调整后的收盘价（adjusted close values），这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。同时，它也包含了原始（未经调整）的OHLCV值。在金融领域，这种OHLCV数据常被称为“蜡烛图”数据。",

            "数据类型": "历史行情数据 (OHLCV，包含复权数据)",
            "数据粒度": "日频（Daily）",
            "应用场景": "股票长期表现分析、真实回报计算、量化交易策略回测（尤其是需要考虑复权因素的策略）、财务报表分析、K线图绘制（复权后更准确）。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "日频", "日线", "历史行情", "OHLCV", "K线数据", "蜡烛图",
            "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "复权", "调整后收盘价", "股票拆分", "股息派发", "除权除息", "调整数据", "修正数据", "真实回报",
            "美股", "欧洲股票", "全球市场", "交易日数据",
            "长期表现分析", "真实回报计算", "量化交易", "策略回测", "财务报表分析", "K线图绘制", "技术分析",
            "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "TIME_SERIES_WEEKLY",
        "description": {
            "功能描述": "获取指定全球股票的周频（Weekly）时间序列数据，包括每周最后一个交易日的开盘价、最高价、最低价、收盘价和成交量（OHLCV）。此接口提供超过20年的历史数据，适用于对股票进行中长期趋势分析和策略回测。在金融领域，这种OHLCV数据常被称为“蜡烛图”数据。",

            "数据类型": "历史行情数据 (OHLCV)",
            "数据粒度": "周频（Weekly）",
            "应用场景": "股票中长期趋势分析、宏观市场研究、周线级别量化交易策略回测、长期投资决策辅助、绘制周K线图。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "周频", "周线", "历史行情", "OHLCV", "K线数据", "蜡烛图",
            "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "原始数据", "未经调整",
            "美股", "欧洲股票", "全球市场",
            "中长期分析", "宏观市场研究", "周线级别", "量化交易", "策略回测", "长期投资", "趋势分析", "周K线图", "技术分析",
            "Alpha Vantage", "金融数据", "时间序列"
        ]
    },
    {
        "name": "TIME_SERIES_WEEKLY_ADJUSTED",
        "description": {
            "功能描述": "获取指定全球股票的周频（Weekly）时间序列数据。此接口的核心特点是提供了经过股票拆分和股息派发事件调整后的数据，包括每周最后一个交易日的开盘价、最高价、最低价、收盘价、周调整收盘价（weekly adjusted close）、周成交量以及周股息（weekly dividend）。这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。它涵盖了20多年的历史数据。",

            "数据类型": "历史行情数据 (OHLCV，包含复权数据及股息信息)",
            "数据粒度": "周频（Weekly）",
            "应用场景": "股票长期表现分析、真实回报计算、量化交易策略回测（尤其是需要考虑复权和股息因素的策略）、财务报表分析、绘制周K线图（复权后更准确）、股息收益分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "周频", "周线", "历史行情", "历史数据", "时间序列", "OHLCV", "K线数据", "蜡烛图",
            "开盘价", "最高价", "最低价", "收盘价", "周调整收盘价", "成交量", "交易量", "周成交量", "周股息", "股息信息",
            "复权", "调整数据", "修正数据", "真实回报", "股票拆分", "股息派发", "除权除息", "复权收盘价",
            "美股", "欧洲股票", "全球市场", "长期表现分析", "真实回报计算", "量化交易", "策略回测",
            "财务报表分析", "周K线图绘制", "股息收益分析", "技术分析", "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "TIME_SERIES_MONTHLY",
        "description": {
            "功能描述": "获取指定全球股票的周频（Weekly）时间序列数据。此接口的核心特点是提供了经过股票拆分和股息派发事件调整后的数据，包括每周最后一个交易日的开盘价、最高价、最低价、收盘价、周调整收盘价（weekly adjusted close）、周成交量以及周股息（weekly dividend）。这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。它涵盖了20多年的历史数据。",

            "数据类型": "历史行情数据 (OHLCV，包含复权数据及股息信息)",
            "数据粒度": "周频（Weekly）",
            "应用场景": "股票长期表现分析、真实回报计算、量化交易策略回测（尤其是需要考虑复权和股息因素的策略）、财务报表分析、绘制周K线图（复权后更准确）、股息收益分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "月频", "月线", "历史行情", "历史数据", "时间序列", "OHLCV", "K线数据", "蜡烛图",
            "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "原始数据", "未经调整", "非复权",
            "美股", "欧洲股票", "全球市场", "长期趋势分析", "宏观市场研究", "月线级别", "量化交易",
            "策略回测", "长期投资", "月K线图绘制", "技术分析", "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "TIME_SERIES_MONTHLY_ADJUSTED",
        "description": {
            "功能描述": "获取指定全球股票的月频（Monthly）时间序列数据。此接口的核心特点是提供了经过股票拆分和股息派发事件调整后的数据，包括每月最后一个交易日的开盘价、最高价、最低价、收盘价、月调整收盘价（monthly adjusted close）、月成交量以及月股息（monthly dividend）。这意味着历史价格数据已经过修正，消除了这些公司行为对股价连续性的影响，从而能更准确地反映股票的长期表现和真实回报。它涵盖了20多年的历史数据。",

            "数据类型": "历史行情数据 (OHLCV，包含复权数据及股息信息)",
            "数据粒度": "月频（Monthly）",
            "应用场景": "股票长期表现分析、真实回报计算、量化交易策略回测（尤其是需要考虑复权和股息因素的策略）、财务报表分析、绘制月K线图（复权后更准确）、股息收益分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "月频", "月线", "历史行情", "历史数据", "时间序列", "OHLCV", "K线数据", "蜡烛图",
            "开盘价", "最高价", "最低价", "收盘价", "月调整收盘价", "成交量", "交易量", "月成交量", "月股息", "股息信息",
            "复权", "调整数据", "修正数据", "真实回报", "股票拆分", "股息派发", "除权除息", "复权收盘价",
            "美股", "欧洲股票", "全球市场", "长期表现分析", "真实回报计算", "量化交易", "策略回测",
            "财务报表分析", "月K线图绘制", "股息收益分析", "技术分析", "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "Quote Endpoint Trending",
        "description": {
            "功能描述": "获取指定单一股票代码的最新实时或准实时价格（如最新成交价、买卖价）和成交量信息。此接口设计用于查询单个股票的即时行情数据，不适用于批量查询。",

            "数据类型": "实时行情数据 / 报价数据",
            "数据粒度": "最新（Current / Latest）",
            "应用场景": "实时股票监控、交易决策辅助、个人投资组合跟踪、股票行情展示、单一股票信息查询。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "股票", "行情", "报价", "实时行情", "即时行情", "准实时", "最新行情", "当前行情",
            "最新价格", "成交价", "买价", "卖价", "成交量", "交易量", "最新数据",
            "单一股票", "单个股票", "指定股票", "股票代码", "非批量查询",
            "美股", "欧洲股票", "全球市场", "市场动态", "实时监控", "交易决策",
            "投资组合跟踪", "行情展示", "信息查询", "Alpha Vantage", "金融数据"
        ]
    },
    {
        "name": "HISTORICAL_OPTIONS",
        "description": {
            "功能描述": "获取指定股票在特定日期的完整历史期权链数据。此接口提供期权的隐含波动率（Implied Volatility, IV）以及常见的希腊字母（如 Delta, Gamma, Theta, Vega, Rho）。期权链数据会按到期日按时间顺序排序，同一到期日内的合约则按行权价从低到高排序。",

            "数据类型": "历史期权链数据（包括看涨/看跌期权、行权价、到期日、成交量、未平仓量、隐含波动率、希腊字母）",
            "数据粒度": "特定日期（每日快照），提供超过15年的历史数据",
            "应用场景": "期权策略回测、波动率分析、期权定价模型开发与验证、风险管理（通过希腊字母）、量化交易策略开发（基于期权数据）、历史期权市场行为研究、教育与学术研究。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "期权", "期权链", "历史期权", "期权数据", "股票期权", "历史数据", "过往数据",
            "看涨期权", "看跌期权", "行权价", "到期日", "成交量", "未平仓量", "持仓量",
            "隐含波动率", "IV", "波动率", "希腊字母", "Delta", "Gamma", "Theta", "Vega", "Rho",
            "每日快照", "15年历史", "完整期权链", "历史分析", "时间序列数据",
            "期权策略", "策略回测", "波动率分析", "期权定价", "定价模型", "风险管理",
            "量化交易", "策略开发", "市场行为研究", "美股期权", "欧洲期权", "衍生品", "金融衍生品",
            "Alpha Vantage"
        ]
    },
    {
        "name": "OVERVIEW",
        "description": {
            "功能描述": "获取指定股票的公司信息、财务比率以及其他关键指标。",

            "数据类型": "公司基本面数据、财务数据、关键指标",
            "数据粒度": "按公司/财报周期更新",
            "应用场景": "公司基本面分析、财务健康度评估、投资决策支持。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "公司信息", "公司概览", "股票信息", "基本面", "基本面分析", "财务比率", "关键指标",
            "财务数据", "财报周期", "最新财报", "公司评估", "财务健康度", "投资决策", "股票分析",
            "全球股票", "欧美市场", "非中国公司", "主要股票市场", "企业信息", "公司简介",
            "Alpha Vantage"
        ]
    },
    {
        "name": "INCOME_STATEMENT",
        "description": {
            "功能描述": "获取目标公司的年度和季度损益表数据，包含映射到美国证券交易委员会（SEC）所采用的 GAAP 和 IFRS 分类体系的标准化字段。",

            "数据类型": "财务报表数据 (损益表)",
            "数据粒度": "年度和季度",
            "应用场景": "公司财务分析、盈利能力评估、投资研究、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "损益表", "利润表", "财务报表", "财务数据", "年度损益表", "季度损益表",
            "GAAP", "IFRS", "SEC", "标准化字段", "财报", "公司财报", "会计准则",
            "财务分析", "盈利能力", "盈利评估", "投资研究", "基本面分析", "公司分析",
            "全球股票", "欧美市场", "非中国公司", "企业财务", "Alpha Vantage"
        ]
    },
    {
        "name": "BALANCE_SHEET",
        "description": {
            "功能描述": "获取目标公司的年度和季度资产负债表数据，包含映射到美国证券交易委员会（SEC）所采用的 GAAP 和 IFRS 分类体系的标准化字段。",

            "数据类型": "财务报表数据 (资产负债表)",
            "数据粒度": "年度和季度",
            "应用场景": "公司财务分析、资产负债结构评估、偿债能力分析、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "资产负债表", "财务报表", "财务数据", "年度资产负债表", "季度资产负债表",
            "GAAP", "IFRS", "SEC", "标准化字段", "财报", "公司财报", "会计准则",
            "财务分析", "资产负债结构", "偿债能力", "基本面分析", "公司分析", "财务健康",
            "全球股票", "欧美市场", "非中国公司", "企业财务", "Alpha Vantage"
        ]
    },
    {
        "name": "CASH_FLOW",
        "description": {
            "功能描述": "获取目标公司的年度和季度现金流量表数据，包含映射至美国证券交易委员会（SEC）所采纳的 GAAP 和 IFRS 分类体系的标准化字段。",

            "数据类型": "财务报表数据 (现金流量表)",
            "数据粒度": "年度和季度",
            "应用场景": "公司财务分析、现金流分析、经营活动评估、投资研究、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "现金流量表", "现金流", "财务报表", "财务数据", "年度现金流", "季度现金流",
            "GAAP", "IFRS", "SEC", "标准化字段", "财报", "公司财报", "会计准则",
            "财务分析", "现金流分析", "经营活动", "投资研究", "基本面分析", "公司分析",
            "全球股票", "欧美市场", "非中国公司", "企业财务", "Alpha Vantage"
        ]
    },
    {
        "name": "EARNINGS",
        "description": {
            "功能描述": "获取目标公司的年度和季度每股收益（EPS）数据。季度数据还包括分析师预测和盈余意外指标。",

            "数据类型": "财务数据 (每股收益、分析师预测、盈余意外)",
            "数据粒度": "年度和季度",
            "应用场景": "公司盈利能力分析、投资决策、分析师预期与实际业绩对比、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "每股收益", "EPS", "盈利", "业绩", "年度EPS", "季度EPS",
            "分析师预测", "盈余意外", "盈利预测", "业绩预期", "预期差", "实际业绩",
            "财务数据", "盈利能力", "投资决策", "基本面分析", "公司分析", "财报数据",
            "全球股票", "欧美市场", "非中国公司", "Alpha Vantage"
        ]
    },
    {
        "name": "DIVIDENDS",
        "description": {
            "功能描述": "获取历史和未来（已宣布的）分红分配信息。",

            "数据类型": "分红数据",
            "数据粒度": "按分红事件",
            "应用场景": "股息投资策略、现金流预测、股票估值、基本面分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "分红", "股息", "派息", "分红信息", "股息分配", "派息记录",
            "历史分红", "未来分红", "已宣布分红", "分红事件", "股息率",
            "股息投资", "现金流预测", "股票估值", "基本面分析", "投资策略", "收益",
            "全球股票", "欧美市场", "非中国公司", "Alpha Vantage"
        ]
    },
    {
        "name": "SPLITS",
        "description": {
            "功能描述": "获取历史拆股（Stock Split）事件信息。",

            "数据类型": "公司事件数据 (拆股)",
            "数据粒度": "按事件",
            "应用场景": "股票历史数据调整（例如，调整历史股价和成交量以反映拆股影响）、财务分析、投资组合管理、技术分析数据校准。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "拆股", "股票拆分", "股份拆分", "合股", "历史拆股", "公司事件",
            "数据调整", "股价调整", "成交量调整", "复权", "除权", "除权除息",
            "财务分析", "投资组合管理", "技术分析", "数据校准", "历史数据",
            "全球股票", "欧美市场", "非中国公司", "Alpha Vantage"
        ]
    },
    {
        "name": "EARNINGS_CALENDAR",
        "description": {
            "功能描述": "返回未来 3、6 或 12 个月内预计发布财报的公司列表。",

            "数据类型": "财报发布日程/日历",
            "数据粒度": "未来3、6或12个月",
            "应用场景": "市场事件跟踪、投资策略规划（如财报季交易）、风险管理、公司基本面研究。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "财报日历", "财报发布", "业绩预告", "财报日程", "财报季", "发布时间",
            "未来财报", "公司列表", "即将发布", "季度财报", "年度财报",
            "市场事件", "事件跟踪", "投资策略", "风险管理", "基本面研究", "公司研究",
            "全球股票", "欧美市场", "非中国公司", "Alpha Vantage"
        ]
    },
    {
        "name": "NEWS_SENTIMENT",
        "description": {
            "功能描述": "提供来自全球顶级新闻媒体的实时及历史市场金融新闻和情绪数据，涵盖股票、加密货币、外汇等多种资产类别，以及财政政策、并购、IPO等广泛主题。此API可用于训练LLM模型或增强交易策略，结合其他金融数据API可提供全面的市场洞察。",

            "数据类型": "市场新闻、情绪数据",
            "数据粒度": "实时及历史新闻事件",
            "应用场景": "LLM模型训练、量化交易策略增强、市场情绪分析、宏观经济研究、特定资产新闻追踪。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "市场新闻", "金融新闻", "新闻数据", "情绪数据", "新闻情绪", "舆情分析",
            "实时新闻", "历史新闻", "新闻事件", "新闻追踪", "新闻源", "顶级媒体",
            "股票新闻", "加密货币新闻", "外汇新闻", "数字货币新闻", "资产类别",
            "财政政策", "并购", "IPO", "市场主题", "经济新闻", "宏观经济",
            "LLM模型训练", "量化交易", "策略增强", "市场情绪", "情绪分析", "市场洞察",
            "数据筛选", "数据排序", "Alpha Vantage", "全球市场"
        ]
    },
    {
        "name": "CURRENCY_EXCHANGE_RATE",
        "description": {
            "功能描述": "获取任意一对数字货币或实体货币的实时汇率。此API提供即时的货币兑换比率，适用于多种金融场景。",

            "数据类型": "实时汇率数据",
            "数据粒度": "实时",
            "应用场景": "货币兑换、外汇交易、加密货币交易、金融应用开发、汇率监控、投资组合管理。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "实时汇率", "即时汇率", "最新汇率", "货币兑换", "汇率查询", "兑换比率",
            "外汇", "外汇市场", "外汇交易", "实体货币", "法定货币", "USD", "EUR",
            "数字货币", "加密货币", "虚拟货币", "数字资产", "Bitcoin", "BTC",
            "汇率监控", "金融应用", "投资组合", "Alpha Vantage"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_DAILY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每日历史时间序列数据。此API提供每日的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每日午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",

            "数据类型": "每日历史行情数据 (OHLCV)",
            "数据粒度": "每日（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "虚拟货币", "数字资产", "历史数据", "每日数据", "日线数据",
            "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "每日行情", "历史行情", "时间序列", "历史价格", "交易市场", "交易对", "BTC/EUR",
            "美元报价", "货币报价", "策略回测", "市场趋势", "价格分析", "Alpha Vantage"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_WEEKLY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每周历史时间序列数据。此API提供每周的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每周午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",

            "数据类型": "每周历史行情数据 (OHLCV)",
            "数据粒度": "每周（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "虚拟货币", "数字资产", "历史数据", "每周数据", "周线数据",
            "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "每周行情", "历史行情", "时间序列", "历史价格", "交易市场", "交易对",
            "美元报价", "货币报价", "策略回测", "市场趋势", "价格分析", "Alpha Vantage"
        ]
    },
    {
        "name": "DIGITAL_CURRENCY_MONTHLY",
        "description": {
            "功能描述": "获取指定数字/加密货币在特定交易市场上的每月历史时间序列数据。此API提供每月的开盘价、最高价、最低价、收盘价和成交量（OHLCV）数据，每月午夜（UTC）刷新。价格和成交量同时以市场特定货币和美元报价。",

            "数据类型": "每月历史行情数据 (OHLCV)",
            "数据粒度": "每月（Daily）",
            "应用场景": "数字货币历史价格分析、加密货币交易策略回测、市场趋势研究、跨市场价格比较、投资组合表现分析。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "数字货币", "加密货币", "虚拟货币", "数字资产", "历史数据", "每月数据", "月线数据",
            "OHLCV", "开盘价", "最高价", "最低价", "收盘价", "成交量", "交易量",
            "每月行情", "历史行情", "时间序列", "历史价格", "交易市场", "交易对",
            "美元报价", "货币报价", "策略回测", "市场趋势", "价格分析", "Alpha Vantage"
        ]
    },
    {
        "name": "NewsAPI Foreign Company News Search (English)",
        "description": {
            "功能描述": "查询指定国外公司的相关英文新闻资讯。",

            "数据类型": "新闻资讯 / 文章列表。",
            "数据粒度": "文章级别。",
            "应用场景": "国际公司舆情监控、全球市场趋势分析、国际投资决策、跨国企业研究。",
            "API提供商": "NewsAPI",
        },
        "keywords": [
            "国外公司", "国际公司", "非中国公司", "英文新闻", "英文资讯", "国际媒体",
            "新闻资讯", "新闻搜索", "公司新闻", "金融新闻", "市场新闻", "新闻文章",
            "舆情监控", "媒体分析", "全球市场", "市场趋势", "投资决策", "跨国企业",
            "新闻标题", "新闻来源", "发布日期", "新闻链接", "实时新闻", "历史新闻", "NewsAPI"
        ]
    },
    {
        "name": "AKShare Baidu Stock Trading Halt and Resumption Information (HK/Other Markets)",
        "description": {
            "功能描述": "获取指定日期的股票停复牌数据，提供股票代码、股票简称、交易所、停牌时间、复牌时间以及停牌事项说明等详细信息",

            "数据类型": "股票交易状态信息 (停牌、复牌)",
            "数据粒度": "指定日期",
            "应用场景": "港股及其他非A股市场的交易策略制定、风险管理、市场事件跟踪、投资组合调整。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "非A股", "港股", "香港股市", "其他市场", "股票停牌", "股票复牌", "停牌信息", "复牌信息",
            "停复牌", "停复牌信息", "交易状态", "交易提醒", "百度股市通",
            "股票代码", "股票简称", "交易所", "停牌时间", "复牌时间", "停牌事项",
            "风险管理", "市场事件", "交易策略", "投资组合调整", "AKShare"
        ]
    },
    {
        "name": "AKShare Baidu Stock Trading Dividend Information (HK/Other Markets)",
        "description": {
            "功能描述": "获取指定日期的股票分红派息数据，提供股票代码、除权日、分红、送股、转增、实物派发、交易所和报告期等详细信息",

            "数据类型": "股票分红派息信息",
            "数据粒度": "指定日期",
            "应用场景": "港股及其他非A股市场的投资收益分析、股息策略制定、除权除息日跟踪、投资组合管理。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "非A股", "港股", "香港股市", "其他市场", "分红", "派息", "股息", "分红派息",
            "分红信息", "股息信息", "派息数据", "百度股市通", "交易提醒",
            "股票代码", "除权日", "送股", "转增", "实物派发", "报告期",
            "投资收益", "股息策略", "除权除息", "除权除息日", "投资组合管理", "AKShare"
        ]
    },
    {
        "name": "AKShare Baidu Stock Trading Financial Report Release Information (HK/Other Markets)",
        "description": {
            "功能描述": "获取指定日期的股票财报发行数据，提供股票代码、交易所、股票简称和财报期等信息。",

            "数据类型": "公司财报发布时间信息",
            "数据粒度": "指定日期，主要针对港股及其他非A股市场",
            "应用场景": "港股及其他非A股市场的投资决策、财报季关注、市场事件跟踪、基本面分析辅助。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "非A股", "港股", "香港股市", "其他市场", "财报发布", "财报发行", "财报时间",
            "财报日期", "业绩发布", "业绩预告", "发布日程", "发布日历", "公司财报",
            "股票代码", "股票简称", "交易所", "财报期", "百度股市通", "交易提醒",
            "投资决策", "财报季", "市场事件", "基本面分析", "公司研究", "AKShare"
        ]
    },
    {
        "name": "AKShare European Central Bank (ECB) Interest Rate Report",
        "description": {
            "功能描述": "获取欧洲央行利率决议的历史报告数据。该接口提供自1999年1月1日至今的欧洲央行利率决议报告，包含商品名称、日期、今值、预测值和前值。",

            "数据类型": "历史利率决议数据",
            "数据粒度": "每次决议会议",
            "应用场景": "欧洲央行货币政策分析、历史利率走势研究、经济数据回溯、市场预期与实际结果对比。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "欧洲央行", "ECB", "利率", "利率决议", "央行利率", "基准利率", "政策利率",
            "宏观经济", "欧元区", "经济数据", "金融数据", "报告数据",
            "历史数据", "历史报告", "历次决议", "1999年", "时间序列",
            "今值", "预测值", "前值", "公布值", "市场预期",
            "货币政策", "货币政策分析", "利率走势", "经济数据回溯", "AKShare"
        ]
    },
    {
        "name": "AKShare New Zealand Reserve Bank (RBNZ) Interest Rate Report",
        "description": {
            "功能描述": "获取新西兰联储利率决议的历史报告数据。该接口提供自1999年4月1日至今的新西兰联储利率决议报告，包含商品名称、日期、今值、预测值和前值。",

            "数据类型": "历史利率决议数据",
            "数据粒度": "每次决议会议",
            "应用场景": "新西兰联储货币政策分析、历史利率走势研究、经济数据回溯、市场预期与实际结果对比。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "新西兰联储", "RBNZ", "利率", "利率决议", "央行利率", "基准利率", "政策利率",
            "宏观经济", "新西兰经济", "经济数据", "金融数据", "报告数据",
            "历史数据", "历史报告", "历次决议", "1999年", "时间序列",
            "今值", "预测值", "前值", "公布值", "市场预期",
            "货币政策", "货币政策分析", "利率走势", "经济数据回溯", "AKShare"
        ]
    },
    {
        "name": "AKShare Swiss National Bank (SNB) Interest Rate Report",
        "description": {
            "功能描述": "获取瑞士央行利率决议的历史报告数据。该接口提供自2008年3月13日至今的瑞士央行利率决议报告，包含商品名称、日期、今值、预测值和前值。",

            "数据类型": "历史利率决议数据",
            "数据粒度": "每次决议会议",
            "应用场景": "瑞士央行货币政策分析、历史利率走势研究、经济数据回溯、市场预期与实际结果对比。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "瑞士央行", "SNB", "利率", "利率决议", "央行利率", "基准利率", "政策利率",
            "宏观经济", "瑞士经济", "经济数据", "金融数据", "报告数据",
            "历史数据", "历史报告", "历次决议", "2008年", "时间序列",
            "今值", "预测值", "前值", "公布值", "市场预期",
            "货币政策", "货币政策分析", "利率走势", "经济数据回溯", "AKShare"
        ]
    },
    {
        "name": "AKShare UK Bank of England (BoE) Interest Rate Report",
        "description": {
            "功能描述": "获取英国央行利率决议的历史报告数据。该接口提供自1970年1月1日至今的英国央行利率决议报告，包含商品名称、日期、今值、预测值和前值。",

            "数据类型": "历史利率决议数据",
            "数据粒度": "每次决议会议",
            "应用场景": "英国央行货币政策分析、历史利率走势研究、经济数据回溯、市场预期与实际结果对比。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "英国央行", "BoE", "英格兰银行", "利率", "利率决议", "央行利率", "政策利率",
            "宏观经济", "英国经济", "货币政策", "经济数据", "金融数据",
            "历史数据", "历史报告", "历史利率走势", "利率走势", "数据回溯",
            "今值", "预测值", "前值", "市场预期", "实际结果", "央行会议",
            "AKShare"
        ]
    },
    {
        "name": "AKShare Australia Reserve Bank (RBA) Interest Rate Report",
        "description": {
            "功能描述": "获取澳洲联储利率决议的历史报告数据。该接口提供自1980年2月1日至今的澳洲联储利率决议报告，包含商品名称、日期、今值、预测值和前值。",

            "数据类型": "历史利率决议数据",
            "数据粒度": "每次决议会议",
            "应用场景": "澳洲联储货币政策分析、历史利率走势研究、经济数据回溯、市场预期与实际结果对比。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "澳洲联储", "RBA", "澳大利亚央行", "利率", "利率决议", "央行利率", "政策利率",
            "宏观经济", "澳大利亚经济", "货币政策", "经济数据", "金融数据",
            "历史数据", "历史报告", "历史利率走势", "利率走势", "数据回溯",
            "今值", "预测值", "前值", "市场预期", "实际结果", "央行会议",
            "AKShare"
        ]
    },
    {
        "name": "AKShare Japan Bank of Japan (BoJ) Interest Rate Report",
        "description": {
            "功能描述": "获取日本央行利率决议的历史报告数据。该接口提供自2008年2月14日至今的日本央行利率决议报告，包含商品名称、日期、今值、预测值和前值。",

            "数据类型": "历史利率决议数据",
            "数据粒度": "每次决议会议",
            "应用场景": "日本央行货币政策分析、历史利率走势研究、经济数据回溯、市场预期与实际结果对比。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "日本央行", "BoJ", "利率", "利率决议", "央行利率", "政策利率",
            "宏观经济", "日本经济", "货币政策", "经济数据", "金融数据",
            "历史数据", "历史报告", "历史利率走势", "利率走势", "数据回溯",
            "今值", "预测值", "前值", "市场预期", "实际结果", "央行会议",
            "AKShare"
        ]
    },
    {
        "name": "AKShare Russia Central Bank (CBR) Interest Rate Report",
        "description": {
            "功能描述": "获取俄罗斯央行利率决议的历史报告数据。该接口提供自2003年6月1日至今的俄罗斯央行利率决议报告，包含商品名称、日期、今值、预测值和前值。",

            "数据类型": "历史利率决议数据",
            "数据粒度": "每次决议会议",
            "应用场景": "俄罗斯央行货币政策分析、历史利率走势研究、经济数据回溯、市场预期与实际结果对比。",
            "API提供商": "AKShare",
        },
        "keywords": [
            "俄罗斯央行", "CBR", "利率", "利率决议", "央行利率", "政策利率",
            "宏观经济", "俄罗斯经济", "货币政策", "经济数据", "金融数据",
            "历史数据", "历史报告", "历史利率走势", "利率走势", "数据回溯",
            "今值", "预测值", "前值", "市场预期", "实际结果", "央行会议",
            "AKShare"
        ]
    },
    {
       "name": "AKSHARE_MACRO_INFO_WS",
        "description": {
            "功能描述": "获取指定日期的全球宏观经济日历事件数据。此接口提供特定日期内发生的各类宏观经济事件的详细信息，包括发布时间、所属地区、事件名称、重要性评级、实际公布值、市场预期值、前值以及事件详情链接。",

            "数据类型": "宏观经济日历，经济事件，金融新闻",
            "数据粒度": "日度（Daily），具体事件发生时间精确到小时分钟",
            "应用场景": "宏观经济事件跟踪、市场情绪分析、交易策略制定（基于事件驱动）、经济数据发布时间查询、历史事件回顾与分析。",
            "API提供商": "Akshare",
        },
       "keywords": [
           "宏观日历", "经济日历", "全球宏观", "宏观经济", "经济事件", "金融日历",
           "华尔街见闻", "财经日历", "经济数据", "数据发布", "事件跟踪",
           "发布时间", "所属地区", "事件名称", "重要性", "今值", "预期值", "前值",
           "市场情绪", "事件驱动", "交易策略", "历史事件", "数据查询", "全球经济",
           "AKShare"
       ]
    },
    {
       "name": "AKSHARE_NEWS_ECONOMIC_BAIDU",
       "description": {
           "功能描述": "获取指定日期的全球宏观经济重大事件数据。此接口提供特定日期内发生的各类宏观经济事件的详细信息，包括发布日期、时间、所属地区、事件名称、实际公布值、市场预期值、前值以及重要性评级。",

           "数据类型": "宏观经济日历，经济事件，金融新闻",
           "数据粒度": "日度（Daily），具体事件发生时间精确到小时分钟",
           "应用场景": "宏观经济事件跟踪、市场情绪分析、交易策略制定（基于事件驱动）、经济数据发布时间查询、历史事件回顾与分析。",
           "API提供商": "Akshare",
       },
       "keywords": [
           "全球宏观事件", "宏观指标", "经济日历", "宏观经济", "经济事件", "金融日历",
           "百度股市通", "财经日历", "经济数据", "数据发布", "事件跟踪",
           "发布日期", "发布时间", "所属地区", "事件名称", "公布值", "预期值", "前值", "重要性",
           "市场情绪", "事件驱动", "交易策略", "历史事件", "数据查询", "全球经济",
           "AKShare"
       ]
    },
    {
        "name": "AKSHARE_CRYPTO_JS_SPOT",
        "description": {
            "功能描述": "获取全球主要加密货币交易所的实时交易数据，包括最近报价、涨跌额、涨跌幅、24小时最高价、24小时最低价、24小时成交量以及更新时间。",

            "数据类型": "实时行情数据",
            "数据粒度": "当前时点（实时快照）",
            "应用场景": "加密货币实时价格监控、市场动态分析、交易决策辅助、数据看板展示、自动化交易系统的数据输入。",
            "API提供商": "AKShare (数据源为金十数据)",
        },
        "keywords": [
            "加密货币", "数字货币", "虚拟货币", "数字资产", "主流币",
            "实时行情", "即时行情", "现货行情", "实时交易", "实时价格", "最新报价",
            "交易所", "交易对", "BTC", "比特币", "LTC", "莱特币", "BCH", "比特币现金", "ETH", "以太坊",
            "涨跌额", "涨跌幅", "24小时最高价", "24小时最低价", "24小时成交量", "交易量",
            "市场动态", "价格监控", "交易决策", "数据看板", "自动化交易",
            "AKShare"
        ]
    },
    {
        "name": "AKSHARE_CRYPTO_BITCOIN_HOLD_REPORT",
        "description": {
            "功能描述": "获取全球主要机构实时的比特币持仓报告数据。此报告提供了各机构的比特币持仓量、持仓市值、持仓成本、占总市值比重等详细信息，以及机构的分类和所在国家/地区。",
            "数据类型": "实时报告数据（快照）",
            "数据粒度": "当前时点",
            "应用场景": "了解全球机构对比特币的投资和配置情况、分析大型机构的比特币持仓趋势、评估机构资金对比特币市场的影响、进行宏观加密货币市场分析、研究比特币的机构采用率，还可用于监控市场的比特币资金流向和投资者行为分析。",
            "API提供商": "AKShare (数据源为金十数据)",
        },
        "keywords": [
            "比特币", "加密货币", "数字货币", "虚拟货币", "数字资产",
            "持仓", "机构持仓", "比特币持仓", "持仓报告", "实时报告", "持仓量", "持仓市值", "持仓成本", "持仓占比",
            "机构投资", "机构配置", "大型机构", "资金流向", "投资者行为", "市场影响",
            "宏观分析", "市场分析", "机构采用率", "公司持仓", "国家地区",
            "AKShare"
        ]
    },
    {
        "name": "MARKET_STATUS",
        "description": {
            "功能描述": "获取全球主要股票、外汇和加密货币交易场所的当前市场状态（开放或关闭）。此接口提供了一个实时的快照，显示哪些市场正在交易，哪些已经休市。",
            "数据类型": "市场状态 / 实时状态",
            "数据粒度": "实时快照（Current Snapshot）",
            "应用场景": "交易日历查询、自动化交易系统的前置检查（判断市场是否开盘）、跨市场交易策略的协调、用户界面中显示市场开放状态、全球市场概览。",
            "API提供商": "Alpha Vantage",
        },
        "keywords": [
            "市场状态", "交易时间", "市场开放", "市场关闭", "开盘", "休市", "实时状态", "当前状态",
            "股票市场", "外汇市场", "加密货币市场", "全球市场", "交易场所", "股市", "汇市", "数字货币",
            "实时数据", "即时快照", "金融市场", "交易日历", "市场概览", "市场查询", "交易判断",
            "Alpha Vantage", "交易时间表", "开放状态", "关闭状态"
        ]
    },
]

ALL_SPECS = [AM_API_SPECS, CN_API_SPECS, HK_API_SPECS, OTHERS_API_SPECS]


def create_api_documents(api_specs: List[Dict]) -> List[Document]:
    """
    将API规范转换为LangChain Documents

    Args: 需要转换的API文档

    Returns: 转换完毕的LangChain Documents
    """
    documents = []
    for spec in api_specs:
        # 用于存储加权后的描述部分
        weighted_description_parts = []

        # 遍历 'description' 字典的所有值，根据权重拼接
        if isinstance(spec.get('description'), dict):
            for key, value in spec['description'].items():
                current_text = ""
                if isinstance(value, str):
                    current_text = value
                elif isinstance(value, list):
                    current_text = "\n".join(value)

                if key == "功能描述":
                    # 功能描述权重为2，重复2次
                    weighted_description_parts.append((current_text + " ") * 2)
                elif key == "数据粒度":
                    # 数据粒度权重为3，重复5次
                    weighted_description_parts.append((current_text + " ") * 5)
                else:
                    # 其他描述字段权重为1
                    weighted_description_parts.append(current_text)

        # 将所有加权后的描述部分拼接起来
        combined_description = "\n".join(part.strip() for part in weighted_description_parts if part.strip())

        # 将keywords列表转换为一个字符串，并添加到内容中
        keywords_str = ""
        if 'keywords' in spec and isinstance(spec['keywords'], list):
            # 关键词权重为1
            keywords_str = " ".join(spec['keywords'])

        # 构建Document的page_content,API名称、组合后的描述和关键词都包含在内
        content = f"API 名称: {spec['name']}\n"
        content += f"描述: {combined_description.strip()}\n"
        if keywords_str:
            content += f"关键词: {keywords_str}\n"

        documents.append(Document(page_content=content, metadata={"name": spec['name']}))
    return documents

def create_api_documents_keywords_only(api_specs: List[Dict]) -> List[Document]:
    """
    将API规范转换为LangChain Documents，仅包含关键词

    Args: 需要转换的API文档

    Returns: 转换完毕的LangChain Documents，仅包含关键词
    """
    documents = []
    for spec in api_specs:
        keywords_str = ""
        if 'keywords' in spec and isinstance(spec['keywords'], list):
            # 用空格连接所有关键词
            keywords_str = " ".join(spec['keywords'])

        # 构建Document的page_content，仅包含关键词
        content = f"API 名称: {spec['name']}\n"
        if keywords_str:
            content += f"关键词: {keywords_str}\n"
        else:
            # 没有关键词
            content += "无关键词信息。\n"


        documents.append(Document(page_content=content, metadata={"name": spec['name']}))
    return documents

if __name__ == "__main__" :
    DashScope_API_KEY = os.getenv("DASHSCOPE_API_KEY")

    try:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=DashScope_API_KEY
        )
        print("DashScopeEmbeddings model initialized successfully.")
    except Exception as e:
        print(f"Error initializing DashScopeEmbeddings: {e}")
        # 如果初始化失败，程序退出
        exit()

    # 构建完整文档
    index_list = ["AM", "CN", "HK", "OT"]
    for API_SPECS, index in zip(ALL_SPECS, index_list):

        documents = create_api_documents(API_SPECS)
        print(f"准备了 {len(documents)} 个文档用于嵌入。")

        # 嵌入并存储到FAISS向量数据库
        database_path = f"./{index}_APISPECS_faiss_index"

        try:
            database = FAISS.from_documents(documents, embeddings)
            print("\nAPI 规范已成功嵌入并存储到 FAISS 向量数据库中。")

            # 保存数据库到本地
            database.save_local(database_path)
            print(f"FAISS 数据库已保存到: {database_path}")

        except Exception as e:
            print(f"\n嵌入或存储到 FAISS 失败: {e}")
            # 构建数据库失败
            database = None

        # 构建关键词文档

        key_documents = create_api_documents_keywords_only(API_SPECS)
        print(f"准备了 {len(key_documents)} 个key_文档用于嵌入。")

        # 嵌入并存储到FAISS向量数据库
        key_database_path = f"./{index}_APISPECS_key_faiss_index"

        try:
            key_database = FAISS.from_documents(key_documents, embeddings)
            print("\nkey_API 规范已成功嵌入并存储到 FAISS 向量数据库中。")

            # 保存数据库到本地
            key_database.save_local(key_database_path)
            print(f"FAISS 数据库已保存到: {key_database_path}")

        except Exception as e:
            print(f"\n嵌入或存储到 FAISS 失败: {e}")
            # 构建数据库失败
            key_database = None
