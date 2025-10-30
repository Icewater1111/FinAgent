# import os
#
# import requests
# from alpha_vantage.fundamentaldata import FundamentalData
# from alpha_vantage.timeseries import TimeSeries
# from alpha_vantage.alphaintelligence import AlphaIntelligence
# from alpha_vantage.cryptocurrencies import CryptoCurrencies
# from dotenv import load_dotenv
# import akshare as ak
# load_dotenv(override=True)
#
# MAX_ROWS_TO_DISPLAY = 10
#
# Alpha_Vantage_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
# fd = FundamentalData(key=Alpha_Vantage_API_KEY, output_format="pandas")
#
# symbol = '科创版'
#
# df = ak.stock_market_pe_lg(symbol=symbol)
# print(df)

def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: int
    """

    a = {}
    ans = 0
    n = 0
    k = 0
    for i, st in enumerate(s):
        print(a, st)
        if st in a and a[st] >= k:

            # print("a")
            n -= a[st] - k + 1
            k = a[st] + 1
        a[st] = i
        n += 1
        print(n)
        ans = max(ans, n)
        #print(ans)
    return ans

s = "aab"

print(lengthOfLongestSubstring(s))