# import akshare as ak
#
# data = ak.stock_profile_cninfo(symbol="00700")
# if(data.empty):
#     print(data)
# info = data.iloc[0]
# print(info.get('公司名称'))


# import csv
# import requests
#
# # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# CSV_URL = 'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol=IBM&apikey=XXS43OQQWEGBJ5TS'
#
# with requests.Session() as s:
#     download = s.get(CSV_URL)
#
#     decoded_content = download.content.decode('utf-8')
#     cr = csv.reader(decoded_content.splitlines(), delimiter=',')
#     my_list = list(cr)
#     for row in my_list:
#         print(row)

str = (f"序号: {1},"
        f"代码: {"111"}")


print(str)

i = 1
print(-i)