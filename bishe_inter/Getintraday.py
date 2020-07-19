import tushare
import pandas
import datetime
import os
import time

token = '5d9ec9dc6d71031a48a24b0e0f6c87e84fd2caf6bf15ac5845df7177'
tushare.set_token(token)
def stockPriceIntraday(ticker,folder):
    # 1. 获得intraday股票的数据
    pro = tushare.pro_api()
    intraday = pro.index_daily(ts_code=ticker, start_date='20000101', end_date='20200331')#fields='ts_code,trade_date,open,high,low,close,vol,amount'
    # 2. 如果股票存在，就将它添加到列表末尾就好
    filename = folder + '/' + ticker + '.csv'
    print(filename)
    #  如果存在历史数据，就将历史数据放在查找出来的intraday之后  index_col :用作行索引的列编号或者列名
    if os.path.exists(filename):
        history = pandas.read_csv(filename,index_col=2)
        intraday.append(history)

    # 3. tushare是最新的数据在最上面，现在将数据按时间倒过来  
    intraday = intraday[::-1]
    intraday.reset_index(drop=True, inplace=True)#把每行的索引改为“0、1、2……”
    #print(intraday.head())
    
    # 4. 存数据
    intraday.to_csv(filename)
    print('Intraday for [' + ticker+'] got.')

def moneyFlowIntraday(ticker,folder):
    # 1. 获得个股资金流向的数据
    pro = tushare.pro_api(token)
    print('dayin')
    moneyflow = pro.moneyflow(ts_code=ticker,fields='ts_code,trade_date,buy_lg_vol,buy_elg_vol,sell_lg_vol,sell_elg_vol,net_mf_vol')
    # 2. 如果股票存在，就将它添加到列表末尾就好
    file = folder + '/' + ticker + '.csv'
    #  如果存在历史数据，就将历史数据放在查找出来的intraday之后  index_col :用作行索引的列编号或者列名
    if os.path.exists(file):
        history = pandas.read_csv(file,index_col=2)
        moneyflow.append(history)

    # 3. tushare是最新的数据在最上面，现在将数据按时间倒过来  
    moneyflow = moneyflow[::-1]
    moneyflow.reset_index(drop=True, inplace=True)#把每行的索引改为“0、1、2……”
    print(moneyflow.head())
    
    # 4. 存数据
    moneyflow.to_csv(file)
    print('moneyflow for [' + ticker+'] got.')
# 1. 获得 股票

pro = tushare.pro_api()

tickersRawData = pro.stock_basic()
#因为code是作为索引出现的，所以要提取索引
#chuangye = tickersRawData[(tickersRawData.market=='创业板')]
tickers = tickersRawData[('ts_code')].tolist() 

# 2. 存储 股票
dateToday = datetime.datetime.today().strftime('%Y%m%d')
file = 'D:/software/python/stock/Data/bishe/BT/total/stocklist/test/'+  +'.csv'
tickersRawData.to_csv(file)
print('tickers saved.')

# 3. 获得股票价格（intraday）
for i,ticker in enumerate(tickers):
    if i>299:
        break
    try:
        print('Intraday',i,'/',ticker)
        stockPriceIntraday(ticker,folder='D:/software/python/stock/Data/bishe/BT/total/ml_test')
        #moneyFlowIntraday(ticker,folder='D:/software/python/stock/Data/IntradayCN/moneyflow')
        time.sleep(2)
    except:
        pass
print('intraday for all stocks got.')

stockPriceIntraday('000300.SH',folder='D:/software/python/project/stock/Data/bishe/BT/stocklist')