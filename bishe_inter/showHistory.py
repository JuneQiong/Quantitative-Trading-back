import cx_Oracle
import os
import json
from flask_cors import *
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
from flask import Flask
from flask_cors import CORS
from PIL import Image
from flask import request, jsonify
import base64

import pandas
import matplotlib
import numpy as np
import datetime
import time
import mpl_finance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import tushare as ts
import pandas as pd
import os
import com_LSTM
import LSTM_pre

from matplotlib.ticker import MultipleLocator,FormatStrFormatter

matplotlib.style.use('ggplot')
matplotlib.style.use('seaborn-darkgrid')

app = Flask(__name__)
CORS(app,resources=r'/*')

def getlist():
    sh_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sh-usual.csv')
    sz_cy = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-cy.csv')
    sz_zxb = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-zxb.csv')
    sz_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-usual.csv')

    data1 = sh_usual[['ts_code','name']]
    dict_1 = data1.set_index('ts_code').T.to_dict('list')
    data2 = sz_cy[['ts_code','name']]
    dict_2 = data2.set_index('ts_code').T.to_dict('list')
    data3 = sz_zxb[['ts_code','name']]
    dict_3 = data3.set_index('ts_code').T.to_dict('list')
    data4 = sz_usual[['ts_code','name']]
    dict_4 = data4.set_index('ts_code').T.to_dict('list')
    dict_list = {**dict_1,**dict_2}
    dict_list = {**dict_list,**dict_3}
    dict_list = {**dict_list,**dict_4}

    return dict_list


def stockPricePlot(code,start_date,date):
    pro = ts.pro_api('5d9ec9dc6d71031a48a24b0e0f6c87e84fd2caf6bf15ac5845df7177')               #token可以在新版tushare的网站上找到
    history = pro.query('daily', ts_code = code, start_date = start_date, end_date = date)
    history = history[::-1].reset_index()
    print(history.head())
    # 获取股票列表
    stocklist = getlist()
    name = []
    print('stocklist:', len(stocklist))
    if code in stocklist.keys():
        name = stocklist[code]
        print('name', name)
    # 2. 数据操作
    close = history[['close','open']]
    # 将索引时间变成数据的一部分

    close.dropna(inplace=True) 
    date_tickers = history['trade_date']
    date_tickers = np.array(date_tickers.values)

    # 变成时间戳

    ohlc = history[['open','high','low','close']]

    ohlc.reset_index()
    ohlc['date'] = ohlc.index
    # 将索引时间变成数据的一部分
    ohlc = ohlc.dropna()
    #print(ohlc)
    # 3. 画图 k线图candle stick plot，散点图 scatter plot，
    # 3.1 散点图  2行1列的图，散点图从0，0开始，占一行一列  x是tradedate，y是close，蓝色的

    print(close[0:5],ohlc[0:5])
    subplot1 = plt.subplot2grid((2,1),(0,0),rowspan=1,colspan=1)
    
    # x轴变成date
    #subplot1.xaxis_date()
    subplot1.plot(date_tickers,close['open'],'black')
    tick_spacing = 1
    tick_spacing = 7
    subplot1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('\"'+code+'-'+str(name[0])+u'\"的股票历史记录 ')
    # 3.2 k线图 为了在放大K线图的时候放大散点图，要共享x轴
    subplot2 = plt.subplot2grid((2,1),(1,0),rowspan=1,colspan=1,sharex=subplot1)
    a = mpf.candlestick_ohlc(ax=subplot2,quotes=ohlc[['date','open','high','low','close']].values,width=0.7,colorup='r',colordown='green', alpha=1)
    plt.savefig('D:/software/python/project/stock/learn_envs/bishe_inter/result/history.png')
    plt.show()
    
def mergeList():
    mergelist = []
    sh_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sh-usual.csv')
    sz_cy = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-cy.csv')
    sz_zxb = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-zxb.csv')
    sz_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-usual.csv')
    mergelist.append(sh_usual[('ts_code')].tolist())
    mergelist.append(sz_cy[('ts_code')].tolist())
    mergelist.append(sz_zxb[('ts_code')].tolist())
    mergelist.append(sz_usual[('ts_code')].tolist())
    mergelist = sum(mergelist,[])
    #print(mergelist)
    return mergelist    

@app.route('/showhistory', methods=["GET"])
def main():
    #code = input("请输入6位代码：") #输入股票代码
    code = request.args.get("code")
    start_date = request.args.get("startdate")
    end_date = request.args.get("enddate")
    stocklist = mergeList()
    if code in stocklist:
        print('in')
    else:
        print('not in')

    print(code,start_date)
    # code = code + '.SZ'
    
    #start_date = input("请输入展示起始日期（eg：20190101）：") #输入预测多少天后的价格
    import time
    date = time.strftime('%Y%m%d',time.localtime(time.time())) #获取当天日期
    stockPricePlot(code,start_date,end_date)
    url = 'D:/software/python/project/stock/learn_envs/bishe_inter/result/history.png'
    image = Image.open(url)
    with open(url, 'rb') as f:
            res = base64.b64encode(f.read())
            resp = jsonify(res)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Access-Control-Allow-Methods'] = 'GET'
            resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
            return resp

#
# os.system('com_LSTM.py')
# os.system('LSTM_pre.py')



if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

    # app.run()
    