import pandas
import matplotlib
import numpy as np
import datetime
import time
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import tushare as ts
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from collections import Counter
from sklearn import svm,model_selection,neighbors,preprocessing
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import tushare as ts
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
import keras

# ----接口
app = Flask(__name__)
CORS(app,resources=r'/*')


def getData(code):
    filename_day = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_svm_test/day/'+ code +'.csv'
    filename_week = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_svm_test/week/'+ code +'.csv'

    history = pd.read_csv(filename_day)
    weekdata = pd.read_csv(filename_week)

    return weekdata,history

# 获取所有ticker 列表
def mergeList():
    mergelist = []
    sh_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sh-usual.csv')
    sz_cy = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-cy.csv')
    sz_zxb = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-zxb.csv')
    sz_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-usual.csv')
    mergelist.append(sh_usual[('ts_code')][0:10].tolist())
    mergelist.append(sz_cy[('ts_code')][0:10].tolist())
    mergelist.append(sz_zxb[('ts_code')][0:10].tolist())
    mergelist.append(sz_usual[('ts_code')][0:10].tolist())
    mergelist = sum(mergelist,[])
    #print(mergelist)
    return mergelist

def mergeListTest():
    mergelist = []
    sh_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sh-usual.csv')
    sz_cy = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-cy.csv')
    sz_zxb = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-zxb.csv')
    sz_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-usual.csv')
    mergelist.append(sh_usual[('ts_code')][300:600].tolist())
    mergelist.append(sz_cy[('ts_code')][150:300].tolist())
    mergelist.append(sz_zxb[('ts_code')][250:500].tolist())
    mergelist.append(sz_usual[('ts_code')][300:600].tolist())
    mergelist = sum(mergelist,[])
    #print(mergelist)
    return mergelist

def updateFile(tickers):
    for i,ticker in enumerate(tickers):
        print(i,ticker)
        filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml/'+ ticker +'.csv'
        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0'], axis=1)
        #diff列表示本日和上日开盘价的差
        
        df['diff'] = df["open"]-df["open"].shift(1)
        df['diff'].fillna(0, inplace = True)
        #up列表示本日是否上涨,1表示涨，0表示跌
        df['up'] = df['diff']
        df['up'][df['diff']>0] = 1
        df['up'][df['diff']<=0] = 0
        df['predictForUp'] = 0
        df.to_csv(filename)

# 获取训练集和验证集，用来训练模型
def handleData(tickers):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i,ticker in enumerate(tickers):
        print(i,ticker)
        filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml/'+ ticker +'.csv'
        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0'], axis=1)
        
        # 获取XY
        length = len(df)
        trainNum = int(length * 0.8)
        predictNum = length - trainNum
        pre_df = df.iloc[:,2:13].pct_change()
        pre_df = pre_df.replace([np.inf,-np.inf],0)
        pre_df.fillna(0,inplace = True)
        
        X_train.extend(list(pre_df[0:trainNum].values))
        Y_train.extend(list(df.up.values[0:trainNum]))
        X_test.extend(list(pre_df[trainNum:].values))
        Y_test.extend(list(df.up.values[trainNum:]))
    print(len(X_train),len(Y_train),len(X_test),len(Y_test))
    return X_train,X_test,Y_train,Y_test

# 获取测试数据,并且经过模型预测出结果，保存到本地。
def handleTestData(tickers):
    clf = joblib.load("svm_model.m")
    for i,ticker in enumerate(tickers):
        print(i,ticker)
        filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_svm_test/day/'+ ticker +'.csv'
        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0'], axis=1)
        # 获取XY
        length = len(df)
        pre_df = df.iloc[:,2:13].pct_change()
        pre_df = pre_df.replace([np.inf,-np.inf],0)
        pre_df.fillna(0,inplace = True)
        # 逐条预测

        for j in range(length):
            predictForUp = clf.predict(pre_df[j:j+1])
            df.loc[j,'predictForUp']=predictForUp 
        df.to_csv(filename)

# 
def model_ml(X_train,Y_train):
    clf=svm.LinearSVC(C=10)
    clf.fit(X_train,Y_train)
    # prediction = clf.predict(X_test)
    # 保存模型
    joblib.dump(clf, "svm_model.m")
    return clf

def plot(start,end):
    # 画图
    
    transactions = pd.read_csv('D:/software/python/project/stock/learn_envs/bishe_inter/transactions_SVM.csv')
    pro = ts.pro_api('5d9ec9dc6d71031a48a24b0e0f6c87e84fd2caf6bf15ac5845df7177')
    print(start,end)
    hs300 = pro.index_daily(ts_code='000300.SH', start_date=start, end_date=end)
    hs300 = hs300[::-1].reset_index()
    # hs300 = hs300.drop(['Unnamed: 0'], axis=1)
    hs300_close = hs300['close'].values
    # print(hs300_close[0])
    hs300_date = hs300['trade_date']   
    hs300_date = np.array(hs300_date.values)
    print('hs_300',hs300.head())

    if (transactions.empty):
        
        plt.plot(hs300_date,hs300['close'],color='blue')
        plt.legend(['hs_300'], loc='upper left')
    else:
        tran_date = transactions['date'].values
        tran_date = list(map(str, tran_date))
        profit_new = transactions['profit_new'].values
        profit = transactions.iloc[-1].profit
        print('492', profit)
        prorate = profit / 10000 * 100
        prorate = round(prorate, 2)
        buy_date = transactions[transactions['behavior'] == 'buy']['date'].values
        buy_profit = transactions[transactions['behavior'] == 'buy']['profit_new'].values
        buy_date = list(map(str, buy_date))
        sell_date = transactions[transactions['behavior'] == 'sell']['date'].values
        sell_profit = transactions[transactions['behavior'] == 'sell']['profit_new'].values
        sell_date = list(map(str, sell_date))
        # print('else hs300',hs300_date,hs300['close'])
        plt.plot(hs300_date, hs300['close'], color='blue')
        plt.plot(tran_date, profit_new, color='orange')

        # 绘制 ^ 买入  v 卖出
        plt.plot(buy_date, buy_profit, '^', color='red', markersize=7)
        plt.plot(sell_date, sell_profit, 'v', color='green', markersize=7)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.legend([u'沪深300指数', u'模型收益', u'买点', u'卖点'], loc='upper left')
        plt.suptitle(u'SVM 买卖点回测结果 --> 收益率：' + str(prorate) + '%',fontsize = 14)

    plt.savefig('D:/software/python/project/stock/learn_envs/bishe_inter/result/SVM_com.png')
    plt.show()
    plt.close()

# 测试
def test():
    tickers = mergeList()
    X_train,X_test,Y_train,Y_test = handleData(tickers)
    model = model_ml(X_train,Y_train)
    Y_pre = []
    Y_pre_train = []
    for i in range(len(X_test)):
        X_test[i] = X_test[i].reshape((1,X_test[i].shape[0]))
        #print(i)
        predictForUp = model.predict( X_test[i])
        Y_pre.append(int(predictForUp))
    correct = np.zeros(len(Y_pre))
    
    Y_test = list(map(int,Y_test))
    # print(len(Y_pre),Y_test)
    for i in range(len(Y_pre)):
        if int(Y_pre[i]) == int(Y_test[i]):
            correct[i]=1
    accu = np.sum(correct) / len(correct) * 100


    for i in range(len(X_train)):
        X_train[i] = X_train[i].reshape((1,X_train[i].shape[0]))
        #print(i)
        predictForUp = model.predict( X_train[i])
        Y_pre_train.append(int(predictForUp))
    correct_train = np.zeros(len(Y_pre_train))

    Y_train = list(map(int,Y_train))
    # print(len(Y_pre),Y_test)
    for i in range(len(Y_pre_train)):
        if int(Y_pre_train[i]) == int(Y_train[i]):
            correct_train[i]=1
    accu_train = np.sum(correct_train) / len(correct_train) * 100
    print('train 预测涨或跌的正确率：', accu_train)
    print('test 预测涨或跌的正确率：', accu)

# 计算买卖点，收益
@app.route('/svmcom', methods=["GET"])  
def main():
    begin_input = request.args.get("begin")
    begin_input=begin_input.split('-')
    end_input = request.args.get('end')
    end_input=end_input.split('-')
    print('----:',begin_input,end_input)
    
    tickers = mergeListTest()
    # handleTestData(tickers)
    begin = datetime.date(int(begin_input[0]),int(begin_input[1]),int(begin_input[2]))
    end = datetime.date(int(end_input[0]),int(end_input[1]),int(end_input[1]))

    d = begin
    delta = datetime.timedelta(days=1)
    end = end - delta - delta
    #交易记录
    #交易记录
    transactions = pd.DataFrame(columns = ['date','behavior','code','price','num','cash','close','profit','profit_new']) 
    init_money = 10000
    num = 0
    cash = 10000
    
    start=time.time()

    begin_input = begin_input[0]+begin_input[1]+begin_input[2]
    end_input = end_input[0]+end_input[1]+end_input[2]

    pro = ts.pro_api('5d9ec9dc6d71031a48a24b0e0f6c87e84fd2caf6bf15ac5845df7177')
    hs300 = pro.index_daily(ts_code='000300.SH', start_date=begin_input, end_date=end_input)
    hs300 = hs300[::-1].reset_index()
    # hs300 = hs300.drop(['Unnamed: 0'], axis=1)
    hs300_close = hs300['close'].values
    print(hs300)

    while d <= end:
        # 存放每天满足条件的股票 ｛‘code’，open｝
        print(d)
        f = d.strftime("%Y%m%d")
        d += delta
        # 20000830 2000-08-31
        price_buy = 0
        price_sell = 0
        ticker = ''
        # 如果没有持仓，判断是否买入
        if num == 0:
            # print('come in ')
            day_kdj = dict()
            # print(type(day_kdj),'311')
            for k, ticker in enumerate(tickers):
                # print(k,ticker)
                weekdata, pdatas = getData(ticker)
                # print(f,k,ticker)
                # print(f,f in pdatas.trade_date.values)
                # 如果股票中有当前时间，就进行kdj判断，若满足买入条件，就 记录到trasaction 中
                if (int(f) in pdatas.trade_date.values):
                    # print(ticker,'time is exist')
                    # 日线的索引 i
                    i = pdatas[pdatas.trade_date.values == int(f)].index.values
                    i = int(str(i).strip('[]'))
                    # pdatas.predictForUp[i]==1 机器学习模型预测是第二天会涨
                    if pdatas.predictForUp[i] == 1:
                        # print('i',i,ticker,f)
                        # 周线的索引 j
                        # if (int(f) in weekdata.trade_date.values):
                        #     j = weekdata[weekdata.trade_date.values == int(f)].index.values
                        #     j = int(str(j).strip('[]'))
                        #     #print(ticker,'week is exist')
                        #     #日线的索引 i
                        #     i = pdatas[pdatas.trade_date.values == int(f)].index.values
                        #     i = int(str(i).strip('[]'))
                        #     #周线的索引 j
                        #     j = weekdata[weekdata.trade_date.values == int(f)].index.values
                        #     j = int(str(j).strip('[]'))
                        #     # print(ticker,j,f)
                        #     # 判断买卖点   weekdata.K[j-2] < weekdata.D[j-2]) & (weekdata.K[j] > weekdata.D[j]
                        #     if j>2:
                        if ((pdatas.K[i - 2] < pdatas.D[i - 2]) & (pdatas.K[i] > pdatas.D[i])):
                            if (pdatas.position[i - 1] == 0):
                                # 在第i天以开盘价买入
                                price_in = pdatas.loc[i, 'open']
                                # print('Zai:',date_in,'buy:',price_in)
                                day_kdj[ticker] = price_in
                else:
                    continue
            if day_kdj:
                sort_day = sorted(day_kdj.values())
                print(day_kdj, '342')
                min_kdj = min(zip(day_kdj.values(), day_kdj.keys()))
                print(min_kdj)
                ticker = min_kdj[1]
                filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_svm_test/day/' + ticker + '.csv'
                data = pd.read_csv(filename)
                i = data[data.trade_date.values == int(f)].index.values
                i = int(str(i).strip('[]'))
                data.loc[i, 'position'] = 1
                data.loc[i, 'flag'] = 1
                price_close = data.loc[i, 'close']
                data.to_csv(filename)
                # print(f,data.loc[i,'position'])
                price_buy = day_kdj[ticker]
                num = cash // price_buy // 100 * 100
                # 手里的现金
                cash = cash - num * price_buy
                # 手里所有的钱
                cash_current = cash + num * price_close
                transactions = transactions.append(pd.DataFrame(
                    {'date': [f], 'behavior': ['buy'], 'price': [price_buy], 'num': [num], 'code': [ticker],
                     'cash': [cash_current], 'close': [price_close], 'profit': [cash_current - init_money],
                     'profit_new': [0]}), ignore_index=True)
                # print('transactions-buy:',transactions)
        else:
            # num ！= 0 ， 就可以卖出了
            code = transactions.iloc[-1].code
            weekdata, pdatas = getData(code)
            filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_svm_test/day/' + code + '.csv'
            if (int(f) in pdatas.trade_date.values):
                # print('time is exits')
                i = pdatas[pdatas.trade_date.values == int(f)].index.values
                i = int(str(i).strip('[]'))
                pdatas.loc[i, 'position'] = 1
                pdatas.loc[i, 'flag'] = 1
                pdatas.to_csv(filename)
                # print(f,pdatas.loc[i,'position'])
                # print('check0:',pdatas.position[i-1])
                # 日线的索引 i

                if ((pdatas.K[i - 2] > pdatas.D[i - 2]) & (pdatas.K[i] < pdatas.D[i])):
                    if (pdatas.position[i - 1] == 1):
                        pdatas.loc[i, 'flag'] = -1  # 卖出
                        pdatas.loc[i, 'position'] = 0  # 不持有
                        pdatas.to_csv(filename)
                        # print('Zai:',pdatas.trade_date[i],'sell:',pdatas.loc[i,'open'])
                        price_sell = pdatas.loc[i, 'open']
                if price_sell:
                    cash = cash + num * price_sell
                    num = 0
                    price_close = data.loc[i, 'close']
                    transactions = transactions.append(pd.DataFrame(
                        {'date': [f], 'behavior': ['sell'], 'price': [price_sell], 'num': [num], 'code': [code],
                         'cash': [cash], 'close': ['--'], 'profit': [cash - init_money], 'profit_new': [0]}),
                                                       ignore_index=True)

                    # print('transactions-sell:',transactions)
                else:
                    open_price = pdatas.loc[i, 'open']
                    price_close = data.loc[i, 'close']
                    cash_current = cash + num * price_close
                    transactions = transactions.append(pd.DataFrame(
                        {'date': [f], 'behavior': ['--'], 'price': [open_price], 'num': [num], 'code': [code],
                         'cash': [cash_current], 'close': [price_close], 'profit': [cash_current - init_money],
                         'profit_new': [0]}), ignore_index=True)

    endtime=time.time()
    if transactions.empty:
        transactions.to_csv('transactions_SVM.csv')
        print('没有买卖记录，盈利为0')
        
    else:
        cash = transactions['cash'].tolist()
        cash = list(map(int,cash))
        print(cash,type(cash))
        profit_new = {}
        print(hs300_close[0])
        profit_new[0] = hs300_close[0]
        print(profit_new[0])
        # df['diff'] = df['close'] - df['close'].shift(1)
        for i in range(len(cash)):
            if i > 0:
                temp = (cash[i] - cash[i-1])/(cash[i-1])
                # print(temp)
                profit_new[i] = profit_new[i-1] * (1+temp)
                # print(profit_new)
        transactions['profit_new'] = list(profit_new.values())
        print('transactions',transactions)
        transactions.to_csv('transactions_SVM.csv')
        
        profit = transactions.iloc[-1].profit
        if(profit > 0):
            print('赚了：',profit)
        else:
            print('亏了：',profit)
    print('Running time: %s Seconds'%(endtime-start))
    print('Running time: %s hour %s minutes %s Seconds'%(((endtime-start)//3600),((endtime-start)%3600)//60,(endtime-start)%3600%60)),

    plot(begin_input,end_input)

    # ----接口
    url = 'D:/software/python/project/stock/learn_envs/bishe_inter/result/SVM_com.png'
    image = Image.open(url)

    with open(url, 'rb') as f:
            res = base64.b64encode(f.read())
            resp = jsonify(res)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Access-Control-Allow-Methods'] = 'GET'
            resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
            return resp

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5022,
        debug=True
    )



