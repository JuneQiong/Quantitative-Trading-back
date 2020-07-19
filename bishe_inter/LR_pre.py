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
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
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

def format_date(x,pos=None):
	if x<0 or x>len(y_visual_date_new)-1:
		return ''
	return y_visual_date_new[int(x)]

def time2stamp(cmnttime):   #转时间戳函数
    cmnttime=datetime.datetime.strptime(cmnttime,'%Y%m%d')
    stamp=int(datetime.datetime.timestamp(cmnttime))
    return stamp

def com_predict_time(day):
    date = datetime.datetime.now().date() #获取当天日期
    print(date)
    date_inter =date + datetime.timedelta(days=0) 
    predict_time=[]
    for i in range(day):
        date_inter =date_inter + datetime.timedelta(days=1) 
        #print(date_inter)
        predict_time.append([date_inter.strftime('%Y-%m-%d')])
    #predict_time.append([date + datetime.timedelta(days=7)])
    #predict_time.columns('time')
    predict_time = pd.DataFrame(predict_time, columns=['time']) 
    # 生成横轴的刻度名字
    predic_time = predict_time.time.values
    return predic_time


def updateFile(code,date):
        pro = ts.pro_api('5d9ec9dc6d71031a48a24b0e0f6c87e84fd2caf6bf15ac5845df7177')               #token可以在新版tushare的网站上找到
        df = pro.query('daily', ts_code = code, start_date = '20150101', end_date = date)
        df = df[::-1] #倒序，使日期靠前的排在前面
        df.reset_index(drop=True, inplace=True) #把每行的索引改为“0、1、2……”
        print(df.head())
        # df = df.drop(['Unnamed: 0'], axis=1)
        #diff列表示本日和上日开盘价的差
        
        df['diff'] = df["open"]-df["open"].shift(1)
        df['diff'].fillna(0, inplace = True)
        #up列表示本日是否上涨,1表示涨，0表示跌
        df['up'] = df['diff']
        df['up'][df['diff']>0] = 1
        df['up'][df['diff']<=0] = 0
        df['predictForUp'] = 0
        return df
# 提取训练集测试集
def handleData(df,day):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_apply = []
    Y_date = []
    # df = df.drop(['Unnamed: 0'], axis=1)
    Y_date=df['trade_date']
    Y_date = list(Y_date)
    Y_date = list(Y_date)
    Y_date = list(map(int,Y_date))
    
    # 获取XY
    length = len(df)
    trainNum = int(length * 0.9)
    predictNum = length - trainNum
    pre_df = df.iloc[:,3:13].pct_change()
    pre_df = pre_df.replace([np.inf,-np.inf],0)
    pre_df.fillna(0,inplace = True)
    X_train.extend(list(pre_df[0:trainNum].values))
    Y_train.extend(list(df.up.values[0:trainNum]))
    X_test.extend(list(pre_df[trainNum:-day].values))
    Y_test.extend(list(df.up.values[trainNum:-day]))
    X_apply.extend(list(pre_df[-day:].values))
    Y_date = Y_date[trainNum:-day]
    Y_date = np.array(Y_date)
    print(len(X_train),len(Y_train),len(X_test),len(Y_test))
    return X_train,X_test,Y_train,Y_test,X_apply,Y_date
# 训练
def model_ml(code,day,X_train,Y_train):
    clf = LogisticRegression()
    clf.fit(X_train,Y_train)
    # prediction = clf.predict(X_test)
    # 保存模型
    joblib.dump(clf, code+"lr_model-"+ day+".m")
    return clf

# 测试
def handleTestData(model,X_test,Y_test):
    Y_pre = []
    for i in range(len(X_test)):
        X_test[i] = X_test[i].reshape((1,X_test[i].shape[0]))

        #print(i)
        predictForUp = model.predict( X_test[i])
        Y_pre.append(int(predictForUp))
    correct = np.zeros(len(Y_pre))
    
    Y_test = list(map(int,Y_test))
    print(len(Y_pre),Y_test)
    for i in range(len(Y_pre)):
        if int(Y_pre[i]) == int(Y_test[i]):
            correct[i]=1
    accu = np.sum(correct) / len(correct) * 100
    print('预测涨或跌的正确率：', accu)
    return Y_pre

def apply(model,X_apply):
    Y_apply = []
    for i in X_apply:
        i = np.array(i)
        i = i.reshape((1,i.shape[0]))
        predictForUp = model.predict(i)
        Y_apply.append(int(predictForUp))
    return Y_apply

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

def visualize(code,predict_time,Y_date,Y_test, Y_pre, Y_apply):
    #print(visualY2,visualPredict2)
    Y_date = [str(x) for x in Y_date]
    # 获取股票列表
    stocklist = getlist()
    name = []
    print('stocklist:', len(stocklist))
    if code in stocklist.keys():
        name = stocklist[code]
        print('name', name)
    subplot1 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
    #subplot1.xaxis_date()
    line1=subplot1.plot(Y_date,Y_test,color='blue')
    line2=subplot1.plot(Y_date,Y_pre,color='orange')
    #subplot1.set_xticklabels(y_visual_date_new)
    subplot1.legend([u'真实值', u'模型预测值'], loc='upper left')
    tick_spacing = 1
    tick_spacing = 7
    subplot1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.suptitle('\"' + code + '-' + str(name[0]) + u'\"的 LR 算法预测结果',fontsize = 14)
    # ----接口
    plt.savefig('D:/software/python/project/stock/learn_envs/bishe_inter/result/LR_pre.png')
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

@app.route('/lrpre', methods=["GET"])  
def main():
    # code = input("请输入6位代码：") #输入股票代码
    code = request.args.get("code")
    # code = code + '.SZ'
    
    # day = input("请输入预测天数：") #输入预测多少天后的价格
    day = request.args.get("day")
    try:
        day = int(day)
    except:
        day = 5
        day = int(day)
    
    import time
    date = time.strftime('%Y%m%d',time.localtime(time.time())) #获取当天日期
    predict_time = com_predict_time(day)
    # print('predict_time',predict_time)
    
    df = updateFile(code,'20200301')
    X_train,X_test,Y_train,Y_test,X_apply,Y_date = handleData(df,day)
    print(type(X_test))
    
    day = str(day)

    # 为了使用接口的时候第二次可以成功
    keras.backend.clear_session()
    try:
        # 载入模型
        model = joblib.load(code+"lr_model-"+ day+".m")
        print('model1',model)
    except:
        print('第一次预测%d天内'%(int(day)) + code + '的估价，需要一点时间建模')
        model = model_ml(code,day,X_train,Y_train)

    Y_pre =handleTestData(model,X_test,Y_test)
    Y_apply = apply(model,X_apply)

    visualize(code,predict_time,Y_date,Y_test, Y_pre, Y_apply)

    # ----接口
    url = 'D:/software/python/project/stock/learn_envs/bishe_inter/result/LR_pre.png'
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
        port=5011,
        debug=True
    )

