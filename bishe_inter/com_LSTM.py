import pandas
import matplotlib
import numpy as np
import datetime
import time
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from keras.callbacks import LearningRateScheduler
from matplotlib.ticker import FuncFormatter
import tushare as ts
import keras
from keras.layers import Input, Dense, LSTM, Reshape
from keras.models import Model
from keras import regularizers, callbacks
from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from collections import Counter
from sklearn import svm,model_selection,neighbors,preprocessing
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import math
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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

# 一次性将数据直接喂给模型



def getData(code):
    filename_day = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_nn_test/day/'+ code +'.csv'
    filename_week = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_nn_test/week/'+ code +'.csv'

    history = pd.read_csv(filename_day)
    weekdata = pd.read_csv(filename_week)

    return weekdata,history

# 获取所有训练集的ticker 列表
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

# # 获取所有测试集的ticker 列表
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

# 更新文件
def updateFile(tickers):
    for i,ticker in enumerate(tickers):
        print(i,ticker)
        filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml/'+ ticker +'.csv'
        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0'], axis=1)
        # df = df.drop(['diff'], axis=1)
        # df = df.drop(['up'], axis=1)
        # df = df.drop(['predictForUp'], axis=1)
        #diff列表示本日和上日开盘价的差
        df['real_close'] = df['close'].shift(-1)
        
        #up列表示本日是否上涨,1表示涨，0表示跌
        df['diff'] = df['close'] - df['close'].shift(1)
        df['diff'].fillna(0, inplace = True)
        df['up'] = df['diff']
        df['up'][df['diff']>0] = 1
        df['up'][df['diff']<=0] = 0
        df['predic_close'] = 0
        df['predic_close'] = df['predic_close'].shift(1)
        df['predic_close'].fillna(0, inplace = True)
        df['predic_diff'] = df['predic_close'] - df['close'].shift(1)
        df['predictForUp'] = df['predic_diff']
        df['predictForUp'][df['predic_diff']>0] = 1
        df['predictForUp'][df['predic_diff']<=0] = 0

        df.to_csv(filename)

# 获取训练集和验证集，用来训练模型
def handleData(tickers):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i,ticker in enumerate(tickers):
        # print(i,ticker)
        filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml/'+ ticker +'.csv'
        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0'], axis=1)
        df = df.replace([np.inf,-np.inf],0)
        df.fillna(0,inplace = True)
        # 取最近10年的数据 
        df = df.loc[(df['trade_date']>20100101)&(df['trade_date']<20200101)]
        # 获取XY
        length = len(df)
        trainNum = int(length * 0.8)
        predictNum = length - trainNum
        pre_df = df.iloc[:,2:13].copy()
        pre_df = pre_df.replace([np.inf,-np.inf],0)
        pre_df.fillna(0,inplace = True)
        X_train.extend(list(pre_df[0:trainNum].values))
        Y_train.extend(list(df.up.values[0:trainNum]))

        X_test.extend(list(pre_df[trainNum:].values))
        Y_test.extend(list(df.up.values[trainNum:]))
    X_train,X_test,Y_train,Y_test = np.array(X_train),np.array(X_test),np.array(Y_train),np.array(Y_test)
    return X_train,X_test,Y_train,Y_test

# 进一步处理训练集和验证集数据，使其可以喂给模型
def handleData_new(X_train,X_test,Y_train,Y_test,window):
    dataX = []
    dataY = []

    #print(Y_train[6])
    for i in range(0, len(X_train) - window):
        _xy = X_train[i:i + window] #包括用于计算的seq_length天的数据，以及day天后的价格
        dataX.append(_xy)
        #print(Y_train[int(i + window)])
        dataY.append(Y_train[int(i + window)])
        
    
    #调整数据的shape，-1是自动计算行数   vstack将数组在垂直方向上叠加。
    dataX = np.vstack(dataX).reshape(-1, window, 11)
    dataY = np.vstack(dataY).reshape(-1, 1)
    

    dataX_test = []
    dataY_test = []

    
    for i in range(0, len(X_test) - window):
        _xy = X_test[i:i + window] #包括用于计算的seq_length天的数据，以及day天后的价格
        dataX_test.append(_xy)
        dataY_test.append(Y_test[i + window])

    
    #调整数据的shape，-1是自动计算行数   vstack将数组在垂直方向上叠加。
    dataX_test = np.vstack(dataX_test).reshape(-1, window, 11)
    dataY_test = np.vstack(dataY_test).reshape(-1,1)
    # print(dataX.shape,dataY.shape)  #(5254, 31, 10) (5254, 31)
    # print(dataX_test.shape,dataY_test.shape)  # (3495, 31, 10) (3495, 31)
    
    #先处理训练集的数据
    scaler = MinMaxScaler() # 归一化
    X_train = dataX.reshape((-1, 11)) #先变成2维，才能transform
    X_train_new = scaler.fit_transform(X_train) #预处理，按列操作，每列最小值为0，最大值为1
    X_train_new = X_train_new.reshape((-1, window, 11)) #变回3维
    Y_train_new = scaler.fit_transform(dataY) #预处理，按列操作，每列最小值为0，最大值为1
    #Y_close_train = scaler.fit_transform(closeY_train)
    
    # 再处理测试集的数据
    X_test = dataX_test.reshape((-1, 11)) #先变成2维，才能transform
    X_test_new = scaler.fit_transform(X_test) #预处理，按列操作，每列最小值为0，最大值为1
    X_test_new = X_test_new.reshape((-1, window, 11)) #变回3维
    Y_test_new = scaler.fit_transform(dataY_test)
    #Y_close = scaler.fit_transform(closeY_test)
    # print(X_test_new.shape,Y_test_new.shape)
    
    return X_train_new,Y_train_new,X_test_new,Y_test_new,scaler


# 获取测试数据,并且经过模型预测出结果，保存到本地。
def handleTestData(tickers,window):
    # tickers = ['000783.SZ', '000785.SZ'] #600006.SH', '600007.SH', '600008.SH', '600009.SH']'
    model = load_model('lstm_model.h5')
    for i,ticker in enumerate(tickers):
        print(i,ticker)
        filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_nn_test/day/'+ ticker +'.csv'
        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0'], axis=1)
        # 获取XY
        length = len(df)
        pre_df = df.iloc[:,2:13].copy()
        pre_df = pre_df.replace([np.inf,-np.inf],0)
        pre_df.fillna(0,inplace = True)
        pre_df = np.array(pre_df)
        # _y = np.array(df.up.values)
        dataX = []
        dataY = []
        for i in range(0, len(pre_df) - window):
            _xy = pre_df[i:i + window] #包括用于计算的seq_length天的数据，以及day天后的价格
            dataX.append(_xy)
            # dataY.append(_y[i:i + window])
        #调整数据的shape，-1是自动计算行数   vstack将数组在垂直方向上叠加。
        # print(len(pre_df),len(dataX))
        dataX = np.vstack(dataX).reshape(-1, window, 11)
        # print(len(pre_df),dataX[0])
        # dataY = np.vstack(dataY).reshape(-1, 1)
        scaler = MinMaxScaler() # 归一化
        X_train = dataX.reshape((-1, 11)) #先变成2维，才能transform
        X_train_new = scaler.fit_transform(X_train) #预处理，按列操作，每列最小值为0，最大值为1
        X_train_new = X_train_new.reshape((-1, window, 11))
        # Y_train_new = scaler.fit_transform(dataY)
        # Y_train_new = Y_train_new.flatten()
        # print('predict:',predict,len(predict))
        # print('Y_train_new:',Y_train_new,len(predict))

        predictions_train = []
        prediclss=[]
        # temp = 0
        for i in range(len(X_train_new)):
            data = X_train_new[i]
            predicted = model.predict(data[np.newaxis,:,:])[0,0]
            prediclss.append(predicted)

        avg_train = np.mean(prediclss)
        print(avg_train)
        for i in range(len(prediclss)):
            predicted = 0
            if (prediclss[i] < avg_train):
                predicted = 0  # 0 将lstm预测结果与svm预测结果 进行 与运算
            else:
                predicted = 1
            predictions_train.append(predicted)
        #print('predictions_train',predictions_train)
        # print(len(X_train_new),len(predictions_train),len(df))
        for i in range(len(predictions_train)):
            df.loc[i+window,'predictForUp'] = predictions_train[i]
       
        df.to_csv(filename)

# 建立模型
def ml_lstm(trainX, trainY):
    
    model = Sequential()
    # first layer   50output feed second layer  所有的股价是一个一个进入的，所以input——dim=1   output的50个中间变量都会喂给第二层
    model.add(LSTM(
        input_dim = 11,
        output_dim = 30,
        return_sequences = True
    ))

    # 避免模型过于 overfit training set  随机删除20%数据
    model.add(Dropout(0.1))
    
    # second layer 不需要喂给下一层，所以是false，并且不需要output

    
    model.add(LSTM(
        50,
        return_sequences = True
    ))
    model.add(Dropout(0.1))

    model.add(LSTM(
        50,
        return_sequences = True
    ))
    model.add(Dropout(0.1))

    model.add(LSTM(
        50,
        return_sequences = False
    ))
    model.add(Dropout(0.1))

    model.add(Dense(512,activation='relu', kernel_regularizer=regularizers.l2(0.0)))
    model.add(Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.0)))
    #model.add(Dense(256,activation='tanh', kernel_regularizer=regularizers.l2(0.0)))
    # last layer  只需要输出一个数字，用线性函数， 优化器用 rmsprop，防止lose function 掉入局部最小值
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer = 'RMSProp',metrics=['accuracy'])

    # 3. 训练模型，每一个包为512个数据，可以根据内存变。训练次数 = 5， 我们用5%的数据用来校正查验。
    
    # '''
    # 第一种
    # monitor：被监测的量
    # factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
    # patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。

    # '''
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, mode='auto')

    # 第二种lr
    
    def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 2 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            print(lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)
    
    reduce_lr = LearningRateScheduler(scheduler)

    # validation_split参数指的是从训练集中选出一部分比例的数据，来进行测试
    history = model.fit(trainX,trainY,
        batch_size=32,
        nb_epoch=4,
        validation_split=0.05,
        callbacks=[reduce_lr]
    )
    

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print('model fit loss:',history.history['loss'])
    # print('model fit acc:',history.history['acc'])
    
    print('model fit val_loss:',history.history['val_loss'])
    # print('model fit val_acc:',history.history['val_acc'])
    # 保存模型
    model.save('lstm_model.h5')   # HDF5文件，pip install h5py
    # print('model',model)
    return model,loss,val_loss

# 训练模型，检测测试集的正确率,以及保存模型
def trainModel():
    start=time.time()
    window=5
    tickers = mergeList()
    X_train,X_test,Y_train,Y_test = handleData(tickers)

    print('handle train finish')
    X_train_new,Y_train_new,X_test_new,Y_test_new,scaler = handleData_new(X_train,X_test,Y_train,Y_test,window)
    print('data finished')
    print(X_train_new[0])

    # 训练模型
    model,loss,val_loss = ml_lstm(X_train_new,Y_train_new)
    print('model finished')
    # 验证集验证  predictionsNormalized为验证结果
    predictions_train = []
    prediclss=[]
    # temp = 0
    for i in range(len(X_train_new)):
        data = X_train_new[i]
        predicted = model.predict(data[np.newaxis,:,:])[0,0]
        prediclss.append(predicted)

    avg_train = np.mean(prediclss)
    print(avg_train)
    for i in range(len(prediclss)):
        predicted = 0
        if (prediclss[i] < avg_train):
            predicted = 0  # 0 将lstm预测结果与svm预测结果 进行 与运算
        else:
            predicted = 1
        predictions_train.append(predicted)

    # --------------------------train-test
    #print('predictions_train',predictions_train)
    predictions_test = []
    prediclss0 = []
    # temp0 = []
    for i in range(len(X_test_new)):
        data = X_test_new[i]
        predicted = model.predict(data[np.newaxis,:,:])[0,0]
        prediclss0.append(predicted)

    avg_test = np.mean(prediclss0)
    print(avg_test)
    for i in range(len(prediclss0)):
        predicted = 0
        if (prediclss0[i] < avg_test):
            predicted = 0  # 0 将lstm预测结果与svm预测结果 进行 与运算
        else:
            predicted = 1
        predictions_test.append(predicted)
    
    print('predictions_test',predictions_test)
    predictions_train = np.array(predictions_train)
    # predictions_train = predictions_train / 10 * scaler.data_range_[0] + scaler.data_min_[0] #放大和scale的逆运算
    predictions_test = np.array(predictions_test)
    # predictions_test = predictions_test / 10 * scaler.data_range_[0] + scaler.data_min_[0] #放大和scale的逆运算
    predictions_train = predictions_train.flatten()
    predictions_test = predictions_test.flatten()

    correct = np.zeros(len(predictions_train))
    correct0 = np.zeros(len(predictions_test))
    print('predictions_test',predictions_test)
    for i in range(len(predictions_train)):
        if (predictions_train[i] == Y_train_new[i]):
            correct[i] = 1
    accuracy = np.sum(correct) / len(correct) * 100
    accuracy = round(accuracy, 2)

    for i in range(len(predictions_test)):
        if (predictions_test[i] == Y_test_new[i]):
            correct0[i] = 1
    accuracy0 = np.sum(correct0) / len(correct0) * 100
    accuracy0 = round(accuracy0, 2)

    print('pre-LSTM预测涨或跌的正确率--training set：', accuracy)
    print('pre-LSTM预测涨或跌的正确率--testing set：', accuracy0)
    end = time.time()
    print('Running time: %s Seconds'%(end-start))
    print('Running time: %s hour %s minutes %s Seconds'%(((end-start)//3600),((end-start)%3600)//60,(end-start)%3600%60)),

    return Y_test_new,predictions_test,accuracy,accuracy0,loss,val_loss


# 调参数  获取最优搭配
def tune_param():
    tickers = mergeList()
    X_train,X_test,Y_train,Y_test,Y_close,Y_close_train = handleData(tickers)
    X_train_new,Y_train_new,X_test_new,Y_test_new,scaler,Y_close_new,Y_close_train_new = handleData_new(X_train,X_test,Y_train,Y_test,Y_close,Y_close_train)
    # r = 10*np.random.rand()
    # hidden = [50,200,10]
    epoch = [10,30,50]
    batch_size = [32,64,128,256]
    activate = ['relu','tanh','softsign']

    transactions = pd.DataFrame(columns = ['hidden','epoch','batch','acti','mse','train_accu','test_accu','loss']) 

    for epoch in epoch:
        for batch in batch_size:
            for acti in range(len(activate)):
                print('hidden:'+str(50)+',--epoch:'+str(epoch)+',--batch:'+str(batch)+',--acti:'+activate[acti])
                Y_test_new,predictions_test,accuracy,accuracy0,loss = trainModel(epoch,batch,activate[acti])
                mse_predict = mean_squared_error(Y_test_new,predictions_test)
                transactions = transactions.append(pd.DataFrame({'hidden':['50'],'epoch':[epoch],'batch':[batch],'acti':[activate[acti]],'mse':[mse_predict],'train_accu':[accuracy],'test_accu':[accuracy0],'loss':[loss[-1]]}),ignore_index=True)
                print('transactions:')
                print(transactions)
                print('-----------------------------------------------')
    transactions.to_csv('tuneparam.csv')
    # Y_test_new,predictions_test,accuracy0 = trainModel()
    # mse_predict = mean_squared_error(Y_test_new,predictions_test)

# 画图  和沪深300的收益对比图
def plot(start,end):
    # 画图
    transactions = pd.read_csv('D:/software/python/project/stock/learn_envs/bishe_inter/transactions_LSTM.csv')
    pro = ts.pro_api('5d9ec9dc6d71031a48a24b0e0f6c87e84fd2caf6bf15ac5845df7177')
    print(start,end)
    hs300 = pro.index_daily(ts_code='000300.SH', start_date=start, end_date=end)
    hs300 = hs300[::-1].reset_index()
    # hs300 = hs300.drop(['Unnamed: 0'], axis=1)
    hs300_close = hs300['close'].values
    # print(hs300_close[0])
    hs300_date = hs300['trade_date']   
    hs300_date = np.array(hs300_date.values)
    # print('hs_300',hs300.head())
    profit = transactions.iloc[-1].profit
    profit1=transactions['profit'].values
    print('492',profit,profit1)
    if (transactions.empty):
        
        plt.plot(hs300_date,hs300['close'],color='blue')
        plt.legend(['hs_300'], loc='upper left')
    else:

        tran_date = transactions['date'].values
        tran_date = list(map(str,tran_date))
        profit_new = transactions['profit_new'].values
        profit = transactions.iloc[-1].profit
        print('492',profit)
        prorate = profit/10000*100
        prorate = round(prorate,2)
        buy_date = transactions[transactions['behavior'] == 'buy']['date'].values
        buy_profit = transactions[transactions['behavior'] == 'buy']['profit_new'].values
        buy_date = list(map(str,buy_date))
        sell_date = transactions[transactions['behavior'] == 'sell']['date'].values
        sell_profit = transactions[transactions['behavior'] == 'sell']['profit_new'].values
        sell_date = list(map(str,sell_date))
        # print('else hs300',hs300_date,hs300['close'])
        plt.plot(hs300_date,hs300['close'],color='blue')
        plt.plot(tran_date,profit_new,color='orange')
        
        # 绘制 ^ 买入  v 卖出
        plt.plot(buy_date,buy_profit,'^',color='red',markersize = 7)
        plt.plot(sell_date,sell_profit,'v',color='green',markersize = 7)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.legend([u'沪深300指数', u'模型收益',u'买点',u'卖点'], loc='upper left')
        plt.suptitle(u'LSTM 买卖点回测结果 --> 收益率：' + str(prorate)+'%',fontsize = 14)
        # plt.title(u'LSTM-buy-sell show  -> init:10000 -> profit:'+str(prorate)+'%')

    plt.savefig('D:/software/python/project/stock/learn_envs/bishe_inter/result/LSTM_com.png')
    plt.show()
    plt.close()



# 计算买卖点，收益
@app.route('/lstmcom', methods=["GET"])  
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
    end = end-delta-delta

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

        #20000830 2000-08-31
        price_buy = 0
        price_sell = 0
        ticker = ''
        #如果没有持仓，判断是否买入
        if num==0:
            #print('come in ')
            day_kdj = dict()
            #print(type(day_kdj),'311')
            for k,ticker in enumerate(tickers):
                # print(k,ticker)
                weekdata,pdatas = getData(ticker)
                #print(f,k,ticker)
                #print(f,f in pdatas.trade_date.values)
                #如果股票中有当前时间，就进行kdj判断，若满足买入条件，就 记录到trasaction 中
                if (int(f) in pdatas.trade_date.values):
                    #print(ticker,'time is exist')
                    #日线的索引 i
                    i = pdatas[pdatas.trade_date.values == int(f)].index.values
                    i = int(str(i).strip('[]'))
                    # pdatas.predictForUp[i]==1 机器学习模型预测是第二天会涨
                    if pdatas.predictForUp[i]==1:
                        #print('i',i,ticker,f)
                        # 周线的索引 j 
                        # if (int(f) in weekdata.trade_date.values):
                            # j = weekdata[weekdata.trade_date.values == int(f)].index.values
                            # j = int(str(j).strip('[]'))
                            # #print(ticker,'week is exist')
                            # #日线的索引 i
                            # i = pdatas[pdatas.trade_date.values == int(f)].index.values
                            # i = int(str(i).strip('[]'))
                            # #周线的索引 j 
                            # j = weekdata[weekdata.trade_date.values == int(f)].index.values
                            # j = int(str(j).strip('[]'))
                            # # print(ticker,j,f)
                            # # 判断买卖点   weekdata.K[j-2] < weekdata.D[j-2]) & (weekdata.K[j] > weekdata.D[j]
                            # if j>2:
                                if ((pdatas.K[i-2] < pdatas.D[i-2]) & (pdatas.K[i] > pdatas.D[i])):
                                    if  (pdatas.position[i-1]==0):
                                        #在第i天以开盘价买入
                                        price_in = pdatas.loc[i,'open']
                                        #print('Zai:',date_in,'buy:',price_in)
                                        day_kdj[ticker] = price_in
                else:
                    continue
            if day_kdj:
                sort_day = sorted(day_kdj.values())
                print(day_kdj,'342')
                min_kdj = min(zip(day_kdj.values(), day_kdj.keys()))
                print(min_kdj)
                ticker = min_kdj[1]
                filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_nn_test/day/'+ ticker +'.csv'
                data = pd.read_csv(filename)
                i = data[data.trade_date.values == int(f)].index.values
                i = int(str(i).strip('[]'))
                data.loc[i,'position'] = 1
                data.loc[i,'flag'] = 1
                price_close = data.loc[i,'close']
                data.to_csv(filename)
                #print(f,data.loc[i,'position'])
                price_buy = day_kdj[ticker]
                num = cash // price_buy//100*100
                # 手里的现金
                cash = cash - num * price_buy
                # 手里所有的钱
                cash_current = cash + num * price_close
                transactions = transactions.append(pd.DataFrame({'date':[f],'behavior':['buy'],'price':[price_buy],'num':[num],'code':[ticker],'cash':[cash_current],'close':[price_close],'profit':[cash_current - init_money],'profit_new':[0]}),ignore_index=True)
                #print('transactions-buy:',transactions)
        else:
            #num ！= 0 ， 就可以卖出了
            code = transactions.iloc[-1].code
            weekdata,pdatas = getData(code)
            filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_nn_test/day/'+ code +'.csv'
            if (int(f) in pdatas.trade_date.values):
                #print('time is exits')
                i = pdatas[pdatas.trade_date.values == int(f)].index.values
                i = int(str(i).strip('[]'))
                pdatas.loc[i,'position'] = 1
                pdatas.loc[i,'flag'] = 1
                pdatas.to_csv(filename)
                #print(f,pdatas.loc[i,'position'])
                #print('check0:',pdatas.position[i-1])
                #日线的索引 i
                
                    #print('sell')
                    #print('i',i)
                    #周线的索引 j 
                if ((pdatas.K[i-2] > pdatas.D[i-2]) & (pdatas.K[i] < pdatas.D[i])):    
                    if(pdatas.position[i]==1):
                        pdatas.loc[i,'flag'] = -1  # 卖出
                        pdatas.loc[i,'position'] = 0  # 不持有
                        pdatas.to_csv(filename)
                        #print('Zai:',pdatas.trade_date[i],'sell:',pdatas.loc[i,'open'])
                        price_sell = pdatas.loc[i,'open']
                if price_sell:
                    cash = cash + num * price_sell
                    num = 0
                    price_close = data.loc[i,'close']
                    transactions = transactions.append(pd.DataFrame({'date':[f],'behavior':['sell'],'price':[price_sell],'num':[num],'code':[code],'cash':[cash],'close':['--'],'profit':[cash - init_money],'profit_new':[0]}),ignore_index=True)

                    #print('transactions-sell:',transactions)
                else:
                    open_price = pdatas.loc[i,'open']
                    price_close = data.loc[i,'close']
                    cash_current = cash + num * price_close
                    transactions = transactions.append(pd.DataFrame({'date':[f],'behavior':['--'],'price':[open_price],'num':[num],'code':[code],'cash':[cash_current],'close':[price_close],'profit':[cash_current - init_money],'profit_new':[0]}),ignore_index=True)
        d += delta
    endtime=time.time()
    if transactions.empty:
        transactions.to_csv('transactions_LSTM.csv')
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
        transactions.to_csv('transactions_LSTM.csv')
        
        profit = transactions.iloc[-1].profit
        if(profit > 0):
            print('赚了：',profit)
        else:
            print('亏了：',profit)
    print('Running time: %s Seconds'%(endtime-start))
    print('Running time: %s hour %s minutes %s Seconds'%(((endtime-start)//3600),((endtime-start)%3600)//60,(endtime-start)%3600%60)),

    plot(begin_input,end_input)

    # ----接口
    url = 'D:/software/python/project/stock/learn_envs/bishe_inter/result/LSTM_com.png'
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
        port=5020,
        debug=True
    )



