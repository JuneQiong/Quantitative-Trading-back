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
from keras.layers import Input, Dense, LSTM, Reshape, K
from keras.models import Model
from keras import regularizers, callbacks
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from collections import Counter
from sklearn import svm, model_selection, neighbors, preprocessing
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
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
CORS(app, resources=r'/*')


# 一次性将数据直接喂给模型


def getData(code):
    filename_day = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_nn_test/day/' + code + '.csv'
    filename_week = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_nn_test/week/' + code + '.csv'

    history = pd.read_csv(filename_day)
    weekdata = pd.read_csv(filename_week)

    return weekdata, history


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
    mergelist = sum(mergelist, [])
    # print(mergelist)
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
    mergelist = sum(mergelist, [])
    # print(mergelist)
    return mergelist


# 更新文件
def updateFile(tickers):
    for i, ticker in enumerate(tickers):
        print(i, ticker)
        filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml/' + ticker + '.csv'
        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0'], axis=1)
        # df = df.drop(['diff'], axis=1)
        # df = df.drop(['up'], axis=1)
        # df = df.drop(['predictForUp'], axis=1)
        # diff列表示本日和上日开盘价的差
        df['real_close'] = df['close'].shift(-1)

        # up列表示本日是否上涨,1表示涨，0表示跌
        df['diff'] = df['close'] - df['close'].shift(1)
        df['diff'].fillna(0, inplace=True)
        df['up'] = df['diff']
        df['up'][df['diff'] > 0] = 1
        df['up'][df['diff'] <= 0] = 0
        df['predic_close'] = 0
        df['predic_close'] = df['predic_close'].shift(1)
        df['predic_close'].fillna(0, inplace=True)
        df['predic_diff'] = df['predic_close'] - df['close'].shift(1)
        df['predictForUp'] = df['predic_diff']
        df['predictForUp'][df['predic_diff'] > 0] = 1
        df['predictForUp'][df['predic_diff'] <= 0] = 0

        df.to_csv(filename)


# 获取训练集和验证集，用来训练模型
def handleData(ticker):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    Y_close = []
    Y_test_data = []

    filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml/' + ticker + '.csv'
    df = pd.read_csv(filename)
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.replace([np.inf, -np.inf], 0)
    df.fillna(0, inplace=True)
    # 取最近10年的数据
    df = df.loc[(df['trade_date'] > 20150101) & (df['trade_date'] < 20200301)]
    # 获取XY
    length = len(df)
    trainNum = int(length * 0.9)
    pre_df = df.iloc[:, 2:13].copy()
    pre_df = pre_df.replace([np.inf, -np.inf], 0)
    pre_df.fillna(0, inplace=True)
    X_train.extend(list(pre_df[0:trainNum].values))
    Y_train.extend(list(df.close.values[0:trainNum]))

    X_test.extend(list(pre_df[trainNum:].values))
    Y_test.extend(list(df.close.values[trainNum:]))
    Y_test_data.extend(list(df.trade_date.values[trainNum:]))
    X_train, X_test, Y_train, Y_test,Y_test_data = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test),np.array(Y_test_data)
    return X_train, X_test, Y_train, Y_test,Y_test_data


# 进一步处理训练集和验证集数据，使其可以喂给模型
def handleData_new(X_train, X_test, Y_train, Y_test,Y_test_data, window):
    dataX = []
    dataY = []
    dataClose = []
    # print(Y_train[6])
    for i in range(0, len(X_train) - window):
        _xy = X_train[i:i + window]  # 包括用于计算的seq_length天的数据，以及day天后的价格
        dataX.append(_xy)
        # print(Y_train[int(i + window)])
        dataY.append(Y_train[int(i + window)])


    # 调整数据的shape，-1是自动计算行数   vstack将数组在垂直方向上叠加。
    dataX = np.vstack(dataX).reshape(-1, window, 11)
    dataY = np.vstack(dataY).reshape(-1, 1)

    dataX_test = []
    dataY_test = []
    dataY_date = []

    for i in range(0, len(X_test) - window):
        _xy = X_test[i:i + window]  # 包括用于计算的seq_length天的数据，以及day天后的价格
        dataX_test.append(_xy)
        dataY_test.append(Y_test[i + window])
        dataY_date.append(Y_test_data[i + window])

    # 调整数据的shape，-1是自动计算行数   vstack将数组在垂直方向上叠加。
    dataX_test = np.vstack(dataX_test).reshape(-1, window, 11)
    dataY_test = np.vstack(dataY_test).reshape(-1, 1)
    dataY_date = np.vstack(dataY_date).reshape(-1,1)
    # print(dataX.shape,dataY.shape)  #(5254, 31, 10) (5254, 31)
    # print(dataX_test.shape,dataY_test.shape)  # (3495, 31, 10) (3495, 31)

    # 先处理训练集的数据
    scaler = MinMaxScaler()  # 归一化
    X_train = dataX.reshape((-1, 11))  # 先变成2维，才能transform
    X_train_new = scaler.fit_transform(X_train)  # 预处理，按列操作，每列最小值为0，最大值为1
    X_train_new = X_train_new.reshape((-1, window, 11))  # 变回3维
    Y_train_new = scaler.fit_transform(dataY)  # 预处理，按列操作，每列最小值为0，最大值为1
    # Y_close_train = scaler.fit_transform(closeY_train)

    # 再处理测试集的数据
    X_test = dataX_test.reshape((-1, 11))  # 先变成2维，才能transform
    X_test_new = scaler.fit_transform(X_test)  # 预处理，按列操作，每列最小值为0，最大值为1
    X_test_new = X_test_new.reshape((-1, window, 11))  # 变回3维
    Y_test_new = scaler.fit_transform(dataY_test)

    # Y_close = scaler.fit_transform(closeY_test)
    # print(X_test_new.shape,Y_test_new.shape)

    return X_train_new, Y_train_new, X_test_new, Y_test_new,dataY_date, scaler


# 获取测试数据,并且经过模型预测出结果，保存到本地。
def handleTestData(tickers, window):
    # tickers = ['000783.SZ', '000785.SZ'] #600006.SH', '600007.SH', '600008.SH', '600009.SH']'
    model = load_model('lstm_model.h5')
    filename = 'D:/software/python/project/stock/Data/bishe/BT/total/ml_nn_test/day/' + ticker + '.csv'
    df = pd.read_csv(filename)
    df = df.drop(['Unnamed: 0'], axis=1)
    # 获取XY
    length = len(df)
    pre_df = df.iloc[:, 2:13].copy()
    pre_df = pre_df.replace([np.inf, -np.inf], 0)
    pre_df.fillna(0, inplace=True)
    pre_df = np.array(pre_df)
        # _y = np.array(df.up.values)
    dataX = []
    dataY = []
    for i in range(0, len(pre_df) - window):
        _xy = pre_df[i:i + window]  # 包括用于计算的seq_length天的数据，以及day天后的价格
        dataX.append(_xy)
            # dataY.append(_y[i:i + window])
        # 调整数据的shape，-1是自动计算行数   vstack将数组在垂直方向上叠加。
        # print(len(pre_df),len(dataX))
    dataX = np.vstack(dataX).reshape(-1, window, 11)
        # print(len(pre_df),dataX[0])
        # dataY = np.vstack(dataY).reshape(-1, 1)
    scaler = MinMaxScaler()  # 归一化
    X_train = dataX.reshape((-1, 11))  # 先变成2维，才能transform
    X_train_new = scaler.fit_transform(X_train)  # 预处理，按列操作，每列最小值为0，最大值为1
    X_train_new = X_train_new.reshape((-1, window, 11))
        # Y_train_new = scaler.fit_transform(dataY)
        # Y_train_new = Y_train_new.flatten()
        # print('predict:',predict,len(predict))
        # print('Y_train_new:',Y_train_new,len(predict))

    predictions_train = []
    prediclss = []
        # temp = 0
    for i in range(len(X_train_new)):
        data = X_train_new[i]
        predicted = model.predict(data[np.newaxis, :, :])[0, 0]
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
        # print('predictions_train',predictions_train)
        # print(len(X_train_new),len(predictions_train),len(df))
    for i in range(len(predictions_train)):
        df.loc[i + window, 'predictForUp'] = predictions_train[i]

    df.to_csv(filename)


# 建立模型
def ml_lstm(trainX, trainY):
    model = Sequential()
    # first layer   50output feed second layer  所有的股价是一个一个进入的，所以input——dim=1   output的50个中间变量都会喂给第二层
    model.add(LSTM(
        input_dim=11,
        output_dim=30,
        return_sequences=True
    ))

    # 避免模型过于 overfit training set  随机删除20%数据
    model.add(Dropout(0.1))

    # second layer 不需要喂给下一层，所以是false，并且不需要output

    model.add(LSTM(
        50,
        return_sequences=True
    ))
    model.add(Dropout(0.1))

    model.add(LSTM(
        50,
        return_sequences=True
    ))
    model.add(Dropout(0.1))

    model.add(LSTM(
        50,
        return_sequences=False
    ))
    model.add(Dropout(0.1))

    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
    # model.add(Dense(256,activation='tanh', kernel_regularizer=regularizers.l2(0.0)))
    # last layer  只需要输出一个数字，用线性函数， 优化器用 rmsprop，防止lose function 掉入局部最小值
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer='RMSProp', metrics=['accuracy'])

    # 3. 训练模型，每一个包为512个数据，可以根据内存变。训练次数 = 5， 我们用5%的数据用来校正查验。

    # '''
    # 第一种
    # monitor：被监测的量
    # factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
    # patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。

    # '''
    # from keras.callbacks import ReduceLROnPlateau
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, mode='auto')

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
    history = model.fit(trainX, trainY,
                        batch_size=32,
                        nb_epoch=20,
                        validation_split=0.05,
                        callbacks=[reduce_lr]
                        )

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print('model fit loss:', history.history['loss'])
    # print('model fit acc:',history.history['acc'])

    print('model fit val_loss:', history.history['val_loss'])
    # print('model fit val_acc:',history.history['val_acc'])
    # 保存模型
    model.save('lstm_model.h5')  # HDF5文件，pip install h5py
    # print('model',model)
    return model


# 训练模型，检测测试集的正确率,以及保存模型
@app.route('/lstmpre', methods=["GET"])
def trainModel():
    start = time.time()
    window = 5
    tickers = request.args.get("code")
    # code = code + '.SZ'
    X_train, X_test, Y_train, Y_test,Y_test_data = handleData(tickers)

    print('handle train finish')
    X_train_new, Y_train_new, X_test_new, Y_test_new,dataY_date, scaler = handleData_new(X_train, X_test, Y_train, Y_test,Y_test_data, window)

    # dataClose_new =  dataClose_new/ 10 * scaler.data_range_[0] + scaler.data_min_[0]
    print('data finished')
    # print(X_train_new[0])
    print('366',len(dataY_date),len(Y_test_new)) # 468 468
    # 训练模型
    model = ml_lstm(X_train_new, Y_train_new)
    print('model finished')
    # 验证集验证  predictionsNormalized为验证结果
    predictions_train = []
    temp = 0
    for i in range(len(X_train_new)):
        data = X_train_new[i]
        if i > 0:
            data[window - 1][4] = temp
        predicted = model.predict(data[np.newaxis, :, :])[0, 0]
        temp = predicted
        predictions_train.append(predicted)

    correct = np.zeros(len(predictions_train))
    for i in range(len(predictions_train)):
        predicted = 0
        if i >0 :
            if np.sign(predictions_train[i] - predictions_train[i-1]) == np.sign(Y_train_new[i] - Y_train_new[i-1]):
                # 如果对涨或跌的判断准确，这里用正负符号判断
                correct[i] = 1  # 就加1
    accuracy = np.sum(correct) / len(correct) * 100
    accuracy = round(accuracy, 2)
    print('LSTM 训练集预测涨或跌的正确率：', accuracy)

    # --------------------------train-test
    # print('predictions_train',predictions_train)
    predictions_test = []

    temp0 = 0
    for i in range(len(X_test_new)):
        if i > 0:
            data[window - 1][4] = temp0
        data = X_test_new[i]
        predicted = model.predict(data[np.newaxis, :, :])[0, 0]
        predictions_test.append(predicted)

    correct0 = np.zeros(len(predictions_test))
    for i in range(len(predictions_test)):
        predicted = 0
        if i > 0:
            if np.sign(predictions_test[i] - predictions_test[i - 1]) == np.sign(Y_test_new[i] - Y_test_new[i - 1]):
                # 如果对涨或跌的判断准确，这里用正负符号判断
                correct0[i] = 1  # 就加1
    accuracy0 = np.sum(correct0) / len(correct0) * 100
    accuracy0 = round(accuracy0, 2)
    print('LSTM 测试集预测涨或跌的正确率：', accuracy0)


    predictions_train = np.array(predictions_train)
    # predictions_train = predictions_train / 10 * scaler.data_range_[0] + scaler.data_min_[0] #放大和scale的逆运算
    predictions_test = np.array(predictions_test)
    predictions_test = predictions_test.reshape(-1,1)
    predictions_test = scaler.inverse_transform(predictions_test) #放大和scale的逆运算
    Y_test_new = scaler.inverse_transform(Y_test_new)
    predictions_train = predictions_train.flatten()
    predictions_test = predictions_test.flatten()
    Y_test_new = Y_test_new.flatten()
    # print(prediclss0)

    print('pre-LSTM预测涨或跌的正确率--training set：', accuracy)
    print('pre-LSTM预测涨或跌的正确率--testing set：', accuracy0)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    print('Running time: %s hour %s minutes %s Seconds' % (
    ((end - start) // 3600), ((end - start) % 3600) // 60, (end - start) % 3600 % 60)),
    predictions_train = predictions_train.reshape(-1,1)
    dataY_date = [str(x).strip('[]') for x in dataY_date]
    print(dataY_date)
    # tran_date = transactions['date'].values
    # tran_date = list(map(str, tran_date))

    stocklist = getlist()
    name = []
    print('stocklist:', len(stocklist))
    if tickers in stocklist.keys():
        name = stocklist[tickers]
        print('name', name)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    subplot1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    line1 = subplot1.plot(dataY_date, Y_test_new, color='blue')
    line2 = subplot1.plot(dataY_date, predictions_test, color='orange')
    # ax1= plt.plot(dataY_date, Y_test_new, color='blue')
    # ax1=plt.plot(dataY_date, predictions_test, color='orange')
    tick_spacing = 7
    subplot1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.legend([u'真实值', u'模型预测值'], loc='upper left',fontsize = 14)
    plt.suptitle('\"' + tickers + '-' + str(name[0]) + u'\"的LSTM 算法预测结果', fontsize=18)

    plt.savefig('D:/software/python/project/stock/learn_envs/bishe_inter/result/lstm_pre.png')
    plt.show()

    url = 'D:/software/python/project/stock/learn_envs/bishe_inter/result/lstm_pre.png'
    image = Image.open(url)

    with open(url, 'rb') as f:
        res = base64.b64encode(f.read())
        resp = jsonify(res)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET'
        resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
        return resp
    return Y_test_new, predictions_test, accuracy, accuracy0

def getlist():
    sh_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sh-usual.csv')
    sz_cy = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-cy.csv')
    sz_zxb = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-zxb.csv')
    sz_usual = pd.read_csv('D:/software/python/project/stock/Data/bishe/BT/stocklist/sz-usual.csv')

    data1 = sh_usual[['ts_code', 'name']]
    dict_1 = data1.set_index('ts_code').T.to_dict('list')
    data2 = sz_cy[['ts_code', 'name']]
    dict_2 = data2.set_index('ts_code').T.to_dict('list')
    data3 = sz_zxb[['ts_code', 'name']]
    dict_3 = data3.set_index('ts_code').T.to_dict('list')
    data4 = sz_usual[['ts_code', 'name']]
    dict_4 = data4.set_index('ts_code').T.to_dict('list')
    dict_list = {**dict_1, **dict_2}
    dict_list = {**dict_list, **dict_3}
    dict_list = {**dict_list, **dict_4}

    return dict_list

# # 计算买卖点，收益
# @app.route('/lstmcom', methods=["GET"])
#
#
#
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5010,
        debug=True
    )


