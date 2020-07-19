import tushare as ts
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Input, Dense, LSTM, Reshape
from keras.models import Model
from keras import regularizers, callbacks
import matplotlib.pyplot as plt
import matplotlib
import datetime
from pandas.plotting import register_matplotlib_converters
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
register_matplotlib_converters()
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


app = Flask(__name__)
CORS(app,resources=r'/*')
#from datetime import datetime

y_visual_date_new=[]

def time2stamp(cmnttime):   #转时间戳函数
    cmnttime=datetime.datetime.strptime(cmnttime,'%Y%m%d')
    stamp=int(datetime.datetime.timestamp(cmnttime))
    return stamp

def com_predict_time(day):
    date = datetime.datetime.now().date() #获取当天日期
    print(date)
    date_inter =date + datetime.timedelta(days=0) 
    print(date_inter)
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


def get_data(code, date):
    #数据准备/data preparation 
    #变量选取Open,High,Low,Close,Volume等，以浦发银行股票为例
    pro = ts.pro_api('5d9ec9dc6d71031a48a24b0e0f6c87e84fd2caf6bf15ac5845df7177')               #token可以在新版tushare的网站上找到
    stock_data = pro.query('daily', ts_code = code, start_date = '20100101', end_date = date)
    stock_data = stock_data[::-1] #倒序，使日期靠前的排在前面
    stock_data.reset_index(drop=True, inplace=True) #把每行的索引改为“0、1、2……”
    print("stock-data")
    print(stock_data.head())
    return stock_data

def preprocess(stock_data, day, seq_length, data_dim):
    '''
    day : 要预测多少天的数据
    seq_length ： 时间窗大小20
    data_dim ： 输入数据维度7
    output_dim: 输出数据维度1
    xxy1:去除时间的数据
    X_date: 为日期
    '''
    xy = stock_data[['trade_date','open', 'close', 'high', 'low', 'vol', 'pct_chg', 'amount']] #选取需要的features
    xy.reset_index()  
    print("xy.head0()")
    print(xy.head())
    #xy['trade_date'] =xy['trade_date'].apply(time2stamp)
    print(type(xy))
    print("xy.head()")
    X_label_date=xy['trade_date']
    #print(X_label_date)
    xy_a = stock_data[['open', 'close', 'high', 'low', 'vol', 'pct_chg', 'amount']] #选取需要的features
    xy_a.reset_index() #选取需要的features
    #xy1.reset_index()  
    xy_a = np.array(xy_a.values) #转为array
    X_label_date = np.array(X_label_date.values)
    #print('71',xy_a,X_label_date)

    X_label_date = list(X_label_date)
    X_label_date = list(map(int,X_label_date))
    #print('X_label_date',len(X_label_date))
    X_label_date = np.array(X_label_date)
    # print('78',xy_a,X_label_date)
    # 此刻的xy里第一列是时间戳。
    #xy_a = xy_a.flatten()
    # print(xy_a,X_label_date.shape) # (4474,7) (4474,)
    print('88',len(xy_a),len(X_label_date)) # 4474 4474

    # dataXY 为seq_length 和 day 一共的数据
    dataXY = []
    dataxy_date = []
    for i in range(0, len(xy_a) - seq_length - day + 1):

        _xy = xy_a[i:i + seq_length + day] #包括用于计算的seq_length天的数据，以及day天后的价格
        dataXY.append(_xy)
        dataxy_date.append(X_label_date[i:i + seq_length + day])
    #_xy = xy_a[0:len(xy_a) - seq_length - day + 1] #包括用于计算的seq_length天的数据，以及day天后的价格
    #dataXY.append(_xy)
    #dataxy_date.append(X_date[0:len(xy_a) - seq_length - day + 1])
    #print('99',len(dataXY),len(dataxy_date))# 4450  4450
    #调整数据的shape，-1是自动计算行数   vstack将数组在垂直方向上叠加。
    dataXY = np.vstack(dataXY).reshape(-1, seq_length+day, data_dim)
    dataxy_date = np.vstack(dataxy_date).reshape(-1, seq_length+day)
    #print('105',dataXY.shape,dataxy_date.shape) #(4450, 25, 7) (4450, 25)
    print('119',dataxy_date[0])

    app_dataX = []
    app_dataX_date = []
    for i in range(len(xy_a) - seq_length - day + 1, len(xy_a) - seq_length + 1): # 30 -5-8+1=18 ,25
        print(i,len(xy_a))
        _x = xy_a[i:i + seq_length]  # 包括用于计算的seq_length天的数据  18-23,C19-24,...25-30
        app_dataX.append(_x)
        app_dataX_date.append(X_label_date[i:i + seq_length])
    #_x = xy_a[len(xy_a) - seq_length - day + 1: len(xy_a) - seq_length + 1] #包括用于计算的seq_length天的数据
    #app_dataX.append(_x)
    #调整数据的shape
    print(app_dataX_date)
    app_dataX = np.array(app_dataX)
    # app_dataX = app_dataX.reshape(-1,seq_length,data_dim)
    # print('appxdatex',app_dataX.shape) # (5,20)
    app_dataX = np.vstack(app_dataX).reshape(-1, seq_length,data_dim)
    print('appxdatex', app_dataX.shape) # (5,20,7)
  
    
    #dataxy_date=np.array(dataxy_date)
    print("dataxy---")
    
    #print(dataXY.shape,dataxy_date.shape)
    dataxy_date=np.array(dataxy_date)
    visual_window = int(len(dataXY) * 0.2)
    xy_visual = np.copy(dataXY[- visual_window:]) #取最近visual_window天的数据，用于最后的画图
    xy_visual_date = np.copy(dataxy_date[- visual_window:])
    #print('xy_visual',xy_visual.shape,xy_visual)
    #print('xy_visual_date',xy_visual_date.shape,xy_visual_date)
    # print(xy_visual_date.shape)
    np.random.shuffle(dataXY) #打乱顺序
    
    #切分训练集合测试集/split to train and testing
    train_size = int(len(dataXY) * 0.8) #训练集长度
    test_size = len(dataXY) - train_size #测试集长度
    xy_train, xy_test = np.array(dataXY[0:train_size]), np.array(dataXY[train_size:len(dataXY)]) #划分训练集、测试集
    
    #先处理训练集的数据
    scaler = MinMaxScaler() # 归一化
    xy_train = xy_train.reshape((-1, data_dim)) #先变成2维，才能transform
    xy_train_new = scaler.fit_transform(xy_train) #预处理，按列操作，每列最小值为0，最大值为1
    xy_train_new = xy_train_new.reshape((-1, seq_length + day, data_dim)) #变回3维
    
    x_new = xy_train_new[:,0:seq_length] #features
    y_new = xy_train_new[:,-1,-1] * 10 #取最后一天的收盘价，用作label，适当放大，便于训练
    
    trainX, trainY = x_new, y_new
    print('xy_test',xy_test.shape,) # (1335,25,7)
    print('151',trainX.shape,trainY.shape)   # (3129, 20, 7) (3129,) #然后处理测试集的数据
    xy_test = xy_test.reshape((-1, data_dim))
    xy_test_new = scaler.transform(xy_test) #使用训练集的scaler预处理测试集的数据
    xy_test_new = xy_test_new.reshape((-1, seq_length + day, data_dim))
    #print('xy_test_new',xy_test_new.shape,xy_test_new)
    x_new = xy_test_new[:, 0:seq_length]
    print('157',type(x_new))
    y_new = xy_test_new[:, -1, -1] * 10
    print(len(x_new),x_new.shape)# 1335 (1342,25,7)
    print(len(y_new),y_new.shape)# 1335  (1342,)
    #以下3项用于计算收入
    close_price = xy_test_new[:, seq_length - 1, -1]
    buy_price = xy_test_new[:, seq_length, 0]
    sell_price = xy_test_new[:, -1, -1]
    
    testX, testY, test_close, test_buy, test_sell = x_new, y_new, close_price, buy_price, sell_price
    
    #再处理应用集
    x_app = app_dataX.reshape((-1, data_dim))
    appX = scaler.transform(x_app) #用训练集的scaler进行预处理
    appX = appX.reshape((-1, seq_length, data_dim))
    print('appx',appX.shape) #(5,5,7)

    #最后处理用于画图的数据
    print('xy_visual',xy_visual.shape)#(200,25,7)
    xy_visual = xy_visual.reshape((-1, data_dim))
    print('xy_visual',xy_visual.shape)#(5000,7)
    print('xy_visual_date',xy_visual_date.shape)#(200,25)

    xy_visual_new = scaler.transform(xy_visual) #使用训练集的scaler预处理
    print('xy_visual_new:',xy_visual_new.shape)#(5000,7)
    xy_visual_date_new = xy_visual_date
    
    xy_visual_new = xy_visual_new.reshape((-1, seq_length + day, data_dim))
    #xy_visual_date_new = xy_visual_date_new.reshape((-1, seq_length + day, data_dim))
    #print('xy_visual_new:',type(xy_visual_new),xy_visual_new.shape)#(200, 25, 7)
    #print('xy_visual_date_new:',type(xy_visual_date_new),xy_visual_date_new.shape)#(200, 25)
    x_new = xy_visual_new[:, 0:seq_length]
    y_new = xy_visual_new[:, -1, -1] * 10
    print('y_new',y_new.shape)
    y_visual_date_new = xy_visual_date_new[:, -1]
    # print('xy_visual_date_new',xy_visual_date_new)
    print('y_visual_date_new',y_visual_date_new.shape)
    
    visualX, visualY = x_new, y_new
    # visualX=np.array(visualX)
    visualY=np.array(visualY)
    # y_visual_date_new=xy_visual_date_new
    y_visual_date_new = np.array(y_visual_date_new)

    #print('len(v):',visualY.shape,visualY.shape)
    #print('len(y):',y_visual_date_new.shape,y_visual_date_new.shape)
    
    return y_visual_date_new,trainX, trainY, testX, testY, appX, scaler, test_close, test_buy, test_sell, visualX, visualY,visual_window

def train(code, day, trainX, trainY, seq_length, data_dim, output_dim):
    # 构建神经网络层 1层Dense层+1层LSTM层+4层Dense层
    rnn_units = 32
    print('208',seq_length,data_dim) # 5,7
    # Dense_input = Input(shape=(seq_length, 1), name='dense_input') #输入层
    #shape: 形状元组（整型）不包括batch size。表示了预期的输入将是一批（seq_len,data_dim）的向量。
    Dense_input = Input(shape=(seq_length,data_dim), name='dense_input')
    Dense_output_1 = Dense(rnn_units, activation='relu', kernel_regularizer=regularizers.l2(0.0), name='dense1')(Dense_input) #全连接网络

    lstm_input = Reshape(target_shape=(seq_length, rnn_units), name='reshape2')(Dense_output_1) 
    #改变Tensor形状，改变后是（None，seq_length, rnn_units）
    lstm_output = LSTM(rnn_units, activation='softsign', dropout=1.0, name='lstm')(lstm_input) #LSTM网络
    #units: Positive integer,dimensionality of the output space.
    #dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
    
    Dense_input_2 = Reshape(target_shape=(rnn_units,), name='reshape3')(lstm_output) 
    #改变Tensor形状，改变后是（None，rnn_units）
    Dense_output_2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0), name='dense2')(Dense_input_2) #全连接网络
    Dense_output_3 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0), name='dense3')(Dense_output_2) #全连接网络
    Dense_output_4 = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0), name='dense4')(Dense_output_3) #全连接网络
    predictions = Dense(output_dim, activation=None, kernel_regularizer=regularizers.l2(0.0), name='dense5')(Dense_output_4) #全连接网络
    
    model = Model(inputs=Dense_input, outputs=predictions)
    #This model will include all layers required in the computation of output given input.
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #Configures the model for training.
    #optimizer: String (name of optimizer) or optimizer instance. See optimizers.
    #loss: String (name of objective function) or objective function.The loss value will be minimized by the model.
    #metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use  metrics=['accuracy'].
    
    ES = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    print('236',trainX.shape) #236 (3129, 20, 7)
    model.fit(trainX, trainY, batch_size=512, epochs=5, verbose=0, callbacks=[ES], validation_split=0.1)
    #Trains the model for a given number of epochs (iterations on a dataset).
    #verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    
    # 保存模型
    model.save(code + '(1)' + str(day) + '.h5')   # HDF5文件，pip install h5py
    print('model')
    print('model',model)
    return model

def test(model, testX, testY, scaler, day, close_price, buy_price, sell_price, visualX, visualY):
    print('testX',testX.shape)
    testPredict = model.predict(testX) #查看测试结果
    
    testPredict2 = testPredict / 10 * scaler.data_range_[1] + scaler.data_min_[1] #放大和scale的逆运算
    testY2 = testY / 10 * scaler.data_range_[1] + scaler.data_min_[1] #放大和scale的逆运算
    
    #以下3项用于计算收入
    #今天的收盘价，用于判断买不买
    close_price2 = close_price * scaler.data_range_[1] + scaler.data_min_[1]
    #明天的开盘价，如果买需要付多少钱
    buy_price2 = buy_price * scaler.data_range_[0] + scaler.data_min_[0]
    #持有day天之后的收盘价，这时卖能卖多少钱
    sell_price2 = sell_price * scaler.data_range_[1] + scaler.data_min_[1]
    
    #平均误差（%）
    mean_error = np.mean(abs(testPredict2 - testY2) / testY2 * 100) # 预测结果和实际结果的比值
    mean_error = round(mean_error, 2) # round函数俩位小数四舍五入
    print('平均误差（%）：', mean_error)
    
    #最大误差（%）
    max_error = np.max(abs(testPredict2 - testY2) / testY2 * 100)
    max_error = round(max_error, 2)
    print('最大误差（%）：', max_error)
    
    count = 0 #绝对误差小于1%的比例
    correct = np.zeros(len(testPredict2)) #预测涨或跌的正确率
    model_income = 0 #模型能挣多少钱
    trade = 0 #计算交易频率
    max_income = 0 #最理想的状况下，能挣多少钱
    random_income = 0 #随机购买，能挣多少钱
    
    tolerance = 1
    for i in range(len(testY2)):
        #计算绝对误差小于 tolerance% 的比例
        if abs(testPredict2[i] - testY2[i]) / testY2[i] * 100 <= tolerance:
            count += 1
        #计算对转折点的预测正确率
        if np.sign(testPredict2[i] - close_price2[i]) == np.sign(testY2[i] - close_price2[i]):
            #如果对涨或跌的判断准确，这里用正负符号判断
            correct[i] = 1 #就加1
        #如果对“day”天后的预测价格高于今天的收盘价，就买进并持有“day”天，计算能挣多少钱
        if testPredict2[i] > close_price2[i]:
            model_income = model_income + sell_price2[i] - buy_price2[i]
            trade += 1
        #最理想的状况下，能挣多少钱
        if testY2[i] > close_price2[i]:
            max_income = max_income + sell_price2[i] - buy_price2[i]
        #随机购买，能挣多少钱
        buy = np.random.randint(0, 2) #随机产生0或1
        if buy: #如果是1就买
            random_income = random_income + sell_price2[i] - buy_price2[i]
    
    count = count / len(testY2) * 100
    count = round(count, 2)
    print('误差小于' + str(tolerance) + '%的比例：', count)
    
    accuracy = np.sum(correct) / len(correct) * 100
    accuracy = round(accuracy, 2)
    print('LSTM预测涨或跌的正确率：', accuracy)
    
    print('模型的购买策略是，如果对%d天之后的预测值大于今天的收盘价，就在明天开市时买进1股，并且持有%d天，再卖出'%(day, day))
    frequency = trade / len(testPredict2) * 100
    model_income = round(float(model_income), 2)
    frequency = round(frequency, 2)
    print('在%d天中，模型交易了%d次，交易频率为%g'%(len(testPredict2), trade, frequency) + '%')
    print('按照模型进行操作所得的收入：', model_income)
    
    max_income = round(float(max_income), 2)
    print('最理想状况下的收入：', max_income)
    
    random_income = round(float(random_income), 2)
    print('随机购买的收入：', random_income)
    
    visualPredict = model.predict(visualX)
    visualPredict2 = visualPredict / 10 * scaler.data_range_[1] + scaler.data_min_[1] #放大和scale的逆运算
    visualPredict2=np.array(visualPredict2)
    #print('visualX:',visualX)
    #print('visualPredict2:',visualPredict2)
    visualY2 = visualY / 10 * scaler.data_range_[1] + scaler.data_min_[1] #放大和scale的逆运算
    #visualY2.astype(int)
    #print('visualY2:',visualY2, visualPredict2)
    return visualY2, visualPredict2

def apply(day,model, appX, scaler):
    #查看应用结果
    appPredict = model.predict(appX)
    appPredict2 = appPredict / 10 * scaler.data_range_[1] + scaler.data_min_[1] #放大和scale的逆运算
    #print('appPredict2',appPredict2)
    appPredict2 = appPredict2.reshape((day,1))
    
    #print('appPredict2',appPredict2)
    return appPredict2

def visualize(code,predict_time,y_visual_date_new,visualY2, visualPredict2, appPredict2, visual_window, day):
    #print(visualY2,visualPredict2)
    y_visual_date_new = [str(x) for x in y_visual_date_new]

    # print('visualPredict2:',type(visualPredict2))
    subplot1 = plt.subplot2grid((2,1),(0,0),rowspan=1,colspan=1)
    #subplot1.xaxis_date()
    line1=subplot1.plot(y_visual_date_new,visualY2,color='blue')
    line2=subplot1.plot(y_visual_date_new,visualPredict2,color='orange')
    subplot1.legend([u'真实值', u'模型预测值'], loc='upper left',fontsize = 14)
    # 获取股票列表
    stocklist = getlist()
    name = []
    print('stocklist:', len(stocklist))
    if code in stocklist.keys():
        name = stocklist[code]
        print('name', name)
    tick_spacing = 1
    subplot1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    subplot2 = plt.subplot2grid((2,1),(1,0),rowspan=1,colspan=1)
    subplot2.scatter(predict_time, appPredict2, color='red')
    subplot2.legend([u'预测值'], loc='upper left')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.suptitle('\"' + code + '-' + str(name[0]) + u'\"的LSTM 算法预测结果',fontsize = 18)
    plt.savefig('D:/software/python/project/stock/learn_envs/bishe_inter/result/lstm_pre.png')
    plt.show()

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

@app.route('/lstmpre', methods=["GET"])
def main():
    #code = input("请输入6位代码：") #输入股票代码
    code = request.args.get("code")
    
    # code = code + '.SZ'
    
    #day = input("请输入预测天数：") #输入预测多少天后的价格
    day = request.args.get("day")
    try:
        day = int(day)
    except:
        day = 5
        day = int(day)
    
    print('day',day)
    
    import time
    date = time.strftime('%Y%m%d',time.localtime(time.time())) #获取当天日期
    predict_time = com_predict_time(day)
    print('predict_time',predict_time)
    #参数设置/parameter setting
    timesteps = seq_length = 5 #时间窗/window length
    data_dim = 7 #输入数据维度/dimension of input data
    output_dim = 1 #输出数据维度/dimension of output data
    # visual_window = 200
    keras.backend.clear_session()
    try:
        stock_data = get_data(code, date)
    except:
        print('代码不正确或无法获得该股票的数据')
        return
    
    if len(stock_data) == 0:
        print('代码不正确或无法获得该股票的数据')
        return
    
    y_visual_date_new,trainX, trainY, testX, testY, appX, scaler, test_close, test_buy, test_sell, visualX, visualY,visual_window = preprocess(
        stock_data, day, seq_length, data_dim)
    
    try:
        # 载入模型
        from keras.models import load_model
        model = load_model(code + '(1)' + str(day) + '.h5')
        print('model1',model)
    except:
        print('第一次预测%d天内'%(day) + code + '的估价，需要一点时间建模')
        model = train(
            code, day, trainX, trainY, seq_length, data_dim, output_dim)
    
    visualY2, visualPredict2 = test(model, testX, testY, scaler, day, test_close, test_buy, test_sell, visualX, visualY)
    appPredict2 = apply(day,model, appX, scaler)
    
    visualize(code,predict_time,y_visual_date_new,visualY2, visualPredict2, appPredict2, visual_window, day)

    url = 'D:/software/python/project/stock/learn_envs/bishe_inter/result/lstm_pre.png'
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
        port=5010,
        debug=True
    )