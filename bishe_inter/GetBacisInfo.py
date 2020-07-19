#import GetIntraday
import tushare
import pandas
import datetime
import os
import time

token = '5d9ec9dc6d71031a48a24b0e0f6c87e84fd2caf6bf15ac5845df7177'
tushare.set_token(token)
pro = tushare.pro_api()

# 利润表  n_income（含）total_profit  total_profit-income_tax  revenue
def getLirun(ticker,folder):
    # 1. 获取intraday股票的利润表
    lirun = pro.income(ts_code=ticker,period='20191231',fields='ts_code,ann_date,f_ann_date,n_income,total_profit,total_profit,income_tax,revenue')
    # 2. 存利润
    file = folder + ticker 
    if not os.path.exists(file):
        os.makedirs(file)
    file_path = file +'/'+ ticker + '- lirun' +'.csv'
    lirun.to_csv(file_path)

# 财务指标数据  undist_profit_ps  capital_rese_ps  surplus_rese_ps
def getCaiwu(ticker,folder):
    caiwu = pro.fina_indicator(ts_code=ticker,period='20191231',fields='ts_code,ann_date,end_date,undist_profit_ps,capital_rese_ps,surplus_rese_ps')
    file = folder + ticker 
    if not os.path.exists(file):
        os.makedirs(file)
    file_path = file +'/'+ ticker + '- caiwu' +'.csv'
    caiwu.to_csv(file_path)
# 每日指标circ_mv  float_share流通股本（万股） pb pb
def getMeiri(ticker,folder):
    meiri = pro.daily_basic(ts_code=ticker,fields='ts_code,trade_date,circ_mv,float_share,pb,pe')
    file = folder + ticker 
    if not os.path.exists(file):
        os.makedirs(file)
    file_path = file +'/'+ ticker + '- meiri' +'.csv'
    meiri.to_csv(file_path)
# 现金流量表  c_recp_return_invest（取得投资收益收到的现金） n_cashflow_act（经营活动产生的现金流量净额）
def getCash(ticker,folder):
    cash = pro.cashflow(ts_code=ticker,period='20191231',fields='ts_code,ann_date,f_ann_date,end_date,c_recp_return_invest,n_cashflow_act')
    file = folder + ticker 
    if not os.path.exists(file):
        os.makedirs(file)
    file_path = file +'/'+ ticker + '- cash' +'.csv'
    cash.to_csv(file_path)

# 业绩快报  bps、diluted_eps、diluted_roe、open_net_assets（期初净资产）、operate_profit、total_assets、yoy_sales（同比增长率：营业收入）、yoy_tp
def getYeji(ticker,folder):
    yeji = pro.express(ts_code=ticker,period='20191231',fields='ts_code,ann_date,end_date,bps,diluted_eps,diluted_roe,open_net_assets,operate_profit,total_assets,yoy_sales,yoy_tp')
    file = folder + ticker 
    if not os.path.exists(file):
        os.makedirs(file)
    file_path = file +'/'+ ticker + '- yeji' +'.csv'
    yeji.to_csv(file_path)

# 资产负债表 accounts_receiv cap_rese intan_assets total_cur_assets  total_cur_liab total_share undistr_porfit
def getZichan(ticker,folder):
    zichan = pro.balancesheet(ts_code=ticker,period='20191231',fields='ts_code,ann_date,f_ann_date,end_date,accounts_receiv,cap_rese,intan_assets,total_cur_assets,total_cur_liab,total_share,undistr_porfit')
    file = folder + ticker 
    if not os.path.exists(file):
        os.makedirs(file)
    file_path = file +'/'+ ticker + '- zichan' +'.csv'
    zichan.to_csv(file_path)

# 查找所有股票列表信息。

def stockInfo():
    stockInfo = pro.stock_basic()
    tickers = stockInfo['ts_code'].tolist() 
    dateToday = datetime.datetime.today().strftime('%Y%m%d')
    file = 'D:/software/python/stock/Data/bishe/_stockinfo.csv'
    stockInfo.to_csv(file)

def companyInfo(exchange):
    companyInfo = pro.stock_company(exchange=exchange)
    file = 'D:/software/python/stock/Data/bishe/_companyinfo.csv'
    companyInfo.to_csv(file)

# 上市公司管理层
def stk_managers(ts_code):
    managerinfo = pro.stk_managers(ts_code=ts_code)
    file = 'D:/software/python/project/stock/Data/bishe/KG/company_manager.csv'
    managerinfo.to_csv(file)

# 获取管理层薪酬
def stk_rewards(ts_code):
    stk_rewards = pro.stk_rewards(ts_code=ts_code)
    file = 'D:/software/python/project/stock/Data/bishe/KG/manager_reward.csv'
    stk_rewards.to_csv(file)


#tickers = GetIntraday.getIntraday()
def get():
    folder='D:/software/python/stock/Data/bishe/test'
    stockInfo = pandas.read_csv('D:/software/python/project/stock/Data/bishe/KG/_stockinfo.csv')
    tickers = stockInfo[('ts_code')].tolist()
    string = ','.join(tickers[0:100])
    stk_managers(str(string))
    stk_rewards(str(string))

# 获取知识图谱数据
def main():
    stockInfo = pandas.read_csv('D:/software/python/project/stock/Data/bishe/KG/_stockinfo.csv')
    tickers = stockInfo[('ts_code')].tolist()
    lirun_res = pandas.DataFrame(columns=('ts_code', 'ann_date', 'f_ann_date','n_income','total_profit','income_tax','revenue'))
    caiwu_res = pandas.DataFrame(columns=('ts_code','ann_date','end_date','undist_profit_ps','capital_rese_ps','surplus_rese_ps'))
    meiri_res = pandas.DataFrame(columns=('ts_code','trade_date','circ_mv','float_share','pb','pe'))
    cash_res = pandas.DataFrame(columns=('ts_code','ann_date','f_ann_date','end_date','c_recp_return_invest','n_cashflow_act'))
    yeji_res = pandas.DataFrame(columns=('ts_code','ann_date','end_date','bps','diluted_eps','diluted_roe','open_net_assets','operate_profit','total_assets','yoy_sales','yoy_tp'))
    zichan_res = pandas.DataFrame(columns=('ts_code','ann_date','f_ann_date','end_date','accounts_receiv','cap_rese','intan_assets','total_cur_assets','total_cur_liab','total_share','undistr_porfit'))
    for i,ticker in enumerate(tickers):
        print(i,ticker)
        lirun = pro.income(ts_code=ticker,period='20191231',fields='ts_code,ann_date,f_ann_date,n_income,total_profit,income_tax,revenue')
        caiwu = pro.fina_indicator(ts_code=ticker,period='20191231',fields='ts_code,ann_date,end_date,undist_profit_ps,capital_rese_ps,surplus_rese_ps')
        meiri = pro.daily_basic(ts_code=ticker,fields='ts_code,trade_date,circ_mv,float_share,pb,pe')
        cash = pro.cashflow(ts_code=ticker,period='20191231',fields='ts_code,ann_date,f_ann_date,end_date,c_recp_return_invest,n_cashflow_act')
        yeji = pro.express(ts_code=ticker,period='20191231',fields='ts_code,ann_date,end_date,bps,diluted_eps,diluted_roe,open_net_assets,operate_profit,total_assets,yoy_sales,yoy_tp')
        zichan = pro.balancesheet(ts_code=ticker,period='20191231',fields='ts_code,ann_date,f_ann_date,end_date,accounts_receiv,cap_rese,intan_assets,total_cur_assets,total_cur_liab,total_share,undistr_porfit')
        
        if lirun.empty:
            pass
        else:
            lirun_res = lirun_res.append(lirun.iloc[0],ignore_index=True)
        print('lirun_res:',lirun,lirun_res)
        if caiwu.empty:
            pass
        else:
            caiwu_res = caiwu_res.append(caiwu.iloc[0],ignore_index=True)
        if meiri.empty:
            pass
        else:
            meiri_res = meiri_res.append(meiri.iloc[0],ignore_index=True)
        if cash.empty:
            pass
        else:
            cash_res = cash_res.append(cash.iloc[0],ignore_index=True)
        if yeji.empty:
            pass
        else:
            yeji_res = yeji_res.append(yeji.iloc[0],ignore_index=True)
        if zichan.empty:
            pass
        else:
            zichan_res = zichan_res.append(zichan.iloc[0],ignore_index=True)
    lirun_res.to_csv('D:/software/python/project/stock/Data/bishe/KG/lirun_res.csv')
    caiwu_res.to_csv('D:/software/python/project/stock/Data/bishe/KG/caiwu_res.csv')
    meiri_res.to_csv('D:/software/python/project/stock/Data/bishe/KG/meiri_res.csv')
    cash_res.to_csv('D:/software/python/project/stock/Data/bishe/KG/cash_res.csv')
    yeji_res.to_csv('D:/software/python/project/stock/Data/bishe/KG/yeji_res.csv')
    zichan_res.to_csv('D:/software/python/project/stock/Data/bishe/KG/zichan_res.csv')

get()
print('all for all stocks got.')