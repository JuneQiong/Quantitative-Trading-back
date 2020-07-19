import os
import csv
import hashlib
import numpy as np

def get_md5(string):
    """Get md5 according to the string
    """
    byte_string = string.encode("utf-8")
    md5 = hashlib.md5()
    md5.update(byte_string)
    result = md5.hexdigest()
    return result

# 新建股票信息表
def build_stock(stock_info, lirun_res,caiwu_res,meiri_res,cash_res,yeji_res,zichan_res,stock_import):
    """Create an 'stock' file in csv format that can be imported into Neo4j.
    format -> company_id:ID,name,code,:LABEL
    label -> Company,ST
    """
    print('Writing to {} file...'.format(stock_import.split('/')[-1]))
    stock = set()  # 'code,name'
    lirun_1 = set()
    caiwu_1 = set()
    meiri_1 = set()
    cash_1 = set()
    yeji_1 = set()
    zichan_1 = set()
    with open(stock_info, 'r', encoding='utf-8') as file_prep:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            code_name = '{},{},{}'.format(row[1], row[3],row[4])
            # print(type(code_name))
            stock.add(code_name)

    with open(lirun_res, 'r', encoding='utf-8') as file_prep:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            lirun = '{},{},{},{},{}'.format(row[1],row[4],row[5],row[6],row[7])
            # print(type(lirun))
            lirun_1.add(lirun)
    
    with open(caiwu_res, 'r', encoding='utf-8') as file_prep:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            caiwu = '{},{},{},{}'.format(row[1],row[4],row[5],row[6])
            # print(type(lirun))
            caiwu_1.add(caiwu)

    with open(meiri_res, 'r', encoding='utf-8') as file_prep:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            meiri = '{},{},{},{},{}'.format(row[1],row[3],row[4],row[5],row[6])
            # print(type(lirun))
            meiri_1.add(meiri)

    with open(cash_res, 'r', encoding='utf-8') as file_prep:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            cash = '{},{},{}'.format(row[1],row[5],row[6])
            # print(type(lirun))
            cash_1.add(cash)
    
    with open(yeji_res, 'r', encoding='utf-8') as file_prep:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            yeji = '{},{},{},{},{}'.format(row[1],row[5],row[6],row[8],row[9])
            # print(type(lirun))
            yeji_1.add(yeji)
    
    with open(zichan_res, 'r', encoding='utf-8') as file_prep:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            zichan = '{},{},{},{},{},{},{},{}'.format(row[1],row[5],row[6],row[7],row[8],row[9],row[10],row[11])
            # print(type(lirun))
            zichan_1.add(zichan)

    with open(stock_import, 'w', encoding='utf-8',newline='') as file_import:
        file_import_csv = csv.writer(file_import, delimiter=',')
        # 
        headers = ['stock_id:ID', 'name','area','n_income','total_profit','income_tax','revenue','undist_profit_ps','capital_rese_ps','surplus_rese_ps',\
        'circ_mv','float_share','pb','pe','c_recp_return_invest','n_cashflow_act','diluted_eps','diluted_roe','operate_profit','total_assets',\
        'accounts_receiv','cap_rese','intan_assets','total_cur_assets','total_cur_liab','total_share','undistr_porfit',':LABEL']
        file_import_csv.writerow(headers)
        for s in stock:
            info = []
            lirun = []
            caiwu = []
            meiri = []
            cash = []
            yeji = []
            zichan = []
            split_s = s.split(',')
            info = [split_s[0], split_s[1], split_s[2]]
            
            for l in lirun_1:
                split_l = l.split(',')
                if(split_s[0] == split_l[0]):
                    lirun = [split_l[1],split_l[2],split_l[3],split_l[4]]
            if lirun:
                pass
            else:
                lirun=[0,0,0,0]

            for c in caiwu_1:
                split_c = c.split(',')
                if(split_s[0] == split_c[0]):
                    caiwu = [split_c[1],split_c[2],split_c[3]]
            if caiwu:
                pass
            else:
                caiwu=[0,0,0]

            for m in meiri_1:
                split_m = m.split(',')
                if(split_s[0] == split_m[0]):
                    meiri = [split_m[1],split_m[2],split_m[3],split_m[4]]
            if meiri:
                pass
            else:
                meiri=[0,0,0,0]

            for ca in cash_1:
                split_ca = ca.split(',')
                if(split_s[0] == split_ca[0]):
                    cash = [split_ca[1],split_ca[2]]      
            if cash:
                pass
            else:
                cash=[0,0]

            for y in yeji_1:
                split_y = y.split(',')
                if(split_s[0] == split_y[0]):
                    yeji=[split_y[1],split_y[2],split_y[3],split_y[4]]
            if yeji:
                pass
            else:
                yeji=[0,0,0,0]

            for z in zichan_1:
                split_z = z.split(',')
                if(split_s[0] == split_z[0]):
                    zichan = [split_z[1],split_z[2],split_z[3],split_z[4],split_z[5],split_z[6],split_z[7]]
            if zichan:
                pass
            else:
                zichan=[0,0,0,0,0,0,0]

            # print('--',lirun,caiwu,meiri,zichan)
            info.extend(lirun)
            info.extend(caiwu)
            info.extend(meiri)
            info.extend(cash)
            info.extend(yeji)
            info.extend(zichan)
            info.append('stock')
            #print(info)
            file_import_csv.writerow(info)
    print('- done.')

# 新建行业信息表
def build_industry(stock_info, industry_import):
    """Create an 'industry' file in csv format that can be imported into Neo4j.
    format -> industry_id:ID,name,:LABEL
    label -> Industry
    """
    print('Write to {} file...'.format(industry_import.split('/')[-1]))
    with open(stock_info, 'r', encoding="utf-8") as file_prep, \
        open(industry_import, 'w', encoding='utf-8',newline='') as file_import:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        file_import_csv = csv.writer(file_import, delimiter=',')
        headers = ['industry_id:ID', 'name', ':LABEL']
        file_import_csv.writerow(headers)

        industries = set()
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            industries.add(row[5])
        print(industries)
        for industry in industries:
            industry_id = get_md5(industry)
            new_row = [industry_id, industry, 'Industry']
            file_import_csv.writerow(new_row)
    print('- done.')

# 新建管理员信息表
def build_manager(company_manager, manager_reward,manager_import):
    """Create an 'manager' file in csv format that can be imported into Neo4j.
    format -> manager_id:ID,name,gender,age,title,reward,:LABEL
    label -> Industry
    """
    print('Write to {} file...'.format(manager_import.split('/')[-1]))
    manager = set()
    reward_title = set()

    with open(company_manager, 'r',  encoding='utf-8') as file_prep:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            # name gender age
            mana_name = '{},{},{}'.format(row[3],row[4],row[9])
            # print(type(code_name))
            manager.add(mana_name)
    with open(manager_reward, 'r',  encoding='gbk') as file_prep:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            # name title reward
            reward = '{},{},{}'.format(row[4],row[5],row[6])
            # print(type(code_name))
            reward_title.add(reward)
    with open(manager_import,'w',encoding='utf-8',newline='') as file_import:
        file_import_csv = csv.writer(file_import, delimiter=',')
        headers = ['manager_id:ID','name','title','reward','gender','age',':LABEL']
        file_import_csv.writerow(headers)
        for r in reward_title:
            split_r = r.split(',')
            # print(split_r[0])
            manager_id = get_md5(split_r[0])# name
            name = []
            if split_r[2]:
                info = [manager_id,split_r[0],str(split_r[1]),split_r[2]]
            else:
                info = [manager_id,split_r[0],str(split_r[1]),0]
            for m in manager:
                split_m = m.split(',')
                #print(split_m[0])
                age = 0
                if(len(str(split_m[2]))==4):
                    age = int(2020) - int(split_m[2])
                elif(len(str(split_m[2]))==6):
                    age = int(2020) - int(int(split_m[2])/100)
                elif(len(str(split_m[2]))==8):
                    age = int(2020) - int(int(split_m[2])/10000)
                if(split_r[0] == split_m[0]):
                    name = [split_m[1],age]
            if name:
                pass
            else:
                name=[0,0]
            info.extend(name)
            info.append('manager')
            file_import_csv.writerow(info)
    print('- done.')

# 新建股票-行业关系表
def build_stock_industry(stock_industry_prep, relation_import):
    """Create an 'stock_industry' file in csv format that can be imported into Neo4j.
    format -> :START_ID,:END_ID,:TYPE
               stock   industry
    type -> industry_of
    """
    with open(stock_industry_prep, 'r', encoding='utf-8') as file_prep, \
        open(relation_import, 'w', encoding='utf-8') as file_import:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        file_import_csv = csv.writer(file_import, delimiter=',')
        headers = [':START_ID', ':END_ID', ':TYPE']
        file_import_csv.writerow(headers)

        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            industry = row[5]
            start_id = row[1]  # code
            end_id = get_md5(industry)
            relation = [start_id, end_id, 'industry_of']
            file_import_csv.writerow(relation)

# 新建股票-管理员关系表
def build_stock_manager(manager_reward, relation_import):
    """Create an 'stock_industry' file in csv format that can be imported into Neo4j.
    format -> :START_ID,:END_ID,:TYPE
               stock   industry
    type -> industry_of
    """
    with open(manager_reward, 'r', encoding='gbk') as file_prep, \
        open(relation_import, 'w', encoding='utf-8',newline='') as file_import:
        file_prep_csv = csv.reader(file_prep, delimiter=',')
        file_import_csv = csv.writer(file_import, delimiter=',')
        headers = [':START_ID', ':END_ID', ':TYPE']
        file_import_csv.writerow(headers)

        for i, row in enumerate(file_prep_csv):
            if i == 0:
                continue
            manager = row[4]
            print(row[4])
            start_id = get_md5(manager)  # code
            end_id = row[1]
            relation = [start_id, end_id, 'employee']
            file_import_csv.writerow(relation)


if __name__ == '__main__':
    import_path = 'data/import'
    if not os.path.exists(import_path):
        os.makedirs(import_path)
    #build_industry('kg/stock_info.csv','kg/import/industry.csv')
    print('industry finish--')
    #build_stock('kg/stock_info.csv', 'kg/lirun_res.csv','kg/caiwu_res.csv','kg/meiri_res.csv','kg/cash_res.csv','kg/yeji_res.csv','kg/zichan_res.csv','kg/import/stock.csv')
    print('stock finish--')
    build_manager('kg/company_manager.csv','kg/manager_reward.csv','kg/import/manager.csv')
    print('manager finish--')
    # build_stock_industry('kg/stock_info.csv','kg/import/stock_industry.csv')
    print('build_stock_industry finish--')
    build_stock_manager('kg/manager_reward.csv','kg/import/stock_manager.csv')
    print('build_stock_manager finish--')