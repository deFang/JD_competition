# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:03:06 2017

@author: Daniel
"""

import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
os.chdir('e:/Codes/JDcomp/JD-daniel')
#os.chdir('/home/ubuntu/JDcomp/JD-daniel')

action_1_path = "../data/JData_Action_201602.csv"
action_2_path = "../data/JData_Action_201603.csv"
action_3_path = "../data/JData_Action_201604.csv"
comment_path = "../data/JData_Comment.csv"
product_path = "../data/JData_Product.csv"
user_path = "../data/JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]
brands = [489, 214]
modelid = [0.0, 216.0, 217.0,  27.0,  26.0, 218.0, 211.0,  24.0,  29.0,
           21.0, 111.0,  17.0, 210.0, 219.0, 222.0,  23.0,  19.0,  31.0,
           220.0,  13.0,  14.0,  25.0, 119.0,  28.0,  11.0, 221.0,  16.0,
           223.0, 115.0, 224.0, 110.0,  18.0, 225.0,  12.0, 120.0,  34.0,
           114.0,  22.0,  15.0,  32.0,  35.0,  33.0, 113.0, 124.0,  36.0,
           121.0,  38.0,  37.0]

def load_data(path, *args, **kwargs):
    try:
        df = pd.read_pickle(path + '.pkl')
    except:
        df = pd.read_csv(path, *args, **kwargs)
        df.to_pickle(path + '.pkl')
    return df
#%%
def _convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1

def _clean_actions():
    dump_path = './cache/clean_actions.pkl'
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        action1 = load_data(action_1_path)
        action2 = load_data(action_2_path)
        action3 = load_data(action_3_path)
        actions = pd.concat([action1, action2, action3], axis=0).reset_index(drop=True)

        deadline_time ='2016-04-07'

        user_after = set(actions[actions['time']>=deadline_time]['user_id'])
        user_before =set(actions[actions['time']<deadline_time]['user_id'])
        new_user = user_after - user_before

        df = pd.get_dummies(actions['type'], prefix='action')
        df = pd.concat([actions[['user_id','sku_id']],df], axis=1)
        df['action_all'] = 1
        user = df.drop('sku_id', axis=1)
        sku = df.drop('user_id', axis=1)

        user = user.groupby('user_id', as_index=False).agg('sum')
        sku = sku.groupby('sku_id', as_index=False).agg('sum')
        user_list = user[(user['action_all']>=2500) & (user['action_4']==0) & ~(user['user_id'].isin(new_user))]
        sku_list = sku[(sku['action_all']>=5000) & (sku['action_4']==0)]
        user_list = pd.DataFrame(user_list['user_id'])
        sku_list = pd.DataFrame(sku_list['sku_id'])
        actions['ind'] = actions.index
        user_index = actions.merge(user_list, on='user_id', how='inner')['ind']
        sku_index = actions.merge(sku_list, on='sku_id', how='inner')['ind']
        actions = actions.drop(user_index, axis=0)
        actions = actions.drop(sku_index, axis=0, errors='ignore')
        actions = actions.reset_index(drop=True)
        del actions['ind']
        print ('get_clean_action done.')
        print (actions.head(3))
        actions.to_pickle(dump_path)
    return actions



#%%
def get_actions(start_date, end_date):
    dump_path = './cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        clean_action_dump_path = './cache/clean_actions.pkl'
        if os.path.exists(clean_action_dump_path):
            actions = pd.read_pickle(clean_action_dump_path)
        else:
            actions = _clean_actions()

        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        print ('get_action {},{} done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

# ui行为计数求和
def get_ui_actions_sum(start_date, end_date):
    dump_path = './cache/action_ui_sum_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id', 'sku_id', 'action_1', 'action_2', 'action_3',
                'action_4', 'action_5', 'action_6']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions[['user_id','sku_id']], df], axis=1)
        actions = actions.groupby(['user_id','sku_id'], as_index=False).sum()

        actions = actions[features]
        print ('get_ui_action {},{} done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

# 衰减系数
def get_decayed_actions(start_date, end_date):
    dump_path = './cache/action_decayed_%s_%s.pkl' % (start_date, end_date)
    #dump_path = './cache/action_decayed_1_2016-01-31_2016-04-11.pkl'
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame
        #近期行为按时间衰减
        ##%%
        actions['date'] = pd.to_datetime(actions['time'])
        end =  datetime.strptime(end_date, '%Y-%m-%d')
        actions['days_togo'] = [(end-x).total_seconds()/86400 for x in actions['date']]
        actions['weights'] = actions['days_togo'].map(lambda x: 0.95**x)
#
#        actions['weights'] = actions['days_togo'].map(lambda x: 0.7**x)
#
        actions['action_1'] = actions['action_1'] * actions['weights']
        actions['action_2'] = actions['action_2'] * actions['weights']
        actions['action_3'] = actions['action_3'] * actions['weights']
        actions['action_4'] = actions['action_4'] * actions['weights']
        actions['action_5'] = actions['action_5'] * actions['weights']
        actions['action_6'] = actions['action_6'] * actions['weights']
        del actions['model_id']
        del actions['type']
        del actions['weights']
        del actions['cate']
        del actions['brand']
        del actions['date']

        print ('get_decayed_action {},{} done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#ui行为计数（衰减）
def get_accumulate_decayed_actions(start_date, end_date, actions):
    actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    del actions['time']
    del actions['days_togo']
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    print ('get_accumulate_decayed_action {},{} done.'.format(start_date, end_date))
    return actions

#ui,user,sku最近活跃时间
def get_days_togo(start_date, end_date, actions):
    actions = actions[['user_id','sku_id', 'days_togo']]
    actions_ui = actions.groupby(['user_id', 'sku_id'], as_index=False).agg(min)

    actions_user =  actions[['user_id','days_togo']]
    actions_user = actions_user.groupby(['user_id'], as_index=False).agg(min)
    actions_user = actions_user.rename(columns={'days_togo':'days_togo_user'})

    actions_product =  actions[['sku_id','days_togo']]
    actions_product = actions_product.groupby(['sku_id'], as_index=False).agg(min)
    actions_product = actions_product.rename(columns={'days_togo':'days_togo_product'})

    actions= pd.merge(actions_ui, actions_user, on='user_id')
    actions = pd.merge(actions, actions_product, on='sku_id')
    print ('get_days_togo {},{} done.'.format(start_date, end_date))
    return actions

#user行为计数（衰减）
def get_accumulate_decayed_users(start_date, end_date, actions):
    actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    del actions['time']
    del actions['sku_id']
    del actions['days_togo']
    actions = actions.groupby(['user_id'], as_index=False).sum()
    print ('get_acumulate_decayed_users{},{} done.'.format(start_date, end_date))
    return actions

#sku行为计数（衰减）
def get_accumulate_decayed_product(start_date, end_date, actions):
    actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    del actions['time']
    del actions['user_id']
    del actions['days_togo']
    print ('get_accumulate_decayed_product {},{} done.'.format(start_date, end_date))
    actions = actions.groupby(['sku_id'], as_index=False).sum()

    return actions
#%%
#user基本
def get_basic_user_feat(start_date, end_date):
    dump_path = './cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pd.read_pickle(dump_path)
    else:
        user = pd.read_csv(user_path, encoding='gb18030')
        user['age'] = user['age'].map(_convert_age)
        user['user_reg_tm'] = pd.to_datetime(user['user_reg_tm'])
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        user['account_age'] = user['user_reg_tm'].map(lambda x: (end_date-x).days/30 if not pd.isnull(x) else None)
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user = pd.concat([user[['user_id','age', 'account_age','user_lv_cd']], sex_df], axis=1)
        user['user_activity'] = user['account_age'] / user['user_lv_cd']

        print ('get_basic_user_feat({},{}) done.'.format(start_date, end_date))
        print (user.head(3))
        pickle.dump(user, open(dump_path, 'wb'))
    return user


#user行为占比
def get_accumulate_user_feat(start_date, end_date):
    dump_path = './cache/user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_ui_actions_sum(start_date, end_date)
        del actions['sku_id']
        actions = actions.groupby(['user_id'], as_index=False).sum()
        action_cols = {'action_1':'user_action_1', 'action_2':'user_action_2', 'action_3':'user_action_3',
                       'action_4':'user_action_4', 'action_5':'user_action_5', 'action_6':'user_action_6'}
        actions = actions.rename(columns=action_cols)
        actions['user_action_all'] = actions[list(action_cols.values())].apply(sum, axis=1)
        actions['user_action_1_ratio'] = actions['user_action_1'] / actions['user_action_all']
        actions['user_action_2_ratio'] = actions['user_action_2'] / actions['user_action_all']
        actions['user_action_3_ratio'] = actions['user_action_3'] / actions['user_action_all']
        actions['user_action_4_ratio'] = actions['user_action_4'] / actions['user_action_all']
        actions['user_action_5_ratio'] = actions['user_action_5'] / actions['user_action_all']
        actions['user_action_6_ratio'] = actions['user_action_6'] / actions['user_action_all']

        actions['user_o2browsing'] = actions['user_action_4'] / actions['user_action_1']
        actions['user_o2addcart'] = actions['user_action_4'] / actions['user_action_2']
        actions['user_o2favor'] = actions['user_action_4'] / actions['user_action_5']
        actions['user_o2click'] = actions['user_action_4'] / actions['user_action_6']

        actions.replace(np.inf, -1, inplace=True)
        print ('get_accumulate_user_feat({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

### user行为占比（cate=8）
def get_accumulate_user_cate_feat(start_date, end_date):
    dump_path = './cache/user_cate_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate']==8]
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        action_cols = {'action_1':'user_action_1_cate', 'action_2':'user_action_2_cate', 'action_3':'user_action_3_cate',
                       'action_4':'user_action_4_cate', 'action_5':'user_action_5_cate', 'action_6':'user_action_6_cate'}
        actions = actions.rename(columns=action_cols)
        actions['user_action_all_cate'] = actions[list(action_cols.values())].apply(sum, axis=1)
        actions['user_action_1_cate_ratio'] = actions['user_action_1_cate'] / actions['user_action_all_cate']
        actions['user_action_2_cate_ratio'] = actions['user_action_2_cate'] / actions['user_action_all_cate']
        actions['user_action_3_cate_ratio'] = actions['user_action_3_cate'] / actions['user_action_all_cate']
        actions['user_action_4_cate_ratio'] = actions['user_action_4_cate'] / actions['user_action_all_cate']
        actions['user_action_5_cate_ratio'] = actions['user_action_5_cate'] / actions['user_action_all_cate']
        actions['user_action_6_cate_ratio'] = actions['user_action_6_cate'] / actions['user_action_all_cate']

        actions['user_o2browsing_cate'] = actions['user_action_4_cate'] / actions['user_action_1_cate']
        actions['user_o2addcart_cate'] = actions['user_action_4_cate'] / actions['user_action_2_cate']
        actions['user_o2favor_cate'] = actions['user_action_4_cate'] / actions['user_action_5_cate']
        actions['user_o2click_cate'] = actions['user_action_4_cate'] / actions['user_action_6_cate']
        actions.replace(np.inf, -1, inplace=True)

        print ('get_accumulate_user_cate_feat({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


# user小时占比
def get_accumulate_user_hour_ratio(start_date, end_date):
    dump_path = './cache/user_hour_ratio_accumulate_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id', 'hour_00_ratio', 'hour_01_ratio', 'hour_02_ratio', 'hour_03_ratio', 'hour_04_ratio',
                 'hour_05_ratio', 'hour_06_ratio', 'hour_07_ratio', 'hour_08_ratio', 'hour_09_ratio',
                 'hour_10_ratio', 'hour_11_ratio', 'hour_12_ratio', 'hour_13_ratio', 'hour_14_ratio',
                 'hour_15_ratio', 'hour_16_ratio', 'hour_17_ratio', 'hour_18_ratio', 'hour_19_ratio',
                 'hour_20_ratio', 'hour_21_ratio', 'hour_22_ratio', 'hour_23_ratio']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        user_hour_acc = pd.DataFrame(actions['user_id'], index=actions.index)
        user_hour_acc['hour'] = [item[11:13] for item in actions['time']] #提取小时时间戳
        hour_dummy = pd.get_dummies(user_hour_acc['hour'], prefix='hour')
        user_hour_acc = pd.concat([user_hour_acc['user_id'], hour_dummy], axis=1)
        user_hour_acc['all_action'] = 1
        user_hour_acc = user_hour_acc.groupby('user_id', as_index=False).sum()
        hour_cols = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04',
                     'hour_05', 'hour_06', 'hour_07', 'hour_08', 'hour_09',
                     'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14',
                     'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19',
                     'hour_20', 'hour_21', 'hour_22', 'hour_23']
        for col in hour_cols:
            user_hour_acc[col+'_ratio'] = user_hour_acc[col]/user_hour_acc['all_action']

        actions = user_hour_acc[features]


        print ('get_accumulate_user_hour_ratio({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        actions.to_pickle(dump_path)
    return actions

#user星期占比
def get_accumulate_user_week_ratio(start_date, end_date):
    dump_path = './cache/user_week_ratio_accumulate_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id', 'weekday_mon_ratio', 'weekday_tue_ratio','weekday_wed_ratio',
                'weekday_thu_ratio', 'weekday_fri_ratio', 'weekday_sat_ratio', 'weekday_sun_ratio',
                'weekday_ratio', 'weekend_ratio']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        features = ['user_id', 'weekday_mon_ratio', 'weekday_tue_ratio','weekday_wed_ratio',
                    'weekday_thu_ratio', 'weekday_fri_ratio', 'weekday_sat_ratio', 'weekday_sun_ratio',
                    'weekday_ratio', 'weekend_ratio']
        actions = pd.DataFrame(actions[['user_id', 'time']], index=actions.index)
        actions['time'] = pd.to_datetime(actions['time'])
        actions['weekday'] = actions['time'].map(lambda x: x.isoweekday() if not pd.isnull(x) else 0)
        df = pd.get_dummies(actions['weekday'], prefix='weekday')
        actions = pd.concat([actions['user_id'],df], axis=1)
        actions = actions.groupby('user_id', as_index=False).sum()
        action_cols = {'weekday_1':'weekday_mon', 'weekday_2':'weekday_tue', 'weekday_3':'weekday_wed',
                       'weekday_4':'weekday_thu', 'weekday_5':'weekday_fri', 'weekday_6':'weekday_sat',
                       'weekday_7':'weekday_sun'}
        actions = actions.rename(columns=action_cols)
        actions['weekday_all'] = actions[list(action_cols.values())].apply(sum, axis=1)
        actions['weekday_mon_ratio'] = actions['weekday_mon']/actions['weekday_all']
        actions['weekday_tue_ratio'] = actions['weekday_tue']/actions['weekday_all']
        actions['weekday_wed_ratio'] = actions['weekday_wed']/actions['weekday_all']
        actions['weekday_thu_ratio'] = actions['weekday_thu']/actions['weekday_all']
        actions['weekday_fri_ratio'] = actions['weekday_fri']/actions['weekday_all']
        actions['weekday_sat_ratio'] = actions['weekday_sat']/actions['weekday_all']
        actions['weekday_sun_ratio'] = actions['weekday_sun']/actions['weekday_all']

        actions['weekday_ratio'] = actions['weekday_mon_ratio'] + actions['weekday_tue_ratio'] + actions['weekday_wed_ratio']\
                                    + actions['weekday_thu_ratio'] + actions['weekday_fri_ratio']
        actions['weekend_ratio'] = actions['weekday_sat_ratio'] + actions['weekday_sun_ratio']

        actions = actions[features]


        print ('get_accumulate_user_week_ratio({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

# user活跃天数，针对全品类，和单品类
def get_user_session(start_date, end_date):
    dump_path = './cache/user_session_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id', 'sku_id', 'ui_active_days', 'user_active_days', 'user_cate_active_days']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(lambda x: x[:10])
        actions = actions[['user_id','sku_id','time','cate']].drop_duplicates()
        actions_ui = actions.groupby(['user_id', 'sku_id', 'cate'], as_index=False).agg('count')
        actions_ui = actions_ui.rename(columns={'time': 'ui_active_days'})
        del actions_ui['cate']

        actions_user = actions[['user_id','time']].drop_duplicates()
        actions_user = actions_user.groupby('user_id', as_index=False).agg('count')
        actions_user = actions_user.rename(columns={'time': 'user_active_days'})

        actions_user_cate = actions[actions['cate']==8].copy()
        actions_user_cate = actions_user_cate[['user_id','time']].drop_duplicates()
        actions_user_cate = actions_user_cate.groupby('user_id', as_index=False).agg('count')
        actions_user_cate = actions_user_cate.rename(columns={'time': 'user_cate_active_days'})

        actions = pd.merge(actions_ui, actions_user, on='user_id')
        actions = pd.merge(actions, actions_user_cate, on='user_id')

        actions = actions[features]
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


#sku基本
def get_basic_product_feat():
    dump_path = './cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pd.read_pickle(dump_path)
    else:
        actions = _clean_actions()
        actions = actions[(actions['type']==4)&(actions['cate']==8)]
        brand_sales = actions['brand'].value_counts()
        brand_sales = pd.DataFrame({'brand':brand_sales.index, 'brand_sales':brand_sales.values})
        product = pd.read_csv(product_path)
        product = pd.merge(product[['sku_id','brand']], brand_sales, on='brand', how='left')
        product = product.fillna(0)

        del product['brand']
        print ('get_basic_product done.')
        print (product.head(3))
        pickle.dump(product, open(dump_path, 'wb'))
    return product

#sku行为占比
def get_accumulate_product_feat(start_date, end_date):
    dump_path = './cache/product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_ui_actions_sum(start_date, end_date)
        del actions['user_id']
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        action_cols = {'action_1':'product_action_1', 'action_2':'product_action_2', 'action_3':'product_action_3',
                       'action_4':'product_action_4', 'action_5':'product_action_5', 'action_6':'product_action_6'}
        actions = actions.rename(columns=action_cols)
        actions['product_action_all'] = actions[list(action_cols.values())].apply(sum, axis=1)
        actions['product_action_1_ratio'] = actions['product_action_1'] / actions['product_action_all']
        actions['product_action_2_ratio'] = actions['product_action_2'] / actions['product_action_all']
        actions['product_action_3_ratio'] = actions['product_action_3'] / actions['product_action_all']
        actions['product_action_4_ratio'] = actions['product_action_4'] / actions['product_action_all']
        actions['product_action_5_ratio'] = actions['product_action_5'] / actions['product_action_all']
        actions['product_action_6_ratio'] = actions['product_action_6'] / actions['product_action_all']

        actions.replace(np.inf, -1, inplace=True)


        print ('get_accumulate_product_feat({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#sku转换率
def get_product_throughrate(start_date, end_date):
    dump_path = './cache/product_throughrate_%s_%s.pkl' % (start_date, end_date)
    features = ['sku_id', 'product_o2browsing', 'product_o2addcart', 'product_o2favor',
                'product_o2click']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate']==8]
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()

        actions['product_o2browsing'] = actions['action_4'] / actions['action_1']
        actions['product_o2addcart'] = actions['action_4'] / actions['action_2']
        actions['product_o2favor'] = actions['action_4'] / actions['action_5']
        actions['product_o2click'] = actions['action_4'] / actions['action_6']

        actions.replace(np.inf, -1, inplace=True)
        actions = actions[features]


        print ('get_product_throughrate({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#sku人均浏览量
def get_action_product_on_user_avg(start_date, end_date):
    dump_path = './cache/action_product_on_user_avg_%s_%s.pkl' % (start_date, end_date)
    #features = ['sku_id', 'product_action_1_avg', 'product_action_2_avg', 'product_action_3_avg',
    #'product_action_4_avg','product_action_5_avg','product_action_6_avg', 'user_count']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_ui_actions_sum(start_date, end_date)
        user_count = actions.groupby('sku_id', as_index=False)['user_id'].agg(pd.Series.nunique)

        del actions['user_id']
        act = pd.DataFrame()
        cols = ['action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']
        for col in cols:

            act['product_'+ col + '_avg'] = actions.groupby('sku_id')[col].mean()
            act['product_'+ col + '_min'] = actions.groupby('sku_id')[col].min()
            act['product_'+ col + '_max'] = actions.groupby('sku_id')[col].max()
            act['product_'+ col + '_std'] = actions.groupby('sku_id')[col].std()

        actions = act.reset_index()
        actions = pd.merge(actions, user_count, on='sku_id')
        actions = actions.rename(columns={'user_id':'user_count'})
        print ('get_action_product_on_user_avg({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#sku浏览量偏移
def get_action_product_on_user_normal(start_date, end_date):
    dump_path = './cache/action_product_on_user_normal_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id','sku_id','ui_action_1_normal_product','ui_action_2_normal_product',
                'ui_action_3_normal_product','ui_action_4_normal_product','ui_action_5_normal_product',
                'ui_action_6_normal_product']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate']==8]

        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame
        actions = actions.groupby(['user_id','sku_id'], as_index=False).sum()

        product_user_avg = get_action_product_on_user_avg(start_date, end_date)

        actions = pd.merge(actions, product_user_avg, on='sku_id', how='left')
        actions['ui_action_1_normal_product'] = actions['action_1'] - actions['product_action_1_avg']
        actions['ui_action_2_normal_product'] = actions['action_2'] - actions['product_action_2_avg']
        actions['ui_action_3_normal_product'] = actions['action_3'] - actions['product_action_3_avg']
        actions['ui_action_4_normal_product'] = actions['action_4'] - actions['product_action_4_avg']
        actions['ui_action_5_normal_product'] = actions['action_5'] - actions['product_action_5_avg']
        actions['ui_action_6_normal_product'] = actions['action_6'] - actions['product_action_6_avg']
        actions = actions[features]

        print ('get_action_product_on_user_normal({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#user对商品的平均浏览量
def get_action_user_on_product_avg(start_date, end_date):
    dump_path = './cache/action_user_on_product_avg_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id', 'user_action_1_avg', 'user_action_2_avg', 'user_action_3_avg',
    'user_action_4_avg','user_action_5_avg', 'user_action_6_avg', 'product_count','brand_count','prod_each_brand']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_ui_actions_sum(start_date, end_date)

        actions = actions.groupby(['user_id','sku_id'], as_index=False).sum()
        del actions['sku_id']
        act = pd.DataFrame()
        cols = ['action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']
        for col in cols:

            act['user_'+ col + '_avg'] = actions.groupby('user_id')[col].mean()
            act['user_'+ col + '_min'] = actions.groupby('user_id')[col].min()
            act['user_'+ col + '_max'] = actions.groupby('user_id')[col].max()
            act['user_'+ col + '_std'] = actions.groupby('user_id')[col].std()

        actions = act.reset_index()

        brand_actions = get_actions(start_date,end_date)
        brand_actions = brand_actions.drop_duplicates(subset=['user_id','brand','sku_id'])
        brand_actions = brand_actions.groupby(['user_id','brand'], as_index=False)['sku_id'].count()

        product_count = brand_actions.groupby('user_id', as_index=False)['sku_id'].sum()
        brand_count = brand_actions.groupby('user_id', as_index=False)['brand'].count()


        actions = pd.merge(actions, product_count, on='user_id')
        actions = pd.merge(actions, brand_count, on='user_id')
        actions = actions.rename(columns={'sku_id':'product_count','brand':'brand_count'})


        act = pd.DataFrame()
        act['sku_brand_mean'] = brand_actions.groupby('user_id')['sku_id'].mean()
        act['sku_brand_max'] = brand_actions.groupby('user_id')['sku_id'].max()
        act['sku_brand_min'] = brand_actions.groupby('user_id')['sku_id'].min()
        act['sku_brand_std'] = brand_actions.groupby('user_id')['sku_id'].std()
        act = act.reset_index()

        actions = pd.merge(actions, act, on='user_id')

        actions.replace(np.inf, 0, inplace=True)
        print ('get_action_user_on_product_avg({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#user对商品的浏览量偏移
def get_action_user_on_product_normal(start_date, end_date):
    dump_path = './cache/action_user_on_product_normal_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id','sku_id','ui_action_1_normal_user','ui_action_2_normal_user',
                'ui_action_3_normal_user','ui_action_4_normal_user','ui_action_5_normal_user',
                'ui_action_6_normal_user','ui_action_1_normal_ratio_user','ui_action_2_normal_ratio_user',
                'ui_action_3_normal_ratio_user','ui_action_4_normal_ratio_user','ui_action_5_normal_ratio_user',
                'ui_action_6_normal_ratio_user']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate']==8]

        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame
        actions = actions.groupby(['user_id','sku_id'], as_index=False).sum()

        user_on_product_avg = get_action_user_on_product_avg(start_date, end_date)

        actions = pd.merge(actions, user_on_product_avg, on='user_id', how='left')
        actions['ui_action_1_normal_user'] = actions['action_1'] - actions['user_action_1_avg']
        actions['ui_action_2_normal_user'] = actions['action_2'] - actions['user_action_2_avg']
        actions['ui_action_3_normal_user'] = actions['action_3'] - actions['user_action_3_avg']
        actions['ui_action_4_normal_user'] = actions['action_4'] - actions['user_action_4_avg']
        actions['ui_action_5_normal_user'] = actions['action_5'] - actions['user_action_5_avg']
        actions['ui_action_6_normal_user'] = actions['action_6'] - actions['user_action_6_avg']

        actions['ui_action_1_normal_ratio_user'] = actions['action_1'] / actions['user_action_1_avg']
        actions['ui_action_2_normal_ratio_user'] = actions['action_2'] / actions['user_action_2_avg']
        actions['ui_action_3_normal_ratio_user'] = actions['action_3'] / actions['user_action_3_avg']
        actions['ui_action_4_normal_ratio_user'] = actions['action_4'] / actions['user_action_4_avg']
        actions['ui_action_5_normal_ratio_user'] = actions['action_5'] / actions['user_action_5_avg']
        actions['ui_action_6_normal_ratio_user'] = actions['action_6'] / actions['user_action_6_avg']

        actions = actions.replace([np.inf], 0)
        actions = actions[features]

        print ('get_action_user_on_product_normal({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#重复购买
def get_action_repeat_buyer(start_date, end_date):
    dump_path = './cache/action_repeat_buyer_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id', 'sku_id', 'repeat_buynum']

    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[(actions['type']==4) & (actions['cate']==8)]
        actions = actions.drop_duplicates()
        duplicated = actions.duplicated(subset=['user_id','sku_id'], keep=False) #set keep to mark all duplicates
        actions = actions[duplicated]
        actions = actions[['user_id','sku_id']]
        actions['repeat_buynum'] = 1
        actions = actions.groupby(['user_id','sku_id'], as_index=False).agg('count')
        actions['repeat_buynum'] = actions['repeat_buynum']-1
        actions = actions[features]
        print ('get_action_repeat_buyer({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#ui对商品行为数除以该用户总行为数
def get_action_ratio_user(start_date, end_date):
    dump_path = './cache/action_ratio_user_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id', 'sku_id', 'ui_action_1_ratio', 'ui_action_2_ratio', 'ui_action_3_ratio',
    'ui_action_4_ratio','ui_action_5_ratio', 'ui_action_6_ratio', 'ui_action_1', 'ui_action_2', 'ui_action_3',
    'ui_action_4', 'ui_action_5', 'ui_action_6', 'ui_action_all', 'ui_rank']
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        cols = {'action_1':'ui_action_1', 'action_2':'ui_action_2', 'action_3':'ui_action_3',
        'action_4':'ui_action_4', 'action_5':'ui_action_5', 'action_6':'ui_action_6'}
        ui_action = get_ui_actions_sum(start_date, end_date)
        ui_action = ui_action.rename(columns=cols)

        ui_rank = ui_action[['user_id','sku_id']]
        ui_rank['ui_action_all'] = ui_action[['ui_action_1','ui_action_2','ui_action_3','ui_action_4','ui_action_5','ui_action_6']].apply(sum, axis=1)
        ui_rank['ui_rank'] = ui_rank.groupby('user_id')['ui_action_all'].rank(method='max')
        user_action = ui_action.copy()
        del user_action['sku_id']
        cols = {'ui_action_1':'action_1_all', 'ui_action_2':'action_2_all', 'ui_action_3':'action_3_all',
                'ui_action_4':'action_4_all', 'ui_action_5':'action_5_all', 'ui_action_6':'action_6_all'}
        user_action = user_action.groupby('user_id', as_index=False).sum()
        user_action = user_action.rename(columns=cols)

        actions = pd.merge(ui_action, user_action, on='user_id')
        actions = pd.merge(actions, ui_rank, on=['user_id','sku_id'])
        actions['ui_action_1_ratio'] = actions['ui_action_1'] / actions['action_1_all']
        actions['ui_action_2_ratio'] = actions['ui_action_2'] / actions['action_2_all']
        actions['ui_action_3_ratio'] = actions['ui_action_3'] / actions['action_3_all']
        actions['ui_action_4_ratio'] = actions['ui_action_4'] / actions['action_4_all']
        actions['ui_action_5_ratio'] = actions['ui_action_5'] / actions['action_5_all']
        actions['ui_action_6_ratio'] = actions['ui_action_6'] / actions['action_6_all']

        actions.replace(np.inf, 0, inplace=True)
        actions = actions[features]

        print ('get_action_ratio_user({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#user最（次）活跃天数和sku_id最（次）活跃天数距离预测日距离
def get_user_most_active_days(start_date, end_date):
    dump_path = './cache/action_user_active_days_%s_%s.pkl' % (start_date, end_date)
    features = ['user_id', 'user_first_peak_time', 'user_first_peak_action','user_second_peak_time', 'user_second_peak_action']

    if os.path.exists(dump_path):
        user = pd.read_pickle(dump_path)
    else:

        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(lambda x: x[:10])
        actions = actions[['user_id','sku_id','time']]
        actions['peak_action'] = 1
        actions = actions.groupby(['user_id','sku_id','time'], as_index=False).sum()

        #user
        user = actions[['user_id','time','peak_action']]
        user = user.groupby(['user_id','time'], as_index=False).sum()
        user = user.sort_values(['user_id','peak_action'], ascending=False)

        user_first_peak = user.drop_duplicates(subset='user_id', keep='first').index
        user_tmp = user.drop(user_first_peak)
        user_second_peak = user_tmp.drop_duplicates(subset='user_id', keep='first').index

        user_first = user.loc[user_first_peak]
        end_time = datetime.strptime(end_date, '%Y-%m-%d')
        user_first['time'] = pd.to_datetime(user['time'])
        user_first['time'] = user_first['time'].map(lambda x: (end_time-x).days)
        user_first = user_first.rename(columns = {'peak_action':'user_first_peak_action', 'time':'user_first_peak_time'})

        user_second = user.loc[user_second_peak]
        user_second['time'] = pd.to_datetime(user['time'])
        user_second['time'] = user_second['time'].map(lambda x: (end_time-x).days)
        user_second = user_second.rename(columns = {'peak_action':'user_second_peak_action', 'time':'user_second_peak_time'})

        user = pd.merge(user_first, user_second, on='user_id', how='left')
        user = user.fillna(9999)
        user = user[features]

        print ('get_user_most_active_days({},{}) done.'.format(start_date, end_date))
        print (user.head(3))
        pickle.dump(user, open(dump_path, 'wb'))
    return user


def get_sku_most_active_days(start_date, end_date):
    dump_path = './cache/action_sku_active_days_%s_%s.pkl' % (start_date, end_date)
    features = ['sku_id', 'sku_peak_time', 'sku_peak_action']

    if os.path.exists(dump_path):
        sku = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(lambda x: x[:10])
        actions = actions[['user_id','sku_id','time']]
        actions['peak_action'] = 1
        actions = actions.groupby(['user_id','sku_id','time'], as_index=False).sum()
        #sku
        sku = actions[['sku_id','time','peak_action']]
        sku = sku.groupby(['sku_id','time'], as_index=False).sum()
        sku = sku.sort_values(['sku_id','peak_action'], ascending=False)
        sku = sku.drop_duplicates(subset='sku_id', keep='first')

        end_time = datetime.strptime(end_date, '%Y-%m-%d')
        sku['time'] = pd.to_datetime(sku['time'])
        sku['time'] = sku['time'].map(lambda x: (end_time-x).days)
        sku = sku.rename(columns = {'peak_action':'sku_peak_action', 'time':'sku_peak_time'})
        sku = sku[features]
        print ('get_sku_most_active_days({},{}) done.'.format(start_date, end_date))
        print (sku.head(3))
        pickle.dump(sku, open(dump_path, 'wb'))
    return sku

#user浏览/点击/加购/收藏到购买的（平均）时间差
def get_user_timegap(start_date, end_date):
    dump_path = './cache/user_timegap_%s_%s.pkl' % (start_date, end_date)

    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        full_ui_list = actions[['user_id','sku_id']].drop_duplicates()
        ui_list = actions[actions['type']==4][['user_id','sku_id']].drop_duplicates()
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions[['user_id','sku_id','time']], df], axis=1)
        actions['time'] = pd.to_datetime(actions['time'])

        actions = pd.merge(actions, ui_list, on=['user_id', 'sku_id'], how='inner')

        action_1 = actions[actions['action_1']==1]
        action_2 = actions[actions['action_2']==1]
        action_4 = actions[actions['action_4']==1][['user_id','sku_id','time']]
        action_4_first = action_4.sort_values('time', ascending=True).drop_duplicates(subset=['user_id','sku_id'], keep='first').rename(columns={'time':'buy_time_first'})
        action_4_last = action_4.sort_values('time', ascending=False).drop_duplicates(subset=['user_id','sku_id'], keep='first').rename(columns={'time':'buy_time_last'})
        action_4 = pd.merge(action_4_first, action_4_last, on=['user_id','sku_id'])

        action_5 = actions[actions['action_5']==1]
        action_6 = actions[actions['action_6']==1]

        action_1_before = action_1.groupby(['user_id','sku_id'], as_index=False)['time'].min().rename(columns={'time':'action_1_before_time'})
        action_1_after = action_1.groupby(['user_id','sku_id'], as_index=False)['time'].max().rename(columns={'time':'action_1_after_time'})
        action_2_before = action_2.groupby(['user_id','sku_id'], as_index=False)['time'].min().rename(columns={'time':'action_2_before_time'})
        action_2_after = action_2.groupby(['user_id','sku_id'], as_index=False)['time'].max().rename(columns={'time':'action_2_after_time'})


        action_5_before = action_5.groupby(['user_id','sku_id'], as_index=False)['time'].min().rename(columns={'time':'action_5_before_time'})
        action_5_after = action_5.groupby(['user_id','sku_id'], as_index=False)['time'].max().rename(columns={'time':'action_5_after_time'})
        action_6_before = action_6.groupby(['user_id','sku_id'], as_index=False)['time'].min().rename(columns={'time':'action_6_before_time'})
        action_6_after = action_6.groupby(['user_id','sku_id'], as_index=False)['time'].max().rename(columns={'time':'action_6_after_time'})

        action = None
        l = [action_1_before, action_1_after, action_2_before, action_2_after, action_5_before, action_5_after, action_6_before, action_6_after]
        for df in l:
            if action is None:
                action = pd.merge(action_4, df, on=['user_id','sku_id'], how='left')
            else:
                action = pd.merge(action, df, on=['user_id','sku_id'], how='left')

        action['browsing_to_buy'] = action['buy_time_first']-action['action_1_before_time']
        action['buy_to_browsing'] = action['action_1_after_time'] - action['buy_time_last']
        action['addcart_to_buy'] = action['buy_time_first']-action['action_2_before_time']
        action['buy_to_addcart'] = action['action_2_after_time'] - action['buy_time_last']
        action['favor_to_buy'] = action['buy_time_first']-action['action_5_before_time']
        action['buy_to_favor'] = action['action_5_after_time'] - action['buy_time_last']
        action['click_to_buy'] = action['buy_time_first']-action['action_6_before_time']
        action['buy_to_click'] = action['action_6_after_time'] - action['buy_time_last']

        features = ['browsing_to_buy', 'buy_to_browsing', 'addcart_to_buy',
                    'buy_to_addcart', 'favor_to_buy', 'buy_to_favor', 'click_to_buy',
                    'buy_to_click']
        action[features] = action[features].applymap(lambda x: x.total_seconds()/(86400))

        df = action[features].applymap(lambda x: None if x<0 else x)
        df = pd.concat([action[['user_id','sku_id']], df], axis=1)
        df = pd.merge(full_ui_list, df, on=['user_id','sku_id'], how='left')
        df = df.fillna(9999)

        df2 = pd.concat([action['user_id'], action[features]], axis=1)
        df2[features] = df2[features].applymap(lambda x: None if x<0 else x)
        df2 = df2.groupby('user_id', as_index=False).mean()
        df2 = df2.fillna(9999)

        actions = pd.merge(df,df2,on='user_id', how='left', suffixes=('', '_mean'))
        actions = actions.fillna(9999)
        print ('get_user_timegap({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


#user浏览/点击/收藏到加购的时间差
def get_user_timegap_addcart(start_date, end_date):
    dump_path = './cache/user_timegap_addcart_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        full_ui_list = actions[['user_id','sku_id']].drop_duplicates()
        ui_list = actions[actions['type']==2][['user_id','sku_id']].drop_duplicates()
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions[['user_id','sku_id','time']], df], axis=1)
        actions['time'] = pd.to_datetime(actions['time'])

        actions = pd.merge(actions, ui_list, on=['user_id', 'sku_id'], how='inner')

        action_1 = actions[actions['action_1']==1]

        action_2 = actions[actions['action_2']==1][['user_id','sku_id','time']]
        action_2_first = action_2.sort_values('time', ascending=True).drop_duplicates(subset=['user_id','sku_id'], keep='first').rename(columns={'time':'addcart_time_first'})
        action_2_last = action_2.sort_values('time', ascending=False).drop_duplicates(subset=['user_id','sku_id'], keep='first').rename(columns={'time':'addcart_time_last'})
        action_2 = pd.merge(action_2_first, action_2_last, on=['user_id','sku_id'])

        action_5 = actions[actions['action_5']==1]
        action_6 = actions[actions['action_6']==1]

        action_1_before = action_1.groupby(['user_id','sku_id'], as_index=False)['time'].min().rename(columns={'time':'action_1_before_time'})
        action_1_after = action_1.groupby(['user_id','sku_id'], as_index=False)['time'].max().rename(columns={'time':'action_1_after_time'})


        action_5_before = action_5.groupby(['user_id','sku_id'], as_index=False)['time'].min().rename(columns={'time':'action_5_before_time'})
        action_5_after = action_5.groupby(['user_id','sku_id'], as_index=False)['time'].max().rename(columns={'time':'action_5_after_time'})
        action_6_before = action_6.groupby(['user_id','sku_id'], as_index=False)['time'].min().rename(columns={'time':'action_6_before_time'})
        action_6_after = action_6.groupby(['user_id','sku_id'], as_index=False)['time'].max().rename(columns={'time':'action_6_after_time'})


        action = None
        l = [action_1_before, action_1_after, action_5_before, action_5_after, action_6_before, action_6_after]
        for df in l:
            if action is None:
                action = pd.merge(action_2, df, on=['user_id','sku_id'], how='left')
            else:
                action = pd.merge(action, df, on=['user_id','sku_id'], how='left')

        action['browsing_to_addcart'] = action['addcart_time_first']-action['action_1_before_time']
        action['addcart_to_browsing'] = action['action_1_after_time'] - action['addcart_time_last']
        action['favor_to_addcart'] = action['addcart_time_first']-action['action_5_before_time']
        action['addcart_to_favor'] = action['action_5_after_time'] - action['addcart_time_last']
        action['click_to_addcart'] = action['addcart_time_first']-action['action_6_before_time']
        action['addcart_to_click'] = action['action_6_after_time'] - action['addcart_time_last']

        features = ['browsing_to_addcart', 'addcart_to_browsing', 'favor_to_addcart', 'addcart_to_favor', 'click_to_addcart',
                    'addcart_to_click']
        action[features] = action[features].applymap(lambda x: x.total_seconds()/(86400))

        df = action[features].applymap(lambda x: None if x<0 else x)
        df = pd.concat([action[['user_id','sku_id']], df], axis=1)
        df = pd.merge(full_ui_list, df, on=['user_id','sku_id'], how='left')
        df = df.fillna(9999)

        df2 = pd.concat([action['user_id'], action[features]], axis=1)
        df2[features] = df2[features].applymap(lambda x: None if x<0 else x)
        df_mean = df2.groupby('user_id', as_index=False).mean()
        df_std = df2.groupby('user_id', as_index=False).std()
        df_min = df2.groupby('user_id', as_index=False).min()
        df_max = df2.groupby('user_id', as_index=False).max()

        actions = pd.merge(df,df_mean,on='user_id', how='left', suffixes=('', '_mean'))
        actions = pd.merge(actions,df_std,on='user_id', how='left', suffixes=('', '_std'))
        actions = pd.merge(actions,df_min,on='user_id', how='left', suffixes=('', '_min'))
        actions = pd.merge(actions,df_max,on='user_id', how='left', suffixes=('', '_max'))
        actions = actions.fillna(9999)
        print ('get_user_timegap_addcart({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#点击modelid比例
def get_click_modelid(start_date, end_date):
    dump_path = './cache/action_click_modelid_%s_%s.pkl' % (start_date, end_date)

    if os.path.exists(dump_path):
        modelid_df = pd.read_pickle(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type']==6].reset_index(drop=True)
#        modelid = actions['model_id'].value_counts()
#        modelid = modelid[modelid>=5000].index
        modelid_df = pd.DataFrame(index=actions.index)
        max_num = 20
        modelid_col = []
        cols = []

        for i,mid in enumerate(modelid):
            if i<max_num:
                print ('start {}'.format(mid))
                col = 'modelid/'+str(mid)
                modelid_col.append(mid)
                cols.append(col)
                modelid_df[col] = [1 if item==mid else 0 for item in actions['model_id']]
            else:
                cols.append('modelid/other')
                modelid_df['modelid/other'] = [1 if item not in modelid_col else 0 for item in actions['model_id']]
                break

        modelid_df['modelid/all'] = 1
        modelid_df = pd.concat([actions[['user_id','sku_id']], modelid_df], axis=1)
        modelid_df = modelid_df.groupby(['user_id','sku_id'], as_index=False).sum()

        for col in cols:
            modelid_df[col] = modelid_df[col]/modelid_df['modelid/all']


        print ('get_click_modelid({},{}) done.'.format(start_date, end_date))
        print (modelid_df.head(3))
        pickle.dump(modelid_df, open(dump_path, 'wb'))
    return modelid_df

#最近N天新用户flag
def get_new_user_tag(start_date, end_date, day_num=10):
    dump_path = './cache/action_user_new_tag_%s_%s.pkl' % (start_date, end_date)

    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        start_time = datetime.strptime(start_date, '%Y-%m-%d')
        end_time = datetime.strptime(end_date, '%Y-%m-%d')
        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(lambda x: x[:10])
        actions['time'] = pd.to_datetime(actions['time'])
        actions = actions[['user_id','sku_id','time']].drop_duplicates()
        actions_user = actions[['user_id','time']].drop_duplicates()

        user_tag = None
        for i in range(1,day_num):
            dead_date = end_time-timedelta(days=i)
            user_before = set(actions_user[actions_user['time']<dead_date]['user_id'])
            user_after = set(actions_user[actions_user['time']>=dead_date]['user_id'])

            new_user = user_after - user_before

            if user_tag is None:
                user_tag = pd.DataFrame({'user_id':list(new_user), 'new_user':1})
            else:
                user_tag_new = pd.DataFrame({'user_id':list(new_user), 'new_user':1})
                user_tag = pd.merge(user_tag, user_tag_new, on='user_id', how='outer', suffixes=('','_'+str(i)+'_days_before'))

        ui_tag = None
        for i in range(1,day_num):
            dead_date = end_time-timedelta(days=i)
            ui_before = set([(x,y) for x,y in zip(actions[actions['time']<dead_date]['user_id'], actions[actions['time']<dead_date]['sku_id'])])
            ui_after = set([(x,y) for x,y in zip(actions[actions['time']>=dead_date]['user_id'], actions[actions['time']>=dead_date]['sku_id'])])
            new_ui = ui_after - ui_before

            user_id, sku_id = [],[]
            for item in new_ui:
                user_id.append(item[0])
                sku_id.append(item[1])
            if ui_tag is None:
                ui_tag = pd.DataFrame({'user_id':user_id,'sku_id':sku_id,'new_ui':1})
            else:
                ui_tag_new = pd.DataFrame({'user_id':user_id,'sku_id':sku_id,'new_ui':1})
                ui_tag = pd.merge(ui_tag, ui_tag_new, on=['user_id','sku_id'], how='outer', suffixes=('','_'+str(i)+'_days_before'))

        actions = pd.merge(ui_tag, user_tag, on='user_id', how='left')
        print ('get_new_user_tags({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

#用户发生的所有购买时间距离现在的天数的平均值以及方差
def user_buy_time_statistic(start_date,end_date):
    dump_path = './cache_jp/user_buy_time_statistic_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:

        actions = get_actions(start_date,end_date)
        actions = actions[actions.type==4]
        actions['time'] = (pd.to_datetime(actions['time']) - pd.to_datetime(end_date)).apply(lambda x:x.days)
        act = pd.DataFrame()
        act['aver_time'] = actions.groupby(['user_id'])['time'].mean()
        act['var_time'] = actions.groupby(['user_id'])['time'].std()
        act['min_time'] = actions.groupby(['user_id'])['time'].min()
        act['max_time'] = actions.groupby(['user_id'])['time'].max()
        actions = act.reset_index()
        pickle.dump(actions, open(dump_path, 'wb'))

    return actions


#基本comment
def get_comments_product_feat(start_date, end_date):
    dump_path = './cache/comments_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = pd.read_pickle(dump_path)
    else:
        comments = pd.read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]

        comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num']]

        print ('get_comments_product_feat({},{}) done.'.format(start_date, end_date))
        print (comments.head(3))
        pickle.dump(comments, open(dump_path, 'wb'))
    return comments

#标签
def get_labels(start_date, end_date):
    dump_path = './cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_pickle(dump_path)
    else:
        product = get_basic_product_feat()
        product = product['sku_id'].to_frame()
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        actions = pd.merge(actions, product, on='sku_id', how='inner') #筛选出cate=8的ui-label

        print ('get_labels({},{}) done.'.format(start_date, end_date))
        print (actions.head(3))
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions




#%%
if __name__ == "__main__":

    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'
    users, actions, labels, user_users, user_actions, user_labels = \
    make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)


#%%
start_date = '2016-03-10'
end_date = '2016-04-11'


#%%


