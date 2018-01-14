# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:27:07 2018

@author: Leo Mao
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import xgboost as xgb
import datetime
from sklearn.linear_model import LassoLars,LinearRegression




def xgb_kfold(dfTrain,dfPred,predictors,n_splits=5,weight = 1,early_stop = 10,ins_rmse = 0,params = {'max_depth':3, 'eta':0.01, 'silent':0,'objective':'reg:linear','lambda':1,'subsample':0.8,
                         'colsample_bytree':0.8}):  
    kf = KFold(n_splits=n_splits,shuffle=True)
    dpred = xgb.DMatrix(dfPred[predictors].values,label=[0]*len(dfPred),missing=np.nan)
    imp = pd.DataFrame({'variable':predictors,'lk':['f'+str(i) for i in range(len(predictors))]})
    round=0
    if weight>1:
        dfTrain['wgt'] = 1+(weight-1)*(np.abs(dfTrain['Y'] - dfTrain['Y'].quantile(0.5))/np.abs(dfTrain['Y'] - dfTrain['Y'].quantile(0.5)).max())
    else:
        dfTrain['wgt'] = 1
    for train_index, test_index in kf.split(dfTrain):
        round+=1
        train_X = dfTrain.loc[train_index,predictors]
        test_X = dfTrain.loc[test_index,predictors]
        train_Y = dfTrain.loc[train_index,'Y']
        test_Y = dfTrain.loc[test_index,'Y']
        train_wgt = dfTrain.loc[train_index,'wgt']
        test_wgt = dfTrain.loc[test_index,'wgt']

        dtrain = xgb.DMatrix(train_X.values, label=train_Y.values,weight=train_wgt, missing = np.nan)
        dtest = xgb.DMatrix(test_X.values, label=test_Y.values,weight=test_wgt, missing = np.nan)
        param = params 
        evallist  = [(dtrain,'train'),(dtest,'eval')]  
        num_round = 5000
        evals_dict = {}
        if 'eval_metric' in params:
            metric = params['eval_metric']
        else:
            metric ='rmse'
        model = xgb.train(param,dtrain,num_round, evallist,early_stopping_rounds=early_stop,evals_result=evals_dict,verbose_eval =100)
        performance_df = pd.DataFrame({'train':evals_dict['train'][metric],'eval':evals_dict['eval'][metric]})
        performance_df =performance_df.loc[performance_df['train']>=ins_rmse]
        #bst_tree = len(performance_df)-1-early_stop
        bst_tree = performance_df.loc[performance_df['eval']==performance_df['eval'].min()].index.tolist()[0] + 1
        print('Best tree is %d, performance is %f, %f'%(bst_tree,performance_df.loc[bst_tree-1,'train'],performance_df.loc[bst_tree-1,'eval']))
        pred_test = model.predict(dtest,ntree_limit =bst_tree)

        tmp_imp = pd.DataFrame(model.get_score(importance_type='gain'),index=['imp_fold%d'%round]).T
        tmp_imp['lk'] = tmp_imp.index
        imp = imp.merge(tmp_imp,'left','lk').fillna(0)


        pred_score = model.predict(dpred,ntree_limit =bst_tree)
        if round==1:
            test_result = pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})
            result = pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score})
        else:
            test_result = pd.concat([test_result,pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})],axis=0)
            result = result.merge(pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score}),'inner','ID')
    print("Test MSE:",metrics.mean_squared_error(test_result['target'], test_result['score']))
    return test_result,result,imp



def rf_kfold(dfTrain,dfPred,predictors,n_splits=5,num_round = 5000):  
    kf = KFold(n_splits=n_splits,shuffle=True)
    imp = pd.DataFrame({'variable':predictors,'lk':['f'+str(i) for i in range(len(predictors))]})
    round=0
    for train_index, test_index in kf.split(dfTrain):
        round+=1
        train_X = dfTrain.loc[train_index,predictors]
        test_X = dfTrain.loc[test_index,predictors]
        train_Y = dfTrain.loc[train_index,'Y']
        test_Y = dfTrain.loc[test_index,'Y']

        
        model = RandomForestRegressor(n_estimators=num_round, max_features='sqrt',  max_depth=4, random_state=202)
        model.fit(train_X,train_Y)
        
        pred_test = model.predict(test_X)

        imp['imp_fold%d'%round] = model.feature_importances_

        pred_score = model.predict(dfPred[predictors].values)
        if round==1:
            test_result = pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})
            result = pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score})
        else:
            test_result = pd.concat([test_result,pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})],axis=0)
            result = result.merge(pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score}),'inner','ID')
    print("Test MSE:",metrics.mean_squared_error(test_result['target'], test_result['score']))
    return test_result,result,imp


def linear_kfold(dfTrain,dfPred,predictors,n_splits=5):  
    kf = KFold(n_splits=n_splits,shuffle=True)
    
    round=0
    for train_index, test_index in kf.split(dfTrain):
        round+=1
        train_X = dfTrain.loc[train_index,predictors]
        test_X = dfTrain.loc[test_index,predictors]
        train_Y = dfTrain.loc[train_index,'Y']
        test_Y = dfTrain.loc[test_index,'Y']

        
        model = LinearRegression()
        model.fit(train_X,train_Y)
        
        pred_test = model.predict(test_X)


        pred_score = model.predict(dfPred[predictors].values)
        if round==1:
            test_result = pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})
            result = pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score})
        else:
            test_result = pd.concat([test_result,pd.DataFrame({'ID':dfTrain.loc[test_index,'ID'].values,'score':pred_test,'target':test_Y})],axis=0)
            result = result.merge(pd.DataFrame({'ID':dfPred['ID'],'Score_%d'%round:pred_score}),'inner','ID')
    print("Test MSE:",metrics.mean_squared_error(test_result['target'], test_result['score']))
    return test_result,result

def simple_avg(dfTrain,dfPred,predictors,weight=False):
    test_result = dfTrain.copy()
    result = dfPred.copy()
    if weight:
        weightList = []
        for var in predictors:
            weightList.append(1/metrics.mean_squared_error(test_result['Y'], test_result[var]))
        weightList =np.array(weightList)
        weightList = len(predictors)*weightList/weightList.sum()
        for i in range(len(predictors)):
            test_result[predictors[i]] = test_result[predictors[i]]*weightList[i]
            result[predictors[i]] = result[predictors[i]]*weightList[i]
    
    test_result['score'] = test_result[predictors].mean(axis=1)
    result['score'] = result[predictors].mean(axis=1)
    print("Test MSE:",metrics.mean_squared_error(test_result['Y'], test_result['score']))
    return test_result,result[['ID','score']]
    


























































































































