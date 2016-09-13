from __future__ import print_function
import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import time
import copy

np.random.seed(42)

def intersect(a, b):
    return list(set(a) & set(b))

# add derived features here
def derive_features(train, test):
    print("Derive new features...")

    # 1. DO NOT TURN ON
    # meanlist = [
    #     [['act_date'],'mean_by_act_date'],\
    #     [['ppl_date'],'mean_by_ppl_date'],\
    #     [['act_date','ppl_char_38'],'mean_by_ch38_act_date'],
    #     [['act_date','ppl_char_7'],'mean_by_ch7_act_date']]
    # for key in meanlist:
    #     basis = key[0]
    #     label = key[1]
    #     mean_outcome = train.groupby(basis).agg({'outcome':np.nanmean}).reset_index()
    #     mean_outcome = mean_outcome.rename(columns={'outcome':label})
    #     train = pd.merge(train,mean_outcome,on=basis,how='left')
    #     test = pd.merge(test,mean_outcome,on=basis,how='left')
    #     crossval = pd.merge(crossval,mean_outcome,on=basis,how='left')

    # # 2. DO NOT TURN ON
    # basis = ['act_date','ppl_group_1']
    # median_outcome = train.groupby(basis).agg({'outcome':np.nanmedian}).reset_index()
    # median_outcome = median_outcome.rename(columns={'outcome':'median_by_grp_act_date'})
    # train = pd.merge(train,median_outcome,on=basis,how='left')
    # test = pd.merge(test,median_outcome,on=basis,how='left')
    # crossval = pd.merge(crossval,median_outcome,on=basis,how='left')

    # 3. merge top 3 features
    # tomerge = [\
    #     ['ppl_group_1','ppl_char_7','ppl_grp1_char7'],
    #     ['ppl_group_1','ppl_char_38','ppl_grp1_char38'],
    #     ['ppl_group_1','ppl_char_6','ppl_grp1_char6'],
    #     ['ppl_char_38','ppl_char_7','ppl_char38_char7'],
    #     ['ppl_char_38','ppl_char_6','ppl_char38_char6']
    # ]
    # for feat in tomerge:
    #     train['foo1'] = train[feat[0]].astype('str')
    #     train['foo2'] = train[feat[1]].astype('str')
    #     train[feat[2]] = train['foo1'] + train['foo2']
    #     train[feat[2]] = train[feat[2]].astype(np.int32)

    #     test['foo1'] = test[feat[0]].astype('str')
    #     test['foo2'] = test[feat[1]].astype('str')
    #     test[feat[2]] = test['foo1'] + test['foo2']
    #     test[feat[2]] = test[feat[2]].astype(np.int32)

    # del train['foo1'], train['foo2']
    # del test['foo1'], test['foo2']

    return train, test


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('people_id')
    output.remove('activity_id')
    output.remove('act_date')
    output.remove('ppl_date')
    output.remove('act_char_10')
    # output.remove('ppl_group_1')
    return sorted(output)

def read_test_train():
    print("Load people.csv...")
    people = pd.read_csv("../input/people.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])

    print("Load train.csv...")
    train = pd.read_csv("../input/act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])

    print("Load test.csv...")
    test = pd.read_csv("../input/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

    print("Pre-process tables...")
    for table in [train, test]:
        # table['year'] = table['date'].dt.year.astype(np.int32)
        # table['month'] = table['date'].dt.month.astype(np.int32)
        # table['day'] = table['date'].dt.day.astype(np.int32)
        # table['weekday'] = table['date'].dt.weekday.astype(np.int32)
        # table['isweekend'] = (table['weekday'] >= 5).astype(np.int32)
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        for i in range(1, 11):
            cursor = 'char_' + str(i)
            table[cursor].fillna('type 99999',inplace=True)
            table.loc[table[cursor].notnull(),cursor] = \
                table.loc[table[cursor].notnull(),cursor].str.lstrip('type ').astype(np.int32)

    # people['year'] = people['date'].dt.year.astype(np.int32)
    # people['month'] = people['date'].dt.month.astype(np.int32)
    # people['day'] = people['date'].dt.day.astype(np.int32)
    # people['weekday'] = people['date'].dt.weekday.astype(np.int32)
    # people['isweekend'] = (people['weekday'] >= 5).astype(np.int32)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        cursor = 'char_' + str(i)
        people.loc[people[cursor].notnull(),cursor] = \
            people.loc[people[cursor].notnull(),cursor].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        cursor = 'char_' + str(i)
        people.loc[people[cursor].notnull(),cursor] = \
            people.loc[people[cursor].notnull(),cursor].astype(np.int32)

    print("Merge with people...")
    # rename features correspondingly
    people.columns = ['ppl_'+x if x not in ['people_id'] else x for x in people.columns]
    train.columns = ['act_'+x if x not in ['people_id','outcome','activity_id'] else x for x in train.columns]
    test.columns = ['act_'+x if x not in ['people_id','activity_id'] else x for x in test.columns]

    # merge
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    
    return train, test


def run_single(train,test,features,target,valsize):

    # creating sparse matrix
    print('Creating sparse matrix...')
    # train = train.drop_duplicates(features)

    # create a small validation set - unique people id
    if valsize > 0:
        mask = np.random.rand(train.people_id.unique().shape[0]) < valsize/1.e2
        mask = train.people_id.unique()[mask]
        valid = train[train.people_id.isin(mask)]
        train = train[~train.people_id.isin(mask)]
        y_valid = valid[target]
        valid = valid[features]
    
    y_train = train[target]
    
    # if not hot encoding - use these
    # train = train[features]
    # test = test[features]

    # hot encode
    enc = OneHotEncoder(handle_unknown='ignore',dtype='np.int32')
    if valsize > 0:
        enc.fit(pd.concat([train[features],test[features],valid[features]]))
    else:
        enc.fit(pd.concat([train[features],test[features]]))
    train = enc.transform(train[features])
    test = enc.transform(test[features])
    if valsize > 0:
        valid = enc.transform(valid[features])

    dtrain = xgb.DMatrix(train, label = y_train)
    dtest = xgb.DMatrix(test)

    if valsize > 0:
        dvalid = xgb.DMatrix(valid, label = y_valid)

    print('Shape of train: {}'.format(train.shape))
    print('Shape of test: {}'.format(test.shape))

    # tree booster params
    # num_boost_round = 150
    # early_stopping_rounds = 10
    # start_time = time.time()
    # params = {
    #     "objective": "binary:logistic",
    #     "booster" : "gbtree",
    #     "eval_metric": "auc",
    #     "eta": 0.01,
    #     "gamma": 0,
    #     "tree_method": 'exact',
    #     "max_depth": 10,
    #     "min_child_weight": 2,
    #     "subsample": 0.7,
    #     "colsample_bytree": 0.7,
    #     "silent": 1,
    #     "seed": 42
    # }

    # linear booster params
    num_boost_round = 10
    early_stopping_rounds = 10
    start_time = time.time()
    params = {
        "objective": "binary:logistic",
        "booster" : "gblinear",
        "eval_metric": "auc",
        "eta": 0.02,
        "gamma": 0,
        "max_depth": 10,
        "min_child_weight": 0,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "silent": 1,
        "seed": 42,
    }

    print('XGBoost params: {}'.format(params))

    if valsize > 0:
        watchlist = [(dtrain, 'train'), (dvalid,'valid')]
    else:
        watchlist = [(dtrain, 'train')]

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,\
        early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    if valsize > 0:
        check = gbm.predict(dvalid) #, ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_valid.values, check)
    else:
        check = gbm.predict(dtrain) #, ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_train.values, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array:\n{}'.format(imp))

    print("Predict test dataset...")
    test_prediction = gbm.predict(dtest) #, ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score, gbm, imp

def get_importance(gbm, features):
    importance = pd.Series(gbm.get_fscore()).sort_values(ascending=False)
    importance = importance/1.e-2/importance.values.sum()
    return importance

def grid_search_CV(X_train,y_train,X_valid,y_valid):
    print('Launching Grid Search CV...')
    dtrain = xgb.DMatrix(X_train, label = y_train, missing = -999)
    dvalid = xgb.DMatrix(X_valid, label = y_valid, missing = -999)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    eta = 0.1
    max_depth = 10
    subsample = 0.7
    colsample_bytree = 0.7
    min_child_weight = 2
    num_boost_round = 150
    early_stopping_rounds = 10
    test_size = 0.1
    random_state = 42

    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    grid_search_res = []
    #         # 'gamma': [0.1,0.2,0.3,0.4,0.5]},
    #         # 'subsample': [0.6,0.8,1.0],
    #         # 'colsample_bytree': [0.6,0.8,1.0]},

    param_i = 'gamma'
    param_j = 'subsample'
    for i in np.arange(0.0,0.6,0.1):
        for j in np.arange(0.5,1.0,0.1):
            start_time = time.time()
            params[param_i] = i
            params[param_j] = j
            gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,\
                early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
            
            check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
            score = roc_auc_score(y_valid.values, check)
            grid_search_res.append({param_i:i,param_j:j,'score':score})
            print('Training time: {} minutes, score: {:0.4f}'.\
                format(round((time.time() - start_time)/60, 2),score))

    for value in grid_search_res:
        print(value)

    # clf = GridSearchCV(xgb_model,{\
    #         'min_child_weight':[1,2],
    #         'max_depth': [2]},
    #         # 'gamma': [0.1,0.2,0.3,0.4,0.5]},
    #         # 'subsample': [0.6,0.8,1.0],
    #         # 'colsample_bytree': [0.6,0.8,1.0]},
    return grid_search_res

def merge_with_leak(prediction):
    print('Merging with leak dataset...')
    leak = pd.read_csv('../output/leak_predictions_NA.csv')
    leak['pred'] = prediction
    # resetting 0.5 to nan to use xgboost
    leak.loc[leak.outcome == 0.5, 'outcome'] = np.nan
    print('Number of missing values: {}'.format(leak[leak.outcome.isnull()].shape[0]))
    leak['outcome'] = leak['outcome'].fillna(leak['pred'])
    return leak.outcome.values

def create_submission(score, test, pred, model, importance, averaged):
    now = datetime.datetime.now()
    scrstr = "{:0.4f}_{}".format(score,now.strftime("%Y-%m-%d-%H-%M"))
    mod_file = '../output/model_' + scrstr + '.model'
    print('Writing model: ', mod_file)
    model.save_model(mod_file)
    imp_file = '../output/imp_' + scrstr + '.csv'
    print('Writing features: ', imp_file)
    importance.to_csv(imp_file)
    if averaged:
        sub_file = '../output/submit_' + scrstr + '_avg.csv'
    else:
        sub_file = '../output/submit_' + scrstr + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('activity_id,outcome\n')
    total = 0
    for id in test['activity_id']:
        str1 = str(id) + ',' + str(pred[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

if __name__ == '__main__':
    train, test = read_test_train()
    train, test = derive_features(train, test)
    features = get_features(train, test)

    print('Shape of train: {}'.format(train.shape))
    print('Shape of test: {}'.format(test.shape))
    print('Features [{}]: {}'.format(len(features), sorted(features)))

    prediction, score, model, importance = run_single(train,test,features,'outcome',0)
    # # # prediction, score = run_kfold(3, train, test, features, 'outcome')
    # # grid = grid_search_CV(train[features],train['outcome'],\
    # #     crossval[features],crossval['outcome'])

    try:
        pred = merge_with_leak(prediction)
    except:
        print('Merge with leak dataset failed!')

    # pred = prediction
    create_submission(score, test, pred, model, importance, averaged=False)
