from __future__ import print_function
import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import random
import time
import copy

random.seed(42)

def intersect(a, b):
    return list(set(a) & set(b))

# add derived features here
def derive_features(train, test):
    # create a small validation set - 0.5%
    mask = np.random.rand(train.shape[0]) < 0.5/1.e2
    crossval = train[mask]
    train = train[~mask]

    # 1. mean by various groups
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

    # # 2. median outcome by activity date and people group in training
    # basis = ['act_date','ppl_group_1']
    # median_outcome = train.groupby(basis).agg({'outcome':np.nanmedian}).reset_index()
    # median_outcome = median_outcome.rename(columns={'outcome':'median_by_grp_act_date'})
    # train = pd.merge(train,median_outcome,on=basis,how='left')
    # test = pd.merge(test,median_outcome,on=basis,how='left')
    # crossval = pd.merge(crossval,median_outcome,on=basis,how='left')

    return train, test, crossval


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('people_id')
    output.remove('activity_id')
    output.remove('act_date')
    output.remove('ppl_date')
    return sorted(output)

def read_test_train():
    print("Read people.csv...")
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

    print("Process tables...")
    for table in [train, test]:
        table['year'] = table['date'].dt.year
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        # table.drop('date', axis=1, inplace=True)
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        for i in range(1, 11):
            table['char_' + str(i)].fillna('type -999', inplace=True)
            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    people['year'] = people['date'].dt.year
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    # people.drop('date', axis=1, inplace=True)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)

    print("Merge...")

    # rename features correspondingly
    people.columns = ['ppl_'+x if x not in ['people_id'] else x for x in people.columns]
    train.columns = ['act_'+x if x not in ['people_id','outcome','activity_id'] else x for x in train.columns]
    test.columns = ['act_'+x if x not in ['people_id','activity_id'] else x for x in test.columns]

    # merge and replace NANs
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(-999, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(-999, inplace=True)

    # derive new features here and create a cross-validation set out of train
    train, test, crossval = derive_features(train, test)

    # get intersection of features
    features = get_features(train, test)
    
    return train, test, crossval, features


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)

    # importance = gbm.get_fscore(fmap='xgb.fmap')
    # importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    
    importance = pd.Series(gbm.get_fscore()).sort_values(ascending=False)
    importance = importance/1.e-2/importance.values.sum()
    return importance


def run_single(train, test, valid, features, target, random_state=0):
    eta = 0.2
    max_depth = 6
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 300
    early_stopping_rounds = 10
    test_size = 0.1

    # X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    # print('Length train:', len(X_train.index))
    # print('Length valid:', len(X_valid.index))

    X_train = train
    X_valid = valid
    print('Length of traininig:', len(X_train.index))
    print('Length of validation:', len(X_valid.index))
    y_train = train[target]
    y_valid = valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(X_valid[target].values, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array:\n{}'.format(imp))

    print("Predict test dataset...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score, gbm, imp


def run_kfold(nfolds, train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state
    }
    num_boost_round = 50
    early_stopping_rounds = 10

    yfull_train = dict()
    yfull_test = copy.deepcopy(test[['activity_id']].astype(object))
    kf = KFold(len(train.index), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_index, test_index in kf:
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, nfolds))
        X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[test_index]
        y_train, y_valid = train[target].as_matrix()[train_index], train[target].as_matrix()[test_index]
        X_test = test[features].as_matrix()

        print('Length train:', len(X_train))
        print('Length valid:', len(X_valid))

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        
        print("Validating...")
        yhat = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_valid.tolist(), yhat)
        print('Check error value: {:.6f}'.format(score))

        # Each time store portion of precicted data in train predicted values
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = yhat[i]

        imp = get_importance(gbm, features)
        print('Importance array: ', imp)

        print("Predict test dataset...")
        test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)
        yfull_test['kfold_' + str(num_fold)] = test_prediction

    # Copy dict to list
    train_res = []
    for i in sorted(yfull_train.keys()):
        train_res.append(yfull_train[i])

    score = roc_auc_score(train[target], np.array(train_res))
    print('Check error value: {:.6f}'.format(score))

    # Find mean for KFolds on test
    merge = []
    for i in range(1, nfolds+1):
        merge.append('kfold_' + str(i))
    yfull_test['mean'] = yfull_test[merge].mean(axis=1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return yfull_test['mean'].values, score


def merge_with_leak(prediction,averaged=True):
    print('Merging with leak dataset...')
    leak = pd.read_csv('../output/leak_predictions_NA.csv')
    leak['pred'] = prediction
    # to average over group_1 and act_date or not
    if averaged:
        leak = pd.merge(leak,test[['activity_id','act_date','ppl_group_1']],\
            on='activity_id',how='left')
        leak['avg_pred'] = leak.groupby(['act_date','ppl_group_1'])\
            ['pred'].transform('mean')
        leak['outcome'] = leak['outcome'].fillna(leak['avg_pred'])
    else:
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

train, test, crossval, features = read_test_train()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))

prediction, score, model, importance = run_single(train, test, crossval, features, 'outcome')
# prediction, score = run_kfold(3, train, test, features, 'outcome')
try:
    pred = merge_with_leak(prediction, averaged=False)
except:
    print('Merge with leak dataset failed!')
create_submission(score, test, pred, model, importance, averaged=False)