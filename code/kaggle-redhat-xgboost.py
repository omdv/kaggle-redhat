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

# --------------------------------------------------------#
# Processing functions

def reduce_dimen(dataset,column,toreplace):
    for index,i in dataset[column].duplicated(keep=False).iteritems():
        if i==False:
            dataset.set_value(index,column,toreplace)
    return dataset

def process_date(input_df):
    df = input_df.copy()
    return (df.assign(year=lambda df: df.date.dt.year,  # Extract year
                      month=lambda df: df.date.dt.month,  # Extract month
                      day=lambda df: df.date.dt.day, # Extract day
                      weekday=lambda df: df.date.dt.weekday) # Extract workday
            )


def process_activity_category(input_df):
    df = input_df.copy()
    return df.assign(activity_category=lambda df:
                     df.activity_category.str.lstrip('type ').astype(np.int32))


def process_activities_char(input_df, columns_range):
    """
    Extract the integer value from the different char_* columns in the
    activities dataframes. Fill the missing values with 999999 as well
    """
    df = input_df.copy()
    char_columns = ['char_' + str(i) for i in columns_range]
    return (df[char_columns].fillna('type 999999')
            .apply(lambda col: col.str.lstrip('type ').astype(np.int32))
            .join(df.drop(char_columns, axis=1)))


def activities_processing(input_df):
    """
    This function combines the date, activity_category and char_*
    columns transformations.
    """
    df = input_df.copy()
    return (df.pipe(process_date)
              .pipe(process_activity_category)
              .pipe(process_activities_char, range(1, 11)))


def process_group_1(input_df):
    df = input_df.copy()
    return df.assign(group_1=lambda df:
                     df.group_1.str.lstrip('group ').astype(np.int32))


# TODO: Refactor the different *_char functions

def process_people_cat_char(input_df, columns_range):
    """
    Extract the integer value from the different categorical char_*
    columns in the people dataframe.
    """
    df = input_df.copy()
    cat_char_columns = ['char_' + str(i) for i in columns_range]
    return (df[cat_char_columns].apply(lambda col:
                                       col.str.lstrip('type ').astype(np.int32))
                                .join(df.drop(cat_char_columns, axis=1)))


def process_people_bool_char(input_df, columns_range):
    """
    Extract the integer value from the different boolean char_* columns in the
    people dataframe.
    """
    df = input_df.copy()
    boolean_char_columns = ['char_' + str(i) for i in columns_range]
    return (df[boolean_char_columns].apply(lambda col: col.astype(np.int32))
                                    .join(df.drop(boolean_char_columns,
                                                  axis=1)))


# TODO: Extract the magic ranges (1 to 10 and 10 to 38) programmatically

def people_processing(input_df):
    """
    This function combines the date, group_1 and char_*
    columns (inclunding boolean and categorical ones) transformations.
    """
    df = input_df.copy()
    return (df.pipe(process_date)
              .pipe(process_group_1)
              .pipe(process_people_cat_char, range(1, 10))
              .pipe(process_people_bool_char, range(10, 38)))


def merge_with_people(input_df, people_df):
    """
    Merge (left) the given input dataframe with the people dataframe and
    fill the missing values with 999999.
    """
    df = input_df.copy()
    return (df.merge(people_df, how='left', on='people_id',
                     left_index=True, suffixes=('_act', '_ppl'))
            .fillna(999999))

# --------------------------------------------------------#

def intersect(a, b):
    return list(set(a) & set(b))

# add derived features here
def derive_features(train, test):
    print("Derive new features...")
    # Delta in days
    train['date_lag'] = (train.date_act - train.date_ppl).dt.days
    test['date_lag'] = (test.date_act - test.date_ppl).dt.days

    # train['ppl_date_weekend'] = (train.weekday_ppl >= 5)
    # train['act_date_weekend'] = (train.weekday_act >= 5)
    # test['ppl_date_weekend'] = (test.weekday_ppl >= 5)
    # test['act_date_weekend'] = (test.weekday_act >= 5)


    # count of activity by people_id
    # count = train.groupby(['people_id']).count()[['activity_id']]
    # count = count.rename(columns = {'activity_id' : 'count_people_id'}).reset_index()
    # train = train.merge(count, on='people_id', how='left')
    # test = test.merge(count, on='people_id', how='left')

    # count of activity_id by date
    # count = pd.concat([train,test]).groupby(['date_act']).count()[['activity_id']]
    # count = count.rename(columns = {'activity_id' : 'act_id_by_date'}).reset_index()
    # train = train.merge(count, on='date_act', how='left')
    # test = test.merge(count, on='date_act', how='left')

    # count of activity by person
    # count = pd.concat([train,test]).groupby(['people_id']).count()[['activity_id']]
    # count = count.rename(columns = {'activity_id' : 'act_id_by_ppl'}).reset_index()
    # train = train.merge(count, on='people_id', how='left')
    # test = test.merge(count, on='people_id', how='left')

    return train, test

def get_features(train, test):
    features = intersect(train.columns, test.columns)
    features.remove('people_id')
    features.remove('activity_id')
    # features.remove('char_10_act')
    # features.remove('group_1')
    features.remove('date_act')
    features.remove('date_ppl')
    return sorted(features)

def read_test_train():
    print("Load people.csv...")
    people_df = pd.read_csv("../input/people.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])

    print("Load train.csv...")
    train_df = pd.read_csv("../input/act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])

    print("Load test.csv...")
    test_df = pd.read_csv("../input/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

    print("Pre-process tables...")
    
    people = people_df.pipe(people_processing)
    train = (train_df.pipe(activities_processing)
        .pipe(merge_with_people, people))
    test = (test_df.pipe(activities_processing)
        .pipe(merge_with_people, people))

    return train, test


def run_single(train,test,features,target,valsize):

    # create a small validation set - unique people id
    if valsize > 0:
        mask = np.random.rand(train.people_id.unique().shape[0]) < valsize/1.e2
        mask = train.people_id.unique()[mask]
        valid = train[train.people_id.isin(mask)]
        train = train[~train.people_id.isin(mask)]
        y_valid = valid[target]
        valid = valid[features]
    
    y_train = train[target]
    
    # # if not hot encoding - use these
    # train = train[features]
    # test = test[features]

    # hot encode
    print('Creating sparse matrix...')
    if valsize > 0:
        X = pd.concat([train[features],test[features],valid[features]])
    else:
        X = pd.concat([train[features],test[features]])

    categorical=['group_1','char_10_act','activity_category','char_1_act','char_2_act','char_3_act',
        'char_4_act','char_5_act','char_6_act','char_7_act','char_8_act','char_9_act',
        'char_2_ppl','char_3_ppl','char_4_ppl','char_5_ppl','char_6_ppl','char_7_ppl',
        'char_8_ppl','char_9_ppl','date_lag','month_act','month_ppl','weekday_act','weekday_ppl',
        'day_act','day_ppl']
    not_categorical=[]
    for category in X.columns:
        if category not in categorical:
            not_categorical.append(category)
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc = enc.fit(X[categorical])
    train_sparse_cat = enc.transform(train[categorical])
    test_sparse_cat = enc.transform(test[categorical])
    if valsize > 0:
        valid_sparse_cat = enc.transform(valid[categorical])
    
    from scipy.sparse import hstack
    train = hstack((train[not_categorical], train_sparse_cat))
    test = hstack((test[not_categorical], test_sparse_cat))
    if valsize > 0:
        valid = hstack((valid[not_categorical], valid_sparse_cat))

    dtrain = xgb.DMatrix(train, label = y_train, missing = 999999)
    dtest = xgb.DMatrix(test, missing = 999999)

    if valsize > 0:
        dvalid = xgb.DMatrix(valid, label = y_valid, missing = 999999)

    print('Shape of train: {}'.format(train.shape))
    print('Shape of test: {}'.format(test.shape))

    # # tree booster params
    # num_boost_round = 29
    # early_stopping_rounds = 10
    # start_time = time.time()
    # params = {
    #     "objective": "binary:logistic",
    #     "booster" : "gbtree",
    #     "eval_metric": "auc",
    #     "eta": 0.01,
    #     "gamma": 0,
    #     "tree_method": 'exact',
    #     "max_depth": 12,
    #     "min_child_weight": 2,
    #     "subsample": 0.7,
    #     "colsample_bytree": 0.7,
    #     "silent": 1,
    #     "seed": 42
    # }

    # linear booster params
    num_boost_round = 110
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
        check = gbm.predict(dvalid)#, ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_valid.values, check)
    else:
        check = gbm.predict(dtrain)#, ntree_limit=gbm.best_iteration+1)
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
    dtrain = xgb.DMatrix(X_train, label = y_train, missing = 999999)
    dvalid = xgb.DMatrix(X_valid, label = y_valid, missing = 999999)
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

    param_i = 'maxdepth'
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
    # train = train.iloc[0:10,]
    # test = test.iloc[0:10,]
    train, test = derive_features(train, test)
    features = get_features(train, test)

    # reduce dimension for linear case
    train = reduce_dimen(train,'char_10_act',99999)
    train = reduce_dimen(train,'group_1',99999)

    print('Shape of train: {}'.format(train.shape))
    print('Shape of test: {}'.format(test.shape))
    print('Features [{}]: {}'.format(len(features), sorted(features)))

    prediction, score, model, importance = run_single(train,test,features,'outcome',0)
    # # grid = grid_search_CV(train[features],train['outcome'],\
    # #     crossval[features],crossval['outcome'])

    # try:
    #     pred = merge_with_leak(prediction)
    # except:
    #     print('Merge with leak dataset failed!')

    # pred = prediction
    # create_submission(score, test, pred, model, importance, averaged=False)
