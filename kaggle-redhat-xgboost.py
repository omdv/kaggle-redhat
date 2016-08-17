import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import time

ppl_file = 'input/people.csv'
train_file = 'input/act_train.csv'
test_file = 'input/act_test.csv'

bst_params = {
    'booster':'gbtree',
    'nthread':-1,
    'max_depth':6,
    'eta':0.1,
    'min_child_weight':1,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'tree_method':'exact',
    'silent':'true'
}

# read datasets
train = pd.read_csv(train_file, parse_dates=['date'])
submit = pd.read_csv(test_file, parse_dates=['date'])
ppl = pd.read_csv(ppl_file, parse_dates=['date'])

# rename features correspondingly
ppl.columns = ['ppl_'+x if x not in ['people_id'] else x for x in ppl.columns]
train.columns = ['act_'+x if x not in ['people_id','outcome','activity_id'] else x for x in train.columns]
submit.columns = ['act_'+x if x not in ['people_id','activity_id'] else x for x in submit.columns]

train = pd.merge(train, ppl, on='people_id')
submit = pd.merge(submit, ppl, on='people_id')
del ppl

# create a small validation set - 0.3%
msk = np.random.rand(train.shape[0]) < 0.003
test = train[msk]
train = train[~msk]

print "\nTraining on Full set of size: {}".format(train.shape)

# select target and remaining predictors
target = 'outcome'
predictors = pd.Series([x for x in train.columns if x not in [target]])

# Create DMatrix
dtrain = xgb.DMatrix(train[predictors.values].values,label=train[target].values)
dtest = xgb.DMatrix(test[predictors.values].values,label=test[target].values)

print "\nDMatrix initiated..."

# Validation set
evallist  = [(df_train,'train'),(df_test,'eval')]

num_round = 200
num_features = predictors.shape[0]

# TODO - annealed learning rate
# Train and save booster
bst = xgb.train( 
        params,\
        df_train,\
        num_boost_round=num_round,
        early_stopping_rounds=10,\
        evals=evallist,\
        verbose_eval=1)

# Feature importances - merge with predictors
fi_dict = bst.get_fscore()
fi_pred = {}
for key in fi_dict:
    num = int(key.replace('f',''))
    nkey = predictors[num]
    fi_pred[nkey] = fi_dict[key]

feat_imp = pd.Series(fi_pred).sort_values(ascending=False)
feat_imp = feat_imp/1.e-2/feat_imp.values.sum()

print "\nFeatures (full set):"
print feat_imp

# Saving model and feature importances
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
modelfile = '/mnt/gcs-bucket/{}_{}feat_{}'.format(num_round,num_features,timestamp)

bst.save_model(modelfile+'.model')
feat_imp.to_csv(modelfile+'_features.csv')
predictors.to_csv(modelfile+'_predictors.csv')

print "\nCase saved to file: {}".format(modelfile)

