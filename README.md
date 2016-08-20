## Kaggle Redhat

There is a leak in the data, mean outcome by ppl_group_1 and act_date is 0 or 1. With extrapolation it covers 

Missing 69073 rows (13.85%) of test data may be predicted using xgboost.

Ideas:
- average final prediction by act_date, ppl_group_1, i.e. if mean > 0.5 set all rows for this group @ 1 and vice versa

## Code description
* kaggle-redhat-xgboost - xgboost script to predict traditionally
* leak_code.py - python script from Kaggle forum to exploit the leak