## Kaggle Redhat

There is a leak in the data, mean outcome by ppl_group_1 and act_date is 0 or 1. With extrapolation it covers 

Missing 69073 rows (13.85%) of test data may be predicted using xgboost.

Ideas:
- average final prediction by act_date, ppl_group_1, i.e. if mean > 0.5 set all rows for this group @ 1 and vice versa

## Code description
* kaggle-redhat-xgboost - xgboost script to predict traditionally
* leak_code.py - python script from Kaggle forum to exploit the leak

## Results - xgboost only
* 0.9554 with max_depth = 10, min_child_weight = 9 and 150 rounds

0.890677 - linear booster w/o group_1 but with char_10

# Feature engineering for non-leak case
0.892309 @ 7 - full dates, no group_1
0.926427 @ 6 - full dates, no group_1, with date_lag
0.90783 @ 28 - full dates, no group_1, elapsed dates, date lag, count_ppl_id
0.923748 @ 4 - removed elapsed dates
0.921728 @ 6 - removed count_ppl_id, added act_id on train by date
0.910247 @ 1 - removed char_10_act -> best, prior to linear booster
0.910139 @ 1 - removed char_10_act and date_lag in business days
0.911199 @ 2 - cat char_10 with tree booster
0.92821 @ 6 - no group_1, char_10
0.927476 @ 14 - added activity_id by ppl_id
0.937635 @ 29 with 1% of valid

## Linear booster
0.906397 @ 113 - no char_10, date_lag
0.906623 @ 117 - with char_10 (not cat) and date_lag
0.917754 @ 89 - with cat char_10 with reduced dimension
0.917671 @ 88 - with cat char_10 and no reduced dimension

# With group_1:
0.95209 @ 28 - non sparse
0.92521 @ 28 - sparse with tree booster
0.980199 @ 104 - sparse with linear booster with group_1 and char_10
0.979923 @ 120 - sparse with linear with group_1 and no char_10
0.980232 @ 102 - sparse dates, linear with group_1 and char_10
0.980255 @ 106 - all dates sparse, linear


