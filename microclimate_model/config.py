from sklearn.model_selection import RepeatedKFold
# models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm

import numpy as np

# Filepaths.
BASE_PATH = "../data/microclimate_model/combined/dataset2"
THREE_BLDGS_FILENAME = "three_buildings"
TRAIN_FILE_PATH = f"{BASE_PATH}/{THREE_BLDGS_FILENAME}_train.csv"
TEST_FILE_PATH = f"{BASE_PATH}/{THREE_BLDGS_FILENAME}_test.csv"
RESULT_DIR_PATH = f"{BASE_PATH}/result"
MODEL_DIR_PATH = f"{RESULT_DIR_PATH}/model"
SCENARIOS_DIR_PATH = f"{BASE_PATH}/scenarios"

HIGH_ALBEDO_WALLS = "high_albedo_walls"
TREES_EXTREME = "trees_extreme"
COOL_PAVEMENT = "cool_pavement"
TREES_SURROUND = "trees_surround"
WALL_SHADE = "wall_shade"
PV_ROOFTOP_TREES = "pv_rooftop_and_trees"
PV_ROOFTOP = "pv_rooftop"
PV_SIDEWALKS = "pv_sidewalks"


ALL_COLUMNS = ['AirT_Mean', 'KW', 'AbsH_Mean', 'HTmmBTU',
               'ShortW_North', 'ShortW_East', 'ShortW_South', 'ShortW_West',
               'Shade_North', 'Shade_East', 'Shade_South', 'Shade_West',
               'bldgname_ISTB 4', 'bldgname_Psychology North','bldgname_Psychology'
               'CHWTON/SQM', 
               # Additional variables on top of those from first publication.`
               'KW/SQM', 'HTmmBTU/SQM', 'CHWTON',
               'AirT_North', 'AirT_East', 'AirT_South', 'AirT_West',
               'RelH_Mean',   
               'ShortW_Top', 
               'Shade_Top', 
               'SumW_North', 'SumW_East', 'SumW_South', 'SumW_West', 'SumW_Top',
               'LongW_North', 'LongW_East', 'LongW_South', 'LongW_West', 'LongW_Top', 
               'Area_North', 'Area_East', 'Area_South','Area_West', 'Area_Top', 
               ]

TEST_DATE = '2023-07-07'

# Declare constants for the variable names that we use often.
CHWTON_SQM = "CHWTON/SQM"
DATE_TIME = 'Date_Time'
BLDGNAME = "bldgname"
ISTB4 = "bldgname_ISTB 4"
PYSCHOLOGY = "bldgname_Psychology"
PSYCHOLOGY_NORTH = "bldgname_Psychology North"

# MODELS common config:
# For base model
RANDOM_STATE = 42
N_ESTIMATORS = 100

# For randomized search
CV = RepeatedKFold(n_splits=10, n_repeats=3)
CV = 3
N_ITER = 10
SCORING = 'r2'
N_JOBS = -1


# MODELS config:
# RF
rf_name = "rf"
rf_base = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
rf_param = {
    'criterion': ['squared_error', 'absolute_error', 'poisson'],
    'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)],
    'max_features': ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [ 1, 2, 4],
    'bootstrap': [True] 
}

# XGB
xgb_name = "xgb"
xgb_base = XGBRegressor(n_estimators = N_ESTIMATORS, verbosity = 0, random_state = RANDOM_STATE)
xgb_param = {
    'learning_rate' : [0.1, 0,2 ,0.3, 0.4],
    'n_estimators':[ 100, 250, 500, 1000],
    'min_child_weight':[1, 2, 4, 5, 8], 
    'max_depth': [4,6,7,8],
    'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7, 1 ],
    'booster': ['gbtree', 'gblinear'] }

# LGBM
lgbm_name = "lgbm"
lgbm_base = LGBMRegressor(random_state = RANDOM_STATE)
lgbm_param = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
                'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],
                'num_leaves': [20,30,40, 50, 100], 
                'min_child_samples': [5, 10, 20, 30],  # Lowered the lower limit
                'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                'subsample': [0.1, 0.3, 0.8, 1.0], 
                'max_depth': [3, 4, 5, 6, 7, 8],  # Adjusted the range
                'colsample_bytree': [0.4, 0.5, 0.6, 1.0],
                'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


# CATBOOST
cb_name = "catboost"
cb_base = CatBoostRegressor(random_state = RANDOM_STATE, verbose=False)
cb_param = {
        'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
        'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}