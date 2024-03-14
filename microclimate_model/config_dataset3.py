# This is the config file for dataset2
from sklearn.model_selection import RepeatedKFold
# models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm
import numpy as np


BASE_PATH = "../data/dataset3"

# Data file paths.
THREE_BLDGS_FILENAME = "three_buildings"
TRAIN_FILE_PATH = f"{BASE_PATH}/{THREE_BLDGS_FILENAME}_train.csv"
TEST_FILE_PATH = f"{BASE_PATH}/{THREE_BLDGS_FILENAME}_test.csv"
SCENARIOS_DIR_PATH = f"{BASE_PATH}/scenarios"
SELECTORS_DIR_PATH = f"{BASE_PATH}/selectors"
RESULT_DIR_PATH = f"{BASE_PATH}/result"

TEST_DATE = '2023-07-07'
BLDG_NAMES = ["Psychology", "Psychology_North", "ISTB_4"]

# Defining constants for features that we always keep:
CHWTON_SQM = "CHWTON/SQM"
KWM = "KW/SQM"
HTM2 = "HTmmBTU/SQM"
AIRT_MEAN = "AirT_Mean"
ABSH_MEAN = "AbsH_Mean"

FEATURES_MUST_KEEP = [
    CHWTON_SQM, KWM, HTM2, AIRT_MEAN, ABSH_MEAN
]

# Define the constants for features strings that are optionals:
DATE_TIME = "Date_Time"
AIRT = "AirT"
AIRT_TOP = "AirT_Top"
ABSH = "AbsH"
ABSH_TOP = "AbsH_Top"
AIRP = "AirP"
HOUR = "Hour"
LONG_W = "LongW"
PRODUCT_W = "ProductW"
SHORT_W = "ShortW"
SHORT_W_TOP = "ShortW_Top"
WIND = "Wind"
SHADE = "Shade"

FEATURES_OPTIONAL = {
    HOUR: ["Hour"],
    LONG_W: ["LongW_Top", "LongW_North", "LongW_East", "LongW_South", "LongW_West"],
    SHADE: ["Shade_Top", "Shade_North", "Shade_East", "Shade_South", "Shade_West"],
    SHORT_W: ["ShortW_Top", "ShortW_North", "ShortW_East", "ShortW_South", "ShortW_West"],
    SHORT_W_TOP: ["ShortW_Top"],
    PRODUCT_W: ["ProductW_Top", "ProductW_North", "ProductW_East", "ProductW_South", "ProductW_West"],
    ABSH: ["AbsH_North", "AbsH_East", "AbsH_South", "AbsH_West"],
    ABSH_TOP: ["AbsH_Top"],
    AIRT: ["AirT_North", "AirT_East", "AirT_South", "AirT_West",],
    AIRT_TOP: ["AirT_Top"],
    AIRP: ["AirP_Top", "AirP_North", "AirP_East", "AirP_South", "AirP_West"],
    WIND: ["Wind_Top", "Wind_North", "Wind_East", "Wind_South", "Wind_West", "Wind_Mean"],
}

# Define default features:
FEATURES_DEFAULT =  FEATURES_MUST_KEEP + FEATURES_OPTIONAL[SHORT_W]


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
# CV = RepeatedKFold(n_splits=5, n_repeats=3)
# CV = RepeatedKFold(n_splits=10, n_repeats=3)
CV = 3
N_ITER = 10
SCORING = 'r2'
N_JOBS = -1

# Scenarios name.
HIGH_ALBEDO_WALLS = "high_albedo_walls"
TREES_EXTREME = "trees_extreme"
COOL_PAVEMENT = "cool_pavement"
TREES_SURROUND = "trees_surround"
WALL_SHADE = "wall_shade"
PV_ROOFTOP_TREES = "pv_rooftop_and_trees"
PV_ROOFTOP = "pv_rooftop"
PV_SIDEWALKS = "pv_sidewalks"
SCENARIOS = [
    'wall_shade',
    'trees_surround',
    'trees_light',
    'trees_extreme',
    'rooftop_wall_shade',
    # 'rooftop_wall_and_trees_surround', # only until 8pm
    'rooftop_wall_and_trees_light',
    'pv_sidewalks_2',
    'pv_sidewalks',
    'pv_rooftop_and_trees_surround',
    'pv_rooftop_and_trees_light',
    'pv_rooftop',
    'high_albedo_walls',
    'green_roof',
    'cool_pavement',
]


# MODELS config:
# RF
rf_name = "rf"
rf_base = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
rf_param = {
    'criterion': ['squared_error', 'absolute_error', 'poisson'],
    'n_estimators': [100, 200, 300],  # Number of trees in the random forest
    'max_features': ['log2', 'sqrt'],  # Number of features to consider at every split
    'max_depth': [10, 20, 30, None],   # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],     # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]         # Method of selecting samples for training each tree
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


# Helper functions:
def get_features_and_title(optional_feature_names):
    final_features = FEATURES_MUST_KEEP
    final_feature_title = ""
    
    for i, feature_name in enumerate(optional_feature_names):
        final_features += FEATURES_OPTIONAL[feature_name]
        final_feature_title += feature_name
        if i!= len(optional_feature_names)-1:
            final_feature_title += "_"


    if final_feature_title == "":
        final_feature_title = "Default"
    return final_features, final_feature_title


def get_paths(feature_title):
    # Define all paths:
    RESULT_FEATURE_DIR_PATH = f"{RESULT_DIR_PATH}/{feature_title}"
    MODEL_DIR_PATH = f"{RESULT_FEATURE_DIR_PATH}/models"
    SCORES_DIR_PATH = f"{RESULT_FEATURE_DIR_PATH}/scores"
    PLOTS_DIR_PATH = f"{RESULT_FEATURE_DIR_PATH}/plots"
    SUMMARY_FILE_PATH = f"{SCORES_DIR_PATH}/summary.csv"

    return RESULT_FEATURE_DIR_PATH, MODEL_DIR_PATH, SCORES_DIR_PATH, PLOTS_DIR_PATH, SUMMARY_FILE_PATH
    