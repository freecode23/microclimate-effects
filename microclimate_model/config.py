from sklearn.model_selection import RepeatedKFold
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

HIGH_ALBEDO_WALLS_DIR_PATH = f"{SCENARIOS_DIR_PATH}/{HIGH_ALBEDO_WALLS}"
TREES_EXTREME_DIR_PATH = f"{SCENARIOS_DIR_PATH}/{TREES_EXTREME}"
COOL_PAVEMENT_DIR_PATH = f"{SCENARIOS_DIR_PATH}/{COOL_PAVEMENT}"
TREES_SURROUND_DIR_PATH = f"{SCENARIOS_DIR_PATH}/{TREES_SURROUND}"
WALL_SHADE_DIR_PATH = f"{SCENARIOS_DIR_PATH}/{WALL_SHADE}"
PV_ROOFTOP_TREES_DIR_PATH = f"{SCENARIOS_DIR_PATH}/{PV_ROOFTOP_TREES}"
PV_ROOFTOP_DIR_PATH = f"{SCENARIOS_DIR_PATH}/{PV_ROOFTOP}"
PV_SIDEWALKS_DIR_PATH = f"{SCENARIOS_DIR_PATH}/{PV_SIDEWALKS}"

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

# Declare constants for the variable names that we use often.
CHWTON = "CHWTON/SQM"
BLDGNAME = "bldgname"
ISTB4 = "bldgname_ISTB 4"
PYSCHOLOGY = "bldgname_Psychology"
PSYCHOLOGY_NORTH = "bldgname_Psychology North"
DATE_TIME = 'Date_Time'

# MODELS common config:

# For base model
RANDOM_STATE = 42
N_ESTIMATORS = 100

# Fo randomized search
CV = RepeatedKFold(n_splits=10, n_repeats=3)
# CV = 3
N_ITER = 10
