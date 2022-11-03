class Model():
    def __init__(self, base_name, random_name, base_model, random_grid):
        self.base_name = base_name
        self.base = base_model
        
        self.random_name = random_name
        self.random = RandomizedSearchCV(
                            estimator = self.base,
                            param_distributions = random_grid,
                            n_iter = 20,
                            cv = 5,
                            verbose = 2,
                            scoring ='r2',
                            random_state = 42,
                            n_jobs = -1)
    
    def get_base_model(self):
        return self.base_name, self.base
    
    
    def get_random_model(self):
        return self.random_name, self.random
# RF
# A. base model
rf_name = "rf"
rf_base = RandomForestRegressor(n_estimators = 100, random_state = 42)

# B. grid for random model
rf_random_grid = {
                # [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)],
                'max_features': ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
                'criterion': ['squared_error', 'absolute_error', 'poisson'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [ 1, 2, 4],
                'bootstrap': [True] }

# XGB
xgb_name = "xgb"
xgb_base = XGBRegressor(n_estimators = 100, random_state = 42)

xgb_random_grid = {
    'learning_rate' : [0.1, 0,2 ,0.3, 0.4],
    'n_estimators':[ 100, 250, 500, 1000],
    'min_child_weight':[1, 2, 4, 5, 8], 
    'max_depth': [4,6,7,8],
    'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7, 1 ],
    'booster': ['gbtree', 'gblinear'] }

# LGBM
# A. base model
lgbm_name = "lgbm"
lgbm_base = LGBMRegressor(random_state = 42)

# B. grid for random model
lgbm_random_grid = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
                'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],
                'num_leaves': [10,15,20,30,40], 
                'min_child_samples': [10,20,40,50,100],
                'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                'subsample': [0.1, 0.3, 0.8, 1.0], 
                'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
                'colsample_bytree': [0.4, 0.5, 0.6, 1.0],
                'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

# CATBOOST
# A. base model
catboost_name = "catboost"
catboost_base = CatBoostRegressor(random_state = 42, verbose=False)


# B. Randomized tuned model 
catboost_random_grid = {
        'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
        'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}