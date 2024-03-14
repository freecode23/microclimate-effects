import pandas as pd
import numpy as np
import datetime

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm

# Processing
from sklearn.model_selection import train_test_split

# parameters search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Scoring
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Vizualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import figure

# save model
import pickle
import os
import joblib

import config_dataset3 as config

class Data(object):
    '''
    This class encapsulates the datas that we will need for training and testing for each building.
    It uses July 7th for the test data.
    '''
    def __init__(self, bldg_name, df, model_cols):
        '''
        Parameters:
            bldgs_df_list (str) : list of dataframe with each dataframe consisting of a single building. 
            dropped_cols (str) : The path for the test csv file. 
        '''

        # 1. Filter out date before 8am
        df = df.copy()

        # 2. Use July 7th as test set.
        df[config.DATE_TIME] = pd.to_datetime(df[config.DATE_TIME])
        df = df[df['Hour'] >= 8]
        test_data = df[(df[config.DATE_TIME].dt.month == 7) & (df[config.DATE_TIME].dt.day == 7)]
        train_data = df[~((df[config.DATE_TIME].dt.month == 7) & (df[config.DATE_TIME].dt.day == 7))]
        test_data.to_csv(config.BASE_PATH + "/test_data.csv", mode='w')

        # 3. Create train and test data.
        train_data = train_data[model_cols]
        test_data = test_data[model_cols]
        
        # 4. Create X, y train and test datasets.
        self.bldg_name = bldg_name
        self.X_train = train_data.drop(config.CHWTON_SQM, axis=1)
        self.y_train = train_data[config.CHWTON_SQM]
        self.X_test = test_data.drop(config.CHWTON_SQM, axis=1)
        self.y_test = test_data[config.CHWTON_SQM]
        self.df = df



class Model():
    '''
    Given a base model and grid or random params, the class will create
    the search grid and assign the name to each of this model.
    '''
    def __init__(self, name, base_model, param, cv, n_iter, search_mode):
        # Base model
        self.name = name
        self.base = base_model
        self.search_mode = search_mode
        self.best = None
        
        self.base_name = self.name + "_base"
        self.best_name = self.name + "_" + self.search_mode
        
        # Randomized search model
        if search_mode == "random":
            self.clf = RandomizedSearchCV(
                estimator = self.base,
                param_distributions = param,
                n_iter = n_iter,
                cv = cv,
                verbose = 0,
                random_state = config.RANDOM_STATE,
                scoring = config.SCORING,
                n_jobs = config.N_JOBS)
            
        # Grid search model
        else:
            self.clf = GridSearchCV(
                estimator=self.base,
                param_grid = param, 
                cv = cv, 
                verbose = 0, 
                scoring = config.SCORING,
                n_jobs = config.N_JOBS)
            
    def save_models(self, bldg_name, model_dir_path):
        '''
        Save the base and best model.
        '''
        models = [(self.base_name, self.base), (self.best_name, self.best)]
        for name, model in models:
            save_dir = f"{model_dir_path}"
            # Create the path to save the models.
            isExist = os.path.exists(save_dir)

            if not isExist:
               # Create a new directory if it does not exist
               os.makedirs(save_dir)

            # Save the mode.
            if "lgbm" in name:
                # Set the filename.
                filename = f"{name}_{bldg_name}.pkl"

                #  Save to the model to filepath.
                save_path = f"{save_dir}/{filename}"
                joblib.dump(model, save_path)

                # To Reload:
                # model = joblib.load(save_path)

            else:
                # Set the filename.
                filename = f"{name}_{bldg_name}.sav"

                # Save to the model to filepath
                save_path = f"{save_dir}/{filename}"
                pickle.dump(model, open(save_path, 'wb'))



class Scores(object):
    '''
    This class stores all scores for all models for all buildings.
    '''
    def __init__(self):
        # Initialized scores dataframe to store the scores for all the models trained.
        self.columns=['model', 'bldg', 'r2_train', 'r2_test', 'rmse_test','mbe_test']
        self.scores_df= pd.DataFrame(columns=self.columns)
        
    def get_MBE(self, y_true, y_pred):
        '''
        Parameters:
            y_true (array): Array of observed values
            y_pred (array): Array of prediction values

        Returns:
            mbe (float): Bias score
        '''
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_true = y_true.reshape(len(y_true),1)
        y_pred = y_pred.reshape(len(y_pred),1)   
        diff = (y_pred-y_true)
        mbe = diff.mean()
        return mbe
    
    def test_and_store_score(self, model, model_name, data):
        # Test and get r2, rmse, and mbe scores.
        y_pred = model.predict(data.X_test)
        r2 = r2_score(data.y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(data.y_test, y_pred))
        mbe = self.get_MBE(data.y_test, y_pred)
        
        # Store all the scores and save as csv.
        new_score_data = {
            'model': model_name,
            'bldg': data.bldg_name,
            "r2_train":r2_train,
            "r2_test":r2,
            'rmse_test':rmse,
            'mbe_test':mbe}
        new_score_row = pd.DataFrame.from_records([new_score_data])
        self.scores_df = pd.concat([self.scores_df, new_score_row])
    
    def train_test_and_store_score(self, model, model_name, data):
        '''
        Function to train the model and make prediction using the data object's X_test df.
        Return the best model after training if the model input is a search classifier.
        '''
        print("\nmodel_name:", model_name)
        
        # Train and get r2 scores.
        model.fit(data.X_train, data.y_train)
        if("random" in model_name) or ("grid" in model_name):
            print("best_params=", model.best_params_)
            # reassign using the best model.
            model = model.best_estimator_
            
        r2_train = model.score(data.X_train, data.y_train)
        print(data.X_train.columns)
        # Test and get r2, rmse, and mbe scores.
        y_pred = model.predict(data.X_test)
        r2 = r2_score(data.y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(data.y_test, y_pred))
        mbe = self.get_MBE(data.y_test, y_pred)
        
        # Store all the scores and save as csv.
        new_score_data = {
            'model': model_name,
            'bldg': data.bldg_name,
            "r2_train":r2_train,
            "r2_test":r2,
            'rmse_test':rmse,
            'mbe_test':mbe}
        new_score_row = pd.DataFrame.from_records([new_score_data])
        self.scores_df = pd.concat([self.scores_df, new_score_row])
        
        # Return the best model.
        return model
    
    def save_as_csv(self, filename, scores_dir_path):        
        # Construct the full file path and save.
        score_file_path = os.path.join(scores_dir_path, f"{filename}.csv")

        # Check if the directory exists, and create it if it doesn't.
        os.makedirs(os.path.dirname(score_file_path), exist_ok=True)
    
        # Check if the file already exists to decide on writing headers or not.
        file_exists = os.path.isfile(score_file_path)
    
        # Append to the file if it exists, otherwise create a new file with headers.
        if file_exists:
            # Append without headers.
            print("Appending scores in", score_file_path)
            self.scores_df.to_csv(score_file_path, mode='a', header=False, index=False)
        else:
            # Create a new file with headers.
            print("Saving new scores in", score_file_path)
            self.scores_df.to_csv(score_file_path, mode='w', header=True, index=False)
        