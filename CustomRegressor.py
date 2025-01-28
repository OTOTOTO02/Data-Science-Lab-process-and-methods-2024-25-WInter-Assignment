import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

def split_gender(X_train_val_df:pd.DataFrame):
    X_train_val_df2 = X_train_val_df.copy()
    
    X_train_male_df = X_train_val_df2[X_train_val_df2['gender'] > 0]
    X_train_female_df = X_train_val_df2[X_train_val_df2['gender'] < 0]

    X_train_male_df = X_train_male_df.drop(columns=['gender'], axis=1)
    X_train_female_df = X_train_female_df.drop(columns=['gender'], axis=1)

    return X_train_male_df, X_train_female_df

def split_ages(X_train_val_df:pd.DataFrame, ages_df:pd.DataFrame):
    ages_df2 = ages_df.copy()
    
    ages_male = ages_df2[X_train_val_df['gender'] > 0]
    ages_female = ages_df2[X_train_val_df['gender'] < 0]

    return ages_male, ages_female

def merge_genders(y_male,y_female, index_male, index_female):
    y_male_s = pd.Series(y_male, index=index_male)
    y_female_s = pd.Series(y_female, index=index_female)

    y = pd.concat([y_male_s, y_female_s], axis=0).sort_index()

    return y

class TripleForestWithGenderDivision(BaseEstimator, RegressorMixin):
    def __init__(self, full_params={}, male_params={}, female_params={}):
        self.full_params = full_params 
        self.male_params = male_params 
        self.female_params = female_params
        self.full_model = RandomForestRegressor(**self.full_params)
        self.male_model = RandomForestRegressor(**self.male_params)
        self.female_model = RandomForestRegressor(**self.female_params)
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame):
        self.full_model.fit(X, y.values.reshape((-1,)))
        
        # Split data by gender
        X_train_male, X_train_female = split_gender(X)
        ages_male, ages_female = split_ages(X, y)

        # Fit gender-specific models
        self.male_model.fit(X_train_male, ages_male.values.reshape((-1,)))
        self.female_model.fit(X_train_female, ages_female.values.reshape((-1)))

        return self

    def predict(self, X:pd.DataFrame):
        # Predict using the full model for fallback
        full_preds = self.full_model.predict(X)
        
        # Gender-specific predictions
        X_male, X_female = split_gender(X)
        
        y_pred = merge_genders(
            self.male_model.predict(X_male), 
            self.female_model.predict(X_female), 
            X_male.index, 
            X_female.index
        ).sort_index()

        return (y_pred.values+full_preds)/2