import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.svm import SVR

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models ={
            "Linear Regression": LinearRegression(),
            "k-Nearest Neighbors": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Regressor": SVR(),
            "XGBRegressor": XGBRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=False),
            "AdaBoostRegressor": AdaBoostRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
            }
        
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber'],
                    'learning_rate':[.1,.01,.05,.001],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64]
                },
                "k-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ["uniform", "distance"],
                    'algorithm': ["auto", "ball_tree", "kd_tree"]
                },
                "Support Vector Regressor": {
                    'kernel': ["linear", "rbf"],
                    'C': [0.1, 1, 10],
                    'gamma': ["scale", "auto"]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models,param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]    

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("no best model found")

            logging.info("best model found for training and test dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)

        


