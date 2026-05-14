import pandas as pd
import numpy as np
import os
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


import matplotlib.pyplot as plt
import seaborn as sns

from src.pipeline.data_pipeline import Data_Pipeline 
import joblib

skewed_cols = [
            # Core logistics
            'delivery_distance',
            'total_product_weight_g',
            'total_volume',
            'freight_ratio',
            'weight_per_item',
            'volume_per_item',

            # Order info
            'n_items',
            'total_price',
            'payment_value',
            'payment_installments',

            # Interactions
            'distance_x_weight',
            'logistics_complexity',

            # Seller behavior
            'seller_previous_order_count',
            'seller_timely_delivery_avg',

            # Structural
            'n_unique_sellers',
            

        ]
normal_cols = []

categorical_cols = ['category','payment_type']
binary_cols = ['is_same_state']

class Prediction_Model:
    def __init__(self):
        path = os.path.join('data','processed')
        training_data_path = os.path.join(path,'training_data.csv')
        testing_data_path = os.path.join(path,'testing_data.csv')
        if (
            not os.path.exists(training_data_path) or 
            not os.path.exists(testing_data_path)   
            ):


            
            dp =  Data_Pipeline()
            dp.generate_train_test()

        
        try:
            dataset = {}
            training_df = pd.read_csv(training_data_path)
            testing_df = pd.read_csv(testing_data_path)

            dataset['training_dataset'] = training_df
            dataset['testing_dataset'] = testing_df

            self.dataset = dataset
            # return (training_df,testing_df)
        
        except Exception as e:

            print(e)


        
    def __data_preprocessing(self):
        
        training_data = self.dataset['training_dataset']
        testing_data = self.dataset['testing_dataset']

        delivery_days_upper_limit =  training_data['delivery_days'].quantile(0.99)
        traing_data_clip  = training_data[training_data['delivery_days']<=delivery_days_upper_limit]
        testing_data_clip  = testing_data[testing_data['delivery_days']<=delivery_days_upper_limit]

       

        x_train = traing_data_clip[skewed_cols+ normal_cols + categorical_cols + binary_cols]
        x_test = testing_data_clip[skewed_cols+ normal_cols + categorical_cols + binary_cols]
        
        y_train = traing_data_clip['delivery_days']
        y_train_transformed = np.log1p(y_train)


        y_test = testing_data_clip['delivery_days']
        y_test_transformed = np.log1p(y_test)



        # removing the outliers.
        training_data =  training_data[training_data['delivery_distance']<4000]
        testing_data = testing_data[testing_data['delivery_distance']<4000]


        top_cols_value =  x_train['category'].value_counts().sort_values(ascending=False).nlargest(10).index.tolist()

        x_train.loc[:,'category'] = x_train['category'].where(x_train['category'].isin(top_cols_value), 'other')

        x_test.loc[:,'category'] = x_test['category'].where(x_test['category'].isin(top_cols_value),'other')


        return (x_train,x_test,y_train,y_test)
    


    def build_model(self):


        x_train,x_test,y_train,y_test =  self.__data_preprocessing()
        print(f'shape of x_train is {x_train.shape}')
        skwed_num_pipeline = Pipeline(steps=[
            ('imputation',SimpleImputer(strategy='median')),
            ('log',PowerTransformer(method='yeo-johnson')),
            ('standardize',StandardScaler())
        ] )

        normal_num_pipeline = Pipeline(steps=[
            ('imputation',SimpleImputer(strategy='median')),
            ('standardize',StandardScaler())
        ] )

        object_pipeline = Pipeline(
            steps = [
            
            ('impute',SimpleImputer(strategy='most_frequent')),
                
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        binary_cols_pipeline = Pipeline(steps=[
            ('imputation',SimpleImputer(strategy='most_frequent')),
        
        ] )



        preprocessor = ColumnTransformer(transformers=[
            ('skewed', skwed_num_pipeline, skewed_cols),
            ('normal', normal_num_pipeline, normal_cols),
            ('categorical', object_pipeline, categorical_cols),
            ('binary', binary_cols_pipeline, binary_cols)
        ])



        model_pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', RandomForestRegressor(n_estimators=170,max_depth=15, random_state=42))
        ])

        model_pipeline.fit(x_train,y_train)
        save_path = os.path.join('models','prediction_model.pkl')
        if not os.path.exists('models'):
            os.makedirs('models')
        try:
            joblib.dump(model_pipeline,save_path, compress=3 )
        except FileNotFoundError as e:
            print("Prediction model saved Directory not exist.")
            print(e)
        print("model saved successfully.")



if __name__ == "__main__":
    pm = Prediction_Model()
    pm.build_model()


    ### Last test: successfully run it.