import streamlit as st
import os
import pandas as pd
import joblib
import numpy as np


def navigator():
    # ── Page Navigation ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Navigate Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.page_link("app.py",          label="📊 Overview",         use_container_width=True)
        st.caption("Sales trends, revenue, orders by state")

    with col2:
        st.page_link("pages/2_Delivery_Analysis.py", label="📦 Delivery Analysis", use_container_width=True)
        st.caption("Delivery times, on-time rates by state")

    with col3:
        st.page_link("pages/3_Prediction.py",        label="🤖 ML Prediction",     use_container_width=True)
        st.caption("Predict delivery days with XGBoost")

    with col4:
        st.page_link("pages/4_Seller_Analysis.py",   label="🏪 Seller Analysis",   use_container_width=True)
        st.caption("Top sellers, performance rankings")

    with col5:
        st.page_link("pages/5_Customer_Analysis.py", label="👥 Customer Analysis", use_container_width=True)
        st.caption("Reviews, satisfaction, payments")



@st.cache_data
def load_datasets(data_path="data/raw"):

    data = {}

    try:
        data['sellers'] = pd.read_csv(os.path.join(data_path,'sellers.csv'))
        data['items'] = pd.read_csv(os.path.join(data_path,'order_items.csv'))
        data['orders'] = pd.read_csv(os.path.join(data_path,'orders.csv'))
        data['reviews'] = pd.read_csv(os.path.join(data_path,'order_reviews.csv'))
        data['geolocation'] = pd.read_csv(os.path.join(data_path,'geolocation.csv'))
        data['customers'] = pd.read_csv(os.path.join(data_path,'customers.csv'))
        data['payments'] = pd.read_csv(os.path.join(data_path,'order_payments.csv'))
        data['products'] = pd.read_csv(os.path.join(data_path,'products.csv'))
        data['category'] = pd.read_csv(os.path.join(data_path,'product_category_name_translation.csv'))

    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        return None

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    return data


@st.cache_data
def load_processed_data(data_path='data/processed'):
    data = {}
    try:
        data['review_category_wise'] = pd.read_csv(os.path.join(data_path,'product_category_reviews.csv'))

    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        return None

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    return data











# delivery prediction model



def calculate_distance(lat1,lng1,lat2,lng2):
    

    # Calucate the difference.
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = (np.sin(dlat/2)**2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2

    c =  2 * np.asin(np.sqrt(a))

    r = 6371

    return np.round((c * r),0)
    


def create_record(inputs:dict, dataset:dict,model):
    unique_geo = dataset['unique_geo']
    seller = dataset['seller']
    product = dataset['product']

    cust_row = unique_geo[unique_geo['geolocation_zip_code_prefix']==inputs['cust_zip_code_inp']]
    customer_latitude,customer_longitude,customer_state = cust_row[['lat','lng','state']].values[0]

    #fetch the select product
    seller_row = product[(product['category_formated']==inputs['category_inp']) & 
                         (product['product_id'] ==inputs['product_inp']) & 
                         (product['seller_id'] ==inputs['seller_id_inp']) ]
    seller_cols =  ['seller_state','seller_latitude','seller_longitude','raw_category','product_volume_cm3','product_weight_g']
    seller_state,seller_latitude,seller_longitude,raw_category,product_volume_cm3,product_weight_g = seller_row[seller_cols].values[0]



    # seller performance features
    seller_performce_row = seller[seller['seller_id'] ==inputs['seller_id_inp']]
    if seller_performce_row.empty:
        st.warning("No seller record available.")
        return -1
    seller_timely_delivery_avg,seller_previous_order_count = seller_performce_row[['seller_timely_delivery_avg','seller_previous_order_count']].values[0]
    

    features ={}
    features['is_same_state'] = (1 if (customer_state == seller_state) else 0)
    features['category'] = raw_category
    features['payment_type'] = inputs['payment_type_inp']
    features['n_unique_sellers'] = 1
    features['payment_installments'] = inputs['payment_installments_inp']
    features['total_price'] = inputs['total_price_inp']
    features['n_items'] = inputs['item_quantity_inp']

    #  # Core logistics
    features['total_product_weight_g'] = (product_weight_g * inputs['item_quantity_inp'])
    features['total_volume'] = (product_volume_cm3* inputs['item_quantity_inp'])
    features['weight_per_item'] = product_weight_g
    features['volume_per_item'] = product_volume_cm3



    features['delivery_distance'] = calculate_distance(customer_latitude,customer_longitude,seller_latitude,seller_longitude)
    features['freight_ratio'] = inputs['total_freight_inp'] / (inputs['total_price_inp'] + 1)
    features['payment_value'] = inputs['payment_value_inp']

    features['logistics_complexity'] =1

    
    features['seller_timely_delivery_avg'] = seller_timely_delivery_avg
    features['seller_previous_order_count'] = seller_previous_order_count
    features['distance_x_weight'] = (features['delivery_distance'] * features['total_product_weight_g'])
    features_df = pd.DataFrame([features])
    print(f'shape of the features is {features_df.shape}')
    y_pred = model.predict(features_df)
  
    y_pred = np.round(y_pred)[0]
    st.session_state.predicted_days = int( y_pred)
    print(f'{y_pred} Delivery days required.')