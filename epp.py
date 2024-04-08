import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


st.title(' Energy Production Prediction  :bar_chart:')
st.subheader("We have used Random Forest Regression") 
#st.subheader('Regression Model To Predict The Energy Generated')

def user_input_features():
  temperature = st.number_input('Temperature:(Degree celcius)', key = "slider1")
  exhaust_vacuum = st.number_input('Exhaust_vacuum:(cm Hg)', key="slider2")
  amb_pressure = st.number_input('Amb_pressure:(millibar)',key ="slider3")
  r_humidity = st.number_input('Relative_humidity:(%)', key ="slider4")
  data = {'temperature' : temperature,
          'exhaust_vacuum' : exhaust_vacuum,
          'amb_pressure' : amb_pressure,
          'r_humidity' : r_humidity}
  features = pd.DataFrame(data,index=[0])
  return features
df2 = pd.read_csv("datasets/energy_production (1).csv",delimiter=";")

df = user_input_features()
cols_to_scale = ['temperature','exhaust_vacuum','amb_pressure','r_humidity']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
df[cols_to_scale] = scaler.transform(df[cols_to_scale])
X = df2.iloc[:,0:4]
y = df2.iloc[:,4:5]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40)
RFE = RandomForestRegressor(n_estimators=200,max_depth = None, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 2)
RFE_Model = RFE.fit(X_train,y_train)
print(RFE_Model)
Output = RFE.predict(df)

st.write(f"Energy Production:(kWh) {Output}")





Output