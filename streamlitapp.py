import pandas as pd 
import joblib
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import streamlit as st
# from sklearn.naive_bayes import GaussianNB

model = joblib.load("liveModelV1.pk1")
data = pd.read_csv('mobile_price_range_data (1).csv')
x = data.iloc[: ,:-1]
y = data.iloc[:, -1]
x_train , x_test, y_train, y_test = train_test_split (x , y, test_size =0.2, random_state = 42)
#make prediction for x_test
y_pred = model.pred(x_test)
#calculate accuracy \
accuracy = accuracy_score(y_test, y_pred)
#page title 
st.title("Model Accuracy and Real-Time Prediction")
#Display accuracy 
st.write(f"Model{accuracy}")
#real time prediction based on user input
st.header("Real-Time Prediction")
input_data = []
for col in x_test.columns:
     input_value = st.number_input(f'Input for feature {col}', value=0)
     input_data.append(input_value)
#convert input data to dataframe 
input_df = pd.DataFrame([input_data],columns=x_test.columns)
#make predictions 
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f'Prediction:{prediction[0]}')