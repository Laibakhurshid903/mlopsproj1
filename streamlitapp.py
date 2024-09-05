import pandas as pd 
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

model = joblib.load("liveModelV1.pk1")
data = pd.read_csv("mobile_price_range_data (1).csv")

X= data.iloc[:, :-1]
y= data.iloc[: , -1]
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)
                                    
#make predictions
y_pred = model.predict(X_test)

#calculate accuracy
accuracy = accuracy_score(y_test , y_pred)

#page title
st.title("Model accuracy and Real time prediction")

#display accuracy
st.write(f"Model Accuracy: {accuracy}")

#real time prediction based user inputs
st.header("Real Time prediction")
input_data = []
for col in X_test.columns:
   input_value = st.number_input(f"input for feature {col}" , value = 0.0)
   input_data.append(input_value)
   
#convert input data to dataframa 
input_df = pd.DataFrame([input_data] , columns = X_test.columns)

#make predictions 
if st.button("Predict"):
   prediction = model.predict(input_df)
   st.write(f"predictions: {prediction[0]}")