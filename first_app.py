# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 08:53:57 2024

@author: ASHNER_NOVILLA
"""

import streamlit as st
import time
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import plotly.express as px


df_iris = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

df_iris_original = df_iris.copy()

le = LabelEncoder()
df_iris['variety'] = le.fit_transform(df_iris['variety'])

## model building

X = df_iris.loc[:,df_iris.columns != 'variety']
y = df_iris['variety'] 

model = LogisticRegression()

model.fit(X, y)



st.title(":blue[_This is my App_] :violet[Iris Dataset] 🌸")
st.markdown("Author: **ASHNER_NOVILLA**")
st.markdown("Created on Sat Dec  7 08:53:57 2024")


st.header("First Part of the Project", divider="violet")

st.write("This is my first application deployed using streamlit")

text_generation = ''' The Iris Dataset contains four features (length and width of sepals and petals) of 
50 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). 
These measures were used to create a linear discriminant model to classify the species.
'''

def stream_data():
    for word in text_generation.split(" "):
        yield word + " "
        time.sleep(0.001)

st.write(stream_data)

st.write("-----------------------------------------------------------------------")

col1, col2 = st.columns([1, 2])

presentation_button = col1.button("Data Presentation", icon="😃", use_container_width=False)

reset_button = col2.button("Data Reset", icon="😡", use_container_width=False)

if presentation_button:
    st.subheader("Iris Data Set")

    st.dataframe(df_iris_original, hide_index=True)

    st.write("-----------------------------------------------------------------------")

    st.subheader("Iris Scatter Plot")

    col1, col2 = st.columns([1, 1])
    

    fig1 = px.scatter(df_iris, x="sepal.width", y="sepal.length")
    fig2 = px.scatter(df_iris, x="petal.width", y="petal.length")

    col1.plotly_chart(fig1)

    col2.plotly_chart(fig2)


elif reset_button:
    st.write(" Data Hidden ")

else:
    st.write(" ")

st.write("-----------------------------------------------------------------------")
st.subheader("Input Data")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])


sepal_len = col1.number_input("Iris Sepal Length", placeholder = ' Sepal Length')

sepal_wid = col2.number_input("Iris Sepal Width", placeholder = ' Sepal Width')

petal_len = col3.number_input("Iris Petal Length", placeholder = ' Petal Length')

petal_wid = col4.number_input("Iris Petal Width", placeholder = ' Petal Width')

if st.button("Predict"):
    input_data = [[sepal_len, sepal_wid, petal_len, petal_wid]]
    prediction = model.predict(input_data)
    predicted_class = le.inverse_transform(prediction)
    st.success(f"The predicted Iris variety is: {predicted_class[0]}")
    



