import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

full_training=pd.read_csv('2016_presidential_polls.csv')
another_set1=pd.read_csv('2012-general-election-romney-vs-obama.csv')
another_set2=pd.read_csv('presidential_polls_2020.csv')

print(full_training)
print(another_set1)
print(another_set2)

x = st.slider('x')
st.write(x, 'squared is', x * x)