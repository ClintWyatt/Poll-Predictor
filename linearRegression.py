import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression,LogisticRegression,Perceptron
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import aux_functions as aux # local library to shorten the amount of code shown


#another_set1=pd.read_csv('2012-general-election-romney-vs-obama.csv')
#another_set2=pd.read_csv('presidential_polls_2020.csv')

full_training=pd.read_csv('2016_sorted_polls.csv')

training_columns = ["cycle","branch","type","matchup","forecastdate","state","startdate",
"enddate","pollster","grade","poll_wt","population","samplesize","rawpoll_johnson",
"rawpoll_mcmullin","adjpoll_clinton","adjpoll_trump","adjpoll_johnson","adjpoll_mcmullin",
"multiversions","url","poll_id","question_id","createddate","timestamp"]

full_training = aux.strip_columns(full_training,training_columns)
#full_training=full_training.drop(columns=["pred_trump"])
#full_training=full_training.drop(columns=["pred_clinton"])

#print(full_training)

#setting the predictions
full_training.loc[full_training["rawpoll_trump"] < full_training["rawpoll_clinton"], "pred_trump"] = 0.0
full_training.loc[full_training["rawpoll_trump"] > full_training["rawpoll_clinton"], "pred_trump"] = 1.0
full_training.loc[full_training["rawpoll_trump"] < full_training["rawpoll_clinton"], "pred_clinton"] = 1.0
full_training.loc[full_training["rawpoll_trump"] > full_training["rawpoll_clinton"], "pred_clinton"] = 0.0
full_training.loc[full_training["rawpoll_trump"] == full_training["rawpoll_clinton"], "pred_clinton"] = 0.0
full_training.loc[full_training["rawpoll_trump"] == full_training["rawpoll_clinton"], "pred_trump"] = 0.0


#full_training.rawpoll_trump = full_training.rawpoll_trump.apply(to_int)
#full_training.actual_trump = full_training.actual_trump.apply(to_int)
#full_training.rawpoll_clinton=full_training.rawpoll_clinton.apply(to_int)
#full_training.actual_clinton=full_training.actual_clinton.apply(to_int)

#creating another column for the correct result
full_training.loc[(full_training["rawpoll_trump"] < full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 1.0
full_training.loc[(full_training["rawpoll_trump"] > full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 0.0
full_training.loc[(full_training["rawpoll_trump"] < full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 0.0
full_training.loc[(full_training["rawpoll_trump"] > full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 1.0
full_training.loc[(full_training["rawpoll_trump"] == full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 0.0
full_training.loc[(full_training["rawpoll_trump"] == full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 0.0

training_target=full_training["rawpoll_trump"]
reg = LinearRegression(fit_intercept=True, normalize=False)
reg.fit(full_training, training_target)
print(reg.score(full_training, training_target))
print(full_training)

st.write(full_training)
st.write('reg score = ', reg.score(full_training, training_target))