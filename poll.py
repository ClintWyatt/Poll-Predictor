import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression,LogisticRegression,Perceptron
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#another_set1=pd.read_csv('2012-general-election-romney-vs-obama.csv')
#another_set2=pd.read_csv('presidential_polls_2020.csv')

def to_int(x):
    
    x = int(x)
    return x

def to_float(x):

    x = float(x)
    return x

full_training=pd.read_csv('2016_sorted_polls.csv')
full_training=full_training.drop(columns=["cycle"])
full_training=full_training.drop(columns=["branch"])
full_training=full_training.drop(columns=["type"])
full_training=full_training.drop(columns=["matchup"])
full_training=full_training.drop(columns=["forecastdate"])
full_training=full_training.drop(columns=["state"])
full_training=full_training.drop(columns=["startdate"])
full_training=full_training.drop(columns=["enddate"])
full_training=full_training.drop(columns=["pollster"])
full_training=full_training.drop(columns=["grade"])#could convert this to floats later on
full_training=full_training.drop(columns=["poll_wt"])#need a method for this column
full_training=full_training.drop(columns=["population"])
full_training=full_training.drop(columns=["samplesize"])
full_training=full_training.drop(columns=["rawpoll_johnson"])
full_training=full_training.drop(columns=["rawpoll_mcmullin"])
full_training=full_training.drop(columns=["adjpoll_clinton"])
full_training=full_training.drop(columns=["adjpoll_trump"])
full_training=full_training.drop(columns=["adjpoll_johnson"])
full_training=full_training.drop(columns=["adjpoll_mcmullin"])
full_training=full_training.drop(columns=["multiversions"])
full_training=full_training.drop(columns=["url"])
full_training=full_training.drop(columns=["poll_id"])
full_training=full_training.drop(columns=["question_id"])
full_training=full_training.drop(columns=["createddate"])
full_training=full_training.drop(columns=["timestamp"])
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

clf=LogisticRegression()
clf.fit(full_training[["pred_trump"]], full_training[["correctResult"]])

x1 = np.linspace(4.5, 8.5, 1000).reshape(-1, 1)
y1=clf.predict_proba(x1)[:,1]
print(clf.score(full_training[["pred_trump"]], full_training[["correctResult"]]))


st.write(full_training)
st.write('reg score = ', reg.score(full_training, training_target))




