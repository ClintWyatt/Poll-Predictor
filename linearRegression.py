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

def run_linReg():
    poll_data = pd.read_csv('2016_sorted_polls.csv')
    full_training = poll_data.sample(frac=0.7,replace=False)
    full_dev = poll_data


    # full_training["state_number"] = pd.Categorical(full_training['state'],full_training['state'])
    # full_training['state_number'] = full_training['state_number'].cat.codes

    training_columns = ["cycle","branch","state","type","matchup","forecastdate","startdate",
    "enddate","pollster","grade","population","samplesize","rawpoll_johnson",
    "rawpoll_mcmullin","adjpoll_johnson","adjpoll_mcmullin",
    "multiversions","url","poll_id","question_id","createddate","timestamp"]

    full_training = aux.strip_columns(full_training,training_columns)
    full_dev = aux.strip_columns(full_dev,training_columns)

    full_dev = full_dev.dropna(axis=1)
    full_training = full_training.dropna(axis=1)


    #full_training=full_training.drop(columns=["pred_trump"])
    #full_training=full_training.drop(columns=["pred_clinton"])

    #print(full_training)

    #setting the predictions
    """
    full_training.loc[full_training["rawpoll_trump"] < full_training["rawpoll_clinton"], "pred_trump"] = 0.0
    full_training.loc[full_training["rawpoll_trump"] > full_training["rawpoll_clinton"], "pred_trump"] = 1.0
    full_training.loc[full_training["rawpoll_trump"] < full_training["rawpoll_clinton"], "pred_clinton"] = 1.0
    full_training.loc[full_training["rawpoll_trump"] > full_training["rawpoll_clinton"], "pred_clinton"] = 0.0
    full_training.loc[full_training["rawpoll_trump"] == full_training["rawpoll_clinton"], "pred_clinton"] = 0.0
    full_training.loc[full_training["rawpoll_trump"] == full_training["rawpoll_clinton"], "pred_trump"] = 0.0
    """

    #full_training.rawpoll_trump = full_training.rawpoll_trump.apply(to_int)
    #full_training.actual_trump = full_training.actual_trump.apply(to_int)
    #full_training.rawpoll_clinton=full_training.rawpoll_clinton.apply(to_int)
    #full_training.actual_clinton=full_training.actual_clinton.apply(to_int)

    #creating another column for the correct result
    """
    full_training.loc[(full_training["rawpoll_trump"] < full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 1.0
    full_training.loc[(full_training["rawpoll_trump"] > full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 0.0
    full_training.loc[(full_training["rawpoll_trump"] < full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 0.0
    full_training.loc[(full_training["rawpoll_trump"] > full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 1.0
    full_training.loc[(full_training["rawpoll_trump"] == full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 0.0
    full_training.loc[(full_training["rawpoll_trump"] == full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 0.0
    """

    training_target_trump=full_training["actual_trump"]
    training_target_clinton=full_training["actual_clinton"]

    dev_target_trump=full_dev["actual_trump"]
    dev_target_clinton=full_dev["actual_clinton"]

    full_dev = full_dev.drop(columns=['actual_trump','actual_clinton'])
    full_training = full_training.drop(columns=['actual_trump','actual_clinton'])
    print(full_training)

    reg_trump = LinearRegression(fit_intercept=True, normalize=False)
    reg_clinton = LinearRegression(fit_intercept=True, normalize=False)

    reg_trump.fit(full_training, training_target_trump)
    reg_clinton.fit(full_training, training_target_clinton)

    # print(reg.score(full_training, training_target))
    # print(full_training)

    # st.write(full_training)

    st.write('linReg_trump training score = ', reg_trump.score(full_training, training_target_trump))
    st.write('linReg_clinton training score = ', reg_clinton.score(full_training, training_target_clinton))

    st.write('linReg_trump dev score = ', reg_trump.score(full_dev, dev_target_trump))
    st.write('linReg_clinton dev score = ', reg_clinton.score(full_dev, dev_target_clinton))