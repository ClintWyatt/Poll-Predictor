import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression,LogisticRegression,Perceptron
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import aux_functions as aux

#another_set1=pd.read_csv('2012-general-election-romney-vs-obama.csv')
#another_set2=pd.read_csv('presidential_polls_2020.csv')
def run_logReg():
    full_training=pd.read_csv('pollsSorted.csv')
    

    pollster_count = full_training.groupby(['pollster'],sort=True).count()
    st.write(pollster_count['samplesize'])

    training_columns = ["cycle","branch","type","matchup","forecastdate","state","startdate",
    "enddate","grade","poll_wt","population","samplesize","rawpoll_johnson",
    "rawpoll_mcmullin","adjpoll_clinton","adjpoll_trump","adjpoll_johnson","adjpoll_mcmullin",
    "multiversions","url","poll_id","question_id","createddate","timestamp"]

    full_training = aux.strip_columns(full_training,training_columns)

    #scatter plot of outputs
    list_polls = ['CVOTER International','Ipsos','Google Consumer Surveys','SurveyMonkey']

    #full_training=full_training.drop(columns=["pred_trump"])
    #full_training=full_training.drop(columns=["pred_clinton"])

    #print(full_training)
    st.text("setting the predictions")
    with st.echo():
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

    st.text("creating another column for the correct result")
    with st.echo():
        full_training.loc[(full_training["rawpoll_trump"] < full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 1.0
        full_training.loc[(full_training["rawpoll_trump"] > full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 0.0
        full_training.loc[(full_training["rawpoll_trump"] < full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 0.0
        full_training.loc[(full_training["rawpoll_trump"] > full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 1.0
        full_training.loc[(full_training["rawpoll_trump"] == full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 0.0
        full_training.loc[(full_training["rawpoll_trump"] == full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 0.0

    aux.create_scatter(full_training,list_polls,len(list_polls))

    full_training=full_training[full_training['pollster'] == 'CVOTER International'] #getting all results for 1 poll. Change as needed
    full_training=full_training.drop(columns=['pollster'])

    with st.echo():
        var="pred_clinton" #change to pred_clinton/pred_trump if needed
        rawPoll="rawpoll_clinton" #change to rawpoll_clinton/rawpoll_trump if needed
        
        #below is the graph output for the trump logistic regression results 
        clf=LogisticRegression()
        clf.fit(full_training[[rawPoll]], full_training[["correctResult"]])

    x1 = np.linspace(4.5, 8.5, 1000).reshape(-1, 1)
    y1=clf.predict_proba(x1)[:,1]
    #=clf.predict(full_training[["rawPoll"]]) # here we are adding a column for the training_target data set where 
    #clf.score(training_target[["rawPoll"]],  training_target["correctResult"]) # getting the score
    fig, (ax1) = plt.subplots(1, figsize=(10, 10))#this shows the plot
    tp=full_training[(full_training["correctResult"]==1.0) & (full_training[var]==1.0)]#true positive
    fp=full_training[(full_training["correctResult"]==0.0) & (full_training[var]==1.0)]#false positive
    tn=full_training[(full_training["correctResult"]==0.0) & (full_training[var]==0.0)]#true negative
    fn=full_training[(full_training["correctResult"]==1.0) & (full_training[var]==0.0)]#false negative

    p = clf.predict_proba(full_training[[rawPoll]])
    ax1.plot([aux.decision_point(clf)]*1000,np.linspace(0, 1, 1000),"--",color="red")#draws a line vertically through the decision point

    ax1.plot(tp[rawPoll],tp["correctResult"],"+",c="green")
    ax1.plot(fp[rawPoll],fp["correctResult"],".",c="red")
    ax1.plot(tn[rawPoll],tn["correctResult"],".",c="green")
    ax1.plot(fn[rawPoll],fn["correctResult"],"+",c="purple")

    ax1.set_title("rawpoll_trump as h(correct result), pollster: CVOTER International",fontsize=18)
    ax1.scatter(full_training[rawPoll], p[:,1], color = 'black')#p[:,1] means only have the second index of the 2d array
    ax1.set_xlabel('rawPoll',fontsize=16)
    ax1.set_ylabel('correctResult',fontsize=16)
    ax1.plot(x1, y1, color='green') #plots what x1 and y1 are set equal to
    ax1.grid(True)#shows the lines through the x and y values representing the axis
    plt.show()

    st.pyplot(fig) ## streamlit function to output matplotlib charts

    log_score = clf.score(full_training[[rawPoll]], full_training[["correctResult"]])

    st.write(log_score)
    print(log_score)
#st.write('reg score = ', reg.score(full_training, training_target))