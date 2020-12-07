# auxillary functions
from sklearn.linear_model import LinearRegression,LogisticRegression,Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def strip_columns(dataset, list_columns): # strip a list of columns from a numpy dataframe
    for column in list_columns:
        dataset = dataset.drop(columns=[column])
    return dataset


# hopefully used to make the correct results lines of code look smaller
#def selectwhere(dataset,columns,query): # select columns and specific samples where query is true 
    #ADD CODE


def decision_point(clf): #
    coef = clf.coef_
    intercept = clf.intercept_
    return (-intercept[0])/coef[0,0]

def to_int(x): #convert variable -> int
    
    x = int(x)
    return x

def to_float(x): #convert variable -> float

    x = float(x)
    return x

def create_scatter(full_training,list_pollsters,num):
    var="pred_clinton" #change to pred_clinton/pred_trump if needed
    rawPoll="rawpoll_clinton" #change to rawpoll_clinton/rawpoll_trump if needed
    # print(full_training)
    #below is the graph output for the trump logistic regression results 

    #=clf.predict(full_training[["rawPoll"]]) # here we are adding a column for the training_target data set where 
    #clf.score(training_target[["rawPoll"]],  training_target["correctResult"]) # getting the score
    fig, (axes) = plt.subplots(num, figsize=(8, num * (3)))#this shows the plot
    fig.subplots_adjust(hspace=0.4, wspace=0.05)
    count = 0

    for ax in axes:
        pollster_data = full_training[full_training['pollster'] == list_pollsters[count]]
        pollster_data = pollster_data.drop(columns=['pollster'])

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_last_row():
            ax.set_xlabel('rawPoll',fontsize=16)

        clf=LogisticRegression()
        clf.fit(pollster_data[[rawPoll]], pollster_data[["correctResult"]])

        x1 = np.linspace(4.5, 8.5, 1000).reshape(-1, 1)
        y1=clf.predict_proba(x1)[:,1]

        tp=pollster_data[(pollster_data["correctResult"]==1.0) & (pollster_data[var]==1.0)]#true positive
        fp=pollster_data[(pollster_data["correctResult"]==0.0) & (pollster_data[var]==1.0)]#false positive
        tn=pollster_data[(pollster_data["correctResult"]==0.0) & (pollster_data[var]==0.0)]#true negative
        fn=pollster_data[(pollster_data["correctResult"]==1.0) & (pollster_data[var]==0.0)]#false negative

        p = clf.predict_proba(pollster_data[[rawPoll]])
        ax.plot([decision_point(clf)]*1000,np.linspace(0, 1, 1000),"--",color="red")#draws a line vertically through the decision point

        ax.plot(tp[rawPoll],tp["correctResult"],"+",c="green")
        ax.plot(fp[rawPoll],fp["correctResult"],".",c="red")
        ax.plot(tn[rawPoll],tn["correctResult"],".",c="green")
        ax.plot(fn[rawPoll],fn["correctResult"],"+",c="purple")

        title = "rawpoll_trump as h(correct result), pollster: " + list_pollsters[count]
        ax.set_title(title,fontsize=10)
        ax.scatter(pollster_data[rawPoll], p[:,1], color = 'black')#p[:,1] means only have the second index of the 2d array
        # ax.set_xlabel('rawPoll',fontsize=16)
        ax.set_ylabel('correctResult',fontsize=12)
        ax.plot(x1, y1, color='green') #plots what x1 and y1 are set equal to
        ax.grid(True)#shows the lines through the x and y values representing the axis

        count += 1
    
    st.pyplot(fig)