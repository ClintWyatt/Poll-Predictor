import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression,LogisticRegression,Perceptron
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import altair as alt

#another_set1=pd.read_csv('2012-general-election-romney-vs-obama.csv')
#another_set2=pd.read_csv('presidential_polls_2020.csv')
def decision_point(clf):
    coef = clf.coef_
    intercept = clf.intercept_
    return (-intercept[0])/coef[0,0]

def to_int(x):
    
    x = int(x)
    return x

def to_float(x):

    x = float(x)
    return x

# get dataset
full_training=pd.read_csv('2016_sorted_polls.csv')

####################################################
### streamlit wrapper begins here###
## streamlit is interspersed throughout the code
st.title("Machine Learning for Poll Prediction")

st.header("Poll Data used for testing")
st.subheader("We used data from the 2016 presidential election to create a model to predict the outcome of the previous election")
st.dataframe(full_training)

st.header("Modeling and Testing")
st.subheader("scatter matrix of dataset features")

### scatter matrix ###
#reset_index puts the index column back into the dataframe
scatter_matrix = alt.Chart(full_training).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color='state',size='samplesize',opacity=alt.value(0.4)
).properties(
    width=150,
    height=150
).repeat( #features can be added here to change matrix size and output
    row=['actual_trump','rawpoll_trump','rawpoll_clinton','actual_clinton'],
    column=['actual_trump','rawpoll_trump','rawpoll_clinton','actual_clinton']
).interactive()

if st.checkbox('View Scatter-matrix'):
    st.altair_chart(scatter_matrix)
### End Intro
#####################################################

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

print(full_training)

#setting the columns for predictions
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

#creating another column for the correct result for correct results
full_training.loc[(full_training["rawpoll_trump"] < full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 1.0
full_training.loc[(full_training["rawpoll_trump"] > full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 0.0
full_training.loc[(full_training["rawpoll_trump"] < full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 0.0
full_training.loc[(full_training["rawpoll_trump"] > full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 1.0
full_training.loc[(full_training["rawpoll_trump"] == full_training["rawpoll_clinton"]) & (full_training["actual_trump"] > full_training["actual_clinton"]), "correctResult"] = 0.0
full_training.loc[(full_training["rawpoll_trump"] == full_training["rawpoll_clinton"]) & (full_training["actual_trump"] < full_training["actual_clinton"]), "correctResult"] = 0.0

##########################################
## section that shows some of the code used for the models
# any code put into the echo is both printed to streamlit and executed as code
st.subheader("regression models")
# using echo() to view code in the app
# echo() both outputs the code to streamlit and executes it
with st.echo():
    #Linear Regression 
    training_target=full_training["rawpoll_trump"]
    reg = LinearRegression(fit_intercept=True, normalize=False)
    reg.fit(full_training, training_target)

print(reg.score(full_training, training_target))
print(full_training)

#using echo to view code in the app
with st.echo():
    #graph probability code for Logistic Regression on Trump data
    clf=LogisticRegression()
    clf.fit(full_training[["rawpoll_trump"]], full_training[["correctResult"]])
    x1 = np.linspace(4.5, 8.5, 1000).reshape(-1, 1)
    y1=clf.predict_proba(x1)[:,1]
##############################################################################


#below is the graph output for the trump logistic regression results 

#=clf.predict(full_training[["rawpoll_trump"]]) # here we are adding a column for the training_target data set where 
#clf.score(training_target[["rawpoll_trump"]],  training_target["correctResult"]) # getting the score
fig, (ax1) = plt.subplots(1, figsize=(10, 10))#this shows the plot
tp=full_training[(full_training["correctResult"]==1.0) & (full_training["pred_trump"]==1.0)]#true positive
fp=full_training[(full_training["correctResult"]==0.0) & (full_training["pred_trump"]==1.0)]#false positive
tn=full_training[(full_training["correctResult"]==0.0) & (full_training["pred_trump"]==0.0)]#true negative
fn=full_training[(full_training["correctResult"]==1.0) & (full_training["pred_trump"]==0.0)]#false negative

p = clf.predict_proba(full_training[["rawpoll_trump"]])
ax1.plot([decision_point(clf)]*1000,np.linspace(0, 1, 1000),"--",color="red")#draws a line horizontally through the decision point

#plot and label the feature samples corresponding to model outcomes
ax1.plot(tp["rawpoll_trump"],tp["correctResult"],"+",c="green", label="True Positives")
ax1.plot(fp["rawpoll_trump"],fp["correctResult"],".",c="orange", label="False Positives")
ax1.plot(tn["rawpoll_trump"],tn["correctResult"],".",c="green", label="True Negatives")
ax1.plot(fn["rawpoll_trump"],fn["correctResult"],"+",c="orange", label="False Negatives")

ax1.set_title("rawpoll_trump as h(correct result)",fontsize=20)
ax1.scatter(full_training['rawpoll_trump'], p[:,1], color = 'black')#p[:,1] means only have the second index of the 2d array
ax1.set_xlabel('rawpoll_trump',fontsize=16)
ax1.set_ylabel('correctResult',fontsize=16)
ax1.legend(loc="upper right") #legend to diplay the meaning of the pointer color and styles
ax1.plot(x1, y1, color='green') #plots what x1 and y1 are set equal to
ax1.grid(True)#shows the lines through the x and y values representing the axis
#plt.show()

st.pyplot(fig) ## streamlit function to output matplotlib charts
######################################################################################

clf_score = clf.score(full_training[["pred_trump"]], full_training[["correctResult"]])
print(clf_score)

############
#hopefully makes a cool looking line chart

#reset index for use in graphs

score_chart = alt.Chart(full_training.reset_index())

###############################################
## chart output using the altair library
# pretty intuitive and quick to use

#st.altair_chart(score_chart.mark_point()
#.encode(alt.X('index'), alt.Y("correctResult",
#scale=alt.Scale(domain =[-1,2], clamp = True)),
#color = 'pred_trump',
#opacity=alt.value(0.5))
#.properties(width = 750,height = 250,)
#.interactive())

################################################

# output some scores to streamlit
st.write('reg score = ', reg.score(full_training, training_target))
st.write('clf model score = ',clf_score)

