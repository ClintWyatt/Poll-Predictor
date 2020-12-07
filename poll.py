from altair.vegalite.v4.schema.channels import Tooltip
import pandas as pd # dataframe manipulation
import numpy as np # numbers and science
import streamlit as st # cool app development
from sklearn.linear_model import LinearRegression,LogisticRegression,Perceptron
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns # for heatmaps
import altair as alt # graphs compatible with streamlit
import aux_functions as aux # local library to shorten the amount of code shown
import linearRegression as local_linReg
import logisticRegression as local_logReg


#another_set1=pd.read_csv('2012-general-election-romney-vs-obama.csv')
#another_set2=pd.read_csv('presidential_polls_2020.csv')

#the functions put here have been moved to the aux_functions file

# get dataset
# full_training=pd.read_csv('2016_sorted_polls.csv')
data_vis=pd.read_csv('2016_sorted_polls.csv')

####################################################
### streamlit wrapper begins here###
## streamlit is interspersed throughout the code
st.title("Machine Learning for Poll Prediction")

st.header("Poll Data used for testing")
st.subheader("We used data from the 2016 presidential election to create a model to predict the outcome of the previous election")
st.dataframe(data_vis)

st.header("Modeling and Testing")
st.subheader("scatter matrix of dataset features")

# st.sidebar()

### scatter matrix ###
#reset_index puts the index column back into the dataframe
scatter_matrix = alt.Chart(data_vis).mark_circle().encode(
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
#trimmed version to keep more numerical and categorical data
vis_columns = ["cycle","branch","type","grade","rawpoll_johnson",
"adjpoll_johnson","rawpoll_mcmullin","adjpoll_mcmullin","matchup",
"forecastdate","startdate","enddate","pollster","population", "multiversions",
"url","poll_id", "question_id","timestamp"]

data_vis = aux.strip_columns(data_vis,vis_columns)
#data_vis["state"] = pd.Categorical(data_vis['state'],data_vis['state'])
#data_vis['state'] = data_vis['state'].cat.codes

#full_training=full_training.drop(columns=["pred_trump"])
#full_training=full_training.drop(columns=["pred_clinton"])
data_vis = data_vis.dropna(axis=1)
data_vis = data_vis.groupby(['state'], sort=True).mean()

st.subheader("Poll data grouped by state")

print(data_vis)
st.write(data_vis)


date_probability = alt.Chart(data_vis.reset_index())

#currently not great
st.altair_chart(date_probability.mark_circle().encode(
    x ='rawpoll_trump',
    y ='rawpoll_clinton',
    #x2='actual_trump',
    #y2='actual_linton',
    color ='state',
    tooltip=['state','actual_trump','actual_clinton'])
.properties(width = 750,height = 500)
.interactive())
##########################################
"""
training_columns = ["cycle","branch","type","matchup","forecastdate","state","startdate",
"enddate","pollster","grade","poll_wt","population","samplesize","rawpoll_johnson",
"rawpoll_mcmullin","adjpoll_clinton","adjpoll_trump","adjpoll_johnson","adjpoll_mcmullin",
"multiversions","url","poll_id","question_id","createddate","timestamp"]

full_training = aux.strip_columns(full_training,training_columns)
"""


## section that shows some of the code used for the models
# any code put into the echo is both printed to streamlit and executed as code
st.subheader("regression models")
# using echo() to view code in the app
# echo() both outputs the code to streamlit and executes it

##############################################################################
local_logReg.run_logReg()
######################################################################################

############
#hopefully makes a cool looking line chart

#reset index for use in graphs
#score_chart = alt.Chart(full_training.reset_index())

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
# st.write('reg score = ', reg.score(full_training, training_target))

local_linReg.run_linReg()
