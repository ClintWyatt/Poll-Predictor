import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
full_training=full_training.drop(columns=["grade"])
full_training=full_training.drop(columns=["population"])
full_training=full_training.drop(columns=["poll_wt"])
full_training=full_training.drop(columns=["rawpoll_mcmullin"])
full_training=full_training.drop(columns=["rawpoll_johnson"])
full_training=full_training.drop(columns=["adjpoll_clinton"])
full_training=full_training.drop(columns=["adjpoll_trump"])
full_training=full_training.drop(columns=["adjpoll_johnson"])
full_training=full_training.drop(columns=["adjpoll_mcmullin"])
full_training=full_training.drop(columns=["poll_id"])
full_training=full_training.drop(columns=["question_id"])
full_training=full_training.drop(columns=["createddate"])
full_training=full_training.drop(columns=["timestamp"])
full_training=full_training.drop(columns=["url"])
full_training=full_training.drop(columns=["multiversions"])
full_training=full_training.drop(columns=["samplesize"])
print(full_training)

training_targets=full_training["rawpoll_trump"]
reg = LinearRegression(fit_intercept=True, normalize=False)




reg.fit(full_training, training_targets)
print(reg.score(full_training, training_targets))

