import pickle
import pandas as pd
import string
import re
import codecs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pit
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.model_selection import train_test_split

input_text = input("Please enter text for prediction: ")
pickled_model = pickle.load(open('model.pkl', 'rb'))
output = pickled_model.predict(pd.Series(input_text))
print("Your entered text is of", output, "language.")
