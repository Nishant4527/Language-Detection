import pickle
import string
import re
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as pit
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("processed_data.csv")
df = df[df['Text'].notna()]
df = df[df['Language'].notna()]

x, y = df.iloc[:, 0], df.iloc[:, 1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 4), analyzer='char')
model = pipeline.Pipeline([('vectorizer', vec),
                                 ('clf', linear_model.LogisticRegression())
                                 ])
model.fit(x_train, y_train)
pickle.dump(model, open('model.pkl', 'wb'))