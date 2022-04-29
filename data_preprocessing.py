import string
import re
import codecs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pit
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("raw_data.csv")
# print(df.Language.unique())
translate_table = dict((ord(char), None) for char in string.punctuation)
# print(translate_table)

LANGAUAGE = ['English', 'Malayalam', 'Hindi', 'Tamil', 'Portugeese', 'French', 'Dutch',
             'Spanish', 'Greek', 'Russian', 'Danish', 'Italian', 'Turkish', 'Sweedish',
             'Arabic', 'German', 'Kannada']
list_df = []
list_lang = []
for lan in LANGAUAGE:
    # print(lan)
    for i, line in df[df["Language"] == lan].iterrows():
        # line = str(line)
        line = line['Text'] 
        # print(line, type(line))
        if len(line) != 0:
            line = line.lower()
            # print(line)
            line = re.sub(r"\d+", "", line)
            # print(line)
            line = line.translate(translate_table)
            list_df.append(line)
            temp = list_lang.append(lan)
print(len(list_df))
print(len(list_lang))

new_df = pd.DataFrame({"Text": list_df, "Language": list_lang})
new_df.to_csv("processed_data.csv", index=False)

