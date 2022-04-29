import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("processed_data.csv")
df = df[df['Text'].notna()]
df = df[df['Language'].notna()]

x, y = df.iloc[:, 0], df.iloc[:, 1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# print(x_test)
pickled_model = pickle.load(open('model.pkl', 'rb'))
output = pickled_model.predict(x_test)
# print(type(output), list(output), type(x_test))
x_test = pd.DataFrame(x_test)
x_test["predicted_value"] = list(output)
print(x_test.head())
x_test.to_csv("predicted_data.csv", index=False)
