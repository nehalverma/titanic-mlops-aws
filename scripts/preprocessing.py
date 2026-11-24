import pandas as pd
import os
from sklearn.model_selection import train_test_split

input_path = "/opt/ml/processing/input/titanic.csv"

df = pd.read_csv(input_path)

df = df[['Pclass', 'Sex', 'Age', 'Survived']]

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Age'] = df['Age'].fillna(df['Age'].median())

train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

out = "/opt/ml/processing/output"

os.makedirs(out + "/train", exist_ok=True)
os.makedirs(out + "/validation", exist_ok=True)
os.makedirs(out + "/test", exist_ok=True)

train.to_csv(out + "/train/train.csv", index=False)
val.to_csv(out + "/validation/val.csv", index=False)
test.to_csv(out + "/test/test.csv", index=False)

print("Preprocessing done")
