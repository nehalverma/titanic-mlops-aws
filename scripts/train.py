
import pandas as pd
import joblib
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

train_path = "/opt/ml/input/data/train/train.csv"
val_path = "/opt/ml/input/data/validation/val.csv"

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)

X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]

X_val = val.drop("Survived", axis=1)
y_val = val["Survived"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", LogisticRegression(max_iter=200))
])

model.fit(X_train, y_train)

pred = model.predict(X_val)

acc = accuracy_score(y_val, pred)

os.makedirs("/opt/ml/model", exist_ok=True)

joblib.dump(model, "/opt/ml/model/model.joblib")

metrics = {
    "accuracy": acc
}

with open("/opt/ml/model/metrics.json", "w") as f:
    json.dump(metrics, f)

print(metrics)
