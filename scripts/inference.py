
import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["inputs"])
    raise Exception("Unsupported type")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    return json.dumps(prediction.tolist())
