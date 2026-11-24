
import json
import os
import tarfile

# ------------------------------------------------------------
# MODEL ARTIFACT PATH
# ------------------------------------------------------------

model_tar_path = "/opt/ml/processing/input/model/model.tar.gz"

# ------------------------------------------------------------
# EXTRACT MODEL ARTIFACTS
# ------------------------------------------------------------

extract_path = "/opt/ml/processing/input/model/extracted"

os.makedirs(extract_path, exist_ok=True)

with tarfile.open(model_tar_path) as tar:
    tar.extractall(path=extract_path)

# ------------------------------------------------------------
# LOAD METRICS
# ------------------------------------------------------------

metrics_path = os.path.join(extract_path, "metrics.json")

with open(metrics_path, "r") as f:
    metrics = json.load(f)

# ------------------------------------------------------------
# CREATE EVALUATION REPORT
# ------------------------------------------------------------

evaluation = {
    "metrics": {
        "accuracy": metrics["accuracy"]
    }
}

# ------------------------------------------------------------
# SAVE EVALUATION OUTPUT
# ------------------------------------------------------------

os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)

with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    json.dump(evaluation, f)

print("Evaluation complete")
