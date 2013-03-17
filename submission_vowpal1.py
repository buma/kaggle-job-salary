import joblib
from data_io import (
    write_submission,
    get_paths
)
import numpy as np
from os.path import join as path_join
paths = get_paths("Settings.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")

model_name = "vowpal_submission"
type_n = "submission"
predictions = joblib.load(path_join(prediction_dir, model_name + "_prediction_" + type_n))
model_name = "vowpal_submission_round"
predictions = np.exp(predictions)
predictions = predictions / 1000
#print predictions[1:10]
predictions = np.round(predictions) * 1000
joblib.dump(predictions, path_join(prediction_dir, model_name + "_prediction_" + type_n))
write_submission("vowpal_fastml_round.csv", "vowpal_submission_round_prediction_submission", unlog=False)
