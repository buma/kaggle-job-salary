import joblib
from data_io import (
    get_paths,
)
from os.path import join as path_join
import numpy as np
paths = get_paths("Settings.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")

model_name = "vowpal_submission"
type_n = "submission"

predictions = np.loadtxt(path_join(data_dir, "code", "from_fastml", "optional", "predictions_submit.txt"))
joblib.dump(predictions, path_join(prediction_dir, model_name + "_prediction_" + type_n))
