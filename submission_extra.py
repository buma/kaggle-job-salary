from data_io import (
    write_submission,
)
model_name = "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_log"
write_submission(model_name + ".csv", model_name + "_prediction_test_subm", unlog=True)
