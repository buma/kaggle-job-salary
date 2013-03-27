from data_io import DataIO
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import Ridge

dio = DataIO("Settings_submission.json")
submission = True
n_trees = 10
min_samples_split = 2
param = """Normal count vector with max 200. New submission which is repeatable.
 and nicer

 """

if submission:
    type_n = "train_full"
    type_v = "valid_full"
else:
    type_n = "train"
    type_v = "valid"

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    max_df=0.5,
    stop_words='english'
)
#short_id = "tfidf_200f_l1"
short_id = "tfidf_05df_stopwords"
tfidf_columns = ["Title", "FullDescription", "LocationRaw"]
dio.make_counts(vectorizer, short_id, tfidf_columns, type_n, type_v)


columns = ["Category", "ContractTime", "ContractType", "Company", "SourceName"]
le_features = dio.get_le_features(columns, "train_and_valid")
extra_features = dio.get_features(columns, type_n, le_features)
extra_valid_features = dio.get_features(columns, type_v, le_features)

features = dio.join_features("%s_" + type_n + "_" + short_id + "_matrix",
                             tfidf_columns,
                             extra_features, sparse=True)
validation_features = dio.join_features("%s_" + type_v + "_" + short_id + "_matrix",
                                        tfidf_columns,
                                        extra_valid_features, sparse=True)

print features.shape
print validation_features.shape

salaries = dio.get_salaries(type_n, log=True)
if not submission:
    valid_salaries = dio.get_salaries(type_v, log=True)

print salaries.shape
classifier = Ridge(alpha=1., tol=1e-2, solver="lsqr")
name = "Ridge_tfidf_05_log"
print name
classifier.fit(features, salaries)
predictions = classifier.predict(validation_features)
if submission:
    dio.save_prediction(name, predictions, type_n=type_v)
    dio.write_submission(name + ".csv", predictions=predictions)
else:
    dio.compare_valid_pred(valid_salaries, predictions)
    metric = dio.error_metric
    mae = metric(valid_salaries, predictions)
    print "MAE validation: ", mae
    dio.save_model(classifier, name, mae)
    dio.save_prediction(name, predictions, type_n=type_v)
    if mae < 6450:
        print "MAE wors then normal"
        os.exit()
        #break
#oob_predictions = classifier.oob_prediction_
#mae_oob = mean_absolute_error(salaries, oob_predictions)
#print "MAE OOB: ", mae_oob
    classifier1 = ExtraTreesRegressor(n_estimators=n_trees,
                                        verbose=1,
                                        n_jobs=3,
                                        oob_score=False,
                                        min_samples_split=min_samples_split,
                                        random_state=3465343)
    scores = cross_val_score(classifier1, features, salaries, cv=3, score_func=metric, verbose=1)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    mae_cv = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    dio.save_model(classifier, name, mae_cv=mae_cv, parameters=param)

