from data_io import DataIO
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pprint import pprint
from time import time
import numpy as np
import joblib

dio = DataIO("Settings.json")
submission = False
n_trees = 10
min_samples_split = 2
param = """Normal count vector with max 200. New submission which is repeatable.
 and nicer

count_vector_titles = TfidfVectorizer(
    read_column(train_filename, column_name),
    max_features=200, norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)
 """

if submission:
    type_n = "train_full"
    type_v = "valid_full"
else:
    type_n = "train"
    type_v = "valid"


vectorizer = CountVectorizer(
    max_features=200,
)
vectorizer = TfidfVectorizer(
    max_features=200,
    norm='l2',
    smooth_idf=True,
    sublinear_tf=False,
    use_idf=True
)
#short_id = "count_200f"
tfidf_columns = ["Title", "FullDescription", "LocationRaw"]


#columns = ["Category", "ContractTime", "ContractType", "Category", "SourceName"]
#le_features = dio.get_le_features(columns, "train_full")
#extra_features = dio.get_features(columns, type_n, le_features)
#extra_valid_features = dio.get_features(columns, type_v, le_features)

#features = dio.join_features("%s_" + type_n + "_count_vector_matrix_max_f_200",
                             #["Title", "FullDescription", "LocationRaw"],
                             #extra_features)
#validation_features = dio.join_features("%s_" + type_v + "_count_vector_matrix_max_f_200",
                                        #["Title", "FullDescription", "LocationRaw"],
                                        #extra_valid_features).astype(np.int64)
short_id = "tfidf_200f_l2"
dio.make_counts(vectorizer, short_id, tfidf_columns, type_n, type_v)
extra_features = []
extra_valid_features = []
features = dio.join_features("%s_" + type_n + "_" + short_id + "_matrix",
                             tfidf_columns,
                             extra_features)
validation_features = dio.join_features("%s_" + type_v + "_" + short_id + "_matrix",
                                        tfidf_columns,
                                        extra_valid_features)

print features.max()
print features.min()
salaries = dio.get_salaries(type_n, log=False).astype(np.int64)
if not submission:
    valid_salaries = dio.get_salaries(type_v, log=False)

def encode_salaries(salaries, bins):
    bin_edges = np.linspace(11500.0, 100000, bins + 1, endpoint=True)
    #hist, bin_edges = np.histogram(salaries, bins)
    print np.diff(bin_edges)
    idxs = np.searchsorted(bin_edges, salaries, side="right")
    return idxs

#bin_edges.insert(0, 0)
#bin_edges.append(salaries.max() + 1)

#print "hist", hist
#print "edges", bin_edges
#idxs = np.searchsorted(bin_edges, salaries, side="right")

#print "idx", idxs
#i = 0
#for idx, salarie in zip(idxs, salaries):
    #prej = bin_edges[idx - 1]
    #try:
        #after = bin_edges[idx]
    #except IndexError as iex:
        #print idx
    #i = i + 1
    #if i < 50:
        #print prej, "<=", salarie, "<", after

    #if not (prej <= salarie < after):
        #print "NI OK"
        #print prej, "<=", salarie, "<", after
        #break



#valid_salaries_enc = encode_salaries(valid_salaries, 4)
#salaries = np.log(salaries)
#valid_salaries = np.log(valid_salaries)
dio.is_log = True



print salaries.shape
metric = dio.error_metric


def make_grid_search(pipeline, parameters, model_name, params):
    print model_name
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=3,
                               #loss_func=f1_score,
                               scoring="f1",
                               iid=False,
                               refit=True)
    #model_name = "ExtraTree_min_sample2_10trees_gridcv_desc_log"

    print("Performing grid search...")
    print("pipeline:", pipeline) # [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(features, salaries_enc)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    best_estimator = pipeline.set_params(**best_parameters)
    params = params + " ", grid_search.cv_scores_
    dio.save_model(best_estimator, model_name, mae_cv=grid_search.best_score_, parameters=params)
    print grid_search.cv_scores_
    prediction = grid_search.predict(validation_features)
    dio.save_prediction(model_name, prediction, "valid_classes")


#for bin_n in [14]:
    #salaries_enc = encode_salaries(salaries, bin_n)
    #nm = "_tfidf_titleFullLoc_bin%d" % bin_n
    #par = " classed from 0-11500 then %d classes to 100 000 and to end\n Tfidf of Title full and location Raw" % (bin_n)
    #if bin_n > 4 :
        #make_grid_search(MultinomialNB(), {"alpha": [0.01, 0.1, 0.5, 1]}, "multinomialnb" + nm, "Multinomial NB" + par)
        #make_grid_search(SGDClassifier(), {'n_iter': [50, 100, 150], 'penalty': ['l2', 'l1']}, "sgd_class" + nm, "SGDClassifier classes" + par)
    ##make_grid_search(KNeighborsClassifier(), {'n_neighbors': range(4,100,20)}, "kneighbour" + nm, "Kneighbour classes" + par)
    #make_grid_search(RandomForestClassifier(random_state=42), {'min_samples_split': [2, 30]}, "randomForest" + nm, "Random Forest" + par)
    ##make_grid_search(GradientBoostingClassifier(), {'learning_rate': [0.1, 0.5], 'subsample': [1,0.8,0.6], 'n_estimators':[100,150]}, "GBM" + nm, "Gradient Boosting Machines " + par)

bin_n = 4
salaries_enc = encode_salaries(salaries, bin_n)
valid_salaries_enc = encode_salaries(valid_salaries, bin_n)
nm = "_tfidf_titleFullLoc_bin%d" % bin_n
model_name = "randomForest" + nm
par = " classed from 0-11500 then %d classes to 100 000 and to end\n Tfidf of Title full and location Raw" % (bin_n)

classifier = RandomForestClassifier(min_samples_split=2, random_state=42)
classifier.fit(features, salaries_enc)
prediction = classifier.predict(validation_features)
dio.save_prediction(model_name, prediction, "valid_classes")
dio.save_model(classifier, model_name, parameters=par)

print (classification_report(valid_salaries_enc, prediction))

print confusion_matrix(valid_salaries_enc, prediction)

prediction = classifier.predict(features)
dio.save_prediction(model_name, prediction, "train_classes")

print (classification_report(salaries_enc, prediction))

print confusion_matrix(salaries_enc, prediction)
