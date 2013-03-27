from data_io import DataIO
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import Bootstrap
from scipy.sparse import csr_matrix
from pprint import pprint
from time import time
import numpy as np
import joblib


import logging
from optparse import OptionParser
import sys
import pylab as pl

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoLars, LassoCV, LassoLarsCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn.metrics import Scorer, mean_absolute_error
#from sklearn import metrics

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")


(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print __doc__
op.print_help()
print


dio = DataIO("Settings_loc5.json")
submission = False
n_trees = 10
min_samples_split = 2

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


#columns = ["Category", "ContractTime", "ContractType"]
columns = ["Category", "ContractTime", "ContractType", "Company", "SourceName"]
le_features = dio.get_le_features(columns, "train_full")
extra_features = dio.get_features(columns, type_n, le_features)
extra_valid_features = dio.get_features(columns, type_v, le_features)
#features = dio.join_features("%s_" + type_n + "_count_vector_matrix_max_f_200",
                             #["Title", "FullDescription", "LocationRaw"],
                             #extra_features)
#validation_features = dio.join_features("%s_" + type_v + "_count_vector_matrix_max_f_200",
                                        #["Title", "FullDescription", "LocationRaw"],
                                        #extra_valid_features)
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
bs = Bootstrap(len(salaries), random_state=45, train_size=0.6)
train_index, test_index = next(iter(bs))
param = """Normal count vector with max 200. New submission which is repeatable.
 and nicer

Bag of Words: %s\n

Encoded cols: %s\n

Logged


vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    max_df=0.5,
    stop_words='english'
)


 """ % (",".join(tfidf_columns), ",".join(columns))

features = csr_matrix(features)

#features = features[train_index]
#salaries = salaries[train_index]

print "features", features.shape
print "valid features", validation_features.shape


def log_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))


metric = dio.error_metric
error_scorer = Scorer(log_mean_absolute_error, greater_is_better=False)
if opts.select_chi2:
    print ("Extracting %d best features by a chi-squared test" %
           opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    features = ch2.fit_transform(features, salaries)
    validation_features = ch2.transform(validation_features)
    print "done in %fs" % (time() - t0)
    print

#features = features.toarray()
#validation_features = validation_features.toarray()

#print "features", features.shape
#print "valid features", validation_features.shape
X_train = features
y_train = salaries
X_test = validation_features
y_test = valid_salaries

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# mapping from integer feature name to original token string
feature_names = None


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print 80 * '_'
    print "Training: "
    print clf
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print "test time:  %0.3fs" % test_time

    score = metric(y_test, pred)
    print "MAE:  %0.3f" % score

    if hasattr(clf, 'alpha_'):
        print "Alpha", clf.alpha_

    try:
        if hasattr(clf, 'coef_'):
            print "density: %f" % density(clf.coef_)
            print "dimensionality: %d" % clf.coef_.shape[0]

            print
    except Exception as ex:
        print ex


    print
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
#for clf, name in (
        #(Ridge(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        ##(Perceptron(n_iter=50, verbose=1, n_jobs=2, random_state=45), "Perceptron"),
        #(PassiveAggressiveRegressor(n_iter=50, verbose=1, random_state=42), "Passive-Aggressive")):
    #print 80 * '='
    #print name
    #results.append(benchmark(clf))

benchmark(Ridge(tol=1e-2, solver="lsqr"))
#results.append(benchmark(PassiveAggressiveRegressor(n_iter=150, verbose=1, random_state=42)))

#for penalty in ["l2", "l1"]:
    #print 80 * '='
    #print "%s penalty" % penalty.upper()
    ## Train Liblinear model
    ##results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            ##dual=False, tol=1e-3)))

    ## Train SGD model
    #results.append(benchmark(SGDRegressor(alpha=.0001, n_iter=50, verbose=1,
                                           #penalty=penalty)))

# Train SGD with Elastic Net penalty
#print 80 * '='
#print "Elastic-Net penalty"
#results.append(benchmark(SGDRegressor(alpha=.0001, n_iter=50, verbose=1,
                                       #penalty="elasticnet")))

# Train NearestCentroid without threshold
#print 80 * '='
#print "NearestCentroid (aka Rocchio classifier)"
#results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
#print 80 * '='
#print "Naive Bayes"
#results.append(benchmark(MultinomialNB(alpha=.01)))
#results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

#print 80 * '='
#print "LinearSVC with L1-based feature selection"
#results.append(benchmark(L1LinearSVC()))


# make some plots

#indices = np.arange(len(results))

#results = [[x[i] for x in results] for i in xrange(4)]

#clf_names, score, training_time, test_time = results
#training_time = np.array(training_time) / np.max(training_time)
#test_time = np.array(test_time) / np.max(test_time)

#pl.title("Score")
#pl.barh(indices, score, .2, label="score", color='r')
#pl.barh(indices + .3, training_time, .2, label="training time", color='g')
#pl.barh(indices + .6, test_time, .2, label="test time", color='b')
#pl.yticks(())
#pl.legend(loc='best')
#pl.subplots_adjust(left=.25)

#for i, c in zip(indices, clf_names):
    #pl.text(-.3, i, c)

#pl.show()




def make_grid_search(pipeline, parameters, model_name, params):
    print model_name
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=3,
                               #loss_func=metric,
                               #scoring="f1",
                               scoring=error_scorer,
                               iid=False,
                               refit=True)
    #model_name = "ExtraTree_min_sample2_10trees_gridcv_desc_log"

    print("Performing grid search...")
    print("pipeline:", pipeline) # [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(features, salaries)
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
    dio.save_prediction(model_name, prediction, type_v)

parameters = {
    "alpha": [0.1, 1, 10],
}

make_grid_search(Ridge(tol=1e-2, solver="lsqr"), parameters, "Ridge_tfidf_05d", param)
#make_grid_search(Lasso(), parameters, "Lasso_tfidf_05d", param)
#make_grid_search(LassoLars(), parameters, "LassoLars_tfidf_05d", param)


a=5/0
benchmark(LassoCV(max_iter=100, verbose=1))
benchmark(LassoLarsCV(n_jobs=-1, max_iter=100, max_n_alphas=50, verbose=1))

n_trees=20
min_samples_split=2
name = "ExtraTrees_min_sample%d_%dtrees_tfidf-05d_BoW-titleFullRaw-AllColumns_new_log" % (min_samples_split, n_trees)
classifier = ExtraTreesRegressor(n_estimators=n_trees,
#classifier = RandomForestRegressor(n_estimators=n_trees,
                                verbose=2,
                                n_jobs=4, # 2 jobs on submission / 4 on valid test
                                oob_score=True,
                                min_samples_split=min_samples_split,
                                random_state=3465343)
classifier.fit(features, salaries)
#classifier = dio.load_model(name)
#predictions = classifier.predict(validation_features)
metric = dio.error_metric
mae = metric(valid_salaries, predictions)
print "MAE validation: ", mae
dio.save_model(classifier, name, mae, parameters=param)
dio.save_prediction(name, predictions, type_n=type_v)
importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
f_names = map(lambda x: "Title(%d)" % x,range(1,201))
f_names.extend(map(lambda x: "Desc(%d)" % x,range(1,201)))
f_names.extend(map(lambda x: "LocR(%d)" % x,range(1,201)))
f_names.extend(columns)

num_feat = len(f_names)

# Print the feature ranking
print "Feature ranking:"

for f in xrange(len(indices)):
    print "%d. feature %d %s (%f)" % (f + 1, indices[f], f_names[indices[f]], importances[indices[f]])

# Plot the feature importances of the forest
import pylab as pl
pl.figure()
pl.title("Feature importances")
pl.bar(xrange(len(indices)), importances[indices],
              color="r", yerr=std[indices], align="center")
pl.xticks(xrange(len(indices)), indices)
pl.xlim([-1, len(indices)])
pl.show()
if mae > 6450:
    print "MAE wors then normal"
    os.exit()
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
