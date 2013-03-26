from data_io import DataIO
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
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
short_id = "count_200f"
tfidf_columns = ["Title", "FullDescription", "LocationRaw"]
#dio.make_counts(vectorizer, short_id, tfidf_columns, type_n, type_v)


columns = ["Category", "ContractTime", "ContractType"]
le_features = dio.get_le_features(columns, "train_full")
extra_features = dio.get_features(columns, type_n, le_features)
extra_valid_features = dio.get_features(columns, type_v, le_features)

features = dio.join_features("%s_" + type_n + "_count_vector_matrix_max_f_200",
                             ["Title", "FullDescription", "LocationRaw"],
                             extra_features)
validation_features = dio.join_features("%s_" + type_v + "_count_vector_matrix_max_f_200",
                                        ["Title", "FullDescription", "LocationRaw"],
                                        extra_valid_features).astype(np.int64)
#features = dio.join_features("%s_" + type_n + "_" + short_id + "_matrix",
                             #tfidf_columns,
                             #extra_features)
#validation_features_1 = dio.join_features("%s_" + type_v + "_" + short_id + "_matrix",
                                        #tfidf_columns,
                                        #extra_valid_features)

#print validation_features
#print validation_features_1

#rez = np.abs(validation_features - validation_features_1)

#print rez

#print rez.max()
#print rez.min()
#print rez.mean()

#print rez[np.where(rez > 0)]

#salaries = np.random.randint(5, 10, size=10)
salaries = dio.get_salaries(type_n, log=False).astype(np.int64)
if not submission:
    valid_salaries = dio.get_salaries(type_v, log=False)

#print "salaries", salaries

#hist, bin_edges = np.histogram(salaries, 10)

#bins = 10

#bin_edges = np.linspace(11500.0, 100000, bins + 1, endpoint=True)

#bin_edges = list(bin_edges)

par = " classed from 0-11500 then 4 classes to 100 000 and to end NoNormal classTypeTime"
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



salaries = encode_salaries(salaries, 4)
valid_salaries = encode_salaries(valid_salaries, 4)
#salaries = np.log(salaries)
#valid_salaries = np.log(valid_salaries)
#dio.is_log = True


parameters = {
    'pca__n_components': range(100, 601, 100),
}

print salaries.shape
metric = dio.error_metric


def make_grid_search(pipeline, parameters, model_name, params):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=3,
                               loss_func=f1_score,
                               iid=False,
                               refit=False)
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
    dio.save_model(best_estimator, model_name, mae_cv=grid_search.best_score_, parameters=params)
    print grid_search.cv_scores_


#make_grid_search(NearestCentroid(), {'metric': 'euclidean'}, "nearest_centroid", "Nearest centroid classes" + par)
make_grid_search(MultinomialNB(), {"alpha": [0.1, 0.5, 1]}, "multinomialnb", "Multinomial NB" + par)
make_grid_search(SGDClassifier(), {'n_iter': [50, 100], 'penalty': ['l2', 'l1']}, "sgd_class", "SGDClassifier classes" + par)
a=5/0
#make_grid_search(KNeighborsClassifier(), {'n_neighbors=':range(4,100,20)}, "kneighbour", "Kneighbour classes" + par)
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
#Best score: 0.586
#Best parameters set:
        #alpha: 0.5
#[CVScoreTuple(parameters={'alpha': 0.1}, mean_validation_score=0.58631130843120649, cv_validation_scores=array([ 0.58841111,  0.58310575,  0.58741707])), CVScoreTuple(parameters={'alpha': 0.5}, mean_validation_score=0.58582062520404232, cv_validation_scores=array([ 0.5877551 ,  0.58263991,  0.58706687])), CVScoreTuple(parameters={'alpha': 1}, mean_validation_score=0.58618103211875761, cv_validation_scores=array([ 0.58805118,  0.58328137,  0.58721055]))]
#SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
       #fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       #loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
       #random_state=None, rho=None, shuffle=False, verbose=0,
       #warm_start=False)
#Best score: 0.600
#Best parameters set:
        #n_iter: 100
        #penalty: 'l2'
#[CVScoreTuple(parameters={'penalty': 'l2', 'n_iter': 50}, mean_validation_score=0.62851084828900494, cv_validation_scores=array([ 0.6453497 ,  0.63742851,  0.60275433])), CVScoreTuple(parameters={'penalty': 'l1', 'n_iter': 50}, mean_validation_score=0.61833034716526936, cv_validation_scores=array([ 0.61887506,  0.61147872,  0.62463726])), CVScoreTuple(parameters={'penalty': 'l2', 'n_iter': 100}, mean_validation_score=0.60045091619136015, cv_validation_scores=array([ 0.60126802,  0.59213141,  0.60795331])), CVScoreTuple(parameters={'penalty': 'l1', 'n_iter': 100}, mean_validation_score=0.62415553952456981, cv_validation_scores=array([ 0.62677737,  0.61433574,  0.63135351]))]
#za TFIDF:
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
#Best score: 0.654
#Best parameters set:
        #alpha: 1
#[CVScoreTuple(parameters={'alpha': 0.1}, mean_validation_score=0.65493955664232339, cv_validation_scores=array([ 0.65673829,  0.65092235,  0.65715803])), CVScoreTuple(parameters={'alpha': 0.5}, mean_validation_score=0.6548238694706302, cv_validation_scores=array([ 0.65621233,  0.65119728,  0.657062  ])), CVScoreTuple(parameters={'alpha': 1}, mean_validation_score=0.65448574030491569, cv_validation_scores=array([ 0.65578231,  0.65094661,  0.6567283 ]))]

#SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
       #fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       #loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
       #random_state=None, rho=None, shuffle=False, verbose=0,
       #warm_start=False)
#Best score: 0.632
#Best parameters set:
        #n_iter: 100
        #penalty: 'l2'
#[CVScoreTuple(parameters={'penalty': 'l2', 'n_iter': 50}, mean_validation_score=0.6335973117724164, cv_validation_scores=array([ 0.66008312,  0.57532641,  0.66538241])), CVScoreTuple(parameters={'penalty': 'l1', 'n_iter': 50}, mean_validation_score=0.64799193767793106, cv_validation_scores=array([ 0.64861211,  0.64349788,  0.65186583])), CVScoreTuple(parameters={'penalty': 'l2', 'n_iter': 100}, mean_validation_score=0.6323284905465183, cv_validation_scores=array([ 0.63202777,  0.60668741,  0.6582703 ])), CVScoreTuple(parameters={'penalty': 'l1', 'n_iter': 100}, mean_validation_score=0.64505184930297021, cv_validation_scores=array([ 0.65421199,  0.62670334,  0.65424022]))]


#np.save("train_200f_noNorm_categoryTimeType_tfidfl2_salaries", salaries)
#np.save("train_200f_noNorm_categoryTimeType_tfidfl2_valid_salaries", valid_salaries)

#joblib.dump(salaries, "train_200f_noNorm_categoryTimeType_tfidfl2_salaries_jl", compress=5)
#joblib.dump(salaries, "train_200f_noNorm_categoryTimeType_tfidfl2_valid_salaries_jl", compress=5)

for n_trees in [10]:
    name = "ExtraTree_min_sample%d_%dtrees_200f_noNorm_categoryTimeType_count_fake_4split_new_log" % (min_samples_split, n_trees)
    print name
    num_classes = salaries_enc.max()
    print "classes:", num_classes
    def predict(class_id):
        print "predicting: ", class_id
        salaries_idx = np.where(salaries_enc == class_id)
        valid_idx = np.where(valid_salaries_enc == class_id)

        if len(salaries_idx[0]) == 0 or len(valid_idx[0]) == 0:
            return [], None

        classifier = ExtraTreesRegressor(n_estimators=n_trees,
                                        verbose=2,
                                        n_jobs=4, # 2 jobs on submission / 4 on valid test
                                        oob_score=False,
                                        min_samples_split=min_samples_split,
                                        random_state=3465343)

        print features[salaries_idx[0], :].shape
        print salaries[salaries_idx].shape
        classifier.fit(features[salaries_idx[0], :], salaries[salaries_idx])
        predictions_part = classifier.predict(validation_features[valid_idx[0]])
        return predictions_part, valid_idx
    predictions = np.zeros_like(valid_salaries)
    for cur_class_id in range(num_classes + 1):
        predictions_part, idx = predict(cur_class_id)
        if idx is not None:
            predictions[idx] = predictions_part
    if submission:
        dio.save_prediction(name, predictions, type_n=type_v)
        dio.write_submission(name + ".csv", predictions=predictions)
    else:
        dio.compare_valid_pred(valid_salaries, predictions)
        metric = dio.error_metric
        mae = metric(valid_salaries, predictions)
        print "MAE validation: ", mae
        dio.save_model(ExtraTreesRegressor(), name, mae)
        dio.save_prediction(name, predictions, type_n=type_v)
#oob_predictions = classifier.oob_prediction_
#mae_oob = mean_absolute_error(salaries, oob_predictions)
#print "MAE OOB: ", mae_oob
        #classifier1 = ExtraTreesRegressor(n_estimators=n_trees,
                                            #verbose=1,
                                            #n_jobs=3,
                                            #oob_score=False,
                                            #min_samples_split=min_samples_split,
                                            #random_state=3465343)
        #scores = cross_val_score(classifier1, features, salaries, cv=3, score_func=metric, verbose=1)
        #print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        #mae_cv = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        #dio.save_model(classifier, name, mae_cv=mae_cv, parameters=param)

