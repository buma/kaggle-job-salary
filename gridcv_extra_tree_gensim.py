from data_io import (
    get_paths,
    read_column,
    save_model,
    #join_features,
    #label_encode_column_fit,
    #label_encode_column_fit_only,
    #label_encode_column_transform
)
from os.path import join as path_join

#import string
import logging
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from pprint import pprint
from time import time
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error


def log_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))

import numpy as np

from scipy.io import mmread
#from gensim.matutils import corpus2csc
paths = get_paths("Settings_loc5.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

salaries = np.array(list(read_column(paths["train_data_path"], "SalaryNormalized"))).astype(np.float64)
salaries = np.log(salaries)
valid_salaries = np.array(list(read_column(paths["valid_data_path"], "SalaryNormalized"))).astype(np.float64)
valid_salaries = np.log(salaries)


#title_corpus_csc = mmread(open(path_join(cache_dir, "train_title_nltk_filtered.corpus.mtx"), "r"))
desc_corpus_csc = mmread(open(path_join(cache_dir, "train_desc_nltk_filtered.corpus.mtx"), "r"))

#print title_corpus_csc.shape
print desc_corpus_csc.shape

pipeline = Pipeline([
    ('pca', RandomizedPCA(random_state=3465343)),
    ('trees', ExtraTreesRegressor(min_samples_split=2, n_estimators=10,
                                  n_jobs=4)),
])

parameters = {
    'pca__n_components': range(100, 601, 100),
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=2,
                               loss_func=log_mean_absolute_error,
                               iid=False,
                               refit=False)
    model_name = "ExtraTree_min_sample2_10trees_gridcv_desc_log"

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(desc_corpus_csc, salaries)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    best_estimator = pipeline.set_params(**best_parameters)
    save_model(best_estimator, model_name, mae_cv=grid_search.best_score_)
    print grid_search.cv_scores_
#title
#[CVScoreTuple(parameters={'pca__n_components': 50}, mean_validation_score=8114.354436894354, cv_validation_scores=array([ 7832.02927486,  8142.59464648,  8368.43938934])), CVScoreTuple(parameters={'pca__n_components': 100}, mean_validation_score=8050.372232184578, cv_validation_scores=array([ 7805.2436147 ,  8050.13238353,  8295.74069832])), CVScoreTuple(parameters={'pca__n_components': 150}, mean_validation_score=7992.0067620542031, cv_validation_scores=array([ 7713.16260219,  8011.95811042,  8250.89957355])), CVScoreTuple(parameters={'pca__n_components': 200}, mean_validation_score=7969.954716946886, cv_validation_scores=array([ 7686.53479712,  7994.52939814,  8228.79995559])), CVScoreTuple(parameters={'pca__n_components': 250}, mean_validation_score=7951.4574455230977, cv_validation_scores=array([ 7667.32401519,  7986.80421311,  8200.24410827])), CVScoreTuple(parameters={'pca__n_components': 300}, mean_validation_score=7944.4481204410049, cv_validation_scores=array([ 7679.24885978,  7956.03557116,  8198.05993038])), CVScoreTuple(parameters={'pca__n_components': 350}, mean_validation_score=7944.1183320537721, cv_validation_scores=array([ 7676.60423802,  7978.78605684,  8176.9647013 ])), CVScoreTuple(parameters={'pca__n_components': 400}, mean_validation_score=7899.1156367965059, cv_validation_scores=array([ 7645.1559318 ,  7910.82353156,  8141.36744703])), CVScoreTuple(parameters={'pca__n_components': 450}, mean_validation_score=7912.5157328026426, cv_validation_scores=array([ 7640.30124334,  7911.32395596,  8185.92199911])), CVScoreTuple(parameters={'pca__n_components': 500}, mean_validation_score=7895.2892790734322, cv_validation_scores=array([ 7605.04498066,  7907.65668509,  8173.16617146]))]
#Best score: 7895.289
#Best parameters set:
        #pca__n_components: 500

#description
#Best score: 9004.720
#Best parameters set:
        #pca__n_components: 200
#[CVScoreTuple(parameters={'pca__n_components': 100}, mean_validation_score=9030.5959214303093, cv_validation_scores=array([ 8729.14388367,  9085.7485522 ,  9276.89532842])), CVScoreTuple(parameters={'pca__n_components': 200}, mean_validation_score=9004.7195356220382, cv_validation_scores=array([ 8712.73164292,  9081.87764045,  9219.54932349])), CVScoreTuple(parameters={'pca__n_components': 300}, mean_validation_score=9064.9514615674289, cv_validation_scores=array([ 8783.89079244,  9134.31360085,  9276.64999141])), CVScoreTuple(parameters={'pca__n_components': 400}, mean_validation_score=9076.2734308900635, cv_validation_scores=array([ 8776.89542476,  9160.65547469,  9291.26939321])), CVScoreTuple(parameters={'pca__n_components': 500}, mean_validation_score=9135.1105593660341, cv_validation_scores=array([ 8850.89803432,  9207.27553593,  9347.15810784])), CVScoreTuple(parameters={'pca__n_components': 600}, mean_validation_score=9124.6361021371431, cv_validation_scores=array([ 8837.96896581,  9150.07743191,  9385.8619087 ]))]
