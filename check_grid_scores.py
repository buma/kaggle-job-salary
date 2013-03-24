from collections import namedtuple

from numpy import array

CVScoreTuple = namedtuple('CVScoreTuple', ('parameters',
                                           'mean_validation_score',
                                           'cv_validation_scores'))
#scores description
scores = [CVScoreTuple(parameters={'pca__n_components': 100}, mean_validation_score=9030.5959214303093, cv_validation_scores=array([8729.14388367, 9085.7485522, 9276.89532842])), CVScoreTuple(parameters={'pca__n_components': 200}, mean_validation_score=9004.7195356220382, cv_validation_scores=array([8712.73164292, 9081.87764045, 9219.54932349])), CVScoreTuple(parameters={'pca__n_components': 300}, mean_validation_score=9064.9514615674289, cv_validation_scores=array([8783.89079244, 9134.31360085, 9276.64999141])), CVScoreTuple(parameters={'pca__n_components': 400}, mean_validation_score=9076.2734308900635, cv_validation_scores=array([8776.89542476, 9160.65547469, 9291.26939321])), CVScoreTuple(parameters={'pca__n_components': 500}, mean_validation_score=9135.1105593660341, cv_validation_scores=array([8850.89803432, 9207.27553593, 9347.15810784])), CVScoreTuple(parameters={'pca__n_components': 600}, mean_validation_score=9124.6361021371431, cv_validation_scores=array([8837.96896581, 9150.07743191, 9385.8619087]))]
#scores title
scores = [CVScoreTuple(parameters={'pca__n_components': 50}, mean_validation_score=8114.354436894354, cv_validation_scores=array([7832.02927486, 8142.59464648, 8368.43938934])), CVScoreTuple(parameters={'pca__n_components': 100}, mean_validation_score=8050.372232184578, cv_validation_scores=array([7805.2436147, 8050.13238353, 8295.74069832])), CVScoreTuple(parameters={'pca__n_components': 150}, mean_validation_score=7992.0067620542031, cv_validation_scores=array([7713.16260219, 8011.95811042, 8250.89957355])), CVScoreTuple(parameters={'pca__n_components': 200}, mean_validation_score=7969.954716946886, cv_validation_scores=array([7686.53479712, 7994.52939814, 8228.79995559])), CVScoreTuple(parameters={'pca__n_components': 250}, mean_validation_score=7951.4574455230977, cv_validation_scores=array([7667.32401519, 7986.80421311, 8200.24410827])), CVScoreTuple(parameters={'pca__n_components': 300}, mean_validation_score=7944.4481204410049, cv_validation_scores=array([7679.24885978, 7956.03557116, 8198.05993038])), CVScoreTuple(parameters={'pca__n_components': 350}, mean_validation_score=7944.1183320537721, cv_validation_scores=array([7676.60423802, 7978.78605684, 8176.9647013])), CVScoreTuple(parameters={'pca__n_components': 400}, mean_validation_score=7899.1156367965059, cv_validation_scores=array([7645.1559318, 7910.82353156, 8141.36744703])), CVScoreTuple(parameters={'pca__n_components': 450}, mean_validation_score=7912.5157328026426, cv_validation_scores=array([7640.30124334, 7911.32395596, 8185.92199911])), CVScoreTuple(parameters={'pca__n_components': 500}, mean_validation_score=7895.2892790734322, cv_validation_scores=array([7605.04498066, 7907.65668509, 8173.16617146]))]


for score in scores:
    score_cv = score.cv_validation_scores
    print score.parameters, "Accuracy: %0.2f (+/- %0.2f)" % (score_cv.mean(), score_cv.std() / 2)
