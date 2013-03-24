from data_io import DataIO
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone
from sklearn.cross_validation import cross_val_score
import numpy as np

dio = DataIO("Settings.json")

title_corpus = dio.read_gensim_corpus("train_title_nltk_filtered.corpus.mtx")
pca = RandomizedPCA(random_state=3465343)
salaries = dio.get_salaries("train", log=True)

columns = ["Category", "ContractTime", "ContractType"]
le_features = dio.get_le_features(columns, "train_full")
extra_features = dio.get_features(columns, "train", le_features)
#extra_valid_features = dio.get_features(columns, "valid", le_features)

param = "RandomizedPCA title 200 Fulldescription 200 " + ",".join(columns)
print map(len, extra_features)
extra_features = map(lambda x: np.reshape(np.array(x),(len(x),1)),extra_features)



print type(title_corpus)
print title_corpus.shape


title_pca = clone(pca)
title_pca.set_params(n_components=200)
title_corpus_pca = title_pca.fit_transform(title_corpus)

print type(title_corpus_pca)

print title_corpus_pca.shape

desc_corpus = dio.read_gensim_corpus("train_desc_nltk_filtered.corpus.mtx")


#print title_pca.explained_variance_ratio_

#import pylab as pl
#pl.clf()
#pl.plot(title_pca.explained_variance_ratio_, linewidth=2)
#pl.axis('tight')
#pl.xlabel('n_components')
#pl.ylabel('explained_variance')

#pl.show()
desc_pca = clone(pca)
desc_pca.set_params(n_components=200)
desc_corpus_pca = desc_pca.fit_transform(desc_corpus)

print desc_corpus_pca.shape
locraw_corpus = dio.read_gensim_corpus("train_locraw_nltk_filtered.corpus.mtx")
locraw_pca = clone(pca)
locraw_pca.set_params(n_components=200)
locraw_corpus_pca = locraw_pca.fit_transform(locraw_corpus)

print locraw_corpus_pca.shape

feature_arrays = [title_corpus_pca, desc_corpus_pca, locraw_corpus_pca]
feature_arrays.extend(extra_features)
features = np.hstack(feature_arrays)

#print desc_pca.explained_variance_ratio_

##pl.clf()
#pl.plot(desc_pca.explained_variance_ratio_, linewidth=2)
#pl.axis('tight')
#pl.xlabel('n_components')
#pl.ylabel('explained_variance')

#pl.show()

print features.shape
for n_trees in [10]:
    for min_samples_split in [2]:
        print n_trees
        name = "ExtraTree_min_sample%d_%dtrees_100f_noNorm_categoryTimeType_log1" % (min_samples_split, n_trees)
        name = "ExtraTree_min_sample%d_%dtrees_200Title200FullDescLocRawcategoryTimeType_log" % (min_samples_split, n_trees)
        print name
        classifier = ExtraTreesRegressor(n_estimators=n_trees,
                                         verbose=2,
                                         n_jobs=3,
                                         oob_score=False,
                                         min_samples_split=min_samples_split,
                                         random_state=3465343)

        #classifier.fit(features, salaries)
        #predictions = classifier.predict(validation_features)
        #dio.compare_valid_pred(valid_salaries, predictions[1:10])
        metric = dio.error_metric
        #mae = metric(valid_salaries, predictions)
        #print "MAE validation: ", mae
        #dio.save_model(classifier, name, mae)
        #dio.save_prediction(name, predictions, type_n="valid")
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
