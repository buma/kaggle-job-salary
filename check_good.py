from data_io import (
    get_paths,
    read_column
)
import joblib
from os.path import join as path_join
import numpy as np
paths = get_paths("Settings.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")
model_name = "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_log"

valid_predictions = joblib.load(path_join(prediction_dir, model_name + "_prediction_valid"))
valid_predictions = np.exp(valid_predictions)
valid_salaries = np.array(list(read_column(paths["valid_data_path"], "SalaryNormalized"))).astype(np.float64)

valid_salaries_ids = np.array(list(read_column(paths["valid_data_path"], "Id"))).astype(np.int64)

diff = np.abs(valid_predictions - valid_salaries)
#print valid_predictions[1:10], valid_salaries[1:10]
#print diff

print "min diff:", diff.min()
print "max diff:", diff.max()
print "mean diff:", diff.mean()

print "Vseh:", valid_salaries.shape[0]

#sorted_diff_indices = np.argsort(diff)
#print diff[sorted_diff_indices[500]]

print diff.shape

diffBiger = diff > 40000


bigIds = valid_salaries_ids[diffBiger]
bigDiffs = diff[diffBiger]
bigSalares = valid_salaries[diffBiger]
bigPredictions = valid_predictions[diffBiger]

print "Vecjih od 40000", bigPredictions.shape[0]

#print bigSalares
#print bigIds

#n = 0
#for bigId, bigSalarie, bigPrediction, bigDiff in zip(bigIds, bigSalares, bigPredictions, bigDiffs):
    #print "ID: %i Salarie: %0.2f Prediction: %0.2f Difference: %0.2f" % (bigId, bigSalarie, bigPrediction, bigDiff)
    #n = n + 1
    #if n > 100:
        #break


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

fig = plt.figure()
ax = fig.add_subplot(411)
diff1 = valid_predictions - valid_salaries

#n, bins, patches = ax.hist(diff1, 100, facecolor='green', alpha=0.75)
#ax.hist2d(x=range(len(diff1)),y=diff1, bins=200)
ax.hist(diff1, bins=200)
#ax.hist2d(valid_salaries, valid_predictions, bins=300)
#ax.hold(True)
#ax.plot(valid_salaries, valid_salaries, color='r')

#bincenters = 0.5*(bins[1:]+bins[:-1])
# add a 'best fit' line for the normal PDF
#y = mlab.normpdf(bincenters, mu, sigma)
#l = ax.plot(bincenters, y, 'r--', linewidth=1)

ax.set_xlabel('Difference between salary and prediction')
ax.set_ylabel('Number of diferences')
ax.set_title('Extra Tree predicions')
#ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#ax.set_xlim(40, 160)
#ax.set_ylim(0, 0.03)
ax.grid(True)
ax.set_ylim(0, 8000)
ax.set_xlim(-50000, 50000)

ax1 = fig.add_subplot(412)
model_name = "vowpall"
#model_name = "Random_forest_min_sample2_20trees_200f_noNorm_categoryTimeType_log"
valid_predictions_vw = joblib.load(path_join(prediction_dir, model_name + "_prediction_valid"))
valid_predictions_vw = np.exp(valid_predictions_vw)
#ax1.hist2d(x=range(len(diff1)),y=(valid_predictions_vw-valid_salaries), bins=200)
ax1.hist((valid_predictions_vw-valid_salaries), bins=200)
#ax1.hist2d(valid_salaries, valid_predictions_vw, bins=300)
#ax1.hold(True)
#ax1.plot(valid_salaries, valid_salaries, color='r')
ax1.set_xlabel('Difference between salary and prediction')
ax1.set_ylabel('Number of diferences')
ax1.set_title('Vowpall predicions')
ax1.grid(True)
ax1.set_ylim(0, 8000)
ax1.set_xlim(-50000, 50000)

ax2 = fig.add_subplot(413)
#model_name = "Random_forest_min_sample2_40trees_200f_noNorm_categoryTimeType_log"
model_name = "Knn_linreg_200centers_Locations"
model_name = "Knn_Dectree_150centers_Locations"
valid_predictions_vw = joblib.load(path_join(prediction_dir, model_name + "_prediction_valid"))
valid_predictions_vw = valid_predictions_vw
#ax2.hist2d(x=range(len(diff1)),y=(valid_predictions_vw-valid_salaries), bins=200)
ax2.hist((valid_predictions_vw-valid_salaries), bins=200)
#ax2.hist2d(valid_salaries, valid_predictions_vw, bins=300)
#ax2.hold(True)
#ax2.plot(valid_salaries, valid_salaries, color='r')
ax2.set_xlabel('Difference between salary and prediction')
ax2.set_ylabel('Number of diferences')
ax2.set_title('Random Forest predicions prediction-valid')
ax2.grid(True)
ax2.set_ylim(0, 8000)
ax2.set_xlim(-50000, 50000)

ax3 = fig.add_subplot(414)
#model_name = "Random_forest_min_sample2_40trees_200f_noNorm_categoryTimeType_log"
model_name = "Knn_linreg_200centers_Locations"
model_name = "Knn_Dectree_150centers_Locations"
model_name = "ExtraTreesRegressor_30trees_split20_notext"
valid_predictions_vw = joblib.load(path_join(prediction_dir, model_name + "_prediction_valid"))
valid_predictions_vw = valid_predictions_vw
#ax3.hist2d(x=range(len(diff1)),y=(valid_predictions_vw-valid_salaries), bins=200)
ax3.hist((valid_predictions_vw-valid_salaries), bins=200)
#ax3.hist2d(valid_salaries, valid_predictions_vw, bins=300)
#ax3.hold(True)
#ax3.plot(valid_salaries, valid_salaries, color='r')
ax3.set_xlabel('Difference between salary and prediction')
ax3.set_ylabel('Number of diferences')
ax3.set_title('Random Forest predicions prediction-valid')
ax3.grid(True)
ax3.set_ylim(0, 8000)
ax3.set_xlim(-50000, 50000)

plt.show()
