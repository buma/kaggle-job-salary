from data_io import DataIO
import numpy as np
import matplotlib.pyplot as plt
dio = DataIO("Settings.json")

model_names = [
    "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_log",
    "vowpall",
    "ExtraTree_min_sample2_10trees_200f_noNorm_categoryTimeType_count_fake_14split_new_log",
    "ExtraTree_min_sample2_10trees_200f_noNorm_categoryTimeType_count_fake_split_new_log"
]

valid_salaries = dio.get_salaries("valid", log=False)

ylim = (0, 8000)
xlim = (-50000, 50000)
grid = True


def my_plot(plot_smt, filename=None, ylim=None, xlim=None, grid=False, xlabel='Difference between salary and prediction', ylabel='Number of diferences'):
    fig = plt.figure()
    num_models = len(model_names)
    for idx, model_name in enumerate(model_names):
        prediction_salaries = dio.get_prediction(model_name=model_name, type_n="valid")
        prediction_salaries = np.exp(prediction_salaries)
        diff = prediction_salaries - valid_salaries
        abs_diff = np.abs(diff)
        print model_name
        print "min diff: {:6,.4f}".format(abs_diff.min())
        print "max diff: {:6,.4f}".format(abs_diff.max())
        print "std diff: {:6,.4f}".format(abs_diff.std())
        print "mean diff: {:6,.4f}".format(abs_diff.mean())
        print "median diff: {:6,.4f}".format(np.median(abs_diff))
        quantile = np.percentile(prediction_salaries, [0, 0.25, 0.5, 0.75, 1])
        print "quantile predictions: ", quantile
        ax = fig.add_subplot(num_models, 1, idx + 1)
        plot_smt(ax, diff, abs_diff, prediction_salaries)
        ax.grid(grid)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        if idx == (num_models - 1):
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(model_name)
        print idx
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_hist(axis, diff, abs_diff, cur_pred):
    axis.hist(diff, bins=200)


def plot_hist2d(axis, diff, abs_diff, cur_pred):
    axis.hist2d(range(len(diff)), diff, bins=200)


def plot_hist2d1(axis, diff, abs_diff, cur_pred):
    axis.hist2d(valid_salaries, cur_pred, bins=200)
    axis.hold(True)
    axis.plot(valid_salaries, valid_salaries, color='r')


def plot(axis, diff, abs_diff, cur_pred):
    axis.plot(diff)


def plot_sorted(axis, diff, abs_diff, cur_pred):
    sort_indices = np.argsort(valid_salaries)
    axis.plot(diff[sort_indices], color='b')
    axis.hold(True)
    axis.plot(valid_salaries[sort_indices], color='g')


def plot_valid_pred_sorted(axis, diff, abs_diff, cur_pred):
    sort_indices = np.argsort(valid_salaries)
    axis.plot(cur_pred[sort_indices], color='r')
    axis.hold(True)
    axis.plot(valid_salaries[sort_indices], color='g')


#my_plot(plot_hist, ylim=ylim, xlim=xlim, grid=True)
#my_plot(plot_hist2d)
#my_plot(plot_hist2d1, xlabel="salarie", ylabel="predicted salarie")
#my_plot(plot, xlabel="salarie", ylabel="predicted salarie")
#my_plot(plot_sorted, xlabel="Ad", ylabel="diff from valid salarie")
my_plot(plot_valid_pred_sorted, xlabel="Ad", ylabel="valid salarie predicted salarie")

#sorted_diff_indices = np.argsort(diff)
#print diff[sorted_diff_indices[500]]

#print diff.shape

#diffBiger = diff > 40000


#bigIds = valid_salaries_ids[diffBiger]
#bigDiffs = diff[diffBiger]
#bigSalares = valid_salaries[diffBiger]
#bigPredictions = valid_predictions[diffBiger]

#print "Vecjih od 40000", bigPredictions.shape[0]

#print bigSalares
#print bigIds

#n = 0
#for bigId, bigSalarie, bigPrediction, bigDiff in zip(bigIds, bigSalares, bigPredictions, bigDiffs):
    #print "ID: %i Salarie: %0.2f Prediction: %0.2f Difference: %0.2f" % (bigId, bigSalarie, bigPrediction, bigDiff)
    #n = n + 1
    #if n > 100:
        #break
