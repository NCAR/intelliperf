import os
import sys
import math
import csv
import argparse
import shutil
from collections import OrderedDict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

LABEL = "LABEL"

def read_csv(csvfile):

    FIELDNAMES = [ 'region', 'group', 'counter', 'xval', 'yval' ]

    # csv data
    csvdata = OrderedDict()
    #cfg['csvdata'] = csvdata

    print("Reading %s"%csvfile)

    with open(csvfile, 'r') as fcsv:

        reader = csv.DictReader(fcsv, fieldnames=FIELDNAMES, delimiter=';')

        for row in reader:

            if row['counter'] == LABEL or \
                not row['counter'].endswith('per_ins'):

                # region
                if row['region'] not in csvdata:
                    region = OrderedDict()
                    csvdata[ row['region'] ] = region
                else:
                    region = csvdata[ row['region'] ]

                # counter
                if row['counter'] not in region:
                    counter = np.zeros(999, dtype=np.float32)
                    region[ row['counter'] ] = counter
                else:
                    counter = region[ row['counter'] ]

                # xval and yval
                xval = int(float(row['xval'])*1000-1)
                yval = float(row['yval'])
                #if xval < 10 or xval > 990:
                #    counter[ xval ] = 0
                if not math.isnan(yval) and not math.isinf(yval):
                    counter[ xval ] = yval

    return csvdata

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description='RandomForest Excersize for Random Number Generator from LRTM')
    parser.add_argument('csvfile', metavar='csvfile', type=str, help='Labelled csv file path')
#    parser.add_argument('-r', '--range', dest='ranges', type=str, action='append',
#        default=[], help='%% range of elapsed time to be labelled as "1"')
    parser.add_argument('-o', '--output', dest='output', type=str,
        default=None, help='path to output file')

    args = parser.parse_args()

    if not os.path.exists(args.csvfile):
        raise Exception("Input file does not exist: %s"%args.csvfile)

    # read data
    csvdata = read_csv(args.csvfile)

    cfg = OrderedDict()

    # create RF
    for region, counters in csvdata.items():


        cfg['feature_ID'] = OrderedDict(list((hwc,idx) for idx, hwc in enumerate(counters.keys()) if hwc != "LABEL"))
        feature_names = cfg['feature_ID'].keys()
        cfg['feature_map'] = OrderedDict(list((hwc,counters[hwc]) for hwc in feature_names))
        features = np.column_stack([counters[hwc] for hwc in feature_names])

        # apply pca ??

        #for n_trees in [50, 100, 1000]:
        for n_trees in [50, 100]:
        #for n_trees in [50]:

            print("")

            task = "%s_%d"%(LABEL, n_trees)

            # split data into train and test set

            X_train, X_test, y_train, y_test = train_test_split(
                features, counters[LABEL], test_size=0.33)
                #features, counters[LABEL], test_size=0.33, random_state=42)

            #rf = RandomForestRegressor(n_estimators = n_trees, oob_score=True, random_state = 42)
            rf = RandomForestRegressor(n_estimators = n_trees, oob_score=True)

            # train RF
            print("Fitting %s"%task)
            rf.fit(X_train, y_train)

            # predict
            print("Predicting %s"%task)
            pred = np.sign(rf.predict(X_test))
            #signed_pred = np.sign(pred)

            #score = rf.score(labels[hwc], features, labels[hwc])
            #import pdb; pdb.set_trace()
            #score = accuracy_score(labels[hwc], pred)
            #score = mean_squared_error(labels[hwc], pred)
            score = rf.oob_score_

            # diff
            #diff = np.abs(signed_pred - y_test)
            diff = np.abs(pred - y_test)

            #nonzeros = np.count_nonzero(diff) 
            #import pdb; pdb.set_trace()
            print("Prediction Error(RMSE) = {0:.2}".format(mean_squared_error(y_test, pred)))
            print("Prediction Error(Not-identical) = {0} of {1}".format(np.nonzero(diff)[0].size, diff.size))

            #diff = pred - labels[hwc]

#            # prepare plot data per lines
#            plines, mlines, tlines = codelines(diff)
#            x = np.asarray(sorted(list(set(plines.keys()) | set(mlines.keys()))), dtype=np.int32)
#            y1 = np.zeros(len(x)) # plus
#            y2 = np.zeros(len(x)) # minus
#            for idx, lineid in enumerate(x):
#                y1val = plines[lineid] if lineid in plines else 0
#                y2val = mlines[lineid] if lineid in mlines else 0
#                y1[idx] = y1val
#                y2[idx] = y2val
#
#            # prepare plot data per feature variables
            importances = list(rf.feature_importances_)
            #importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_names, importances)]
            importances = [(feature, importance) for feature, importance in zip(feature_names, importances)]
            importances = sorted(importances, key = lambda x: x[1], reverse = True)
            print("Important inputs")
            for event_name, importance in importances[:7]:
                print("{0} : {1:.2%}".format(event_name, importance))
            #import pdb; pdb.set_trace()
#
#            print("Plotting %s"%task)
#                # plot diff
#                error_page(fig, ax, diff, score, hwc, n_trees)
#
#                # plot analysis
#                weighted_page(fig, ax, x, y1, y2, hwc, n_trees)
#
#                # write rankings
#                ranking_page(fig, ax, x, y1, y2, tlines, importances, hwc, n_trees)
#                #plt.show()

    return 0

if __name__ == "__main__":
    sys.exit(main())
