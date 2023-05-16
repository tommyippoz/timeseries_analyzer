import random
import time

import numpy
import pandas
import sklearn

from tanalyzer.classifier.RegularClassifier import RegularClassifier
from tanalyzer.classifier.TimeSeriesClassifier import TimeSeriesClassifier
from tanalyzer.dataset.TimeSeriesInstance import TimeSeriesInstance


def read_csv_dataset(dataset_name, label_name="multilabel", limit=numpy.nan, split=True):
    """
    Method to process an input dataset as CSV
    :param normal_tag: tag that identifies normal data
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    df = pandas.read_csv(dataset_name, sep=",")

    # Shuffle
    df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)
    df = df[df.columns[df.nunique() > 1]]

    # Testing Purposes
    if (numpy.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    if split:
        encoding = pandas.factorize(df[label_name])
        y_enc = encoding[0]
        labels = encoding[1]
    else:
        y_enc = df[label_name]

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("\nDataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(
        len(normal_frame.index)) + " normal and " + str(len(numpy.unique(df[label_name]))) + " labels")

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns

    if split:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5,
                                                                                    shuffle=True)
        return x_train, x_test, y_train, y_test, feature_list, numpy.NaN
    else:
        return x_no_cat, y_enc, feature_list, numpy.NaN


def read_csv_binary_dataset(dataset_name, label_name="multilabel", normal_tag="normal", limit=numpy.nan, split=True):
    """
    Method to process an input dataset as CSV
    :param normal_tag: tag that identifies normal data
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    df = pandas.read_csv(dataset_name, sep=",")

    # Shuffle
    df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)
    df = df[df.columns[df.nunique() > 1]]

    # Testing Purposes
    if (numpy.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    # Binarize label
    if split:
        y_enc = numpy.where(df[label_name] == normal_tag, 0, 1)
    else:
        y_enc = df[label_name]

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("\nDataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(
        len(normal_frame.index)) + " normal and 2 labels")
    att_perc = (y_enc == 1).sum() / len(y_enc)

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns

    if split:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5,
                                                                                    shuffle=True)
        return x_train, x_test, y_train, y_test, feature_list, att_perc
    else:
        return x_no_cat, y_enc, feature_list, att_perc


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def get_classifier_name(clf):
    """
    Gets the name of the classifier
    :param clf: classifier to get the name of
    :return: a string
    """
    if isinstance(clf, TimeSeriesClassifier) or isinstance(clf, RegularClassifier):
        return clf.classifier_name()
    else:
        return clf.__class__.__name__


def write_dict(dict_obj, filename, header=None):
    """
    writes dict obj to file
    :param dict_obj: obj to write
    :param filename: file to create
    :param header: optional header of the file
    :return: None
    """
    with open(filename, 'w') as f:
        if header is not None:
            f.write("%s\n" % header)
        write_rec_dict(f, dict_obj, "")


def write_rec_dict(out_f, dict_obj, prequel):
    """
    writes dict obj to file
    :param dict_obj: obj to write
    :param out_f: file object to append data to
    :param prequel: optional prequel to put as header of each new row
    :return: None
    """
    if (type(dict_obj) is dict) or issubclass(type(dict_obj), dict):
        for key in dict_obj.keys():
            if (type(dict_obj[key]) is dict) or issubclass(type(dict_obj[key]), dict):
                if len(dict_obj[key]) > 10:
                    for inner in dict_obj[key].keys():
                        if (prequel is None) or (len(prequel) == 0):
                            out_f.write("%s,%s,%s\n" % (key, inner, dict_obj[key][inner]))
                        else:
                            out_f.write("%s,%s,%s,%s\n" % (prequel, key, inner, dict_obj[key][inner]))
                else:
                    prequel = prequel + "," + str(key) if (prequel is not None) and (len(prequel) > 0) else str(key)
                    write_rec_dict(out_f, dict_obj[key], prequel)
            elif type(dict_obj[key]) is list:
                item_count = 1
                for item in dict_obj[key]:
                    new_prequel = prequel + "," + str(key) + ",item" + str(item_count) \
                        if (prequel is not None) and (len(prequel) > 0) else str(key) + ",item" + str(item_count)
                    write_rec_dict(out_f, item, new_prequel)
                    item_count += 1
            else:
                if (prequel is None) or (len(prequel) == 0):
                    out_f.write("%s,%s\n" % (key, dict_obj[key]))
                else:
                    out_f.write("%s,%s,%s\n" % (prequel, key, dict_obj[key]))
    else:
        if (prequel is None) or (len(prequel) == 0):
            out_f.write("%s\n" % dict_obj)
        else:
            out_f.write("%s,%s\n" % (prequel, dict_obj))


def check_fitted(clf):
    if hasattr(clf, "classes_"):
        return True
    else:
        return False


def get_clf_name(clf):
    if hasattr(clf, "classifier_name"):
        return clf.classifier_name()
    else:
        return clf.__class__.__name__


def compute_feature_importances(clf):
    """
    Outputs feature ranking in building a Classifier
    :return: ndarray containing feature ranks
    """
    if hasattr(clf, 'feature_importances_'):
        return clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        return numpy.sum(numpy.absolute(clf.coef_), axis=0)
    elif isinstance(clf, TimeSeriesClassifier) or isinstance(clf, RegularClassifier):
        return clf.compute_feature_importances()
    return []


def clean_dataset(df, label_name="label", limit=numpy.nan):
    """
    Method to process an input dataset as CSV
    :param limit: integer to cut dataset if needed.
    :param df: the dataframe
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    df = df.fillna(0)
    df = df.replace('null', 0)
    df = df[df.columns[df.nunique() > 1]]

    # Testing Purposes
    if (numpy.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    y = df[label_name]
    df = df.drop(columns=[label_name])
    df = df.select_dtypes(exclude=['object'])
    df[label_name] = y

    return df


def extract_timeseries(df, cooldown=1, normal_tag='normal', label_name='label', shuffle=True):
    """
    Extracts a list of timeseries from a dataframe
    :param df: the dataframe
    :param cooldown: the cooldown after an anomaly
    :param normal_tag: the normal class
    :param label_name: the label column
    :param shuffle: True if series have to be shuffled before the return
    :return: a list of timeseries
    """
    timeseries_list = []
    from_index = 0
    while from_index < len(df.index):
        new_ts = TimeSeriesInstance(df, from_index, normal_tag, label_name)
        if new_ts is not None:
            from_index = from_index + new_ts.get_n_items() + cooldown + 1
            timeseries_list.append(new_ts)
        else:
            print('Error while parsing dataframe')
            break
    if shuffle:
        random.shuffle(timeseries_list)
    return timeseries_list


def compute_ts_gain(test_ts, y_pred, y_test, normal_class='normal', gain_decrease='linear'):
    gain_per_class = {}
    current_index = 0
    for ts in test_ts:
        true_series = y_test[current_index: current_index + len(ts.get_timeseries())]
        an_index = numpy.where(true_series != normal_class)[0]
        if ts.get_anomaly() not in gain_per_class:
            gain_per_class[ts.get_anomaly()] = []
        if len(an_index) > 0:
            an_index = an_index[0]
            pred_an = y_pred[current_index + an_index: current_index + len(ts.get_timeseries())]
            gain_per_class[ts.get_anomaly()].append(
                compute_gain(pred_an, gain_decrease, ts.get_anomaly()))
        else:
            gain_per_class[ts.get_anomaly()].append(0)
        current_index = current_index + len(ts.get_timeseries())
    gain_avg = {gcl: numpy.mean(gain_per_class[gcl]) for gcl in gain_per_class}
    gain_avg['wavg'] = numpy.sum(gain_avg[gcl] * len(gain_per_class[gcl]) for gcl in gain_per_class) / len(test_ts)
    return gain_avg['wavg']


def compute_gain(preds, gain_decrease, an_tag):
    if preds is None or len(preds) == 0:
        return 0
    elif preds[0] == an_tag:
        return 1
    elif an_tag not in preds:
        return 0
    else:
        duration = len(preds)
        first_det = numpy.where(preds == an_tag)[0][0]
        if gain_decrease == 'quadratic':
            gf = lambda t: t ** 2
        elif gain_decrease == 'cubic':
            gf = lambda t: t ** 3
        else:
            gf = lambda t: t
        return gf(first_det / duration)
