import random

import numpy
import pandas
import sklearn.metrics

from tanalyzer.dataset.TimeSeriesInstance import TimeSeriesInstance


class TimeSeriesDataset:
    """
    Class that contains a time series dataset and provides utility methods. Builds upon PANDAS.
    """

    def __init__(self, dataframe: pandas.DataFrame, label_name: str = 'label',
                 timestamp_name: str = 'timestamp', dataset_name: str = None,
                 normal_class: str = 'normal', an_duration: int = 1, cooldown: int = 1, ):
        self.dataframe = dataframe
        self.label_name = label_name
        self.timestamp_name = timestamp_name
        self.an_duration = an_duration
        self.cooldown = cooldown
        self.dataset_name = dataset_name if dataframe is not None else 'ts_dataset'
        if label_name in dataframe.columns:
            self.classes = dataframe[label_name].unique()
        else:
            self.classes = []
        if normal_class not in self.classes:
            print('Normal class \'%s\' does not exist' % normal_class)
            self.normal_class = None
        else:
            self.normal_class = normal_class
        self.feature_scaler = None
        self.timeseries = []
        self.unique_labels = None

    def preprocess(self, normalize: bool = True, split: float = 0.5, avoid_anomalies: list = []):
        """
        Preprocesses dataset and prepares it for analyses
        :param avoid_anomalies: list of tags of anomalies to be discarded when creating train and test splits
        :param normalize: True if feature data has to be normalized
        :param split: float that sets the percentage of train-test split
        """
        self.timeseries = []
        print('Preprocessing Dataset ...')
        # Checking for formatting or numbering errors
        print('Initial Features: %d' % (len(self.dataframe.columns) - 1))
        self.dataframe = self.dataframe.fillna(0)
        self.dataframe = self.dataframe.replace('null', 0)
        self.dataframe = self.dataframe[self.dataframe.columns[self.dataframe.nunique() > 1]]
        dataframe_y = self.dataframe[self.label_name]
        encoding = pandas.factorize(dataframe_y)
        dataframe_y = encoding[0]
        self.unique_labels = encoding[1]
        print('Encoding %d Labels: %s' % (len(self.unique_labels), "\n\t".join(self.unique_labels)))
        dataframe_x = self.dataframe.drop(columns=[self.label_name])
        self.dataframe = dataframe_x.select_dtypes(exclude=['object'])
        self.dataframe[self.label_name] = dataframe_y
        print('Final Features: %d' % (len(self.dataframe.columns) - 1))
        # Normalizing
        if normalize:
            self.dataframe = (self.dataframe - self.dataframe.min()) / \
                             (self.dataframe.max() - self.dataframe.min())
        # Extracting timeseries
        from_index = 0
        while from_index < len(self.dataframe.index):
            new_ts = TimeSeriesInstance(self.dataframe, from_index, self.cooldown, self.an_duration,
                                        numpy.where(self.unique_labels == self.normal_class)[0],
                                        self.label_name)
            if new_ts is not None:
                from_index = new_ts.get_range()['to'] + 1
                if new_ts.get_anomaly() not in avoid_anomalies:
                    self.timeseries.append(new_ts)
            else:
                print('Error while parsing dataframe')
                break
        random.shuffle(self.timeseries)
        train_ts = self.timeseries[0:int(len(self.timeseries) * split)]
        test_ts = self.timeseries[int(len(self.timeseries) * split):]
        train_df = pandas.concat([ds.get_df() for ds in train_ts], ignore_index=True)
        test_df = pandas.concat([ds.get_df() for ds in test_ts], ignore_index=True)
        return train_ts, test_ts, train_df, test_df

    def compute_metrics(self, test_ts, y_pred, y_test):
        m_dict = {}
        y_pred_str = self.unique_labels[y_pred]
        y_test_str = self.unique_labels[y_test]
        if len(numpy.unique(y_test)) > 2:
            multi_dict = sklearn.metrics.classification_report(y_test_str, y_pred_str, output_dict=True)
            for wk in multi_dict['weighted avg']:
                m_dict[wk] = multi_dict['weighted avg'][wk]
        else:
            m_dict['p'] = sklearn.metrics.precision_score(y_test, y_pred)
            m_dict['r'] = sklearn.metrics.recall_score(y_test, y_pred)
            m_dict['f1'] = sklearn.metrics.f1_score(y_test, y_pred)
        m_dict['fpr'] = sum((y_test_str == self.normal_class) * (y_pred_str != self.normal_class)) \
                        / sum(y_test_str == self.normal_class)
        m_dict['accuracy'] = sklearn.metrics.accuracy_score(y_test, y_pred)
        m_dict['mcc'] = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
        m_dict['linear_detection_gain'] = self.compute_timing_metric(test_ts, y_pred_str, y_test_str)
        m_dict['quadratic_detection_gain'] = self.compute_timing_metric(test_ts, y_pred_str, y_test_str,
                                                                        gain_decrease='quadratic')
        m_dict['cubic_detection_gain'] = self.compute_timing_metric(test_ts, y_pred_str, y_test_str,
                                                                    gain_decrease='cubic')
        m_dict['ts_metric'] = 2*(1-m_dict['fpr'])*m_dict['linear_detection_gain']/\
                              (1-m_dict['fpr']+m_dict['linear_detection_gain'])
        return m_dict

    def compute_timing_metric(self, test_ts, y_pred, y_test, gain_decrease='linear'):
        gain_per_class = {}
        current_index = 0
        for ts in test_ts:
            true_series = y_test[current_index: current_index + len(ts.get_timeseries())]
            an_index = numpy.where(true_series != self.normal_class)[0]
            if ts.get_anomaly() not in gain_per_class:
                gain_per_class[ts.get_anomaly()] = []
            if len(an_index) > 0:
                an_index = an_index[0]
                pred_an = y_pred[current_index + an_index: current_index + len(ts.get_timeseries())]
                gain_per_class[ts.get_anomaly()].append(
                    self.compute_gain(pred_an, gain_decrease, self.unique_labels[ts.get_anomaly()]))
            else:
                gain_per_class[ts.get_anomaly()].append(0)
            current_index = current_index + len(ts.get_timeseries())
        gain_avg = {gcl: numpy.mean(gain_per_class[gcl]) for gcl in gain_per_class}
        gain_avg['wavg'] = numpy.sum(gain_avg[gcl] * len(gain_per_class[gcl]) for gcl in gain_per_class) / len(test_ts)
        return gain_avg['wavg']

    def compute_gain(self, preds, gain_decrease, an_tag):
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
