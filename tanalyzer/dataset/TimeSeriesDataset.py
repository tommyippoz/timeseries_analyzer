import random

import numpy
import pandas
import sklearn.metrics

from tanalyzer.dataset.TimeSeriesInstance import TimeSeriesInstance
from tanalyzer.utils import clean_dataset, extract_timeseries, compute_ts_gain


class TimeSeriesDataset:
    """
    Class that contains a time series dataset and provides utility methods. Builds upon PANDAS.
    """

    def __init__(self, dataframe: pandas.DataFrame, label_name: str = 'label',
                 timestamp_name: str = 'timestamp', dataset_name: str = None,
                 normal_class: str = 'normal', cooldown: int = 1):
        # Default Information
        self.dataframe = dataframe
        self.label_name = label_name
        self.timestamp_name = timestamp_name
        self.cooldown = cooldown
        self.dataset_name = dataset_name if dataframe is not None else 'ts_dataset'
        # Additional Information
        if label_name in dataframe.columns:
            self.classes = dataframe[label_name].unique()
        else:
            self.classes = []
        if normal_class not in self.classes:
            print('Normal class \'%s\' does not exist' % normal_class)
            self.normal_class = None
        else:
            self.normal_class = normal_class
        # To be updated later
        self.timeseries = []
        self.train_ts = None
        self.test_ts = None

    def preprocess(self, normalize: bool = True, split: float = 0.5, avoid_anomalies: list = []):
        """
        Preprocesses dataset and prepares it for analyses
        :param avoid_anomalies: list of tags of anomalies to be discarded when creating train and test splits
        :param normalize: True if feature data has to be normalized
        :param split: float that sets the percentage of train-test split
        """
        self.timeseries = []
        print('Preprocessing Dataset ...')
        # Checking for formatting or numbering errors, removing text columns
        print('Initial Features: %d' % (len(self.dataframe.columns) - 1))
        self.dataframe = clean_dataset(self.dataframe, label_name=self.label_name)
        print('Final Features: %d' % (len(self.dataframe.columns) - 1))
        # Normalizing
        if normalize:
            self.dataframe = (self.dataframe - self.dataframe.min()) / \
                             (self.dataframe.max() - self.dataframe.min())
        # Extracting timeseries
        self.timeseries = extract_timeseries(self.dataframe, cooldown=self.cooldown,
                                             normal_tag=self.normal_class, label_name=self.label_name)
        self.train_ts = self.timeseries[0:int(len(self.timeseries) * split)]
        self.test_ts = self.timeseries[int(len(self.timeseries) * split):]
        return self.train_ts, self.test_ts

    def get_data(self, data_type='train', augmentation=None):
        series_group = self.train_ts if data_type == 'train' else self.test_ts
        if augmentation == 'firstorder':
            series_data = [ds.get_firstorder_timeseries() for ds in series_group]
        else:
            series_data = [ds.get_timeseries() for ds in series_group]
        series_data = pandas.concat(series_data, ignore_index=True)
        series_mapping = ['ts_' + data_type + '_' + str(i)
                          for i in range(0, len(series_group)) for _ in range(0, series_group[i].get_n_items())]
        series_mapping = [''.join(ele) for ele in series_mapping]
        y_data = series_data[self.label_name].to_numpy()
        x_data = series_data.drop(columns=[self.label_name])
        return x_data, y_data, series_mapping

    def compute_metrics(self, test_ts, y_pred, y_test):
        m_dict = {}
        if len(numpy.unique(y_test)) > 2:
            multi_dict = sklearn.metrics.classification_report(y_test, y_pred, output_dict=True)
            for wk in multi_dict['weighted avg']:
                m_dict[wk] = multi_dict['weighted avg'][wk]
        else:
            m_dict['p'] = sklearn.metrics.precision_score(y_test, y_pred)
            m_dict['r'] = sklearn.metrics.recall_score(y_test, y_pred)
            m_dict['f1'] = sklearn.metrics.f1_score(y_test, y_pred)
        m_dict['fpr'] = sum((y_test == self.normal_class) * (y_pred != self.normal_class)) \
                        / sum(y_test == self.normal_class)
        m_dict['accuracy'] = sklearn.metrics.accuracy_score(y_test, y_pred)
        m_dict['mcc'] = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
        m_dict['linear_detection_gain'] = compute_ts_gain(test_ts, y_pred, y_test, normal_class=self.normal_class)
        m_dict['quadratic_detection_gain'] = compute_ts_gain(test_ts, y_pred, y_test,
                                                             normal_class=self.normal_class, gain_decrease='quadratic')
        m_dict['cubic_detection_gain'] = compute_ts_gain(test_ts, y_pred, y_test, normal_class=self.normal_class,
                                                         gain_decrease='cubic')
        m_dict['ts_metric'] = 2 * (1 - m_dict['fpr']) * m_dict['linear_detection_gain'] / \
                              (1 - m_dict['fpr'] + m_dict['linear_detection_gain'])
        return m_dict
