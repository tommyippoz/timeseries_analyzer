import random

import numpy
import pandas

from tanalyzer.TimeSeriesInstance import TimeSeriesInstance


class TimeSeriesDataset:
    """
    Class that contains a time series dataset and provides utility methods. Builds upon PANDAS.
    """

    def __init__(self, dataframe: pandas.DataFrame, label_name: str = 'label',
                 timestamp_name: str = 'timestamp', dataset_name: str = None,
                 normal_class: str = 'normal'):
        self.dataframe = dataframe
        self.label_name = label_name
        self.timestamp_name = timestamp_name
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

    def preprocess(self, normalize: bool = True, split: float = 0.5,
                   an_duration: int = 1, cooldown: int = None, avoid_anomalies: list = []):
        """
        Preprocesses dataset and prepares it for analyses
        :param avoid_anomalies: list of tags of anomalies to be discarded when creating train and test splits
        :param an_duration: number of observations in which the anomaly appears
        :param normalize: True if feature data has to be normalized
        :param split: float that sets the percentage of train-test split
        :param cooldown: int that specifies the number of observations between an anomaly
            and the next normal instance
        """
        timeseries = []
        if normalize:
            self.dataframe = (self.dataframe-self.dataframe.min()) / \
                             (self.dataframe.max() - self.dataframe.min())
        from_index = 0
        while from_index < len(self.dataframe.index):
            new_ts = TimeSeriesInstance(self.dataframe, from_index, cooldown, an_duration,
                                        self.normal_tag, self.label_column)
            if new_ts is not None:
                from_index = new_ts.get_range()['to'] + 1
                if new_ts.get_anomaly() not in avoid_anomalies:
                    timeseries.append(new_ts)
            else:
                print('Error while parsing dataframe')
                break
        random.shuffle(timeseries)
        train_ts = timeseries[0:int(len(timeseries) * split)]
        test_ts = timeseries[int(len(timeseries) * split):]
        train_df = pandas.concat([ds.get_df() for ds in train_ts], ignore_index=True)
        test_df = pandas.concat([ds.get_df() for ds in test_ts], ignore_index=True)
        return train_ts, test_ts, train_df, test_df


