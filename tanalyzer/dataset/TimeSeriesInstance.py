import warnings
import numpy
import pandas


class TimeSeriesInstance:
    """
    Class that contains a time series.
    Builds upon PANDAS.
    """

    def __init__(self, dataframe, from_index: int = 0,
                 normal_tag: str = 'normal', label_column: str = 'label'):
        self.series = None
        self.anomaly_tag = None
        self.label_column = label_column
        self.anomaly_indexes = []
        self.next_row = None
        if dataframe is not None and isinstance(dataframe, pandas.DataFrame):
            if 0 <= from_index < len(dataframe.index):
                anomalies = dataframe[label_column].values[from_index:] != normal_tag
                an_index = from_index + anomalies.argmax()
                self.anomaly_tag = dataframe[label_column][an_index]
                while (an_index < len(dataframe.index)) \
                        and (dataframe[label_column].values[an_index] == self.anomaly_tag):
                    self.anomaly_indexes.append(an_index)
                    an_index = an_index + 1
                self.series = dataframe.iloc[from_index:an_index, :]
            else:
                print('Start index %d of the timeseries is not valid' % from_index)
        else:
            print('You need to provide a dataframe to extract data from')

    def get_anomaly(self) -> str:
        """
        Gets the type of anomaly contained in the timeseries
        :return: a string
        """
        return self.anomaly_tag

    def get_timeseries(self) -> pandas.DataFrame:
        """
        Gets timeseries as pandas object
        :return: the list of observations in the timeseries
        """
        return self.series

    def get_firstorder_timeseries(self) -> pandas.DataFrame:
        """
        Gets timeseries data and adds first order differences (diff wrt previous)
        :return: an updated dataframe
        """
        y = self.series[self.label_column]
        new_series = self.series.drop(columns=[self.label_column])
        new_series = new_series.join(pandas.DataFrame(
            dict([(col + '_diff', new_series[col] - new_series[col].shift(+1)) for col in new_series.columns]),
            index=new_series.index))
        for col in new_series.columns:
            if '_diff' in col:
                new_series.loc[new_series.index[0], col] = 0
        new_series[self.label_column] = y
        return new_series

    def get_secondorder_timeseries(self) -> pandas.DataFrame:
        """
        Gets timeseries data and adds first and second order differences
        :return: an updated dataframe
        """
        y = self.series[self.label_column]
        new_series = self.series.drop(columns=[self.label_column])
        new_series_d1 = dict([(col + '_diff1', new_series[col] - new_series[col].shift(+1))
                              for col in new_series.columns])
        new_series_d2 = dict([(col + '_diff2', new_series[col] - new_series[col].shift(+2))
                              for col in new_series.columns])
        new_series_d1.update(new_series_d2)
        new_series = new_series.join(pandas.DataFrame(new_series_d1, index=new_series.index))
        for col in new_series.columns:
            if '_diff' in col:
                new_series.loc[new_series.index[0], col] = 0
            if '_diff2' in col and len(new_series.index) > 1:
                new_series.loc[new_series.index[1], col] = \
                    new_series.loc[new_series.index[1], col.replace('_diff2', '_diff1')]
        new_series[self.label_column] = y
        return new_series

    def get_movingavg_timeseries(self, n_obs) -> pandas.DataFrame:
        """
        Gets timeseries data and adds a moving average and the difference between observation and average
        :param n_obs: observations to compute average of
        :return: an updated dataframe
        """
        y = self.series[self.label_column]
        new_series = self.series.drop(columns=[self.label_column])
        m_data = [[new_series[col].shift(i).to_numpy() for i in range(1, n_obs+1)] for col in new_series.columns]
        # avoid 'RuntimeWarning: Mean of empty slice'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_data = numpy.transpose(numpy.asarray([numpy.nanmean(x, axis=0) for x in m_data]))
        avg_data[0, :] = new_series.iloc[0].to_numpy()
        movingavg = pandas.DataFrame(avg_data, columns=[col + '_mavg' for col in new_series.columns],
                                     index=new_series.index)
        new_series_diff = dict([(col + '_diff', new_series[col] - movingavg[col + '_mavg'])
                                for col in new_series.columns])
        new_series = new_series.join(movingavg)
        new_series = new_series.join(pandas.DataFrame(new_series_diff, index=new_series.index))
        new_series[self.label_column] = y
        return new_series

    def get_n_items(self) -> int:
        """
        Gets the number of observation that are in the timseries
        :return: an int
        """
        return len(self.series.index)
