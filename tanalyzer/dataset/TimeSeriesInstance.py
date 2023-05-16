import pandas


class TimeSeriesInstance:
    """
    Class that contains a time series Builds upon PANDAS.
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
                self.series.reset_index(inplace=True)
                self.next_row = dataframe.iloc[an_index:an_index+1, :]
                self.next_row.reset_index(inplace=True)
            else:
                print('Start index %d of the timeseries is not valid' % from_index)
        else:
            print('You need to provide a dataframe to extract data from')

    def get_anomaly(self):
        return self.anomaly_tag

    def get_timeseries(self):
        return self.series

    def get_firstorder_timeseries(self):
        y = self.series[self.label_column]
        new_series = self.series.drop(columns=[self.label_column])
        for col in new_series.columns:
            new_series[col + '_diff'] = new_series[col] - new_series[col].shift(-1)
            if self.next_row is not None:
                new_series[len(new_series.index)-1, col + '_diff'] = \
                    new_series[len(new_series.index)-1, col] - self.next_row[col].to_numpy()[0]
            else:
                new_series[col + '_diff', len(new_series.index)-1] = 0
        new_series[self.label_column] = y
        return new_series

    def get_n_items(self):
        return len(self.series.index)
