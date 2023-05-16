import pandas


class TimeSeriesInstance:
    """
    Class that contains a time series Builds upon PANDAS.
    """

    def __init__(self, dataframe, from_index: int = 0, cooldown: int = 1, an_duration: int = 1,
                 normal_tag: str = 'normal', label_column: str = 'label'):
        self.series = None
        self.anomaly_tag = None
        self.anomaly_indexes = None
        self.series_range = {'from': from_index, 'to': None}
        if dataframe is not None and isinstance(dataframe, pandas.DataFrame):
            if 0 <= from_index < len(dataframe.index):
                an_index = (dataframe[label_column].values != normal_tag).argmax()
                self.anomaly_indexes = [i for i in range(an_index, an_index + an_duration)]
                self.anomaly_tag = dataframe[label_column][an_index]
                self.series_range['to'] = an_index + an_duration + cooldown
                self.series = dataframe.iloc[from_index:self.series_range['to'], :]
            else:
                print('Start index %d of the timeseries is not valid' % from_index)
        else:
            print('You need to provide a dataframe to extract data from')

    def get_anomaly(self):
        return self.anomaly_tag

    def get_timeseries(self):
        return self.series

    def get_range(self):
        return self.series_range

    def get_df(self):
        return self.series
