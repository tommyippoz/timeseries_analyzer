import pandas
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from tanalyzer.classifier.RegularClassifier import RegularClassifier, UnsupervisedClassifier, XGB
from tanalyzer.dataset.TimeSeriesDataset import TimeSeriesDataset
from tanalyzer.utils import get_classifier_name

CSV_FILE = "data/Draft_Dataset_5anomalie_3orelog.csv"

LABEL_NAME = 'label'

CLASSIFIERS = [RegularClassifier(model=DecisionTreeClassifier()),
               RegularClassifier(model=LinearDiscriminantAnalysis()),
               RegularClassifier(model=RandomForestClassifier()),
               XGB()
               ]

FEATURE_TYPE = [None, 'firstorder', 'secondorder', 'movingavg_3', 'movingavg_5', 'movingavg_10']

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pandas.read_csv(CSV_FILE)
    ts_d = TimeSeriesDataset(df, timestamp_name='_timestamp', dataset_name='test_dataset', cooldown=5)
    train_ts, test_ts = ts_d.preprocess(normalize=False, split=0.7)
    for ft in FEATURE_TYPE:
        x_train, y_train, ts_train_map = ts_d.get_data(data_type='train', augmentation=ft)
        x_test, y_test, ts_test_map = ts_d.get_data(data_type='test', augmentation=ft)
        print('\nFeature engineering type: %s, using %d features' % (ft, len(x_train.columns)))
        for clf in CLASSIFIERS:
            # Training and Testing classifier
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            # Computing Metrics
            ts_metrics = ts_d.compute_metrics(test_ts, y_pred, y_test)
            print('[%s]: acc=%.3f, \tmcc=%.3f, \tfpr=%.3f, \tlin_gain=%.3f, \tpcz=%.3f' %
                  (get_classifier_name(clf), ts_metrics['accuracy'], ts_metrics['mcc'],
                   ts_metrics['fpr'], ts_metrics['linear_detection_gain'], ts_metrics['ts_metric']))
