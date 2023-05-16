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

CLASSIFIERS = [#RegularClassifier(model=DecisionTreeClassifier()),
               #RegularClassifier(model=LinearDiscriminantAnalysis()),
               #RegularClassifier(model=RandomForestClassifier()),
               XGB(),
               #UnsupervisedClassifier(classifier=COPOD()),
               #UnsupervisedClassifier(classifier=IForest()),
               ]

FEATURE_TYPE = [None, 'firstorder', 'movingavg']

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pandas.read_csv(CSV_FILE)
    ts_d = TimeSeriesDataset(df, timestamp_name='_timestamp', dataset_name='test_dataset', cooldown=5)
    train_ts, test_ts = ts_d.preprocess(normalize=False, split=0.7)
    for ft in FEATURE_TYPE:
        print('\nFeature engineering type: %s\n', ft)
        x_train, y_train, ts_train_map = ts_d.get_data(data_type='train', augmentation=ft)
        x_test, y_test, ts_test_map = ts_d.get_data(data_type='test', augmentation=ft)
        for clf in CLASSIFIERS:
            print('Using classifier: %s' % get_classifier_name(clf))
            # Training and Testing classifier
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            # Computing Metrics
            ts_metrics = ts_d.compute_metrics(test_ts, y_pred, y_test)
            for met_name in ['accuracy', 'mcc', 'fpr', 'linear_detection_gain', 'ts_metric']:
                print('%s: %.3f' % (met_name, ts_metrics[met_name]))
