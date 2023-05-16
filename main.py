import pandas
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from tanalyzer.classifier.RegularClassifier import RegularClassifier, UnsupervisedClassifier
from tanalyzer.dataset.TimeSeriesDataset import TimeSeriesDataset
from tanalyzer.utils import get_classifier_name

CSV_FILE = "data/Draft_Dataset_5anomalie_3orelog.csv"

LABEL_NAME = 'label'

CLASSIFIERS = [RegularClassifier(model=DecisionTreeClassifier()),
               RegularClassifier(model=LinearDiscriminantAnalysis()),
               RegularClassifier(model=RandomForestClassifier()),
               RegularClassifier(model=XGBClassifier()),
               UnsupervisedClassifier(classifier=COPOD()),
               UnsupervisedClassifier(classifier=IForest()),
               ]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pandas.read_csv(CSV_FILE)
    ts_d = TimeSeriesDataset(df, timestamp_name='_timestamp', dataset_name='test_dataset',
                             an_duration=2, cooldown=5)
    train_ts, test_ts, train_df, test_df = ts_d.preprocess(normalize=False, split=0.7)
    x_train = train_df.drop(columns=[LABEL_NAME]).to_numpy()
    y_train = train_df[LABEL_NAME].to_numpy()
    x_test = test_df.drop(columns=[LABEL_NAME]).to_numpy()
    y_test = test_df[LABEL_NAME].to_numpy()
    for clf in CLASSIFIERS:
        print('Using classifier: %s' % get_classifier_name(clf))
        # Training and Testing classifier
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # Computing Metrics
        ts_metrics = ts_d.compute_metrics(test_ts, y_pred, y_test)
        for met_name in ['accuracy', 'mcc', 'fpr', 'linear_detection_gain', 'ts_metric']:
            print('%s: %.3f' % (met_name, ts_metrics[met_name]))
        # for met_name in ts_metrics:
        #     if not isinstance(ts_metrics[met_name], dict):
        #         print('%s: %.3f' % (met_name, ts_metrics[met_name]))
        #     else:
        #         for inner in ts_metrics[met_name]:
        #             print('%s - %s: %.3f' % (met_name, inner, ts_metrics[met_name][inner]))
