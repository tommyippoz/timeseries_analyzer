import numpy
import pandas
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class TimeSeriesClassifier:
    """
    Basic Abstract Class for TimeSeriesClassifiers.
    Abstract methods are only the classifier_name, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self, model):
        """
        Constructor of a generic Classifier
        :param model: model to be used as Classifier
        """
        self.model = model
        self.trained = False
        self._estimator_type = "classifier"
        self.classes_ = None
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, x_train, y_train=None):
        """
        Fits a Classifier
        :param x_train: feature set
        :param y_train: labels
        """
        if y_train is not None:
            if isinstance(x_train, pd.DataFrame):
                self.model.fit(x_train.to_numpy(), y_train)
            else:
                self.model.fit(x_train, y_train)
            self.classes_ = numpy.unique(y_train)
        else:
            if isinstance(x_train, pd.DataFrame):
                self.model.fit(x_train.to_numpy())
            else:
                self.model.fit(x_train)
            self.classes_ = 2
        self.feature_importances_ = self.compute_feature_importances()
        self.trained = True

    def is_trained(self):
        """
        Flags if train was executed
        :return: True if trained, False otherwise
        """
        return self.trained

    def predict(self, x_test):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        if isinstance(x_test, pandas.DataFrame):
            x_t = x_test.to_numpy()
        else:
            x_t = x_test
        return self.model.predict(x_t)

    def predict_proba(self, x_test):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """
        return self.model.predict_proba(x_test)

    def predict_confidence(self, x_test):
        """
        Method to compute confidence in the predicted class
        :return: -1 as default, value if algorithm is from framework PYOD
        """
        return -1

    def compute_feature_importances(self):
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return numpy.sum(numpy.absolute(self.model.coef_), axis=0)
        return []

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        pass


class FeatureRegressor(TimeSeriesClassifier):

    def __init__(self, clf_model, reg_model=DecisionTreeRegressor()):
        """
        Constructor of a generic Classifier
        :param model: model to be used as Classifier
        """
        super().__init__(clf_model)
        self.reg_model = reg_model
        self.regressors = None

    def fit(self, x_train, y_train=None):
        """
        Fits a Classifier
        :param x_train: feature set
        :param y_train: labels
        """

        super().fit(x_train, y_train)

    def predict(self, x_test):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """

        super().predict(x_test)

    def predict_proba(self, x_test):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """

        return super().predict_proba(x_test)

    def update_set(self, x_set):
        """
        To be overridden by classes
        :param x_set:
        :return:
        """

        pass
