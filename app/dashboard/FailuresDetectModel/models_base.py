# models_base.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .models_config import MODEL_COLUMN_CONFIG

class ModelBase:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def get_column_config(self):
        """Méthode pour obtenir la configuration des colonnes pour ce modèle"""
        return MODEL_COLUMN_CONFIG.get(self.__class__.__name__, {})


class RandomForestModel(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class LogisticRegressionModel(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
