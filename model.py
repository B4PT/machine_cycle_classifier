from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from config import MODEL_PARAMETERS


class ModelBase:
    def __init__(self, name, model, params):
        self.name = name
        self.model = model
        self.params = params
        self.best_model = None
        self.grid_search = None

    def train(self, X_train, y_train):
        self.grid_search = GridSearchCV(
            self.model, self.params, cv=5, scoring="accuracy", n_jobs=-1
        )
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_

    def evaluate(self, X_test, y_test):
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }
        return metrics

    def predict_cycle(self, cycle_data):
        if self.best_model is None:
            raise ValueError(f"Model {self.name} has not been trained yet.")
        return self.best_model.predict(cycle_data)


class LogisticRegressionModel(ModelBase):
    def __init__(self):
        super().__init__(
            "LogisticRegression",
            LogisticRegression(),
            MODEL_PARAMETERS["LogisticRegression"],
        )


class RandomForestClassifierModel(ModelBase):
    def __init__(self):
        super().__init__(
            "RandomForestClassifier",
            RandomForestClassifier(),
            MODEL_PARAMETERS["RandomForestClassifier"],
        )


class GradientBoostingClassifierModel(ModelBase):
    def __init__(self):
        super().__init__(
            "GradientBoostingClassifier",
            GradientBoostingClassifier(),
            MODEL_PARAMETERS["GradientBoostingClassifier"],
        )


class SVCModel(ModelBase):
    def __init__(self):
        super().__init__("SVC", SVC(probability=True), MODEL_PARAMETERS["SVC"])


def initialize_models():
    return [
        LogisticRegressionModel(),
        RandomForestClassifierModel(),
        GradientBoostingClassifierModel(),
        SVCModel(),
    ]


def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    model_results = {}
    num_steps = len(models) * 2  # Training and evaluating each model
    step_counter = 0

    for model in models:
        step_counter += 1
        print(f"Step {step_counter}/{num_steps}: Training {model.name}...")
        model.train(X_train, y_train)
        step_counter += 1
        print(f"Step {step_counter}/{num_steps}: Evaluating {model.name}...")
        metrics = model.evaluate(X_test, y_test)
        model_results[model.name] = metrics

    return model_results
