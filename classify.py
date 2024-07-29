import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt

# Define constants
PROFILE_COLUMN_NAMES = [
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure",
    "stable_flag",
]
COOLER_CONDITION_MAP = {3: 0.0, 20: 0.5, 100: 1.0}
VALVE_CONDITION_MAP = {100: 1.0, 90: 0.75, 80: 0.5, 73: 0.25}
PUMP_LEAKAGE_MAP = {2: 0.0, 1: 0.5, 0: 1.0}
ACCUMULATOR_PRESSURE_MAP = {130: 1.0, 115: 0.75, 100: 0.5, 90: 0.25}
STABLE_FLAG_MAP = {0: 1.0, 1: 0.0}


def process_data(profile_path, fs1_path, ps2_path):
    # Load data
    profile = pd.read_csv(profile_path, sep="\t", header=None)
    fs1 = pd.read_csv(fs1_path, sep="\t", header=None)
    ps2 = pd.read_csv(ps2_path, sep="\t", header=None)

    # Add column names to profile data
    profile.columns = PROFILE_COLUMN_NAMES

    # Normalize profile data
    profile["cooler_condition_normalized"] = profile["cooler_condition"].map(
        COOLER_CONDITION_MAP
    )
    profile["valve_condition_normalized"] = profile["valve_condition"].map(
        VALVE_CONDITION_MAP
    )
    profile["pump_leakage_normalized"] = profile["pump_leakage"].map(PUMP_LEAKAGE_MAP)
    profile["accumulator_pressure_normalized"] = profile["accumulator_pressure"].map(
        ACCUMULATOR_PRESSURE_MAP
    )
    profile["stable_flag_normalized"] = profile["stable_flag"].map(STABLE_FLAG_MAP)

    profile_normalized = profile[
        [
            "cooler_condition_normalized",
            "valve_condition_normalized",
            "pump_leakage_normalized",
            "accumulator_pressure_normalized",
            "stable_flag_normalized",
        ]
    ]

    # Normalize fs1 data
    fs1_normalized = (fs1 - fs1.min()) / (fs1.max() - fs1.min())

    # Normalize ps2 data
    ps2_normalized = (ps2 - ps2.min()) / (ps2.max() - ps2.min())

    # Concatenate all data
    data = pd.concat([ps2_normalized, fs1_normalized, profile_normalized], axis=1)

    return data, fs1, ps2


# Example usage
profile_path = "data_subset/profile.txt"
fs1_path = "data_subset/FS1.txt"
ps2_path = "data_subset/PS2.txt"

data, fs1, ps2 = process_data(profile_path, fs1_path, ps2_path)
data.columns = data.columns.astype(str)

# Define Constants
TRAIN_SIZE = 1000
TARGET_COLUMN = "valve_condition_normalized"
MODEL_PARAMETERS = {
    "LogisticRegression": {"C": [0.001, 0.01], "solver": ["liblinear"]},
    "RandomForestClassifier": {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [50, 100],
        "learning_rate": [0.001, 0.01],
        "max_depth": [3, 5],
    },
    "SVC": {
        "C": [0.001, 0.01],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
}


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
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "f1_score": f1_score(y_test, y_pred, average="binary"),
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


def binarize_targets(y, threshold=0.5):
    return (y > threshold).astype(int)


# Split data into train and test
train_data = data.iloc[:TRAIN_SIZE]
test_data = data.iloc[TRAIN_SIZE:]

# Features and target
X_train = train_data.drop(columns=[TARGET_COLUMN])
y_train = train_data[TARGET_COLUMN]
y_train = binarize_targets(y_train)
X_test = test_data.drop(columns=[TARGET_COLUMN])
y_test = test_data[TARGET_COLUMN]
y_test = binarize_targets(y_test)

# Initialize models
models = [
    LogisticRegressionModel(),
    RandomForestClassifierModel(),
    GradientBoostingClassifierModel(),
    SVCModel(),
]

# Train and evaluate models with verbose output
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

print(model_results)
