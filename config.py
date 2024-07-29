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

TRAIN_SIZE = 2000
TARGET_COLUMN = "valve_condition_normalized"
MODEL_PARAMETERS = {
    "LogisticRegression": {"C": [0.001, 0.01, 0.1, 1, 10], "solver": ["liblinear"]},
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 7],
    },
    "SVC": {
        "C": [0.01, 0.1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
}
