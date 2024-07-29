import nbimporter
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from classifier import process_data, binarize_targets


# Mock data for tests
@pytest.fixture
def sample_data():
    # Create example data
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
            "target": [0, 1, 0, 1, 0],
        }
    )
    return data


def test_binarize_targets(sample_data):
    # Prepare the data
    y = sample_data["target"]
    y_binarized = binarize_targets(y)

    # Check that binarization is correct
    assert all(
        y_binarized == label_binarize(y, classes=[0, 1]).flatten()
    ), "Target binarization is incorrect"


def test_train_test_split(sample_data):
    # Prepare the data
    TRAIN_SIZE = 3
    train_data = sample_data.iloc[:TRAIN_SIZE]
    test_data = sample_data.iloc[TRAIN_SIZE:]

    X_train = train_data.drop(columns=["target"])
    y_train = train_data["target"]
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    assert X_train.shape[0] == TRAIN_SIZE, "Number of training samples is incorrect"
    assert (
        X_test.shape[0] == sample_data.shape[0] - TRAIN_SIZE
    ), "Number of test samples is incorrect"


def test_model_training_and_evaluation(sample_data):
    # Prepare the data
    TRAIN_SIZE = 3
    train_data = sample_data.iloc[:TRAIN_SIZE]
    test_data = sample_data.iloc[TRAIN_SIZE:]

    X_train = train_data.drop(columns=["target"])
    y_train = train_data["target"]
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "SVC": SVC(probability=True),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # You can check that the metrics are reasonable
        assert 0 <= accuracy <= 1, f"Accuracy for {name} is outside the range [0, 1]"
        assert 0 <= roc_auc <= 1, f"ROC AUC for {name} is outside the range [0, 1]"


if __name__ == "__main__":
    pytest.main()
