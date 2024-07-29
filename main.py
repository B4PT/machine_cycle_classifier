from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from jinja2 import Template
import model
import processing
from utils import binarize_targets
from config import TRAIN_SIZE, TARGET_COLUMN

# Initialize FastAPI app
app = FastAPI()

# Import data
profile_path = "data_subset/profile.txt"
fs1_path = "data_subset/FS1.txt"
ps2_path = "data_subset/PS2.txt"

data, fs1, ps2 = processing.process_data(profile_path, fs1_path, ps2_path)
data.columns = data.columns.astype(str)

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
models = model.initialize_models()

# Train
model_results = model.train_and_evaluate_models(
    models, X_train, y_train, X_test, y_test
)

# Determine the best model based on F1 score
best_model_name = max(model_results, key=lambda x: model_results[x]["f1_score"])
best_model = next(model for model in models if model.name == best_model_name)

print(
    f"Best model: {best_model.name} with F1 score: {model_results[best_model_name]['f1_score']}"
)
print(model_results)

# HTML Template for using the best model only
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Cycle Prediction</title>
</head>
<body>
    <h1>Cycle Prediction</h1>
    <form action="/predict" method="post">
        <label for="cycle_number">Cycle Number:</label>
        <input type="number" id="cycle_number" name="cycle_number" required>
        <button type="submit">Predict</button>
    </form>
    <h2>Prediction Result</h2>
    <p>Cycle Number: {{ cycle_number }}</p>
    <p>Prediction: {{ prediction }}</p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def read_root():
    template = Template(HTML_TEMPLATE)
    return HTMLResponse(content=template.render())


@app.post("/predict", response_class=HTMLResponse)
async def predict(cycle_number: int = Form(...)):
    cycle_data = data.iloc[[cycle_number]].drop(columns=[TARGET_COLUMN])

    if cycle_data.empty:
        template = Template(HTML_TEMPLATE)
        return HTMLResponse(
            content=template.render(
                cycle_number=cycle_number,
                prediction="No data found for the given cycle number",
            )
        )

    prediction = best_model.predict_cycle(cycle_data)

    if prediction is None:
        template = Template(HTML_TEMPLATE)
        return HTMLResponse(
            content=template.render(
                cycle_number=cycle_number,
                prediction="No prediction found",
            )
        )

    template = Template(HTML_TEMPLATE)
    return HTMLResponse(
        content=template.render(cycle_number=cycle_number, prediction=prediction[0])
    )
