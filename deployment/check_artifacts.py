from pathlib import Path
from importlib.metadata import version
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# Find the project using this script's location.
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "assessment_outputs"

# Read the saved prediction settings.
metadata_path = OUTPUT_DIR / "prediction_metadata.json"

with metadata_path.open(encoding="utf-8") as file:
    metadata = json.load(file)

# Load the fitted preprocessor and trained neural network.
preprocessor = joblib.load(
    OUTPUT_DIR / metadata["preprocessor_file"]
)

model = tf.keras.models.load_model(
    OUTPUT_DIR / metadata["model_file"],
    compile=False
)

print("Package versions:")
for package in ["tensorflow", "scikit-learn", "numpy", "pandas", "joblib"]:
    print(f"  {package}: {version(package)}")

# An invented example for checking the prediction flow.
# We do not know this person's actual income.
person = {
    "age": 40,
    "workclass": "Private",
    "education.num": 13,
    "marital.status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital.gain": 0,
    "capital.loss": 0,
    "hours.per.week": 40,
    "native.country": "United-States",
}

# Preserve the input column order recorded during training.
row = pd.DataFrame([person], columns=metadata["input_columns"])

# Apply the existing scaler and encoder. Do not fit them again.
processed = preprocessor.transform(row).astype(np.float32)

# Check that the preprocessor matches the saved feature information.
actual_names = list(preprocessor.get_feature_names_out())
expected_names = metadata["processed_feature_names"]

if actual_names != expected_names:
    raise ValueError("Preprocessor feature names do not match the metadata.")

if processed.shape[1] != model.input_shape[-1]:
    raise ValueError("Processed feature count does not match the model.")

if not np.isfinite(processed).all():
    raise ValueError("Processed inputs contain invalid numerical values.")

# Run inference using the saved weights.
score = float(model.predict(processed, verbose=0)[0, 0])

if not np.isfinite(score) or not 0 <= score <= 1:
    raise ValueError("The model returned an invalid score.")

threshold = float(metadata["threshold"])
predicted_class = int(score >= threshold)
label = metadata["class_labels"][str(predicted_class)]

print("\nPrediction check:")
print("Selected model:", metadata["selected_model"])
print("Prepared input shape:", processed.shape)
print("Model input shape:", model.input_shape)
print("Model score:", round(score, 4))
print("Threshold:", threshold)
print("Predicted income category:", label)
print("\nArtifact check passed.")