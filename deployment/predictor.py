from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


class IncomePredictor:
    def __init__(self, artifact_dir: Path):
        """Load the saved settings, preprocessor and model once."""
        artifact_dir = Path(artifact_dir)

        with (artifact_dir / "prediction_metadata.json").open(
            encoding="utf-8"
        ) as file:
            self.metadata = json.load(file)

        self.preprocessor = joblib.load(
            artifact_dir / self.metadata["preprocessor_file"]
        )

        self.model = tf.keras.models.load_model(
            artifact_dir / self.metadata["model_file"],
            compile=False
        )

        self.threshold = float(self.metadata["threshold"])

        # Fail early if we accidentally combine mismatched artifacts.
        feature_names = list(self.preprocessor.get_feature_names_out())

        if feature_names != self.metadata["processed_feature_names"]:
            raise ValueError("Preprocessor features do not match metadata.")

        if len(feature_names) != self.model.input_shape[-1]:
            raise ValueError("Preprocessor features do not match model.")

        if not 0 <= self.threshold <= 1:
            raise ValueError("Prediction threshold must be between 0 and 1.")

    def predict(self, person: dict) -> dict:
        """Prepare one person's details and return an income prediction."""
        required_columns = self.metadata["input_columns"]

        missing_columns = [
            column for column in required_columns
            if column not in person
        ]

        if missing_columns:
            raise ValueError(f"Missing input fields: {missing_columns}")

        row = pd.DataFrame(
            [person],
            columns=required_columns
        )

        # Repeat the text-cleaning rules used during training.
        for column in self.metadata["categorical_columns"]:
            value = row.at[0, column]

            if pd.isna(value):
                value = self.metadata["preparation"]["missing_category"]
            else:
                value = str(value).strip()

                if value == "?":
                    value = self.metadata["preparation"]["missing_category"]

            row.at[0, column] = value

        processed = self.preprocessor.transform(row).astype(np.float32)

        if not np.isfinite(processed).all():
            raise ValueError("Inputs contain invalid numerical values.")

        # Use the loaded model directly for a single-record prediction.
        # training=False disables training behaviour such as dropout.
        output = self.model(processed, training=False)
        score = float(output.numpy()[0, 0])

        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Model returned an invalid score.")

        predicted_class = int(score >= self.threshold)

        return {
            "predicted_class": predicted_class,
            "income_category": self.metadata["class_labels"][
                str(predicted_class)
            ],
            "model_score": score,
            "threshold": self.threshold,
            "model_name": self.metadata["selected_model"],
        }