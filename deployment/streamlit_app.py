from pathlib import Path
import json
import os

import httpx
import streamlit as st


# ---------- Page and connection settings ----------

st.set_page_config(
    page_title="Census Income Explorer",
    page_icon="💬",
    layout="centered",
)

GATEWAY_URL = os.getenv(
    "GATEWAY_URL",
    "http://gateway:8000"
).rstrip("/")

METADATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "assessment_outputs"
    / "prediction_metadata.json"
)

with METADATA_PATH.open(encoding="utf-8") as file:
    metadata = json.load(file)


# ---------- Labels and choices ----------

LABELS = {
    "age": "Age",
    "workclass": "Employment type",
    "education.num": "Education",
    "marital.status": "Marital status",
    "occupation": "Occupation",
    "relationship": "Household relationship",
    "race": "Race",
    "sex": "Sex",
    "capital.gain": "Annual capital gains ($)",
    "capital.loss": "Annual capital losses ($)",
    "hours.per.week": "Working hours per week",
    "native.country": "Country of birth",
}

EDUCATION = {
    1: "Preschool",
    2: "1st–4th grade",
    3: "5th–6th grade",
    4: "7th–8th grade",
    5: "9th grade",
    6: "10th grade",
    7: "11th grade",
    8: "12th grade",
    9: "High school graduate",
    10: "Some college",
    11: "Associate degree — vocational",
    12: "Associate degree — academic",
    13: "Bachelor's degree",
    14: "Master's degree",
    15: "Professional school",
    16: "Doctorate",
}

# Minimum, maximum and starting value for numerical questions.
NUMBER_SETTINGS = {
    "age": (1, 120, 40),
    "capital.gain": (0, None, 0),
    "capital.loss": (0, None, 0),
    "hours.per.week": (1, 168, 40),
}

DEMO_PERSON = {
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

COLUMNS = metadata["input_columns"]

# Extract the exact category values from the saved feature names.
CATEGORY_OPTIONS = {}

for column in metadata["categorical_columns"]:
    prefix = f"categorical__{column}_"

    CATEGORY_OPTIONS[column] = [
        name.removeprefix(prefix)
        for name in metadata["processed_feature_names"]
        if name.startswith(prefix)
    ]

    if not CATEGORY_OPTIONS[column]:
        st.error(f"No saved choices were found for {column}.")
        st.stop()


def display_value(column, value):
    """Show readable labels while preserving original model values."""
    if column == "education.num":
        return EDUCATION[int(value)]

    if column in ("capital.gain", "capital.loss"):
        return f"${value:,.0f}"

    return str(value).replace("-", " ")


# ---------- Session state ----------

if "answers" not in st.session_state:
    st.session_state.answers = {}
    st.session_state.step = 0
    st.session_state.result = None


def reset_profile(use_demo=False):
    st.session_state.answers = DEMO_PERSON.copy() if use_demo else {}
    st.session_state.step = len(COLUMNS) if use_demo else 0
    st.session_state.result = None

    # Clear old input widgets when starting a new profile.
    for key in list(st.session_state):
        if key.startswith("input_"):
            del st.session_state[key]


# ---------- Visual styling ----------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #f8fafc, #eef5ff);
    }
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
    }
    .income-hero {
        background: linear-gradient(120deg, #102a43, #2358a5);
        padding: 28px;
        border-radius: 20px;
        margin-bottom: 24px;
        color: white;
    }
    .income-hero h1 {
        color: white;
        font-size: 2rem;
        margin: 0 0 10px 0;
    }
    .income-hero p {
        color: #e0ecff;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="income-hero">
        <h1>💬 Census Income Explorer</h1>
        <p>A guided conversation powered by a trained neural network.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Sidebar ----------

with st.sidebar:
    st.title("Income Explorer")
    st.caption("Adult Census · Deep Learning Demo")

    if st.button("✨ Load sample profile"):
        reset_profile(use_demo=True)
        st.rerun()

    if st.button("↻ Start again"):
        reset_profile()
        st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.write("1. Answer the questions.")
    st.write("2. Review your profile.")
    st.write("3. Request a prediction.")

    st.divider()
    st.caption(
        "Educational demo based on historical US census data. "
        "The result is a prediction, not a verified income assessment."
    )

    if st.button("Check service connection"):
        try:
            with httpx.Client(timeout=10, trust_env=False) as client:
                response = client.get(f"{GATEWAY_URL}/health")
                response.raise_for_status()
            st.success("Gateway and model are reachable.")
        except httpx.HTTPError:
            st.error("Cannot reach a ready prediction service.")


# ---------- Guided conversation ----------

step = st.session_state.step
answers = st.session_state.answers

st.progress(
    step / len(COLUMNS),
    text=f"{step} of {len(COLUMNS)} questions completed",
)

if step > 0:
    if st.button("← Back to previous question"):
        st.session_state.step -= 1
        st.session_state.result = None
        st.rerun()

# Show the three most recent completed exchanges.
for column in COLUMNS[max(0, step - 3):step]:
    with st.chat_message("assistant"):
        st.write(f"What is the person's **{LABELS[column].lower()}**?")

    with st.chat_message("user"):
        st.write(display_value(column, answers[column]))

if step < len(COLUMNS):
    column = COLUMNS[step]

    with st.chat_message("assistant"):
        st.write(
            f"Question {step + 1}: "
            f"What is the person's **{LABELS[column].lower()}**?"
        )

        if column in ("capital.gain", "capital.loss"):
            st.caption(
                "Use annual capital gains or losses, not salary. "
                "Enter 0 if there are none."
            )

    with st.form(f"question_{step}"):
        if column in NUMBER_SETTINGS:
            minimum, maximum, default = NUMBER_SETTINGS[column]

            value = st.number_input(
                LABELS[column],
                min_value=minimum,
                max_value=maximum,
                value=int(answers.get(column, default)),
                step=1,
                key=f"input_{step}",
            )

        elif column == "education.num":
            choices = list(EDUCATION)
            current = answers.get(column, 9)

            value = st.selectbox(
                LABELS[column],
                options=choices,
                index=choices.index(current),
                format_func=lambda item: EDUCATION[item],
                key=f"input_{step}",
            )

        else:
            choices = CATEGORY_OPTIONS[column]
            current = answers.get(column, choices[0])

            value = st.selectbox(
                LABELS[column],
                options=choices,
                index=choices.index(current),
                format_func=lambda item: item.replace("-", " "),
                key=f"input_{step}",
            )

        submitted = st.form_submit_button(
            "Continue →",
            type="primary",
        )

    if submitted:
        answers[column] = value
        st.session_state.step += 1
        st.session_state.result = None
        st.rerun()


# ---------- Review and prediction ----------

else:
    with st.chat_message("assistant"):
        st.write(
            "Your profile is complete. Review the answers below, "
            "then ask the model for a prediction."
        )

    with st.expander("Review all answers", expanded=True):
        st.table([
            {
                "Detail": LABELS[column],
                "Answer": display_value(column, answers[column]),
            }
            for column in COLUMNS
        ])

    field_to_edit = st.selectbox(
        "Want to change an answer?",
        options=COLUMNS,
        format_func=lambda item: LABELS[item],
    )

    if st.button("Edit selected answer"):
        st.session_state.step = COLUMNS.index(field_to_edit)
        st.session_state.result = None
        st.rerun()

    if st.button("Predict income category", type="primary"):
        with st.spinner("Sending your profile to the prediction service…"):
            try:
                with httpx.Client(
                    timeout=30,
                    trust_env=False,
                ) as client:
                    response = client.post(
                        f"{GATEWAY_URL}/predict",
                        json=answers,
                    )

                response.raise_for_status()
                result = response.json()

                required = {
                    "income_category",
                    "model_score",
                    "threshold",
                    "model_name",
                }

                if not isinstance(result, dict) or not required <= result.keys():
                    raise ValueError("Unexpected prediction response.")

                st.session_state.result = result

            except httpx.HTTPStatusError as error:
                st.error(
                    f"The service returned HTTP "
                    f"{error.response.status_code}."
                )
                with st.expander("Response details"):
                    st.text(error.response.text)

            except httpx.RequestError:
                st.error(
                    "Could not contact the gateway. "
                    "Check that the services are running."
                )

            except ValueError:
                st.error("The service returned an unexpected response.")

    result = st.session_state.result

    if result is not None:
        st.divider()

        with st.chat_message("assistant"):
            category = (
                "Above $50,000 per year"
                if result["income_category"] == ">50K"
                else "$50,000 or below per year"
            )

            st.success(f"Predicted category: {category}")

            score_column, threshold_column = st.columns(2)

            score_column.metric(
                "Model score",
                f'{result["model_score"]:.4f}',
            )

            threshold_column.metric(
                "Decision threshold",
                f'{result["threshold"]:.2f}',
            )

            st.caption(
                "Scores at or above the threshold predict >$50K. "
                "This score is not model accuracy or a calibrated probability."
            )

            with st.expander("About this prediction"):
                st.write("Model:", result["model_name"])
                st.write(
                    "The same saved preprocessor and trained weights "
                    "from the assessment produced this result."
                )