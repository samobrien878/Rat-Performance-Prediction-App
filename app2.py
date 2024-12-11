import streamlit as st
import numpy as np
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt
#cd "C:\Users\obrie\Python Senior Thesis"


# Set Streamlit page configuration
st.set_page_config(page_title="Williams Behavioral Neuroscience Lab: Rat Assessment App", page_icon="🐭", layout="centered")

# Define the path to the dataset
file_path = 'hab_data_sessions.csv'

# Load the habituation data
habituation_data = pd.read_csv(file_path)

# Calculate max values for each feature
feature_ranges = habituation_data.describe().loc[['max']].transpose()
feature_ranges.reset_index(inplace=True)
feature_ranges.columns = ['Feature', 'Max']

# Function to round up to the nearest 10
def round_up_to_nearest_10(value):
    return math.ceil(value / 10) * 10

# Map feature ranges to rounded max values and set min to 0
feature_range_dict = feature_ranges.set_index('Feature').to_dict(orient='index')
for feature, ranges in feature_range_dict.items():
    ranges['Min'] = 0  # Set min to 0
    ranges['Max'] = round_up_to_nearest_10(ranges['Max'])  # Round max to nearest 10

# Define the 20 habituation data features used for training
features = [
    "S1 poke event", "S2 poke event", "M1 poke event", "M2 poke event",
    "M3 poke event", "Sp1 corner poke event", "Sp2 corner poke event", "Door event",
    "Match Box event", "Inactive event", "S1 poke duration", "S2 poke duration",
    "M1 poke duration", "M2 poke duration", "M3 poke duration", "Sp1 corner poke duration",
    "Sp2 corner poke duration", "Door duration", "Match Box duration", "Inactive duration"
]

st.markdown(
    """
    <style>
    /* Global Styles */
    body, .main, .stApp {
        background-color: #000000; /* Black background */
        color: #F84C1E; /* UVA Orange text */
    }
    h1, h2, h3, h4, h5, h6, .stText {
        color: #F84C1E; /* UVA Orange headers */
    }
    .stButton>button {
        color: #FFFFFF; /* White text */
        background-color: #232D4B; /* UVA Navy buttons */
        border: none;
        padding: 12px 30px;
        font-size: 18px;
        cursor: pointer;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #F84C1E; /* UVA Orange hover effect */
        transform: scale(1.1);
    }
    .stSlider label {
        color: #F84C1E; /* UVA Orange labels for sliders */
    }
    .stSlider .st-uy {
        background: #232D4B !important; /* Dark slider background */
    }
    .block-container {
        padding-top: 0px; /* Remove padding to make top part black */
    }

    /* Title-Specific Styles */
    .title-container {
        margin-top: 100px; /* Adjust the value to move the title further down */
        text-align: center; /* Center-align the title */
    }
    h1 {
        color: #F84C1E; /* UVA Orange for the title */
    }

    /* Prediction Outcome Styles */
    .outcome-proficient {
        color: #F84C1E; /* UVA Orange for Proficient */
        font-weight: bold;
        font-size: 24px;
    }
    .outcome-generally-proficient {
        color: #FFD700; /* Gold for Generally Proficient */
        font-weight: bold;
        font-size: 24px;
    }
    .outcome-intervention-needed {
        color: #DC143C; /* Crimson Red for Interventions Needed */
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    <div class="title-container">
        <h1>Williams Behavioral Neuroscience Lab: Rat Performance Prediction App</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# Load the Random Forest models
@st.cache_resource
def load_models():
    # Update the model file paths
    model_files = {
        "Performance": r"C:\Users\obrie\Python Senior Thesis\RandomForest_Performance.joblib",
        "Match": r"C:\Users\obrie\Python Senior Thesis\RandomForest_Match.joblib",
        "Latency_Sample": r"C:\Users\obrie\Python Senior Thesis\RandomForest_Latency_Sample.joblib",
        "Latency_Match": r"C:\Users\obrie\Python Senior Thesis\RandomForest_Latency_Match.joblib",
    }
    models = {}
    for model_name, file_path in model_files.items():
        try:
            models[model_name] = joblib.load(file_path)
        except FileNotFoundError:
            st.error(f"Model file for {model_name} not found. Please ensure the model file is located at '{file_path}'.")
            models[model_name] = None
    return models


# Streamlit UI setup
st.header("Input Day 1 Habituation Data")

# Create sliders dynamically for all features
input_data = {}
for feature in features:
    if feature in feature_range_dict:
        min_val = 0  # Set all mins to 0
        max_val = feature_range_dict[feature]['Max']
    else:
        min_val, max_val = 0, 100  # Default values if feature is not in dataset

    # Default slider value set to midpoint of min and max
    default_val = (min_val + max_val) / 2
    input_data[feature] = st.slider(
        f"{feature}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val),
        step=0.1
    )

if st.button("Predict Performance"):
    st.header("Prediction Results")

    # Prepare input data for the model
    input_values = np.array([input_data[feature] for feature in features]).reshape(1, -1)

    # Load the models
    models = load_models()
    if all(models.values()):  # Ensure all models are loaded
        predictions = {}
        details = {}  # To store additional details for false positives

        for model_name, model in models.items():
            try:
                # Predict the classification for each model
                predictions[model_name] = model.predict(input_values)[0]

                # Add specific details for Performance and Match models
                if model_name == "Performance":
                    details["Performance"] = model.predict_proba(input_values)[0][1] * 100  # Probability of false positives
                elif model_name == "Match":
                    details["Match"] = model.predict_proba(input_values)[0][1] * 100  # Probability of false matches
            except ValueError as e:
                st.error(f"Error during prediction with {model_name}: {e}")

        # Count occurrences of each prediction category
        prediction_counts = {
            "Okay/Good": sum(pred == "Okay/Good" for pred in predictions.values()),
            "Poor": sum(pred == "Poor" for pred in predictions.values())
        }

        # Calculate overall performance
        poor_count = prediction_counts["Poor"]

        # Determine dynamic graph title and color
        if poor_count >= 2:
            graph_title = "The Predicted performance of this rat is NOT PROFICIENT"
            graph_color = "red"
        elif poor_count == 1:
            graph_title = "The Predicted performance of this rat is GENERALLY PROFICIENT"
            graph_color = "orange"
        else:
            graph_title = "The Predicted performance of this rat is PROFICIENT"
            graph_color = "green"

        # Create pie chart
        labels = prediction_counts.keys()
        sizes = prediction_counts.values()
        colors = ["green", "red"]
        explode = (0.1, 0)  # Highlight the first slice (Proficient)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')  # Set the figure background to black
        ax.set_facecolor('black')  # Set the axes background to black
        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 12, "color": "white"}  # White text for labels
        )
        ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular
        st.pyplot(fig)

        # Summarize results
        st.markdown(
            f"<div style='text-align: center; font-size: 20px; color: {graph_color}; font-weight: bold;'>"
            f"{graph_title}</div>",
            unsafe_allow_html=True
        )