# Full Python Code with Model Saving

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from joblib import dump

# Load habituation data
hab_data = pd.read_csv('hab_data_sessions.csv')

# Load phase 1 and phase 2 data
phase_data = pd.read_csv('phase_1_2_2023_2024.csv')

# Step 1: Clean phase data and prepare target features (Y)
phase_data['Trial num'] = pd.to_numeric(phase_data['Trial num'], errors='coerce')  # Handle mixed types
cleaned_phase_data = phase_data.dropna(subset=['Trial num', 'Latency to corr sample', 'Latency to corr match'])

# Engineer metrics for target variables
metrics_cleaned = cleaned_phase_data.groupby('RatID').agg(
    total_false_pos_inc_sample=('False pos inc sample', 'sum'),
    total_false_pos_match1=('False pos inc match 1', 'sum'),
    total_false_pos_match2=('False pos inc match 2', 'sum'),
    avg_latency_to_corr_sample=('Latency to corr sample', 'mean'),
    avg_latency_to_corr_match=('Latency to corr match', 'mean'),
    total_sessions=('session', 'nunique')
)

# Normalize false positives by sessions
metrics_cleaned['false_pos_inc_sample_per_session'] = (
    metrics_cleaned['total_false_pos_inc_sample'] / metrics_cleaned['total_sessions']
)
metrics_cleaned['false_pos_inc_match_per_session'] = (
    (metrics_cleaned['total_false_pos_match1'] + metrics_cleaned['total_false_pos_match2']) / metrics_cleaned['total_sessions']
)

# Define classification thresholds based on 66th percentile for each target
performance_threshold = metrics_cleaned['false_pos_inc_sample_per_session'].quantile(0.66)
match_threshold = metrics_cleaned['false_pos_inc_match_per_session'].quantile(0.66)
latency_sample_threshold = metrics_cleaned['avg_latency_to_corr_sample'].quantile(0.66)
latency_match_threshold = metrics_cleaned['avg_latency_to_corr_match'].quantile(0.66)

# Create binary classification targets
metrics_cleaned['Performance_Class'] = np.where(
    metrics_cleaned['false_pos_inc_sample_per_session'] > performance_threshold, 'Poor', 'Okay/Good'
)
metrics_cleaned['Match_Class'] = np.where(
    metrics_cleaned['false_pos_inc_match_per_session'] > match_threshold, 'Poor', 'Okay/Good'
)
metrics_cleaned['Latency_Sample_Class'] = np.where(
    metrics_cleaned['avg_latency_to_corr_sample'] > latency_sample_threshold, 'Poor', 'Okay/Good'
)
metrics_cleaned['Latency_Match_Class'] = np.where(
    metrics_cleaned['avg_latency_to_corr_match'] > latency_match_threshold, 'Poor', 'Okay/Good'
)

# Step 2: Prepare features (X) from session 1 habituation data
hab_session_1 = hab_data[hab_data['Session'] == 1].drop(columns=['Session'])
merged_data = hab_session_1.merge(
    metrics_cleaned[['Performance_Class', 'Match_Class', 'Latency_Sample_Class', 'Latency_Match_Class']],
    on='RatID'
)

excluded_rats = [9, 16, 19]
merged_data = merged_data[~merged_data['RatID'].isin(excluded_rats)]

# Separate features (X) and labels (Y_targets)
X = merged_data.drop(columns=['RatID', 'Performance_Class', 'Match_Class', 'Latency_Sample_Class', 'Latency_Match_Class'])
Y_targets = {
    'Performance': merged_data['Performance_Class'],
    'Match': merged_data['Match_Class'],
    'Latency_Sample': merged_data['Latency_Sample_Class'],
    'Latency_Match': merged_data['Latency_Match_Class']
}

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize dictionaries to store cumulative confusion matrices and feature importances
results = {target: {'cumulative_cm': np.zeros((2, 2), dtype=int), 'accuracies': [], 'feature_importances': []} for target in Y_targets}

# Directory to save models
model_directory = './models/'
os.makedirs(model_directory, exist_ok=True)

# Perform bootstrapping for each target
n_iterations = 100
for target_name, Y in Y_targets.items():
    print(f"Training for target: {target_name}")

    for i in range(n_iterations):
        # Resample the data with replacement
        X_resampled, Y_resampled = resample(X_scaled, Y, random_state=i)

        # Split resampled data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=i)

        # Train the Random Forest model
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, Y_train)

        # Save the model for the first iteration (or overwrite as necessary)
        if i == 0:  # Save the model trained on the first bootstrap sample
            model_path = f"{model_directory}RandomForest_{target_name}.joblib"
            dump(rf_model, model_path)

        # Predict on the test set
        Y_pred = rf_model.predict(X_test)

        # Calculate accuracy for this iteration
        results[target_name]['accuracies'].append(accuracy_score(Y_test, Y_pred))

        # Add confusion matrix for this iteration
        cm = confusion_matrix(Y_test, Y_pred, labels=rf_model.classes_)
        results[target_name]['cumulative_cm'] += cm

        # Track feature importances
        results[target_name]['feature_importances'].append(rf_model.feature_importances_)

# Inform the user that models have been saved
print(f"\nAll models have been saved in the directory: {model_directory}")

# Summarize results for each target
for target_name, metrics in results.items():
    print(f"\nResults for target: {target_name}")

    # Average accuracy
    avg_accuracy = np.mean(metrics['accuracies'])
    std_accuracy = np.std(metrics['accuracies'])
    print(f"Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

    # Cumulative confusion matrix
    print("Cumulative Confusion Matrix:")
    print(metrics['cumulative_cm'])

    # Feature importance
    feature_importances_mean = np.mean(metrics['feature_importances'], axis=0)
    feature_importances_std = np.std(metrics['feature_importances'], axis=0)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean Importance': feature_importances_mean,
        'Std Deviation': feature_importances_std
    }).sort_values(by='Mean Importance', ascending=False)

    print("\nTop Features:")
    print(importance_df.head())

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Mean Importance'], xerr=importance_df['Std Deviation'], color='blue', alpha=0.7)
    plt.title(f'Feature Importance for {target_name}')
    plt.xlabel('Mean Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

