# Adaboost code was used from https://github.com/srijarkoroy/adaboost

import matplotlib.pyplot as plt
from datagenerate import dataset
from boosting import AdaBoost, MultiLabelAdaBoost  # Ensure correct import path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import pickle

def print_model_predictions(X_test, X_test_filtered, y_pred_labels, y_test_labels, mlb):
    """
    Prints model predictions with consistent formatting.
    
    Parameters:
    - X_test: Original test features before filtering
    - X_test_filtered: Filtered test features
    - y_pred_labels: Predicted labels after inverse transforming
    - y_test_labels: Actual labels after inverse transforming
    - mlb: MultiLabelBinarizer instance
    """
    # Define sensor names for clarity
    sensor_names = ['left3', 'left2', 'left1', 'M', 'right1', 'right2', 'right3']
    
    # Calculate the maximum width for each column
    max_idx_width = max(len(str(len(X_test_filtered))), 5)
    max_features_width = 35  # Adjust as needed
    max_pred_label_width = 50
    max_true_label_width = 50
    
    with open("filtered_output.txt", 'w') as file:
        # Print header
        header = f"{'Input':<{max_idx_width}} | {'IR Sensor Values':<{max_features_width}} | {'Predicted':<{max_pred_label_width}} | {'Actual':<{max_true_label_width}}\n"
        file.write(header)
        file.write('-' * len(header))
        file.write("\n")
        # Loop and file.write formatted output
        for idx, (features, pred_labels, true_labels) in enumerate(zip(X_test_filtered, y_pred_labels, y_test_labels)):
            # Convert features array to a list of integers
            features_list = features.astype(int).tolist()
            # Join features into a string with commas
            features_str = ','.join(map(str, features_list))
            # Truncate or pad the features string to fit the column width
            features_str = features_str[:max_features_width].ljust(max_features_width)
            # Convert lists to strings
            pred_labels_str = ', '.join(pred_labels)
            true_labels_str = ', '.join(true_labels)
            # Truncate or pad label strings
            pred_labels_str = pred_labels_str[:max_pred_label_width].ljust(max_pred_label_width)
            true_labels_str = true_labels_str[:max_true_label_width].ljust(max_true_label_width)
            # Format the strings with padding
            file.write(f"{idx:<{max_idx_width}} | {features_str:<{max_features_width}} | {pred_labels_str:<{max_pred_label_width}} | {true_labels_str:<{max_true_label_width}}\n")

# Load the data
X, y, mlb = dataset()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a mask for test samples where at least one IR sensor value >= 10
# mask = np.any(X_test >= 10, axis=1)

# Apply the mask to X_test and y_test
# X_test_filtered = X_test[mask]
# y_test_filtered = y_test[mask]
X_test_filtered = X_test
y_test_filtered = y_test

# Check if the filtered test set is not empty
if X_test_filtered.shape[0] == 0:
    print("No test samples have IR sensor values >= 10.")
else:
    # Initialize MultiLabelAdaBoost
    clf = MultiLabelAdaBoost(n_estimators=200)
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict labels for the filtered test set
    y_pred = clf.predict(X_test_filtered)
    
    # Convert binary indicator matrix back to original zone labels
    y_pred_labels = mlb.inverse_transform(y_pred)
    y_test_labels = mlb.inverse_transform(y_test_filtered)
    
    # Print out the predictions with corresponding inputs and actual labels
    print_model_predictions(X_test, X_test_filtered, y_pred_labels, y_test_labels, mlb)
    
    # Calculate and print accuracy
    # For multi-label, accuracy can be defined in various ways. Here, we'll use subset accuracy.
    accuracy = accuracy_score(y_test_filtered, y_pred)
    print(f"\nTest accuracy (subset accuracy): {accuracy:.2%}")
    
    # Print classification report
    print("\nClassification Report:")
    # print(classification_report(y_test_filtered, y_pred_labels))
    
    # Plot Confusion Matrix
    # Note: Confusion matrix for multi-label can be complex. Here, we show it per class.
    for idx, class_label in enumerate(mlb.classes_):
        cm = confusion_matrix(y_test_filtered[:, idx], y_pred[:, idx], labels=[0,1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {class_label}')
        plt.show()
    
    with open("multilabel_adaboost_model.pkl", "wb") as f:
        pickle.dump(clf, f)
        print("Model saved")
    with open("mlb_object.pkl", "wb") as f:
        pickle.dump(mlb, f)
        print(mlb)
        print("mlb saved")