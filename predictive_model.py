"""
Predictive Modeling Using Machine Learning
Ready-to-run project: Student Performance Prediction
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc
)

DATA_PATH = "data/student_performance.csv"

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# 2. Prepare data
X = df.drop(columns=["Student_ID", "Result"])
y = LabelEncoder().fit_transform(df["Result"])  # Fail=0, Pass=1

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

# 5. Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=150, max_depth=7, random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

def evaluate(name, y_true, y_pred):
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0)
    }
    print("\n" + name)
    print(classification_report(y_true, y_pred, target_names=["Fail", "Pass"]))
    return metrics

results = pd.DataFrame([
    evaluate("Decision Tree", y_test, dt_pred),
    evaluate("Random Forest", y_test, rf_pred)
])
results.to_csv("outputs/model_metrics.csv", index=False)

# 6. Confusion matrix for Random Forest
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=200)
plt.close()

# 7. ROC curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
auc_dt = auc(fpr_dt, tpr_dt)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={auc_dt:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=200)
plt.close()

# 8. Feature importance
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importance.to_csv("outputs/feature_importance.csv")

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=200)
plt.close()

# 9. Example prediction
sample = pd.DataFrame([{
    "Attendance": 85,
    "Study_Hours": 5.0,
    "Previous_Marks": 72,
    "Assignment_Score": 80,
    "Internal_Marks": 75
}])
prediction = rf_model.predict(sample)[0]
probability = rf_model.predict_proba(sample)[0][prediction]
label = "Pass" if prediction == 1 else "Fail"

print("\nExample Student Prediction:")
print("Prediction:", label)
print(f"Model confidence: {probability:.2%}")

# Save a prediction file
sample["Predicted_Result"] = label
sample["Confidence"] = probability
sample.to_csv("outputs/example_prediction.csv", index=False)

print("\nProject completed. Check the outputs/ folder.")
