# Predictive Modeling Using Machine Learning

## Project Title
Student Performance Prediction Using Machine Learning

## Objective
Build a supervised machine-learning model to predict whether a student will Pass or Fail using attendance, study hours, previous marks, assignment score, and internal marks.

## Algorithms
1. Decision Tree Classifier
2. Random Forest Classifier

## Tools
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset
`data/student_performance.csv` contains 500 reproducible sample student records.

## How to Run
1. Install Python 3.9+.
2. Open a terminal in this project folder.
3. Run:
   `pip install -r requirements.txt`
4. Run:
   `python predictive_model.py`

The script creates:
- `outputs/model_metrics.csv`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/feature_importance.png`
- `outputs/feature_importance.csv`
- `outputs/example_prediction.csv`

## Workflow
Dataset -> Preprocessing -> Train/Test Split -> Model Training -> Prediction -> Evaluation -> Visualization

## Note
The included dataset is a synthetic educational dataset created for this project. Replace it with an approved real dataset if your institution requires external data.
