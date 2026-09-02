import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

print("MODEL SAVING AND LOADING")
df = pd.read_csv("/content/Student_performance_data _.csv")

X = df[["StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport"]]
y = df["GradeClass"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)

test_accuracy = pipeline.score(X_test, y_test)
print(f"Model trained. Test accuracy: {test_accuracy*100:.2f}%")
joblib.dump(pipeline, "grade_predictor_model.pkl")
print("\nModel saved as: grade_predictor_model.pkl")
loaded_pipeline = joblib.load("grade_predictor_model.pkl")
print("Model loaded successfully!")
loaded_accuracy = loaded_pipeline.score(X_test, y_test)
print(f"Loaded model accuracy: {loaded_accuracy*100:.2f}%")
print(f"Same as original: {test_accuracy == loaded_accuracy}")
import os
file_size = os.path.getsize("grade_predictor_model.pkl")
print(f"\nModel file size: {file_size/1024:.2f} KB")

print("SIMPLE PREDICTION INTERFACE")

grade_map = {0:"A", 1:"B", 2:"C", 3:"D", 4:"F"}

def predict_student_grade(study_time, absences, tutoring, parental_support):
    """
    Production-style prediction function
    Takes raw inputs, returns clean prediction
    """
    if study_time < 0 or study_time > 40:
        return {"error": "Invalid study time (0-40 hours expected)"}
    if absences < 0 or absences > 30:
        return {"error": "Invalid absences (0-30 expected)"}
    if tutoring not in [0, 1]:
        return {"error": "Tutoring must be 0 or 1"}
    if parental_support not in [0,1,2,3,4]:
        return {"error": "Parental support must be 0-4"}
    input_data = pd.DataFrame({
        "StudyTimeWeekly": [study_time],
        "Absences": [absences],
        "Tutoring": [tutoring],
        "ParentalSupport": [parental_support]
    })
    prediction = loaded_pipeline.predict(input_data)[0]
    probability = loaded_pipeline.predict_proba(input_data)[0]
    confidence = probability[prediction] * 100

    return {
        "predicted_grade": grade_map[prediction],
        "confidence": round(confidence, 2),
        "input_received": {
            "study_time": study_time,
            "absences": absences,
            "tutoring": "Yes" if tutoring==1 else "No",
            "parental_support": parental_support
        }
    }
print("Test 1 - Valid input:")
result1 = predict_student_grade(15, 5, 1, 3)
print(result1)
print("\nTest 2 - Invalid input (negative absences):")
result2 = predict_student_grade(15, -5, 1, 3)
print(result2)
print("\nTest 3 - Invalid input (wrong tutoring value):")
result3 = predict_student_grade(15, 5, 2, 3)
print(result3)

print("MODEL MONITORING - DATA DRIFT CONCEPT")

print("""
What is Data Drift?

Training data (2023): Students study 10 hrs/week average
Production data (2024): Students study 5 hrs/week average
(Maybe due to new school policy, COVID effect, etc.)

Model trained on OLD patterns
New data looks DIFFERENT
Model accuracy silently drops!

This is why models need MONITORING, not just deployment
""")
print("Original Training Data Statistics:")
print(X_train.describe())
np.random.seed(100)
new_data_simulated = pd.DataFrame({
    "StudyTimeWeekly": np.random.uniform(2, 8, 100),
    "Absences": np.random.randint(15, 30, 100),
    "Tutoring": np.random.randint(0, 2, 100),
    "ParentalSupport": np.random.randint(0, 5, 100)
})

print("\nSimulated New Production Data Statistics:")
print(new_data_simulated.describe())
print("\nDRIFT DETECTION:")
for col in ["StudyTimeWeekly", "Absences"]:
    train_mean = X_train[col].mean()
    new_mean = new_data_simulated[col].mean()
    diff_pct = abs(new_mean - train_mean) / train_mean * 100

    print(f"\n{col}:")
    print(f"  Training mean: {train_mean:.2f}")
    print(f"  New data mean: {new_mean:.2f}")
    print(f"  Difference: {diff_pct:.1f}%")

    if diff_pct > 20:
        print(f"  Status: SIGNIFICANT DRIFT DETECTED - retrain recommended")
    else:
        print(f"  Status: Normal variation")

print("END-TO-END ML SYSTEM SUMMARY")

print("""
COMPLETE ML SYSTEM LIFECYCLE:

1. DATA COLLECTION
   - Gather student records (2,392 samples)

2. DATA PREPROCESSING
   - Clean missing values, handle outliers
   - Feature engineering (StudyEfficiency, RiskScore, etc.)

3. MODEL TRAINING
   - Train/test split (80/20)
   - Compare multiple models (LR, DT, RF)
   - Cross-validation for reliable evaluation

4. MODEL EVALUATION
   - Accuracy, F1-score, Confusion Matrix
   - Check for overfitting (train vs test gap)

5. MODEL SAVING
   - joblib.dump() - persist trained model

6. DEPLOYMENT (simplified)
   - Prediction function with input validation
   - (Real world: Flask/FastAPI + cloud hosting)

7. MONITORING
   - Track data drift
   - Monitor prediction accuracy over time
   - Set up alerts if performance drops

8. RETRAINING
   - Periodic retraining with new data
   - A/B testing new model vs old model
""")
print("FINAL SYSTEM TEST - Multiple Students:")

test_students = [
    {"study_time": 18, "absences": 2, "tutoring": 1, "parental_support": 4},
    {"study_time": 5, "absences": 25, "tutoring": 0, "parental_support": 0},
    {"study_time": 12, "absences": 10, "tutoring": 1, "parental_support": 2},
]

for i, student in enumerate(test_students, 1):
    result = predict_student_grade(**student)
    print(f"\nStudent {i}:")
    print(f"  Input: {student}")
    print(f"  Prediction: Grade {result['predicted_grade']} "
          f"(Confidence: {result['confidence']}%)")

print("SYSTEM DESIGN ANSWERS")
print("""
1. How often would you retrain the model? Why?
I would start with monthly retraining, while monitoring the model every week for data drift and performance changes. If significant data drift or a drop in accuracy/F1-score is detected, I would retrain earlier instead of waiting for the monthly schedule.

2. What would trigger an alert to teachers?
I would trigger an alert when the model predicts a high-risk grade such as D or F, especially when the predicted probability is above a threshold such as 70%. I would also prioritize students whose risk remains high across multiple weekly predictions rather than alerting teachers based on a single prediction.

3. What could go wrong with this system?
The model could have bias or fairness issues if the training data is not representative, causing certain groups of students to be incorrectly identified as at-risk more often. There are also privacy concerns because student academic information is sensitive, so access should be restricted and the data should be securely stored and handled.

4. How would you explain a WRONG prediction to a teacher?
I would explain that the model makes a prediction based on patterns learned from historical data and that 75% accuracy means it will still make mistakes. I would show the teacher the student's relevant input factors and predicted probabilities, and clarify that the prediction is meant to support the teacher's decision—not replace their judgment.
""")