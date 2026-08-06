
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("/content/Student_performance_data _.csv")

X = df[["StudyTimeWeekly", "Absences","Tutoring", "ParentalSupport"]]
y = df["GradeClass"].astype(int)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("5-Fold Cross Validation Results:")
print("="*55)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\n{name}:")
    print(f"  Fold scores: {scores.round(3)}")
    print(f"  Mean accuracy: {scores.mean()*100:.2f}%")
    print(f"  Std deviation: {scores.std()*100:.2f}%")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score,f1_score, classification_report,confusion_matrix)

df = pd.read_csv("/content/Student_performance_data _.csv")

# Binary: Pass (GradeClass 0,1,2) vs Fail (3,4)
df["PassFail"] = (df["GradeClass"] <= 2).astype(int)

X = df[["StudyTimeWeekly", "Absences","Tutoring", "ParentalSupport"]]
y = df["PassFail"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Binary Classification — Pass/Fail:")
print(f"Accuracy : {(y_pred==y_test).mean()*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall   : {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1 Score : {f1_score(y_test, y_pred)*100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                Predicted Fail  Predicted Pass")
print(f"Actual Fail  →  TN:{cm[0][0]:<12} FP:{cm[0][1]}")
print(f"Actual Pass  →  FN:{cm[1][0]:<12} TP:{cm[1][1]}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Fail", "Pass"]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_csv("/content/Student_performance_data _.csv")

df["PassFail"] = (df["GradeClass"] <= 2).astype(int)
X = df[["StudyTimeWeekly", "Absences","Tutoring", "ParentalSupport"]]
y = df["PassFail"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc:.4f}")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

print(f"\nROC Curve Sample Points:")
print(f"{'Threshold':<12}{'FPR':<10}{'TPR':<10}")
print("-"*32)
step = len(thresholds) // 5
for i in range(0, len(thresholds), step):
    print(f"{thresholds[i]:<12.3f}{fpr[i]:<10.3f}{tpr[i]:<10.3f}")

print(f"\nAUC Interpretation:")
if auc >= 0.9:
    print(f"  {auc:.3f} → Excellent model!")
elif auc >= 0.8:
    print(f"  {auc:.3f} → Good model")
elif auc >= 0.7:
    print(f"  {auc:.3f} → Fair model")
else:
    print(f"  {auc:.3f} → Poor model")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

df = pd.read_csv("/content/Student_performance_data _.csv")
df["PassFail"] = (df["GradeClass"] <= 2).astype(int)

X = df[["StudyTimeWeekly", "Absences","Tutoring", "ParentalSupport"]]
y = df["PassFail"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100,
                                            random_state=42)
}

print("="*65)
print("COMPLETE MODEL EVALUATION REPORT")
print("="*65)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"\n{name}")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Precision : {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"  Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"  F1 Score  : {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  CV Score  : {cv_scores.mean()*100:.2f}% "
          f"(±{cv_scores.std()*100:.2f}%)")

print("\n" + "="*65)
print("RECOMMENDATION")
print("="*65)
print("Best model for student pass/fail prediction:")
print("→ [Your analysis here based on output]")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("/content/Student_performance_data _.csv")
df["PassFail"] = (df["GradeClass"] <= 2).astype(int)

X = df[["StudyTimeWeekly", "Absences","Tutoring", "ParentalSupport"]]
y = df["PassFail"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

best_threshold = 0
best_f1 = 0

print("Threshold Tuning Results:")
print(f"{'Threshold':<10} {'Precision':<12} {'Recall':<10} {'F1 Score'}")
print("-" * 48)

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        mark = " ← Best F1!"
    else:
        mark = ""

    print(f"{threshold:<10}"
          f"{precision*100:<12.2f}%"
          f"{recall*100:<10.2f}%"
          f"{f1*100:.2f}%{mark}")

print(f"\nBest Threshold: {best_threshold:.1f} (F1: {best_f1*100:.2f}%)")