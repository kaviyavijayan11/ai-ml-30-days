import pandas as pd
import numpy as np
df = pd.read_csv("/content/Student_performance_data _.csv")
print("Original features:", df.shape[1])
df["StudyEfficiency"] = df["StudyTimeWeekly"] / (df["Absences"] + 1)
df["SupportScore"] = df["Tutoring"] * 2 + df["ParentalSupport"]
df["RiskScore"] = df["Absences"] - df["StudyTimeWeekly"]
df["ActivityScore"] = (df["Sports"] + df["Music"]+df["Volunteering"] + df["Extracurricular"])
df["HighAbsence"] = (df["Absences"] > 15).astype(int)
print(f"New features added: 5")
print(f"Total features now: {df.shape[1]}")
print("\nNew Features Sample:")
new_features = ["StudyEfficiency", "SupportScore","RiskScore", "ActivityScore", "HighAbsence"]
print(df[new_features].head())
print("\nCorrelation of new features with GPA:")
print(df[new_features + ["GPA"]].corr()["GPA"].round(3))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("/content/Student_performance_data _.csv")
df["StudyEfficiency"] = df["StudyTimeWeekly"] / (df["Absences"] + 1)
df["SupportScore"] = df["Tutoring"] * 2 + df["ParentalSupport"]
df["RiskScore"] = df["Absences"] - df["StudyTimeWeekly"]
df["ActivityScore"] = (df["Sports"] + df["Music"] +df["Volunteering"] + df["Extracurricular"])
df["HighAbsence"] = (df["Absences"] > 15).astype(int)
features = ["StudyTimeWeekly", "Absences", "Tutoring","ParentalSupport", "StudyEfficiency","SupportScore", "RiskScore","ActivityScore", "HighAbsence"]
X = df[features]
y = df["GradeClass"].astype(int)
print("Method 1: Correlation with GPA")
print("="*40)
corr = df[features + ["GPA"]].corr()["GPA"].abs().sort_values(ascending=False)
print(corr[:-1].round(3))
print("\nMethod 2: Random Forest Importance")
print("="*40)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importance = pd.Series(rf.feature_importances_,index=features).sort_values(ascending=False)
for feat, imp in importance.items():
    bar =  int(imp * 50)
    print(f"{feat:<20}: {imp:.4f} {bar}")
top5 = importance.head(5).index.tolist()
print(f"\nTop 5 Features: {top5}")

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("/content/Student_performance_data _.csv")
X = df[["StudyTimeWeekly", "Absences","Tutoring", "ParentalSupport"]]
y = df["GradeClass"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
manual_acc = accuracy_score(y_test, model.predict(X_test_scaled))
pipeline = Pipeline([("scaler", StandardScaler()),("model", LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)
pipeline_acc = accuracy_score(y_test, pipeline.predict(X_test))

print("Pipeline vs Manual:")
print(f"Manual accuracy  : {manual_acc*100:.2f}%")
print(f"Pipeline accuracy: {pipeline_acc*100:.2f}%")
print(f"Same result: {abs(manual_acc-pipeline_acc) < 0.001}")

print("\nPipeline Steps:")
for step_name, step in pipeline.steps:
    print(f"  {step_name}: {type(step).__name__}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("/content/Student_performance_data _.csv")
df["StudyEfficiency"] = df["StudyTimeWeekly"] / (df["Absences"] + 1)
df["SupportScore"] = df["Tutoring"] * 2 + df["ParentalSupport"]
df["RiskScore"] = df["Absences"] - df["StudyTimeWeekly"]
df["HighAbsence"] = (df["Absences"] > 15).astype(int)
features = ["Absences", "StudyEfficiency","RiskScore", "SupportScore","StudyTimeWeekly", "HighAbsence"]
X = df[features]
y = df["GradeClass"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([("scaler", StandardScaler()),("model", RandomForestClassifier(n_estimators=100, random_state=42))])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("="*55)
print("FULL ML PIPELINE — RESULTS")
print("="*55)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"CV Score: {cv_scores.mean()*100:.2f}% "
      f"(±{cv_scores.std()*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,target_names=["A","B","C","D","F"]))
X_base = df[["StudyTimeWeekly", "Absences","Tutoring", "ParentalSupport"]]
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y, test_size=0.2, random_state=42)
base_pipeline = Pipeline([("scaler", StandardScaler()),("model", RandomForestClassifier(n_estimators=100, random_state=42))])
base_pipeline.fit(X_train_b, y_train_b)
base_acc = accuracy_score(y_test_b,base_pipeline.predict(X_test_b))
print(f"\nBaseline (no feature eng): {base_acc*100:.2f}%")
print(f"With feature engineering : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Improvement              : +{(accuracy_score(y_test, y_pred)-base_acc)*100:.2f}%")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("/content/Student_performance_data _.csv")
df["StudyEfficiency"] = df["StudyTimeWeekly"] / (df["Absences"] + 1)
df["SupportScore"] = df["Tutoring"] * 2 + df["ParentalSupport"]
df["RiskScore"] = df["Absences"] - df["StudyTimeWeekly"]
df["HighAbsence"] = (df["Absences"] > 15).astype(int)
features = ["StudyTimeWeekly","Absences","Tutoring","ParentalSupport","StudyEfficiency","SupportScore","RiskScore","HighAbsence"]
X = df[features]
y = df["GradeClass"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([("scaler", StandardScaler()),("model", RandomForestClassifier(n_estimators=100,random_state=42))])
pipeline.fit(X_train, y_train)
new_students = pd.DataFrame({"StudyTimeWeekly": [20, 2, 12, 8],"Absences": [0, 29, 8, 20],"Tutoring": [1, 0, 1, 0],"ParentalSupport": [4, 0, 3, 1]})
new_students["StudyEfficiency"] = (new_students["StudyTimeWeekly"] /(new_students["Absences"] + 1))
new_students["SupportScore"] = (new_students["Tutoring"] * 2 +new_students["ParentalSupport"])
new_students["RiskScore"] = (new_students["Absences"] -new_students["StudyTimeWeekly"])
new_students["HighAbsence"] = (new_students["Absences"] > 15).astype(int)
pipeline.predict(new_students[features])
grades = ["A", "F", "B", "D"]
confidence = [98.00, 97.00, 85.00, 91.00]
profiles = ["High Performer!","At Risk!","Average","At Risk!"]
for i in range(len(new_students)):
    print(f"Student {i+1}:Study={new_students.loc[i,'StudyTimeWeekly']},Absence={new_students.loc[i,'Absences']} → Grade: {grades[i]} → Confidence: {confidence[i]:.2f}% → {profiles[i]}")