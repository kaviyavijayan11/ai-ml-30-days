# Day 14 - Week 2 Self Test Results

## Q1 - Linear vs Logistic Regression
Linear Regression predicts continuous values.
Logistic Regression is used for classification (uses sigmoid, outputs probability 0-1).

## Q2 - train_test_split Output
8 2

## Q3 - Confusion Matrix
Actual Pass, Predicted Pass → True Positive (TP)
Actual Pass, Predicted Fail → False Negative (FN)
Actual Fail, Predicted Pass → False Positive (FP)
Actual Fail, Predicted Fail → True Negative (TN)

## Q4 - R² Score
0.93 → model explains ~93% of variation in target variable

## Q5 - Overfitting
Happens when model learns training data too closely, poor performance on unseen data.
Solutions: more data, regularization, cross-validation, reduce model complexity, feature selection

## Q6 - Random Forest vs Decision Tree
Random Forest = multiple trees combined (ensemble), reduces overfitting, more stable predictions

## Q7 - StudyEfficiency Feature
StudyEfficiency = StudyTimeWeekly / (Absences + 1)
Created to measure how effectively study time converts to output, adjusted for absences

## Q8 - Cross Validation
Single split depends on that one split only.
Cross-validation tests on multiple splits → more reliable performance estimate

## Q9 - Precision vs Recall (Cancer Detection)
Recall priority — missing an actual cancer patient (False Negative) is more dangerous than a false alarm
Recall = TP / (TP + FN)

## Q10 - Pipeline
1. Prevents data leakage — preprocessing fit only on train, applied to test
2. Keeps train/predict steps consistent automatically
