import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("DROPOUT — PREVENTING OVERFITTING")
df = pd.read_csv("/content/Student_performance_data _.csv")
X = df[["StudyTimeWeekly", "Absences","Tutoring", "ParentalSupport"]].values
y = df["GradeClass"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n What is Dropout?")
print("""
During training:
- Network has neurons: o o o o o
- Dropout(0.5): Randomly disable 50%
  Each forward pass: o X o X o
                     (different neurons!)
  Forces network to learn robust features

During prediction:
- ALL neurons active (scaled appropriately)
- More reliable predictions

Why it works:
1. Prevents co-adaptation (neurons depend too much)
2. Acts like ensemble (multiple models train)
3. Reduces overfitting
""")
print("\n Model WITHOUT Dropout:")
model_no_dropout = Sequential([Dense(128, activation='relu', input_shape=(4,)),Dense(64, activation='relu'),Dense(32, activation='relu'),Dense(5, activation='softmax')])
model_no_dropout.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_no_dropout = model_no_dropout.fit(X_train_scaled, y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=0)
print("\nModel WITH Dropout:")
model_with_dropout = Sequential([Dense(128, activation='relu', input_shape=(4,)),Dropout(0.5),Dense(64, activation='relu'),Dropout(0.3),Dense(32, activation='relu'),Dropout(0.2),Dense(5, activation='softmax')])
model_with_dropout.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_with_dropout = model_with_dropout.fit(X_train_scaled, y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=0)
print("\nComparison:")
print(f"{'Model':<25} {'Test Acc':<12} {'Train Loss':<15} {'Val Loss':<15}")
test_loss, test_acc = model_no_dropout.evaluate(
    X_test_scaled, y_test, verbose=0)
train_loss_1 = history_no_dropout.history['loss'][-1]
val_loss_1 = history_no_dropout.history['val_loss'][-1]

test_loss_2, test_acc_2 = model_with_dropout.evaluate(
    X_test_scaled, y_test, verbose=0)
train_loss_2 = history_with_dropout.history['loss'][-1]
val_loss_2 = history_with_dropout.history['val_loss'][-1]
print(f"{'Without Dropout':<25} {test_acc*100:<12.2f}% {train_loss_1:<15.4f} {val_loss_1:<15.4f}")
print(f"{'With Dropout':<25} {test_acc_2*100:<12.2f}% {train_loss_2:<15.4f} {val_loss_2:<15.4f}")
print("\n Overfitting Check:")
diff_1 = abs(val_loss_1 - train_loss_1)
diff_2 = abs(val_loss_2 - train_loss_2)

print(f"Without Dropout: Train-Val difference = {diff_1:.4f}")
if diff_1 > 0.1:
    print("OVERFITTING DETECTED")
else:
    print("Good generalization")

print(f"With Dropout: Train-Val difference = {diff_2:.4f}")
if diff_2 > 0.1:
    print("OVERFITTING DETECTED")
else:
    print("Good generalization")

from tensorflow.keras import regularizers

print("L1/L2 REGULARIZATION")
print("\nWhat is Regularization?")
print("""
Normal loss = prediction error
Regularized loss = prediction error + penalty for large weights

Loss = MSE + λ × (sum of |weights|)  [L1]
Loss = MSE + λ × (sum of weights²)   [L2]

λ (lambda) = regularization strength
  λ=0   → no regularization (overfitting risk)
  λ=0.1 → strong regularization (underfitting risk)
  λ=0.01 → balanced

L1 vs L2:
L1 = feature selection (some weights → 0)
L2 = weight decay (small weights)
""")
print("\n Model WITH L2 Regularization:")
model_with_l2 = Sequential([
    Dense(128, activation='relu', input_shape=(4,),
          kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    Dense(32, activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    Dense(5, activation='softmax')
])

model_with_l2.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_l2 = model_with_l2.fit(X_train_scaled, y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=0)
test_loss_l2, test_acc_l2 = model_with_l2.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Test Accuracy: {test_acc_l2*100:.2f}%")
print(f"Final Training Loss: {history_l2.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history_l2.history['val_loss'][-1]:.4f}")

print("\n All Methods Comparison:")
print(f"{'Method':<25} {'Test Acc':<12}")
print("-"*37)
print(f"{'No Regularization':<25} {test_acc*100:<12.2f}%")
print(f"{'With Dropout':<25} {test_acc_2*100:<12.2f}%")
print(f"{'With L2 Regularization':<25} {test_acc_l2*100:<12.2f}%")

from tensorflow.keras.callbacks import EarlyStopping

print("EARLY STOPPING")
print("\n What is Early Stopping?")
print("""
Monitor validation loss during training
If val_loss doesn't improve for N epochs → stop

Benefits:
1. Saves training time
2. Prevents overfitting automatically
3. Finds optimal epoch count

Parameters:
monitor='val_loss' → what to monitor
patience=10 → how many epochs with no improvement
restore_best_weights=True → restore best model
""")
print("\n Model WITH Early Stopping:")
early_stop = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
model_with_es = Sequential([
    Dense(128, activation='relu', input_shape=(4,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

model_with_es.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_es = model_with_es.fit(X_train_scaled, y_train,epochs=200,batch_size=32,validation_split=0.2,callbacks=[early_stop],verbose=0)

epochs_trained = len(history_es.history['loss'])
print(f"Stopped at epoch: {epochs_trained}")
print(f"(Would have trained 200 epochs otherwise!)")
test_loss_es, test_acc_es = model_with_es.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_acc_es*100:.2f}%")
print("\n Summary — Best Techniques Combined:")
print(f"{'Technique':<30} {'Test Acc':<12} {'Epochs':<10}")
print(f"{'Baseline (no tricks)':<30} {test_acc*100:<12.2f}% {50:<10}")
print(f"{'Dropout':<30} {test_acc_2*100:<12.2f}% {50:<10}")
print(f"{'L2 Regularization':<30} {test_acc_l2*100:<12.2f}% {50:<10}")
print(f"{'Dropout + Early Stop':<30} {test_acc_es*100:<12.2f}% {epochs_trained:<10}")

print("PRODUCTION-READY NEURAL NETWORK")

final_model = Sequential([
    Dense(128, activation='relu', input_shape=(4,),
          kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.4),
    Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(5, activation='softmax')])

final_model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True,verbose=0)
history_final = final_model.fit(X_train_scaled, y_train,epochs=200,batch_size=32,validation_split=0.2,callbacks=[early_stop],verbose=0)
train_loss_final = history_final.history['loss'][-1]
val_loss_final = history_final.history['val_loss'][-1]
test_loss_final, test_acc_final = final_model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\nFinal Model Performance:")
print(f" Test Accuracy: {test_acc_final*100:.2f}%")
print(f" Training Epochs: {len(history_final.history['loss'])}")
print(f" Training Loss: {train_loss_final:.4f}")
print(f" Validation Loss: {val_loss_final:.4f}")
print(f" Overfitting Status: ", end="")

if val_loss_final > train_loss_final * 1.2:
    print("Some overfitting")
else:
    print("Good generalization")
print(f"\n Deep Learning vs Traditional ML:")
print(f"{'Method':<25} {'Accuracy'}")
print("-"*40)
print(f"{'Logistic Regression (Day 13)':<25} 71.61%")
print(f"{'Neural Network (Day 16)':<25} {test_acc_final*100:.2f}%")
improvement = (test_acc_final - 0.7161) * 100
print(f"{'Improvement':<25} +{improvement:.2f}%")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
df = pd.read_csv("/content/Student_performance_data _.csv")

X = df[["StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport"]].values
y = df["GradeClass"].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
dropout_rates = [0.2, 0.3, 0.4, 0.5]
l2_lambdas = [0.001, 0.01, 0.1]
learning_rates = [0.001, 0.01]
results = []

for dropout_rate in dropout_rates:
    for l2_lambda in l2_lambdas:
        for learning_rate in learning_rates:
            model = Sequential([
                Dense(128,activation="relu",input_shape=(4,),kernel_regularizer=regularizers.l2(l2_lambda)),
                Dropout(dropout_rate),
                Dense(64,activation="relu",kernel_regularizer=regularizers.l2(l2_lambda)),
                Dropout(dropout_rate),
                Dense(32,activation="relu",kernel_regularizer=regularizers.l2(l2_lambda)),
                Dropout(dropout_rate),
                Dense(5, activation="softmax")])
            model.compile(optimizer=Adam(learning_rate=learning_rate),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
            early_stop = EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True)
            history = model.fit(X_train_scaled,y_train,epochs=200,batch_size=32,validation_split=0.2,callbacks=[early_stop],verbose=0)
            test_loss, test_accuracy = model.evaluate(X_test_scaled,y_test,verbose=0)
            train_loss = history.history["loss"][-1]
            val_loss = history.history["val_loss"][-1]
            if val_loss > train_loss * 1.2:
                overfitting = "Some overfitting"
            else:
                overfitting = "Good"
            results.append([dropout_rate,l2_lambda,learning_rate,test_accuracy * 100,overfitting])
results_df = pd.DataFrame(results,columns=["Dropout","L2","LR","Accuracy","Overfitting"])
best_index = results_df["Accuracy"].idxmax()
best = results_df.loc[best_index]
print("\nHyperparameter Tuning Results:\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2f}%".format
        }
    )
)

print("\nBest Combination:\n")

print(
    f"Dropout={best['Dropout']}, "
    f"L2={best['L2']}, "
    f"LR={best['LR']}"
)

print(f"\nAccuracy: {best['Accuracy']:.2f}%")