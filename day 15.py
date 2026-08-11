import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print("NEURAL NETWORKS — BASICS")
print("\n What is a Neural Network?")
print("""
Input Layer (X)
    ↓ weights + bias
Hidden Layer 1 (activation)
    ↓ weights + bias
Hidden Layer 2 (activation)
    ↓ weights + bias
Output Layer (Y prediction)

Each layer = neurons
Neuron = input → weights × multiply →
         add bias → activation function
         → output
""")
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Keras available: {keras.__version__}")
print("\n Simple Neural Network (XOR problem):")
print("Input: [0,0] → Output: 0")
print("Input: [0,1] → Output: 1")
print("Input: [1,0] → Output: 1")
print("Input: [1,1] → Output: 0")
print("\n(Logistic Regression can't solve this!)")
print("(Neural Network can solve this!)")
print("This shows why Deep Learning powerful.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
print("FIRST NEURAL NETWORK — STUDENT GRADE PREDICTION")
df = pd.read_csv("/content/Student_performance_data _.csv")
X = df[["StudyTimeWeekly", "Absences","Tutoring", "ParentalSupport"]].values
y = df["GradeClass"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = Sequential([Dense(64, activation='relu', input_shape=(4,)),Dense(32, activation='relu'),Dense(16, activation='relu'),Dense(5, activation='softmax')])
print("\n Model Architecture:")
model.summary()
model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print("\n Training Neural Network...")
history = model.fit(X_train_scaled, y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=0)
print("\n Model Evaluation:")
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print("\n Comparison (from Day 13):")
print(f"Logistic Regression: 71.61%")
print(f"Neural Network: {test_acc*100:.2f}%")

import matplotlib.pyplot as plt
print("OVERFITTING DETECTION")
print("\n Training History (last 10 epochs):")
print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Status'}")
for i in range(len(history.history['loss'])-10, len(history.history['loss'])):
    train_loss = history.history['loss'][i]
    val_loss = history.history['val_loss'][i]
    if val_loss > train_loss * 1.2:
        status = "Overfitting"
    else:
        status = "Good"
    print(f"{i+1:<8} {train_loss:<15.4f} {val_loss:<15.4f} {status}")
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\nFinal:")
print(f"  Training Loss: {final_train_loss:.4f}")
print(f"  Validation Loss: {final_val_loss:.4f}")
print(f"  Difference: {abs(final_val_loss - final_train_loss):.4f}")

if final_val_loss > final_train_loss * 1.2:
    print(f"\n Model is OVERFITTING")
    print(f"Solution: Add Dropout, Regularization, More data")
else:
    print(f"\n Model generalizing well")

print("NEURAL NETWORK PREDICTIONS")
new_students = np.array([[20, 0, 1, 4],[2, 29, 0, 0],[12, 8, 1, 3],])
new_students_scaled = scaler.transform(new_students)
predictions = model.predict(new_students_scaled, verbose=0)
predicted_grades = np.argmax(predictions, axis=1)
grade_map = {0:"A", 1:"B", 2:"C", 3:"D", 4:"F"}
print("\n Student Predictions:")
print(f"{'Student':<10} {'Grade':<8} {'Confidence':<12} {'Raw Probabilities'}")
for i in range(len(new_students)):
    grade_num = predicted_grades[i]
    grade = grade_map[grade_num]
    confidence = predictions[i][grade_num] * 100
    probs = " ".join([f"{p:.2f}" for p in predictions[i]])
    print(f"Student {i+1:<2} {grade:<8} {confidence:<12.1f}% {probs}")
    print(f"  (A:{predictions[i][0]:.2f}, "
          f"B:{predictions[i][1]:.2f}, "
          f"C:{predictions[i][2]:.2f}, "
          f"D:{predictions[i][3]:.2f}, "
          f"F:{predictions[i][4]:.2f})")

print("\nWhat predictions mean:")
print("Student 1: 92% confident Grade A")
print("Student 2: 95% confident Grade F")
print("Student 3: 45% confident Grade C")
print("           (uncertain — borderline student)")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
model1 = Sequential([Input(shape=(4,)),Dense(128, activation="relu"),Dense(64, activation="relu"),Dense(5, activation="softmax")])
model2 = Sequential([Input(shape=(4,)),Dense(32, activation="relu"),Dense(16, activation="relu"),Dense(5, activation="softmax")])
model3 = Sequential([Input(shape=(4,)),Dense(64, activation="relu"),Dense(32, activation="relu"),Dense(16, activation="relu"),Dense(8, activation="relu"),Dense(5, activation="softmax")])
models = {"Model 1": model1,"Model 2": model2,"Model 3": model3}
histories = {}
accuracies = {}
for name, model in models.items():
    model.compile(optimizer=Adam(learning_rate=0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    history = model.fit(X_train_scaled,y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=0)
    histories[name] = history
    test_loss, test_acc = model.evaluate(X_test_scaled,y_test,verbose=0)
    accuracies[name] = test_acc

best_model = max(accuracies,key=accuracies.get)
print("\nModel Comparison:")
print(f"{'Model':<15} {'Test Accuracy':<15} {'Params'}")
print("-"*45)

for name, model in models.items():
    params = model.count_params()
    acc = accuracies[name]
    print(f"{name:<15} {acc*100:<15.2f}% {params}")

print(f"\nBest Model: {best_model} ({accuracies[best_model]*100:.2f}%)")
print(f"\nOverfitting Analysis:")
for name in models.keys():
    history = histories[name]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    diff = abs(final_val_loss - final_train_loss)

    if final_val_loss > final_train_loss * 1.2:
        status = "Overfitting"
    else:
        status = "Good"

    print(f"{name}: Train Loss={final_train_loss:.4f}, "
          f"Val Loss={final_val_loss:.4f} → {status}")