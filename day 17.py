import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
print("CNN — CONVOLUTIONAL NEURAL NETWORKS")

print("\n What is CNN?")
print("""
Components:
1. Input Layer → Image (28×28×1 for MNIST)

2. Convolution Layer (Conv2D)
   - Filter/kernel slides over image
   - Detects patterns: edges, textures
   - Output: Feature maps

3. Pooling Layer (MaxPooling2D)
   - Reduces dimensions (28×28 → 14×14)
   - Keeps important features
   - Faster computation

4. Flatten Layer
   - Convert 2D to 1D (for dense layers)

5. Dense Layers
   - Classification (like regular NN)

6. Output Layer
   - 10 neurons (digits 0-9)
   - softmax activation

Architecture:
Input(28×28×1)
  ↓ Conv2D(32 filters)
  ↓ MaxPooling2D
  ↓ Conv2D(64 filters)
  ↓ MaxPooling2D
  ↓ Flatten
  ↓ Dense(128)
  ↓ Dense(10) softmax
  ↓ Output (0-9 digit)
""")

print("\n Loading MNIST Dataset (Handwritten Digits):")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Training images: {X_train.shape[0]}")
print(f"Test images: {X_test.shape[0]}")
print(f"Image shape: {X_train.shape[1:]} (28×28 pixels)")
print(f"Classes: {len(np.unique(y_train))} (0-9 digits)")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"\nReshaped for CNN: {X_train.shape}")
print("(batch, height, width, channels)")
print("\nSample Image:")
print(f"First training image shape: {X_train[0].shape}")
print(f"First training label: {y_train[0]}")
print("(Image normalized to 0-1 range)")

print("BUILD & TRAIN CNN MODEL")
print("\n Building CNN Model:")
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu',
           input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

print("Model Architecture:")
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print("\n Training CNN (this may take 1-2 minutes)...")
history = model.fit(X_train, y_train,epochs=10,batch_size=128,validation_split=0.2,verbose=0)

print("Training complete!")
print("\n Model Evaluation:")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {train_acc*100:.2f}%")
print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
if val_acc < train_acc * 0.95:
    print("Overfitting detected: Training >> Validation")
else:
    print("Good generalization: Training ≈ Validation")

print("CNN PREDICTIONS ON TEST IMAGES")
predictions = model.predict(X_test[:10], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = y_test[:10]
print("\nFirst 10 Test Images Predictions:")
print(f"{'Image':<8} {'Predicted':<12} {'Actual':<10} {'Confidence':<12} {'Correct?'}")
correct_count = 0
for i in range(10):
    pred = predicted_classes[i]
    actual = actual_classes[i]
    confidence = predictions[i][pred] * 100
    is_correct = "Yes" if pred == actual else "No"

    if pred == actual:
        correct_count += 1

    print(f"{i+1:<8} {pred:<12} {actual:<10} {confidence:<12.1f}% {is_correct}")

accuracy = (correct_count / 10) * 100
print(f"\nAccuracy on first 10 images: {accuracy:.1f}%")

print("\nConfidence Distribution (first 5 images):")
for i in range(5):
    print(f"\nImage {i+1} (Actual: {actual_classes[i]}):")
    probs = predictions[i]
    for digit in range(10):
        bar_length = int(probs[digit] * 50)
        bar = "=" * bar_length
        print(f"  {digit}: {probs[digit]*100:.1f}% {bar}")

print("CNN vs TRADITIONAL NEURAL NETWORK")
print("\nBuilding Traditional NN (flattened input)...")

model_traditional = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model_traditional.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_trad = model_traditional.fit(X_train, y_train,epochs=10,batch_size=128,validation_split=0.2,verbose=0)
test_loss_trad, test_acc_trad = model_traditional.evaluate(X_test, y_test, verbose=0)
print("\n Comparison:")
print(f"{'Model':<25} {'Test Accuracy':<15} {'Parameters':<15}")
print("-"*55)
print(f"{'CNN':<25} {test_acc*100:<15.2f}% {model.count_params():<15}")
print(f"{'Traditional NN':<25} {test_acc_trad*100:<15.2f}% {model_traditional.count_params():<15}")

improvement = (test_acc - test_acc_trad) * 100
print(f"\nCNN Improvement: +{improvement:.2f}%")

print("\nConclusion:")
print("CNN > Traditional NN for image tasks!")
print("- Better accuracy (preserves spatial structure)")
print("- More parameters but learns better features")
print("- Standard approach for computer vision")

print("CNN ARCHITECTURE COMPARISON")
print("\nBuilding Architecture 1...")

model1 = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu',
           input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model1.fit(X_train,y_train,epochs=5,batch_size=128,validation_split=0.2,verbose=0)
loss1, acc1 = model1.evaluate(X_test, y_test, verbose=0)
params1 = model1.count_params()
print("Building Architecture 2...")

model2 = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu',
           input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model2.fit(X_train,y_train,epochs=5,batch_size=128,validation_split=0.2,verbose=0)
loss2, acc2 = model2.evaluate(X_test, y_test, verbose=0)
params2 = model2.count_params()
print("Building Architecture 3...")

model3 = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu',
           input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model3.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model3.fit(X_train,y_train,epochs=5,batch_size=128,validation_split=0.2,verbose=0)
loss3, acc3 = model3.evaluate(X_test, y_test, verbose=0)
params3 = model3.count_params()
print("CNN ARCHITECTURE COMPARISON")
print(f"{'Architecture':<15} {'Test Accuracy':<18} {'Parameters':<15} {'Complexity'}")
print(f"{'1':<15} {acc1*100:<18.2f}% {params1:<15} Medium")
print(f"{'2':<15} {acc2*100:<18.2f}% {params2:<15} Light")
print(f"{'3':<15} {acc3*100:<18.2f}% {params3:<15} Heavy")
accuracies = [acc1, acc2, acc3]
parameters = [params1, params2, params3]

best_index = accuracies.index(max(accuracies))
best_architecture = best_index + 1
print(f"Best Architecture: {best_architecture}")
if best_architecture == 1:
    print("Reason: Good balance of accuracy and parameters")

elif best_architecture == 2:
    print("Reason: Light model with fewer parameters and good accuracy")

else:
    print("Reason: Highest accuracy, but uses more parameters")