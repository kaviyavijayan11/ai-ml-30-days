import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

print("TRANSFER LEARNING - IMAGE CLASSIFIER")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training images: {X_train.shape[0]}")
print(f"Test images: {X_test.shape[0]}")
print(f"Image shape: {X_train.shape[1:]}")
print(f"Classes: {len(class_names)}")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = y_train.flatten()
y_test = y_test.flatten()
train_subset = 5000
test_subset = 1000

X_train_sub = X_train[:train_subset]
y_train_sub = y_train[:train_subset]
X_test_sub = X_test[:test_subset]
y_test_sub = y_test[:test_subset]

print(f"\nUsing subset for faster training:")
print(f"Training: {train_subset} images")
print(f"Testing: {test_subset} images")
print("\nClass distribution in training subset:")
for i, name in enumerate(class_names):
    count = np.sum(y_train_sub == i)
    print(f"  {name}: {count}")

print("\nBUILDING TRANSFER LEARNING MODEL")

print("""
Transfer Learning Strategy:
1. Load MobileNetV2 (pretrained on ImageNet -
   1.4 million images, 1000 categories)
2. FREEZE its learned weights (dont retrain them)
3. Add NEW layers on top for our 10 categories
4. Only train the NEW layers (fast!)
""")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)
base_model.trainable = False

print(f"Base model layers: {len(base_model.layers)}")
print(f"Base model trainable: {base_model.trainable}")
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

print(f"\nTrainable parameters: {trainable_params:,}")
print(f"Non-trainable (frozen) parameters: {non_trainable_params:,}")
print(f"We only train {trainable_params/(trainable_params+non_trainable_params)*100:.1f}% of total parameters!")

print("\nTRAINING MODEL")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("Training started (this may take 3-5 minutes)...")

history = model.fit(
    X_train_sub, y_train_sub,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

epochs_trained = len(history.history['loss'])
print(f"Training completed in {epochs_trained} epochs")
test_loss, test_acc = model.evaluate(X_test_sub, y_test_sub, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

train_final_loss = history.history['loss'][-1]
val_final_loss = history.history['val_loss'][-1]
print(f"Final Training Loss: {train_final_loss:.4f}")
print(f"Final Validation Loss: {val_final_loss:.4f}")

diff = abs(val_final_loss - train_final_loss)
if diff > 0.3:
    print("Status: Some overfitting detected")
else:
    print("Status: Good generalization")
print(f"\nComparison:")
print(f"Transfer Learning: {test_acc*100:.2f}% accuracy in {epochs_trained} epochs")
print(f"From-scratch CNN would typically need 50-100+ epochs")
print(f"and much more data to reach similar accuracy")

print("\nPREDICTIONS AND ERROR ANALYSIS")

predictions = model.predict(X_test_sub[:20], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = y_test_sub[:20]

print(f"{'Image':<8}{'Predicted':<15}{'Actual':<15}{'Confidence':<12}{'Correct'}")
print("-"*65)

correct = 0
for i in range(20):
    pred_name = class_names[predicted_classes[i]]
    actual_name = class_names[actual_classes[i]]
    confidence = predictions[i][predicted_classes[i]] * 100
    is_correct = "Yes" if predicted_classes[i] == actual_classes[i] else "No"
    if predicted_classes[i] == actual_classes[i]:
        correct += 1

    print(f"{i+1:<8}{pred_name:<15}{actual_name:<15}{confidence:<12.1f}{is_correct}")

print(f"\nAccuracy on these 20 samples: {correct}/20 = {correct/20*100:.1f}%")
full_predictions = model.predict(X_test_sub, verbose=0)
full_predicted = np.argmax(full_predictions, axis=1)

print("\nPer-Class Accuracy:")
for i, name in enumerate(class_names):
    class_mask = y_test_sub == i
    if class_mask.sum() > 0:
        class_correct = (full_predicted[class_mask] == i).sum()
        class_total = class_mask.sum()
        class_acc = class_correct / class_total * 100
        print(f"  {name:<12}: {class_acc:.1f}% ({class_correct}/{class_total})")

readme_content = f"""# Image Classifier - Transfer Learning

## Project Overview
Multi-class image classifier using Transfer Learning with
MobileNetV2, classifying images into 10 CIFAR-10 categories.

## Approach
- Used pretrained MobileNetV2 (trained on ImageNet, 1.4M images)
- Froze base model weights, added custom classification head
- Only trained {trainable_params:,} parameters instead of
  millions - much faster than training from scratch

## Results
- Test Accuracy: {test_acc*100:.2f}%
- Training completed in {epochs_trained} epochs
- Trainable params: {trainable_params/(trainable_params+non_trainable_params)*100:.1f}% of total

## Key Techniques
1. Transfer Learning (MobileNetV2 base)
2. Global Average Pooling
3. Dropout regularization
4. Early Stopping

## Comparison to From-Scratch CNN (Day 17)
Day 17 MNIST CNN: Built from scratch, 98.91% accuracy,
simpler grayscale digits, more epochs needed.

This project: Complex color images (10 real-world categories),
faster training via transfer learning, demonstrates
industry-standard approach for limited data/time scenarios.

## Tools Used
- TensorFlow/Keras
- MobileNetV2 (Transfer Learning)
- CIFAR-10 dataset

## Author
Kaviya V | github.com/kaviyavijayan11
"""

with open("README.md", "w") as f:
    f.write(readme_content)

print("README.md created!")