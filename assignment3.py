import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. DATA PREPROCESSING [cite: 7]
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Normalize to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. MODEL BUILDING FUNCTION [cite: 5, 10]
def build_model(model_type):
    model = models.Sequential()
    
    if model_type == "ResNet50":
        # Pretrained ResNet50 [cite: 8]
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    else: # InceptionV3 [cite: 9]
        # InceptionV3 requires resizing to at least 75x75
        model.add(layers.Resizing(75, 75))
        base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
    
    base_model.trainable = False # Use Transfer Learning [cite: 5]
    
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax')) # 10 classes for CIFAR-10 [cite: 10]
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. TRAINING LOOP [cite: 11, 12]
results = {}
epochs = 10 # Set to 10 (Range is 10-20 per instructions) 

for model_name in ["ResNet50", "InceptionV3"]:
    print(f"\n--- Training {model_name} for {epochs} Epochs ---")
    model = build_model(model_name)
    
    start_time = time.time()
    # Training results to be used for comparison 
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_data=(x_test, y_test))
    end_time = time.time()
    
    # Model size metric 
    model_path = f"{model_name}_model.h5"
    model.save(model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    
    results[model_name] = {
        'history': history.history,
        'time': end_time - start_time,
        'size': model_size,
        'model': model
    }

# 4. PLOTTING PERFORMANCE [cite: 17, 18, 19]
def plot_results(results):
    plt.figure(figsize=(12, 5))
    
    # Accuracy vs Epoch [cite: 18]
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['history']['accuracy'], label=f'{name} Train')
        plt.plot(data['history']['val_accuracy'], label=f'{name} Val', linestyle='--')
    plt.title('Accuracy vs Epoch')
    plt.legend()

    # Loss vs Epoch [cite: 19]
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['history']['loss'], label=f'{name} Train')
        plt.plot(data['history']['val_loss'], label=f'{name} Val', linestyle='--')
    plt.title('Loss vs Epoch')
    plt.legend()
    
    plt.show()

plot_results(results)

# 5. CONFUSION MATRIX 
for name, data in results.items():
    print(f"\nGenerating Confusion Matrix for {name}...")
    y_pred = np.argmax(data['model'].predict(x_test[:1000]), axis=1)
    cm = confusion_matrix(y_test[:1000], y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {name}')
    plt.show()

# 6. FINAL COMPARISON TABLE 
print("\n" + "="*35)
print("FINAL PERFORMANCE SUMMARY")
print("="*35)
for name, data in results.items():
    print(f"MODEL: {name}")
    print(f"- Training Time: {data['time']:.2f} seconds") # 
    print(f"- Model Size: {data['size']:.2f} MB") # 
    print(f"- Final Val Accuracy: {data['history']['val_accuracy'][-1]:.4f}") # 
    print("-" * 20)