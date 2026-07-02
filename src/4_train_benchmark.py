import os
import pickle
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

# --- IMPORT ALL MODELS ---
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    InceptionResNetV2,
    InceptionV3,
    MobileNet,
    ResNet50,
    VGG16,
    VGG19
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
lrn_typ = 0             # 0 = False (Freeze Base Model)
learning_type = "False" # Folder suffix

num_epochs = 50
num_classes = 6
batch_size = 32

models_to_train = {
    "DenseNet121": DenseNet121,
    "EfficientNetB0": EfficientNetB0,
    "InceptionResNetV2": InceptionResNetV2,
    "InceptionV3": InceptionV3,
    "MobileNet": MobileNet,
    "ResNet50": ResNet50,
    "VGG16": VGG16,
    "VGG19": VGG19
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def calculate_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    sens = metrics.recall_score(y_true, y_pred, average='macro')
    prec = metrics.precision_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    return acc, sens, prec, f1

def plot_history(history, model_name, save_dir):
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy')
    loss = history.history['loss']
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(acc) + 1)

    history_data = {
        'Epoch': list(epochs),
        'Train Accuracy': acc,
        'Train Loss': loss
    }
    
    if val_acc: history_data['Val Accuracy'] = val_acc
    if val_loss: history_data['Val Loss'] = val_loss
        
    df_history = pd.DataFrame(history_data)
    csv_path = f"{save_dir}/{model_name}_history_data.csv"
    df_history.to_csv(csv_path, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, acc, 'b', label='Training Accuracy', linewidth=2)
    if val_acc: ax1.plot(epochs, val_acc, 'r', label='Validation Accuracy', linewidth=2, linestyle='-.')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(epochs, loss, 'b', label='Training Loss', linewidth=2)
    if val_loss: ax2.plot(epochs, val_loss, 'r', label='Validation Loss', linewidth=2, linestyle='-.')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(f"{save_dir}/{model_name}_history.png", bbox_inches='tight')
    plt.close()

def visualize_data_with_tsne(X_data, y_data, model, save_path):
    try:
        feature_extractor_model = Sequential()
        for layer in model.layers[:-1]:
            feature_extractor_model.add(layer)

        feature_vectors = feature_extractor_model.predict(X_data, verbose=0)
        
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(feature_vectors)

        plt.figure(figsize=(10, 8))
        if y_data.ndim > 1:
            y_labels = np.argmax(y_data, axis=1)
        else:
            y_labels = y_data

        class_labels = np.unique(y_labels)
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        if len(class_labels) > len(colors):
            colors = colors * (len(class_labels) // len(colors) + 1)

        legend_patches = []
        for i, label in enumerate(class_labels):
            indices = np.where(y_labels == label)
            color = colors[i % len(colors)]
            plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], c=color, label=f"Class {label}", alpha=0.6)
            legend_patches.append(mpatches.Patch(color=color, label=f"Class {label}"))

        plt.legend(handles=legend_patches, loc='best')
        plt.title(f"t-SNE Visualization")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"t-SNE Error: {e}")

def evaluate_and_store(model, X_test, y_test, class_num, model_name, train_duration, train_count, total_params, trainable_params, results_dict, save_dir_matrix):
    test_count = len(X_test)

    start_cold = time.time()
    _ = model.predict(X_test, verbose=0)
    end_cold = time.time()
    duration_cold = end_cold - start_cold

    start_warm = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    end_warm = time.time()
    duration_warm = end_warm - start_warm

    train_ms = (train_duration / (num_epochs * train_count)) * 1000
    test_cold_ms = (duration_cold / test_count) * 1000
    test_warm_ms = (duration_warm / test_count) * 1000

    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc, sens, prec, f1 = calculate_metrics(y_true, y_pred)

    results_dict["Model"].append(model_name)
    results_dict["Cls Acc"].append(acc)
    results_dict["Sens"].append(sens)
    results_dict["Prec"].append(prec)
    results_dict["F1"].append(f1)
    results_dict["Train (ms/img)"].append(train_ms)
    results_dict["Test Cold (ms/img)"].append(test_cold_ms)
    results_dict["Test Warm (ms/img)"].append(test_warm_ms)
    results_dict["Total Params"].append(total_params)
    results_dict["Trainable Params"].append(trainable_params)

    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_norm = normalize(cm, axis=1, norm='l1')
    LABELS = [f"Class {i}" for i in range(class_num)]

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_norm, annot=True, cmap='Blues', xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    plt.title(f'{model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"{save_dir_matrix}/{model_name}_cm.png", bbox_inches='tight')
    plt.close()

    print(f"[{model_name}] Acc: {acc:.4f} | Cold: {test_cold_ms:.2f} ms | Params: {total_params}")

def load_and_split_dataset(base_path, target_size=(224, 224)):
    X, y = [], []
    class_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    for idx, class_name in enumerate(class_folders):
        class_dir = os.path.join(base_path, class_name)
        images = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for img_name in images:
            try:
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size)
                img_array = np.array(img)
                X.append(img_array)
                y.append(idx)
            except Exception:
                pass
    
    X = np.array(X)
    y = np.array(y)
    
    X, y = shuffle(X, y, random_state=42)
    
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    return X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end], X[val_end:], y[val_end:]

def create_custom_model(base_model, num_classes):
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # This line forces Keras to build the model structure immediately
    model.build(input_shape=(None, 224, 224, 3)) 
    
    return model

# ==========================================
# 3. MAIN EXECUTION (MULTI-DATASET LOOP)
# ==========================================

# Set root to data/4_final_datasets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
main_datasets_root = os.path.join(BASE_DIR, 'data', '4_final_datasets')

all_datasets = sorted([d for d in os.listdir(main_datasets_root) if os.path.isdir(os.path.join(main_datasets_root, d))])

print(f"Found Datasets: {all_datasets}")

for dataset_name in all_datasets:
    print("\n" + "#"*60)
    print(f"### PROCESSING DATASET: {dataset_name} ###")
    print("#"*60 + "\n")

    dataset_path = os.path.join(main_datasets_root, dataset_name)
    parent_dir = "main_res"
    
    save_dir_models = os.path.join(parent_dir, f'saved_models_{dataset_name}_{learning_type}')
    save_dir_results = os.path.join(parent_dir, f'saved_results_{dataset_name}_{learning_type}')
    save_dir_matrix = os.path.join(parent_dir, f'saved_matrixes_{dataset_name}_{learning_type}')
    save_dir_tsne = os.path.join(parent_dir, f'saved_TSNE_{dataset_name}_{learning_type}')

    os.makedirs(save_dir_models, exist_ok=True)
    os.makedirs(save_dir_results, exist_ok=True)
    os.makedirs(save_dir_matrix, exist_ok=True)
    os.makedirs(save_dir_tsne, exist_ok=True)

    current_results_data = {
        "Model": [], "Cls Acc": [], "Sens": [], "Prec": [], "F1": [],
        "Train (ms/img)": [], "Test Cold (ms/img)": [], "Test Warm (ms/img)": [],
        "Total Params": [], "Trainable Params": []
    }

    print(f"Step 1: Loading {dataset_name} into RAM...")
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset ({dataset_name}): {e}")
        continue 

    if len(X_train) == 0:
        print(f"WARNING: No training data found in {dataset_name}, skipping.")
        continue

    train_count = len(X_train)
    
    # Preprocessing
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    print(f"Data Loaded for {dataset_name}. Train size: {train_count}")

    for model_name, ModelClass in models_to_train.items():
        print(f"\n--- Training {model_name} on {dataset_name} ---")

        try:
            K.clear_session()
            base_model = ModelClass(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

            for layer in base_model.layers:
                layer.trainable = bool(lrn_typ)

            model = create_custom_model(base_model, num_classes)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            total_params = model.count_params()
            trainable_params = sum([K.count_params(w) for w in model.trainable_weights])

            ckpt_path = f'{save_dir_models}/{model_name}_best.weights.h5'
            
            checkpoint = ModelCheckpoint(
                ckpt_path, 
                monitor='val_loss', 
                save_best_only=True, 
                save_weights_only=True, 
                mode='min'
            )

            start_train = time.time()
            history = model.fit(
                X_train, y_train_cat,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val_cat),
                callbacks=[checkpoint],
                verbose=1
            )
            end_train = time.time()
            train_duration = end_train - start_train

            plot_history(history, model_name, save_dir_results)

            print(f"Loading best weights for {model_name}...")
            model.load_weights(ckpt_path)
            
            evaluate_and_store(model, X_test, y_test_cat, num_classes, model_name,
                               train_duration, train_count, total_params, trainable_params, 
                               current_results_data, save_dir_matrix)

            limit = min(len(X_train), 1000)
            visualize_data_with_tsne(X_train[:limit], y_train_cat[:limit], model,
                                     save_path=f"{save_dir_tsne}/{model_name}.png")

            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                print(f"Cleanup: Temporary weight file for {model_name} has been deleted.")

            del model, base_model, history
            gc.collect()

        except Exception as e:
            print(f"Error training {model_name} on {dataset_name}: {e}")

    df_results = pd.DataFrame(current_results_data)
    df_results["Cls Acc"] = df_results["Cls Acc"].round(4)
    df_results["Sens"] = df_results["Sens"].round(4)
    df_results["Prec"] = df_results["Prec"].round(4)
    df_results["F1"] = df_results["F1"].round(4)
    df_results["Train (ms/img)"] = df_results["Train (ms/img)"].round(4)
    df_results["Test Cold (ms/img)"] = df_results["Test Cold (ms/img)"].round(4)
    df_results["Test Warm (ms/img)"] = df_results["Test Warm (ms/img)"].round(4)
    df_results["Total Params"] = df_results["Total Params"].astype(int)
    df_results["Trainable Params"] = df_results["Trainable Params"].astype(int)

    csv_path = f"{save_dir_results}/Final_Benchmark_{dataset_name}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nCompleted {dataset_name}. Results saved to {csv_path}")
    
    del X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat
    K.clear_session()
    gc.collect()

print("\nAll datasets processed successfully.")

master_list = []
results_root = "main_res"
if os.path.exists(results_root):
    for folder in os.listdir(results_root):
        folder_path = os.path.join(results_root, folder)
        if folder.startswith('saved_results_') and os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.startswith('Final_Benchmark_') and file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(folder_path, file))
                    dataset_name = file.replace('Final_Benchmark_', '').replace('.csv', '')
                    df.insert(0, 'Dataset_Name', dataset_name)
                    master_list.append(df)

if master_list:
    final_df = pd.concat(master_list, ignore_index=True)
    final_df.to_csv("ALL_DATASETS_MASTER_REPORT.csv", index=False)
    print("\nSUCCESS: ALL_DATASETS_MASTER_REPORT.csv has been created.")