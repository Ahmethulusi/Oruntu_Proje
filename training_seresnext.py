"""
SE-ResNeXt50 Transfer Learning — Fungi Sınıflandırma (TensorFlow/Keras)
Sınıflar: H1, H2, H3, H5, H6  (5 sınıf)
Gerekli paketler: pip install image-classifiers optuna
"""

# ── 1. KÜTÜPHANELER ───────────────────────────────────────────────────────────
# import ve Classifiers.get() aynı hücrede olmalı; ayrılırsa NameError çıkar.

from pathlib import Path

import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing import image_dataset_from_directory
from classification_models.tfkeras import Classifiers

SEResNeXt50, preprocess_input = Classifiers.get("seresnext50")

# Mixed precision: float16 hesaplama, float32 ağırlık — VRAM'i ~yarıya indirir.
mixed_precision.set_global_policy("mixed_float16")

print(f"TensorFlow versiyon : {tf.__version__}")
print(f"GPU'lar             : {tf.config.list_physical_devices('GPU')}")
print(f"Precision policy    : {mixed_precision.global_policy()}")
print(f"Model yüklendi      : {SEResNeXt50}")

# ─────────────────────────────────────────────────────────────────────────────


# ── 2. AYARLAR ────────────────────────────────────────────────────────────────

BASE_DIR    = Path.cwd()
DATA_DIR    = BASE_DIR / "data"
CLASSES     = ["H1", "H2", "H3", "H5", "H6"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE             = (224, 224)
BATCH_SIZE           = 8   # Head eğitimi için
BATCH_SIZE_FINETUNE  = 4   # Fine-tune: tüm gradyanlar belleğe sığsın diye küçük
EPOCHS_FROZEN        = 10
EPOCHS_FINETUNE      = 20
N_TRIALS             = 20  # Optuna deneme sayısı

# ─────────────────────────────────────────────────────────────────────────────


# ── 3. VERİ YÜKLEYİCİLER ─────────────────────────────────────────────────────

train_ds = image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASSES,
    seed=42,
)

valid_ds = image_dataset_from_directory(
    DATA_DIR / "valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASSES,
    seed=42,
)

test_ds = image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASSES,
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# Fine-tune aşaması için küçük batch'li yükleyiciler (objective içinde de kullanılır)
train_ds_ft = image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE_FINETUNE,
    label_mode="categorical",
    class_names=CLASSES,
    seed=42,
).prefetch(AUTOTUNE)

valid_ds_ft = image_dataset_from_directory(
    DATA_DIR / "valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE_FINETUNE,
    label_mode="categorical",
    class_names=CLASSES,
    seed=42,
).prefetch(AUTOTUNE)

# ─────────────────────────────────────────────────────────────────────────────


# ── 5. VERİ ARTIRMA (AUGMENTATION) ───────────────────────────────────────────

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="augmentation")

# ─────────────────────────────────────────────────────────────────────────────


# ── 6. MODEL — SE-ResNeXt50 ───────────────────────────────────────────────────
# include_top=False → ImageNet head kaldırılıyor.
# SE bloğu kanal ağırlıklandırması yaparak ResNeXt'in grouped conv'larını güçlendirir.
#
# preprocess_input (keras_applications) TF 2.10 sembolik graph'ıyla uyumsuz;
# SE-ResNeXt'in beklediği 'torch' mode normalizasyonu elle uygulanıyor:
#   [0,255] → [0,1] → ImageNet mean/std ile normalize.

class TorchPreprocess(keras.layers.Layer):
    def call(self, x):
        x = tf.cast(x, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        return (x - mean) / std

def build_model(trainable_base=False, dropout_rate=0.4):
    base = SEResNeXt50(
        input_shape=(*IMG_SIZE, 3),
        weights="imagenet",
        include_top=False,
    )
    base.trainable = trainable_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = TorchPreprocess()(x)
    x = base(x, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    # mixed_float16 politikasında son katman float32 olmalı (sayısal kararlılık)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    return keras.Model(inputs, outputs), base

# ─────────────────────────────────────────────────────────────────────────────


# ── 7. CALLBACK'LER ───────────────────────────────────────────────────────────

def make_callbacks(save_path):
    return [
        keras.callbacks.ModelCheckpoint(
            save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

def _trial_callbacks():
    """ModelCheckpoint olmadan hafif callback seti — Optuna denemeleri için."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

# ─────────────────────────────────────────────────────────────────────────────


# ── 8. OPTUNA OBJEKTİF FONKSİYONU ────────────────────────────────────────────

def objective(trial):
    lr_head      = trial.suggest_float("lr_head",      1e-4, 1e-2, log=True)
    lr_finetune  = trial.suggest_float("lr_finetune",  1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2,  0.6)

    # Aşama 1 — donuk backbone
    model, _ = build_model(trainable_base=False, dropout_rate=dropout_rate)
    model.compile(
        optimizer=keras.optimizers.Adam(lr_head),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=_trial_callbacks(),
        verbose=0,
    )

    # Aşama 2 — backbone son %40 açık
    backbone = next(l for l in model.layers if hasattr(l, "layers"))
    total_layers = len(backbone.layers)
    fine_tune_from = int(total_layers * 0.60)
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= fine_tune_from

    model.compile(
        optimizer=keras.optimizers.Adam(lr_finetune),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds_ft,
        validation_data=valid_ds_ft,
        epochs=EPOCHS_FINETUNE,
        callbacks=_trial_callbacks(),
        verbose=0,
    )

    return min(history.history["val_loss"])

# ─────────────────────────────────────────────────────────────────────────────


# ── 9. OPTUNA ARAŞTIRMASI ─────────────────────────────────────────────────────

print("\n=== Optuna Hiperparametre Arama ===")

study = optuna.create_study(
    direction="minimize",
    study_name="seresnext50_fungi",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_params
print(f"\nEn iyi deneme  : Trial #{study.best_trial.number}")
print(f"En iyi val_loss: {study.best_value:.4f}")
print(f"Parametreler   : {best}")

# ─────────────────────────────────────────────────────────────────────────────


# ── 10. SON EĞİTİM — EN İYİ PARAMETRELERİYLE ─────────────────────────────────

print("\n=== AŞAMA 1: Head Eğitimi (backbone dondurulmuş) ===")

model, base_model = build_model(
    trainable_base=False,
    dropout_rate=best["dropout_rate"],
)
model.compile(
    optimizer=keras.optimizers.Adam(best["lr_head"]),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary(show_trainable=True)

history_frozen = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=make_callbacks("best_seresnext_head.keras"),
)

# ─────────────────────────────────────────────────────────────────────────────


print("\n=== AŞAMA 2: Fine-Tune (backbone son %40 açık) ===")

model = keras.models.load_model(
    "best_seresnext_head.keras",
    custom_objects={"TorchPreprocess": TorchPreprocess},
)

backbone = next(l for l in model.layers if hasattr(l, "layers"))
total_layers = len(backbone.layers)
fine_tune_from = int(total_layers * 0.60)
for i, layer in enumerate(backbone.layers):
    layer.trainable = i >= fine_tune_from
print(f"Backbone: {total_layers} katman, {total_layers - fine_tune_from} tanesi eğitiliyor.")

model.compile(
    optimizer=keras.optimizers.Adam(best["lr_finetune"]),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_finetune = model.fit(
    train_ds_ft,
    validation_data=valid_ds_ft,
    epochs=EPOCHS_FINETUNE,
    callbacks=make_callbacks("best_seresnext.keras"),
)

# ─────────────────────────────────────────────────────────────────────────────


# ── 11. TEST DEĞERLENDİRMESİ ──────────────────────────────────────────────────

print("\n=== Test Değerlendirmesi ===")

best_model = keras.models.load_model(
    "best_seresnext.keras",
    custom_objects={"TorchPreprocess": TorchPreprocess},
)
test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"\nTest Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────


# ── 12. LOSS / ACCURACY GRAFİKLERİ ───────────────────────────────────────────

import matplotlib.pyplot as plt

def plot_history(frozen_hist, finetune_hist):
    train_loss = frozen_hist.history["loss"]     + finetune_hist.history["loss"]
    val_loss   = frozen_hist.history["val_loss"] + finetune_hist.history["val_loss"]
    train_acc  = frozen_hist.history["accuracy"]     + finetune_hist.history["accuracy"]
    val_acc    = frozen_hist.history["val_accuracy"] + finetune_hist.history["val_accuracy"]

    epochs     = range(1, len(train_loss) + 1)
    frozen_end = len(frozen_hist.history["loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss,   label="Val Loss")
    ax1.axvline(frozen_end, color="gray", linestyle="--", linewidth=1, label="Fine-tune başlangıcı")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc, label="Train Accuracy")
    ax2.plot(epochs, val_acc,   label="Val Accuracy")
    ax2.axvline(frozen_end, color="gray", linestyle="--", linewidth=1, label="Fine-tune başlangıcı")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("SE-ResNeXt50 — Eğitim Geçmişi", fontsize=14)
    plt.tight_layout()
    plt.savefig("training_history_seresnext.png", dpi=150)
    plt.show()
    print("Graf kaydedildi → training_history_seresnext.png")

plot_history(history_frozen, history_finetune)

# ─────────────────────────────────────────────────────────────────────────────


# ── 13. CONFUSION MATRIX ──────────────────────────────────────────────────────

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, dataset, class_names):
    y_true, y_pred = [], []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix — SE-ResNeXt50")
    plt.tight_layout()
    plt.savefig("confusion_matrix_seresnext.png", dpi=150)
    plt.show()
    print("Graf kaydedildi → confusion_matrix_seresnext.png")

    return y_true, y_pred

y_true, y_pred = plot_confusion_matrix(best_model, test_ds, CLASSES)

# ─────────────────────────────────────────────────────────────────────────────


# ── 14. CLASSIFICATION REPORT ─────────────────────────────────────────────────

from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
print("\nClassification Report:\n")
print(report)

# ─────────────────────────────────────────────────────────────────────────────
