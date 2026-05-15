import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # ekransiz ortamda PNG kaydetmek icin
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# GPU KONTROLÜ
# ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU bulundu: {len(gpus)} adet")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth aktif edildi.\n")
    except RuntimeError as e:
        print(f"Memory growth ayarlanamadi: {e}\n")
else:
    print("GPU bulunamadi — egitim CPU uzerinde calisacak.\n")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
CLASSES     = ["H1", "H2", "H3", "H5", "H6"]
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE    = 224          # DenseNet169 giriş boyutu
BATCH_SIZE  = 32
EPOCHS_FT   = 20           # frozen backbone fine-tune
EPOCHS_UF   = 30           # unfrozen fine-tune
LR_FT       = 1e-3
LR_UF       = 1e-5
SEED        = 42

VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SAVE_DIR    = os.path.join(BASE_DIR, "outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DOSYA LISTESI — slide bazli bolme icin
# ─────────────────────────────────────────────
records = []
for cls in CLASSES:
    folder = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".jpg"):
            continue
        m = re.match(r"([A-Z]\d+)_(\d+[a-z]?)_(\d+)\.jpg", fname, re.IGNORECASE)
        if m:
            records.append({
                "filepath": os.path.join(folder, fname),
                "class":    cls,
                "label":    CLASS2IDX[cls],
                "slide_id": m.group(2),          # ornek / slide kimlik
            })

df = pd.DataFrame(records)
print(f"Toplam goruntu: {len(df)}")
print(df["class"].value_counts().sort_index())

# ─────────────────────────────────────────────
# 2. SLIDE BAZLI TRAIN / VAL / TEST BOLME
#    Ayni slide'dan kareler ayni kume icinde kalir
# ─────────────────────────────────────────────
# Her slide'a benzersiz grup ID ver (sinif + slide_id)
df["group"] = df["class"] + "_" + df["slide_id"]
groups = df["group"].values

splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=SEED)
train_val_idx, test_idx = next(splitter.split(df, groups=groups))

df_trainval = df.iloc[train_val_idx].reset_index(drop=True)
df_test     = df.iloc[test_idx].reset_index(drop=True)

groups_tv = df_trainval["group"].values
val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)
splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=SEED)
train_idx, val_idx = next(splitter2.split(df_trainval, groups=groups_tv))

df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

print(f"\nTrain: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
print("Train sinif dagilimi:\n", df_train["class"].value_counts().sort_index())

# ─────────────────────────────────────────────
# 3. SINIF AGIRLIKLARI
# ─────────────────────────────────────────────
cw = compute_class_weight("balanced",
                           classes=np.arange(len(CLASSES)),
                           y=df_train["label"].values)
class_weight = {i: float(w) for i, w in enumerate(cw)}
print("\nSinif agirliklari:", class_weight)

# ─────────────────────────────────────────────
# 4. VERI ARTTIRMA — tibbi goruntu icin muhafazakar
#    Geometrik: yatay/dikey cevir, kucuk dondurme
#    Fotometrik: kontrast ve parlaklık — detay kaybi olmadan
# ─────────────────────────────────────────────
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(factor=0.05),          # ±18 derece
    layers.RandomZoom(height_factor=(-0.05, 0.05)),
    # Kontrast: [1-delta, 1+delta] araliginda rastgele olcekler
    layers.RandomContrast(factor=0.15),          # ±%15 kontrast
    # Parlaklık: additive jitter, kucuk tutuyoruz
    layers.RandomBrightness(factor=0.10),        # ±%10 parlaklık
], name="augmentation")

# ─────────────────────────────────────────────
# 5. tf.data PIPELINE
# ─────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(path, label, training=False):
    raw  = tf.io.read_file(path)
    img  = tf.image.decode_jpeg(raw, channels=3)
    img  = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img  = tf.cast(img, tf.float32)
    img  = tf.keras.applications.densenet.preprocess_input(img)
    return img, label

def make_dataset(df_split, training=False):
    paths  = df_split["filepath"].values
    labels = df_split["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(lambda p, l: preprocess(p, l, training),
                num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.batch(BATCH_SIZE).map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    else:
        ds = ds.batch(BATCH_SIZE)

    return ds.prefetch(AUTOTUNE)

ds_train = make_dataset(df_train, training=True)
ds_val   = make_dataset(df_val,   training=False)
ds_test  = make_dataset(df_test,  training=False)

# ─────────────────────────────────────────────
# 6. MODEL — DenseNet169 transfer learning
# ─────────────────────────────────────────────
def build_model(trainable_backbone=False):
    backbone = keras.applications.DenseNet169(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None,
    )
    backbone.trainable = trainable_backbone

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=trainable_backbone)

    # Global Average Pooling + Global Max Pooling birlestirmesi
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x   = layers.Concatenate()([gap, gmp])

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, backbone

# ─────────────────────────────────────────────
# 7. AŞAMA 1: FROZEN BACKBONE EĞİTİMİ
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ASAMA 1: Frozen backbone egitimi")
print("=" * 60)

model, backbone = build_model(trainable_backbone=False)
model.compile(
    optimizer=keras.optimizers.Adam(LR_FT),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy",
             keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")],
)
model.summary(line_length=100)

cb_ft = [
    callbacks.ModelCheckpoint(
        os.path.join(SAVE_DIR, "best_frozen.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
    ),
    callbacks.CSVLogger(os.path.join(SAVE_DIR, "log_frozen.csv")),
]

history_ft = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS_FT,
    class_weight=class_weight,
    callbacks=cb_ft,
)

# ─────────────────────────────────────────────
# 8. AŞAMA 2: UNFREEZE — ince ayar (fine-tuning)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ASAMA 2: Unfreeze fine-tuning")
print("=" * 60)

# Son 50 katmani unfreeze et, oncesini dondur
backbone.trainable = True
for layer in backbone.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(LR_UF),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy",
             keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")],
)

cb_uf = [
    callbacks.ModelCheckpoint(
        os.path.join(SAVE_DIR, "best_finetuned.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=4, min_lr=1e-8, verbose=1,
    ),
    callbacks.CSVLogger(os.path.join(SAVE_DIR, "log_finetuned.csv")),
]

history_uf = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS_UF,
    class_weight=class_weight,
    callbacks=cb_uf,
)

# ─────────────────────────────────────────────
# 9. TEST DEĞERLENDİRMESİ
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST DEGERLENDIRMESI")
print("=" * 60)

best_model = keras.models.load_model(os.path.join(SAVE_DIR, "best_finetuned.keras"))
test_loss, test_acc, test_top2 = best_model.evaluate(ds_test, verbose=1)
print(f"\nTest Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Top-2   : {test_top2:.4f}")

# Per-class metrikler — confusion matrix
y_true, y_pred = [], []
for imgs, labels in ds_test:
    preds = best_model.predict(imgs, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# ─────────────────────────────────────────────
# 10. GRAFİKLER — PNG olarak kaydet
# ─────────────────────────────────────────────

def save_history_plot(h_ft, h_uf, save_dir):
    """İki aşamanın history'sini birleştirip loss ve accuracy grafikleri çizer."""
    acc    = h_ft.history["accuracy"]      + h_uf.history["accuracy"]
    val_acc= h_ft.history["val_accuracy"]  + h_uf.history["val_accuracy"]
    loss   = h_ft.history["loss"]          + h_uf.history["loss"]
    val_loss=h_ft.history["val_loss"]      + h_uf.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    boundary = len(h_ft.history["accuracy"])   # frozen / finetuned siniri

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_vals, val_vals, title, ylabel in [
        (axes[0], acc,  val_acc,  "Accuracy",  "Accuracy"),
        (axes[1], loss, val_loss, "Loss",      "Loss"),
    ]:
        ax.plot(epochs, train_vals, label="Train")
        ax.plot(epochs, val_vals,   label="Val")
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=1,
                   label="Unfreeze")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("DenseNet169 — Egitim Gecmisi", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Grafik kaydedildi: {path}")


def save_confusion_matrix_plot(cm, classes, save_dir):
    """Normalized ve ham deger olmak uzere iki confusion matrix yan yana."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, fmt, title in [
        (axes[0], cm,      "d",    "Ham Degerler"),
        (axes[1], cm_norm, ".2f",  "Normalize (satir %)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    linewidths=0.5, ax=ax, cbar=True)
        ax.set_title(title)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gercek")

    fig.suptitle("DenseNet169 — Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Grafik kaydedildi: {path}")


save_history_plot(history_ft, history_uf, SAVE_DIR)
save_confusion_matrix_plot(cm, CLASSES, SAVE_DIR)

# Modeli kaydet
best_model.save(os.path.join(SAVE_DIR, "densenet169_defungi_final.keras"))
print(f"\nModel kaydedildi: {SAVE_DIR}")
