# GPU Kurulum Rehberi — RTX 4060 (Windows 11)

Bu belge, ResNet ve SE-ResNeXt modellerini RTX 4060 üzerinde eğitmek için gereken
CUDA, cuDNN ve framework kurulumunu adım adım açıklar. PyTorch ve TensorFlow için
ayrı yol haritaları içerir.

---

## İçindekiler

1. [Genel Ön Hazırlık](#1-genel-ön-hazırlık)
2. [PyTorch Yol Haritası](#2-pytorch-yol-haritası)
3. [TensorFlow Yol Haritası](#3-tensorflow-yol-haritası)
   - [Seçenek A — WSL2 (Önerilen)](#seçenek-a--wsl2-önerilen)
   - [Seçenek B — Conda ile Windows Native](#seçenek-b--conda-ile-windows-native)
4. [Model Seçimi: timm ile ResNet / SE-ResNeXt](#4-model-seçimi-timm-ile-resnet--se-resnext)
5. [Kurulumu Doğrulama](#5-kurulumu-doğrulama)
6. [Sürüm Uyumluluk Tabloları](#6-sürüm-uyumluluk-tabloları)
7. [Sık Karşılaşılan Hatalar](#7-sık-karşılaşılan-hatalar)

---

## 1. Genel Ön Hazırlık

Bu adımlar hem PyTorch hem TensorFlow için ortaktır.

### 1.1 NVIDIA Sürücüsünü Güncelle

RTX 4060, CUDA 12.x için **sürücü ≥ 525.60** gerektirir.

1. `nvidia.com/drivers` adresine git.
2. Ürün ailesi: **GeForce RTX 40 Series**, model: **RTX 4060** seç.
3. En güncel **Game Ready Driver** veya **Studio Driver**'ı indir ve kur.
4. Kurulum sonrası bilgisayarı yeniden başlat.

Sürücünün kurulduğunu doğrulamak için:

```powershell
nvidia-smi
```

Çıktıda `CUDA Version: 12.x` ve GPU adı görünmeli.

### 1.2 Python Ortamı

Her framework için ayrı sanal ortam oluşturmanı öneririz (sürüm çakışmalarını önler).

```bash
# conda kullanıyorsan
conda create -n pytorch-env python=3.11
conda create -n tf-env python=3.10

# venv kullanıyorsan
python -m venv pytorch-env
python -m venv tf-env
```

---

## 2. PyTorch Yol Haritası

PyTorch en kolay GPU kurulumunu sunar: **cuDNN'i ayrıca indirmene gerek yok**,
pip paketi her şeyi içinde getirir.

### Adım 1 — CUDA Toolkit Kur (Opsiyonel)

PyTorch'un pip paketi kendi CUDA runtime'ını beraberinde getirir.
Sisteme CUDA Toolkit kurmak zorunda değilsin. Sadece NVIDIA sürücüsü yeterli.

Yine de sistem genelinde CUDA isteniyorsa:
- `developer.nvidia.com/cuda-downloads` → Windows → x86_64 → CUDA **12.4**

### Adım 2 — PyTorch Kur

```bash
# Ortamı etkinleştir
conda activate pytorch-env   # veya: source pytorch-env/bin/activate

# CUDA 12.4 için (önerilen)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1 için
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> Güncel komutu her zaman `pytorch.org/get-started/locally` adresinden al.
> Stable / Windows / Pip / Python / CUDA seçeneklerini ayarla, sayfa komutu otomatik üretir.

### Adım 3 — Doğrulama

```python
import torch

print(torch.__version__)
print("CUDA mevcut:", torch.cuda.is_available())
print("GPU adı:", torch.cuda.get_device_name(0))
print("GPU sayısı:", torch.cuda.device_count())
```

Beklenen çıktı:
```
2.x.x+cu124
CUDA mevcut: True
GPU adı: NVIDIA GeForce RTX 4060
GPU sayısı: 1
```

---

## 3. TensorFlow Yol Haritası

### Önemli Uyarı — Windows Native Kısıtlaması

TensorFlow **2.11 sürümünden itibaren Windows'ta native GPU desteğini sonlandırdı.**
RTX 4060, CUDA 11.8+ gerektirdiğinden TF 2.10 (son native Windows sürümü) ile
doğrudan çalışmaz.

İki seçenek mevcuttur:

---

### Seçenek A — WSL2 (Önerilen)

Windows 11 altında Ubuntu çalıştırarak tam TensorFlow GPU desteği elde edilir.
RTX 4060 ile en kararlı ve güncel kombinasyon budur.

#### A.1 WSL2 Kurulumu

```powershell
# PowerShell'i Yönetici olarak aç
wsl --install
# Yeniden başlat, ardından Ubuntu kullanıcı adı/şifresi belirle
```

Ubuntu sürümünü kontrol et:
```bash
lsb_release -a   # Ubuntu 22.04 önerilen
```

#### A.2 CUDA Toolkit Kurulumu (WSL içinde)

`developer.nvidia.com/cuda-downloads` adresine git:
- **Linux → x86_64 → WSL-Ubuntu → 22.04 → deb (network)**

Sayfanın verdiği komutları sırayla çalıştır (genellikle 4-5 satır).

Ardından PATH'e ekle:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Doğrulama:
```bash
nvcc --version
```

#### A.3 cuDNN Kurulumu (WSL içinde)

`developer.nvidia.com/cudnn` adresine git → **Download cuDNN** → TF sürümüne göre
uygun cuDNN'i seç (bkz. [Sürüm Tablosu](#6-sürüm-uyumluluk-tabloları)).

Local installer (deb) ile kurulum:
```bash
sudo dpkg -i cudnn-local-repo-*.deb
sudo apt-get update
sudo apt-get install libcudnn8 libcudnn8-dev
```

#### A.4 TensorFlow Kurulumu (WSL içinde)

```bash
pip install tensorflow[and-cuda]==2.15.0
```

`tensorflow[and-cuda]` paketi CUDA bağımlılıklarını otomatik çeker (TF 2.15+).

---

### Seçenek B — Conda ile Windows Native

Conda, CUDA ve cuDNN'i izole bir ortamda yönetir; sisteme ayrıca kurmak gerekmez.
Resmi destek bulunmasa da RTX 4060'ta çalışabilir.

```bash
# Ortam oluştur
conda create -n tf-env python=3.10
conda activate tf-env

# CUDA ve cuDNN'i conda üzerinden kur
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6

# TensorFlow kur
pip install tensorflow==2.13.0
```

> **Not:** Bu seçenek deneyseldir. Sorun yaşanırsa WSL2 seçeneğine geç.

---

## 4. Model Seçimi: timm ile ResNet / SE-ResNeXt

`timm` (PyTorch Image Models), önceden eğitilmiş yüzlerce CNN mimarisini tek
komutla kullanmayı sağlar.

```bash
pip install timm
```

### Kullanılabilir Modeller

```python
import timm

# Tüm ResNet varyantlarını listele
print(timm.list_models('*resnet*'))

# Tüm SE-ResNeXt varyantlarını listele
print(timm.list_models('*seresnext*'))
```

### Model Oluşturma (5 sınıf için)

```python
import timm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet varyantları
model_r50  = timm.create_model('resnet50',   pretrained=True, num_classes=5)
model_r101 = timm.create_model('resnet101',  pretrained=True, num_classes=5)
model_r200 = timm.create_model('resnet200d', pretrained=True, num_classes=5)

# SE-ResNeXt varyantları
model_sx50  = timm.create_model('seresnext50_32x4d',  pretrained=True, num_classes=5)
model_sx101 = timm.create_model('seresnext101_32x4d', pretrained=True, num_classes=5)

model = model_sx101.to(device)
```

### Önerilen Kombinasyonlar (RTX 4060 — 8 GB VRAM)

| Model               | Parametre | VRAM (batch=32) | Öneri         |
|---------------------|-----------|-----------------|---------------|
| resnet50            | 25M       | ~3 GB           | Başlangıç     |
| resnet101           | 45M       | ~5 GB           | Orta          |
| seresnext50_32x4d   | 27M       | ~4 GB           | Güçlü baseline|
| seresnext101_32x4d  | 49M       | ~6 GB           | Ana deney     |
| resnet200d          | 64M       | ~7 GB           | Maks kapasite |

> VRAM yetersiz kalırsa `batch_size`'ı düşür (32 → 16 → 8).

---

## 5. Kurulumu Doğrulama

### PyTorch

```python
import torch

assert torch.cuda.is_available(), "CUDA bulunamadı!"
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Basit GPU testi
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print("GPU matris çarpımı başarılı:", z.shape)
```

### TensorFlow

```python
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"Bulunan GPU'lar: {gpus}")

assert len(gpus) > 0, "GPU bulunamadı!"

# Basit GPU testi
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
print("GPU matris çarpımı başarılı:", c.shape)
```

---

## 6. Sürüm Uyumluluk Tabloları

### PyTorch — CUDA Uyumluluğu

| PyTorch | CUDA  | Python    |
|---------|-------|-----------|
| 2.4     | 12.4  | 3.9–3.12  |
| 2.3     | 12.1  | 3.9–3.12  |
| 2.2     | 12.1  | 3.8–3.12  |
| 2.1     | 12.1  | 3.8–3.11  |

### TensorFlow — CUDA / cuDNN Uyumluluğu

| TensorFlow | Python   | CUDA | cuDNN |
|------------|----------|------|-------|
| 2.16       | 3.9–3.12 | 12.3 | 8.9   |
| 2.15       | 3.9–3.11 | 12.2 | 8.9   |
| 2.14       | 3.9–3.11 | 11.8 | 8.7   |
| 2.13       | 3.8–3.11 | 11.8 | 8.6   |
| 2.10 *     | 3.7–3.10 | 11.2 | 8.1   |

\* 2.10: Windows'ta native GPU destekleyen son sürüm.

Kaynak: `tensorflow.org/install/source` — sayfanın altındaki tam tablo.

---

## 7. Sık Karşılaşılan Hatalar

### `torch.cuda.is_available()` → False (PyTorch)

- NVIDIA sürücüsü eski olabilir → `nvidia-smi` çalışıyor mu kontrol et.
- CPU-only PyTorch kurulmuş olabilir → `pip install torch --index-url .../cu124` komutunu tekrar çalıştır.

### `Could not load dynamic library 'libcudart.so'` (TensorFlow / WSL)

CUDA PATH doğru eklenmemiş:
```bash
echo $LD_LIBRARY_PATH   # /usr/local/cuda/lib64 içermeli
source ~/.bashrc
```

### `CUDA out of memory`

Batch size'ı düşür:
```python
# 32 → 16 → 8
train_loader = DataLoader(dataset, batch_size=16, ...)
```

### WSL'de `nvidia-smi` komutu bulunamıyor

WSL içine CUDA kurmak için Windows tarafındaki NVIDIA sürücüsü yeterlidir;
WSL'ye ayrıca sürücü kurma. Sadece CUDA Toolkit'i WSL içinde kur.

### Conda ortamında `DLL load failed` (Windows / TF)

Conda ortamı aktif değilken TF import edilmiş olabilir:
```bash
conda activate tf-env
python -c "import tensorflow as tf"
```

---

## Referans Kaynaklar

| Kaynak | Adres |
|--------|-------|
| PyTorch kurulum sayfası | `pytorch.org/get-started/locally` |
| TF sürüm uyumluluk tablosu | `tensorflow.org/install/source` |
| CUDA Toolkit indirme | `developer.nvidia.com/cuda-downloads` |
| cuDNN indirme | `developer.nvidia.com/cudnn` |
| NVIDIA sürücü indirme | `nvidia.com/drivers` |
| timm model listesi | `github.com/huggingface/pytorch-image-models` |
