# MRI Microrobot Detection — Synthetic Dataset Generation

YOLOv8 ile MRI görüntülerinde manyetik mikrorobot tespiti için sentetik etiketli veri seti üretim pipeline'ı.

## Proje Hakkında

Gerçek hastalarda mikrorobot deneyi etik ve pratik kısıtlamalar nedeniyle mümkün değildir. Bu proje, mevcut klinik MRI arşivlerine fizik tabanlı sentetik mikrorobot artifaktları ekleyerek YOLOv8 eğitimi için etiketli veri üretmektedir.

Manyetik susceptibility artifaktı dipol alan formülüyle simüle edilir. Void (karanlık merkez) ve Halo (parlak kenar) bölgeleri gerçek MRI fiziğiyle uyumludur. Doku dışına taşan artifact pikselleri otomatik olarak bastırılır.

## Desteklenen Organ / Dataset'ler

| Organ | Dataset | Format | Görüntü |
|-------|---------|--------|---------|
| Beyin | MR-ART Dataset (sagittal) | PNG | 148 |
| Kalp | Medical Segmentation Decathlon Task02_Heart | NIfTI → PNG | 2271 |
| Göğüs | Kaggle Breast Cancer MRI (Healthy) | JPG | 700 |
| Diz | KneeMRI Dataset — Rijeka Kliniği (1.5T Siemens) | .pck → PNG | 22.623 |

## Kurulum

```bash
pip install -r requirements.txt
```

Python 3.10+ gereklidir.

## Kullanım — Pipeline Sırası

```bash
# 1. Ham görüntüleri birleştir
py merge_dataset.py

# 2. Sentetik görüntü üret
py brain_mri_synthetic.py

# 3. CNR + Sharpness filtresi
py filter_by_cnr.py

# 4. Train/Val böl (augmentation'dan ÖNCE)
py split_dataset.py

# 5. Sadece train'e augmentation uygula
py augment_dataset.py

# 6. dataset.yaml oluştur (network seçimi sonrası)
# 7. Eğitim
pip install ultralytics
py -c "from ultralytics import YOLO; model=YOLO('yolov8n.pt'); model.train(data='dataset.yaml', epochs=100, imgsz=320, batch=16)"
```

> **Önemli:** Split işlemi augmentation'dan önce yapılmalıdır. Aksi halde aynı görüntünün orijinali val'e, augmented hali train'e düşerek veri sızıntısı (data leakage) oluşur.

> **Not:** `dataset.yaml` dosyası network mimarisi seçimi (YOLOv8 detection vs segmentation) netleştikten sonra oluşturulacaktır.

## Script Açıklamaları

| Script | Görevi |
|--------|--------|
| `merge_dataset.py` | Tüm organ görüntülerini `all_mri/` klasöründe birleştirir. Her çalıştırmada eski klasörü siler. Diz için her seferinde rastgele 5000 görüntü seçer. |
| `pck_to_png_knee.py` | KneeMRI `.pck` formatını PNG slice'larına dönüştürür |
| `brain_mri_synthetic.py` | Ana üretim scripti — artifakt yerleştirme, doku maskesi, YOLO etiketi |
| `filter_by_cnr.py` | CNR ve Laplacian sharpness filtresi, her organdan max 500 görüntü seçer |
| `split_dataset.py` | %80 train / %20 val bölme, pozitif/negatif dengesi korunur |
| `augment_dataset.py` | Sadece train'e flip, brightness, contrast, noise, rotation augmentation |

## Üretim Parametreleri

- Robot sayısı: görüntü başına 2–25 (rastgele dağılım)
- Artifact scale: 0.08–0.13
- Alpha (karartma): 0.93–1.00
- Derinlik (h): 0–2.7 mm
- Rice noise sigma: 2.0–5.0 (görüntü başına rastgele)
- Minimum robot arası mesafe: 25 piksel

## Dataset İstatistikleri (Bir Üretim Turundan)

| Aşama | Görüntü |
|-------|---------|
| Ham üretim | ~8500 |
| Filtre sonrası (pozitif) | 2000 (500/organ) |
| Negatif örnekler | 225 |
| Train (augment sonrası) | 5340 |
| Val (temiz) | 445 |
| **Toplam** | **5785** |

## YOLO Label Formatı

```
# class  x_center  y_center  width  height  (normalize 0-1)
0  0.452341  0.318756  0.064200  0.071300
```

- `x_center`, `y_center`: Gerçek artifact bbox merkezinden hesaplanır (yerleştirme merkezinden değil)
- Bbox sadece doku içinde görünen artifact alanından hesaplanır (`em_placed & tissue_roi`)
- 4 pikselden küçük bbox'lar etiketlenmez

## Klasör Yapısı

```
Examples/
├── merge_dataset.py
├── brain_mri_synthetic.py
├── filter_by_cnr.py
├── split_dataset.py
├── augment_dataset.py
├── pck_to_png_knee.py
├── dataset.yaml  # network seçimi sonrası oluşturulacak
├── magnet_pattern.png
├── synthetic_dataset_yolo_filtered/
│   ├── images/
│   ├── labels/
│   └── images_annotated/
├── synthetic_dataset_yolo_split/
│   ├── train/
│   └── val/
└── synthetic_dataset_yolo_augmented/
    ├── train/
    └── val/
```

## Lisans

MIT License