# Knee Cartilage Segmentation with 3D V-Net

This project implements a 3D convolutional neural network (V-Net) for automatic knee cartilage segmentation from MRI scans. The goal is to replace expensive and time-consuming manual segmentation with an automated, scalable approach.

The system supports both low-resolution clinical MRI data and high-resolution 7T MRI data, using transfer learning to improve performance on the high-resolution dataset.

---

## Project Overview

- 3D V-Net architecture for volumetric segmentation
- Binary segmentation (cartilage vs. background)
- Support for low-resolution and high-resolution (7T) MRI datasets
- Transfer learning from low-resolution to high-resolution data
- Custom preprocessing for medical imaging formats (.im, .seg, .hdf5)
- Dice score evaluation for segmentation quality
- GPU memory profiling for training and inference

---

## Project Structure

```
src/
├── data_loaders/
│   ├── knee_dataset.py
│   ├── dataset_7t.py
│   └── extract_7t_volumes.py
│
├── models/
│   └── vnet.py
│
├── training/
│   ├── train.py
│   └── train_7t.py
│
├── inference/
│   ├── infer_knee.py
│   └── infer_7t.py
│
└── scripts/
    └── *.sbatch
```

---

## Approach

### Low-Resolution Training
- Trained V-Net on labeled MRI volumes
- Converted multi-class segmentation into binary labels:
  - Cartilage = 1
  - Background = 0
- Removed non-relevant anatomical structures (e.g., meniscus)

### High-Resolution (7T) Pipeline
- Loaded raw HDF5 MRI data and reconstructed 3D volumes
- Remapped complex labels into binary cartilage segmentation
- Trained on full volumes due to limited dataset size

### Transfer Learning
- Initialized high-resolution model using pretrained low-resolution weights
- Optional warmup phase freezes layers before full fine-tuning

### Loss Function
Combination of cross-entropy and Dice loss:

Loss = α * CrossEntropy + β * DiceLoss

This improves performance on highly imbalanced medical segmentation tasks.

---

## Training

### Low-Resolution
```
python train.py --data data --save work/vnet.lowres
```

### High-Resolution (7T)
```
python train_7t.py   --data path/to/7T_data.hdf5   --pretrained path/to/lowres_model.pth.tar   --save work/vnet.7t
```

---

## Inference

### Low-Resolution
```
python infer_knee.py   --ckpt model.pth.tar   --im sample.im   --out pred.seg
```

### High-Resolution
```
python infer_7t.py   --ckpt model.pth.tar   --im volume.h5   --seg ground_truth.seg   --dice   --out pred.h5
```

---

## Results

The model was evaluated using the Dice coefficient.

- Low-resolution V-Net (baseline): 0.88 Dice
- High-resolution V-Net (with transfer learning): 0.86 Dice

These results demonstrate that transfer learning enables strong performance on high-resolution MRI data despite limited training samples.

---

## Technical Details

### Model
- 3D V-Net (encoder-decoder architecture)
- Residual and skip connections
- Outputs voxel-wise predictions

### Input / Output
- Input: (1, D, H, W) MRI volume
- Output: (D, H, W) binary segmentation

### Data Formats
- .im: MRI volume (HDF5)
- .seg: segmentation labels
- .hdf5: raw 7T dataset

### Dependencies
- NumPy
- PyTorch
- h5py
- matplotlib

---

## Motivation

Manual cartilage segmentation is:
- Time-consuming
- Expensive
- Subjective

This project demonstrates how deep learning can automate the process, improve consistency, and enable scalable analysis of knee osteoarthritis using MRI data.

---

## Author

James Stringham  
