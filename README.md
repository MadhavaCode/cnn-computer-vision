# Convolution Neural Network

Convolutional Neural Networks from first principles to industry-standard architectures.

## 🏗 Architectures

| Architecture | Implementation |
|---|---|
| Custom CNN | Derived from scratch — feature map formula `(N-F+2P)/S+1` |
| LeNet | 5×5 kernels, 2 conv-pool blocks |
| AlexNet | 96→256→384→384→256 filters, 3 FC layers |
| VGG16 / VGG19 | 13/16 conv layers with 3×3 kernels |
| GoogLeNet | Inception-style deep architecture |
| ResNet50 | Keras Applications (frozen) + custom residual block from scratch |

## 🔑 Key Projects
- **Cats vs Dogs** — binary CNN with `ImageDataGenerator` augmentation (Kaggle dataset)
- **Brain Tumor MRI** — 6-architecture comparison on 5,600 MRI images across 4 classes (T4 GPU)

## ⚡ Optimization
Batch GD vs SGD vs Mini-Batch GD convergence curves + Exponentially Weighted Moving Average (EWMA).

## 🛠 Tech Stack
`Python` · `TensorFlow` · `Keras` · `OpenCV` · `NumPy` · `Pandas` · `kagglehub`

## 👤 Author
**Madhava Narra** — [@MadhavaCode](https://github.com/MadhavaCode) · Built Feb–May 2025
