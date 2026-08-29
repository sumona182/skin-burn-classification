# A Comparative Study of Machine Learning and Deep Learning Models for Skin Burn Severity Prognosis.
skin-burn-classification


## Abstract

Accurate assessment of skin burn severity is essential for determining appropriate treatment and improving patient outcomes, yet conventional diagnostic methods often rely on subjective clinical evaluation, leading to inconsistent and variable diagnoses. Recent advancements in Artificial Intelligence (AI), particularly Machine Learning (ML) and Deep Learning (DL), have provided promising solutions for automated and objective burn severity assessment using medical image analysis. This study presents a comparative framework for skin burn severity classification by evaluating a wide range of ML algorithms, including Logistic Regression, Bayesian Logistic Regression, Ridge Regression, Random Forest, Gradient Boosting, Support Vector Machine (SVM), Voted Perceptron, Multi-Class Classifier, Random Subspace, and clustering-based approaches, alongside state-of-the-art DL architectures such as ResNet50, EfficientNet, Xception, InceptionV3, MobileNet, AlexNet, VGG19, and VGG16. The experimental results demonstrate that while traditional machine learning models provide strong baseline performance and ensemble methods improve classification accuracy, deep learning models consistently achieve superior performance by automatically learning complex hierarchical features from burn images without manual feature engineering. The findings further indicate that deep learning techniques are particularly effective when trained on sufficiently large, diverse, and high-quality datasets, enabling more accurate and robust classification of burn severity. Overall, this research highlights the potential of AI-based image analysis, especially deep learning approaches, to enhance the accuracy, reliability, and efficiency of automated skin burn severity assessment, thereby supporting clinical decision-making and improving burn diagnosis.

This repository presents an empirical comparative evaluation of **twenty-two algorithmic frameworks**—spanning fourteen classical machine learning estimators leveraging dimensionality-reduced handcrafted features and eight deep convolutional neural network (CNN) architectures—for multi-class burn severity categorization. Evaluated on a curated, quality-controlled dataset of 15,456 clinical dermatological images, deep transfer learning paradigms consistently outperformed classical feature-engineering approaches. Among all evaluated topologies, **EfficientNet** achieved state-of-the-art performance with a top multi-class test accuracy of **90.87%**, establishing a robust baseline for computer-assisted dermatological triage.

---

## Key Thesis Contributions

* **Standardized Multi-Source Dataset Curation:** Integrated and standardized 15,456 clinical images across four diagnostic classes (*No Sunburn*, *First-Degree*, *Second-Degree*, and *Third-Degree* burns).
* **Automated Data Quality & Segmentation Pipeline:** Developed a dual-stage quality control protocol employing SHA-256 cryptographic hashing for exact deduplication and Laplacian variance thresholding ($\text{Var} < 130$) for automated out-of-focus image rejection. Region-of-Interest (ROI) segmentation was executed via $a^*$-channel thresholding in the CIELAB color space.
* **Handcrafted Feature Engineering vs. Deep Feature Representation:** Benchmarked handcrafted color metrics, Grey-Level Co-occurrence Matrix (GLCM) texture descriptors, and Discrete Wavelet Transform (DWT) frequencies compressed via Principal Component Analysis (PCA at 95% variance) against deep convolutional feature hierarchies.
* **Comprehensive Multi-Model Benchmarking:** Evaluated 14 classical estimators (e.g., Support Vector Machines, Random Forests, Gradient Boosting) alongside 8 deep CNN topologies (e.g., EfficientNet, ResNet50, InceptionV3, Xception, VGG, MobileNet).

---

## Experimental Dataset Architecture

The unified dataset comprises 15,456 high-resolution clinical images partitioned via class-stratified sampling into distinct training, validation, and hold-out test sets using a fixed seed ($Seed = 42$).

|       Diagnostic Class       |        Clinical Description        |    Training (70%)   |    Validation (15%)    |     Testing (15%)   |      Total     |

|         **Class 0**          |     Healthy Skin / No Sunburn      |        2,730        |          585           |        585          |      3,900     |
|         **Class 1**          |     First-Degree (Superficial)     |        2,450        |          525           |        525          |      3,500     |
|         **Class 2**          |  Second-Degree (Partial-Thickness) |        2,800        |          600           |        600          |      4,000     |
|         **Class 3**          |    Third-Degree (Full-Thickness)   |        2,839        |          608           |        613          |      4,060     |
|          **Total**           |        **Combined Corpus**         |      **10,819**     |       **2,318**        |     **2,323**       |    **15,456**  |

---

## Experimental Results & Comparative Performance

Evaluating model generalizability on the **2,323 unseen hold-out test samples** highlights the empirical advantage of end-to-end spatial representation learning over static feature extraction.

### Deep Convolutional Architectures

|   Architecture   |   Test Accuracy (%)   |   Class 0 Acc. (%)   |   Class 1 Acc. (%)   |   Class 2 Acc. (%)   |   Class 3 Acc. (%)   |

| **EfficientNet** |       **90.87**       |      **100.00**      |       **87.70**      |       **89.67**      |       **92.77**      |
| **InceptionV3**  |         90.36         |         98.65        |         84.68        |         91.66        |         90.83        |
|   **ResNet50**   |         89.88         |        100.00        |         81.65        |         92.81        |         89.58        |
|  **XceptionV3**  |         89.88         |         99.00        |         87.00        |         89.00        |         93.00        |
|    **VGG19**     |         84.85         |         97.97        |         77.02        |         80.92        |         92.78        |
|   **MobileNet**  |         84.50         |         99.32        |         81.65        |         77.89        |         69.86        |

### Classical Machine Learning Estimators (PCA Feature Space)

|             Model Estimator               |   Test Accuracy (%)   |    Class 0 Acc. (%)   |   Class 1 Acc. (%)   |    Class 2 Acc. (%)   |    Class 3 Acc. (%)  |

|       **Random Forest Classifier**        |       **66.03**       |        **50.00**      |       **39.92**      |       **77.58**       |       **71.94**      |
|       **Random Subspace Ensemble**        |         64.49         |          28.00        |         33.00        |         82.00         |         70.00        |
| **Sequential Minimal Optimization (SVM)** |         61.21         |          50.00        |         35.00        |         70.00         |         60.00        |
|      **Gradient Boosting Classifier**     |         57.55         |          47.30        |         36.69        |         67.36         |         60.97        |

---

## Methodology Overview

```text
Raw Image Corpus
       │
       ├──► SHA-256 Deduplication Pipeline
       ├──► Blur Rejection (Laplacian Variance < 130)
       └──► CIELAB Transformation & a*-Channel Otsu ROI Masking
                 │
                 ├──► Classical Path: GLCM + Color Metrics + DWT ──► PCA (95%) ──► ML Estimators
                 └──► Deep Transfer Learning Path: Data Augmentation ──► CNN Backbones 

```



## Project Directory Organization

```text
.
├── DL_models/                  # Deep learning notebook implementations
│   ├── alexnet_burn_classification.ipynb
│   ├── efficientnet_burn_classification.ipynb
│   ├── inceptionv3_burn_classification.ipynb
│   ├── mobilenet_burn_classification.ipynb
│   ├── resnet50_burn_classification.ipynb
│   ├── vgg16_burn_classification.ipynb
│   ├── vgg19_burn_classification.ipynb
│   └── xception_burn_classification.ipynb
├── ML_models/                  # Handcrafted feature extraction & classical pipelines
│   ├── gradient_boosting.ipynb
│   ├── logistic_regression.ipynb
│   ├── naive_bayes.ipynb
│   ├── random_forest_classifier.ipynb
│   ├── random_subspace.ipynb
│   └── smo_svm.ipynb
├── Preprocessing/              # Data sanitation scripts
│   ├── laplacian_blur_detector.py
│   ├── roi_segmentation_lab_otsu.py
│   └── sha256_duplicate_remover.py
├── requirements.txt            # System dependencies
└── README.md                   # Thesis documentation

```


## Reproduction & Setup Guide

### Environment Configuration

Ensure Python 3.8+ and an execution environment with NVIDIA CUDA support are configured.

```bash
# Clone the repository
git clone https://github.com/sumona182/skin-burn-classification.git
cd skin-burn-classification

# Install dependency requirements
pip install -r requirements.txt

```

### Preprocessing and Segmentation

To execute the automated image quality screening and ROI extraction pipeline on raw clinical inputs:

```python
from Preprocessing.roi_segmentation_lab_otsu import segment_burn_roi
import cv2

# Load target sample
image = cv2.imread("sample_burn.jpg")

# Process through CIELAB a*-channel thresholding
roi_masked_image, binary_mask = segment_burn_roi(image)

```

---

## Citation

If you utilize this experimental framework, codebase, or dataset structuring methodology in your research, please cite:

```bibtex
@mastersthesis{burn_severity_classification_2026,
  author       = {Samia Sayed Sumona, S.},
  title        = {A Comparative Study of Machine Learning and Deep Learning Models for Skin Burn Severity Prognosis},
  school       = {Department of Computer Science and Engineering},
  year         = {2026},
  type         = {Undergraduation's Thesis}
}

```



## License

This research project and its source code are distributed under the **MIT License**. See the `LICENSE` file for details.
