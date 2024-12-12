# Fashion-MNIST Classification: Baseline vs Data Augmentation

## Project Overview
This project evaluates the performance of a Convolutional Neural Network (CNN) for image classification on the Fashion-MNIST dataset. Two models were trained and compared:
1. A baseline CNN model without any data augmentation.
2. A model using real-time data augmentation to improve robustness and generalization.

## Dataset Overview
Fashion-MNIST is a dataset of 70,000 grayscale images (28x28 pixels) of 10 fashion categories:
- **Training Samples**: 60,000
- **Testing Samples**: 10,000
- Each category includes examples such as T-shirts, trousers, and shoes.

Dataset source: [Fashion-MNIST by Zalando Research](https://github.com/zalandoresearch/fashion-mnist).

## Project Objectives
1. Train and evaluate a **baseline CNN model** on the Fashion-MNIST dataset.
2. Implement data augmentation techniques and train an **augmented model**.
3. Compare the performance of both models in terms of accuracy, loss, and generalization.

---

## Implementation Details

### 1. Baseline Model
#### Architecture
- **Convolutional Layers**: Extract spatial features using 2D kernels with ReLU activation.
- **Max-Pooling Layers**: Downsample feature maps to reduce spatial dimensions.
- **Flatten Layer**: Convert feature maps into a vector for dense layers.
- **Fully Connected Layers**: Map features to class probabilities.
- **Softmax Output**: Converts logits into probabilities.

#### Training Configuration
- **Optimizer**: Adam Optimizer
- **Loss Function**: Sparse Categorical Crossentropy
- **Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**: Default Adam settings

---

### 2. Data-Augmented Model
To enhance the model's robustness, the following real-time transformations were applied:
- **Random Rotations**: Rotated images within a specified angle range.
- **Horizontal Flips**: Mirrored images along the vertical axis.
- **Shifts**: Random translations in horizontal and vertical directions.
- **Normalization**: Scaled pixel values to the range [0, 1].

#### Augmentation Implementation
Augmentation was applied during training using TensorFlow's `ImageDataGenerator`.

---

## Training and Results

### 1. Baseline Model
- **Final Training Accuracy**: 89.52%
- **Final Validation Accuracy**: 88.11%
- **Final Training Loss**: 0.2987
- **Final Validation Loss**: 0.3564

**Observations**:
- The baseline model showed strong performance with rapid convergence.
- Accuracy varied slightly between simpler classes (e.g., T-shirts) and more complex ones (e.g., Dresses).

---

### 2. Augmented Model
- **Final Training Accuracy**: 85.34%
- **Final Validation Accuracy**: 86.72%
- **Final Training Loss**: 0.3951
- **Final Validation Loss**: 0.3667

**Observations**:
- Data augmentation slowed convergence but improved generalization.
- The model performed better on test images with slight distortions or variations.

---

## Results Comparison

| Metric                    | Baseline Model | Augmented Model |
|---------------------------|----------------|-----------------|
| Final Training Accuracy   | 89.52%         | 85.34%          |
| Final Validation Accuracy | 88.11%         | 86.72%          |
| Final Training Loss       | 0.2987         | 0.3951          |
| Final Validation Loss     | 0.3564         | 0.3667          |

**Key Insights**:
- The baseline model achieved higher accuracy due to the lack of additional noise in the data.
- The augmented model displayed better generalization, especially on distorted or unseen data.

---

## How to Run

### Prerequisites
Ensure the following are installed:
- Python 3.7+
- TensorFlow/Keras
- NumPy
- Matplotlib

### Steps
1. Clone the repository:
   ```bash
   git clone 
