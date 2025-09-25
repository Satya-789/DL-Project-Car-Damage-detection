🚗 Vehicle Damage Detection (Transfer Learning with PyTorch)
📌 Project Overview

This project develops a Deep Learning image classification system to automate vehicle damage assessment, focusing on damage location and severity.

It leverages Transfer Learning with a pre-trained ResNet or VGG model in PyTorch, significantly reducing training time while improving accuracy.

All processes, from data handling to training and evaluation, are contained in:
📓 damage_prediction.ipynb

🎯 Objectives

🔹 Classification → Identify damage location (front, rear, side) and severity (minor, moderate, severe).

🔹 Transfer Learning → Use pre-trained CNNs (ResNet/VGG) and fine-tune for the specific task.

🔹 PyTorch Implementation → Build a complete, end-to-end deep learning pipeline.

🔹 Evaluation → Visualize performance with a Confusion Matrix.

🛠 Methodology

The project follows a standard computer vision pipeline, enhanced with transfer learning techniques:

1️⃣ Data Preparation & Augmentation

Data Structure: Images organized by class in subfolders (e.g., data/train/front_damage/).

Transformations: Resize, normalize, and augment images (rotations, flips) using torchvision.transforms.

Data Loading: Use ImageFolder and DataLoader for efficient batching.

2️⃣ Model Architecture (Transfer Learning)

Load a pre-trained CNN (e.g., ResNet50 or VGG16) from torchvision.models.

Replace the classifier head with a new fully connected layer matching the number of output classes.

Optionally freeze base layers and train only the new classifier initially.

3️⃣ Training & Evaluation

Device: GPU (cuda) if available.

Loss Function: Cross-Entropy Loss (nn.CrossEntropyLoss) for multi-class classification.

Optimizer: Adam (torch.optim.Adam).

Metrics: Track training/validation loss and accuracy per epoch.

Confusion Matrix: Visualize prediction performance across all classes.

4️⃣ Model Persistence

Save the trained model with torch.save(model.state_dict(), ...) for deployment or future use.
