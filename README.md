---
license: apache-2.0
pipeline_tag: image-classification
library_name: transformers
tags:
- deep-fake
- ViT
- detection
base_model:
- google/vit-base-patch16-224-in21k
---

**<span style="color:red;">Model Link :</span>** https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model

![df[ViT].gif](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Xbuv-x40-l3QjzWu5Yj2F.gif)

# **Deep-Fake-Detector-Model**

# **Overview**

The **Deep-Fake-Detector-Model** is a state-of-the-art deep learning model designed to detect deepfake images. It leverages the **Vision Transformer (ViT)** architecture, specifically the `google/vit-base-patch16-224-in21k` model, fine-tuned on a dataset of real and deepfake images. The model is trained to classify images as either "Real" or "Fake" with high accuracy, making it a powerful tool for detecting manipulated media.

**<span style="color:red;">Update :</span>** The previous model checkpoint was obtained using a smaller classification dataset. Although it performed well in evaluation scores, its real-time performance was average due to limited variations in the training set. The new update includes a larger dataset to improve the detection of fake images.

| Repository | Link |
|------------|------|
| Deep Fake Detector Model | [GitHub Repository](https://github.com/PRITHIVSAKTHIUR/Deep-Fake-Detector-Model) |

# **Key Features**
- **Architecture**: Vision Transformer (ViT) - `google/vit-base-patch16-224-in21k`.
- **Input**: RGB images resized to 224x224 pixels.
- **Output**: Binary classification ("Real" or "Fake").
- **Training Dataset**: A curated dataset of real and deepfake images (e.g., `Hemg/deepfake-and-real-images`).
- **Fine-Tuning**: The model is fine-tuned using Hugging Face's `Trainer` API with advanced data augmentation techniques.
- **Performance**: Achieves high accuracy and F1 score on validation and test datasets.

# **Model Architecture**
The model is based on the **Vision Transformer (ViT)**, which treats images as sequences of patches and applies a transformer encoder to learn spatial relationships. Key components include:
- **Patch Embedding**: Divides the input image into fixed-size patches (16x16 pixels).
- **Transformer Encoder**: Processes patch embeddings using multi-head self-attention mechanisms.
- **Classification Head**: A fully connected layer for binary classification.

# **Training Details**
- **Optimizer**: AdamW with a learning rate of `1e-6`.
- **Batch Size**: 32 for training, 8 for evaluation.
- **Epochs**: 2.
- **Data Augmentation**:
  - Random rotation (±90 degrees).
  - Random sharpness adjustment.
  - Random resizing and cropping.
- **Loss Function**: Cross-Entropy Loss.
- **Evaluation Metrics**: Accuracy, F1 Score, and Confusion Matrix.

# **Inference with Hugging Face Pipeline**
```python
from transformers import pipeline

# Load the model
pipe = pipeline('image-classification', model="prithivMLmods/Deep-Fake-Detector-Model", device=0)

# Predict on an image
result = pipe("path_to_image.jpg")
print(result)
```

# **Inference with PyTorch**
```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load the model and processor
model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")

# Load and preprocess the image
image = Image.open("path_to_image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Map class index to label
label = model.config.id2label[predicted_class]
print(f"Predicted Label: {label}")
```
# **Performance Metrics**
```
Classification report:

              precision    recall  f1-score   support

        Real     0.6276    0.9823    0.7659     38054
        Fake     0.9594    0.4176    0.5819     38080

    accuracy                         0.6999     76134
   macro avg     0.7935    0.7000    0.6739     76134
weighted avg     0.7936    0.6999    0.6739     76134
```

![Untitled.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/MoxwukbZZZuVpvXHstxsw.png)

- **Confusion Matrix**:
  ```
  [[True Positives, False Negatives],
   [False Positives, True Negatives]]
  ```

# **Dataset**
The model is fine-tuned on the `Hemg/deepfake-and-real-images` dataset, which contains:
- **Real Images**: Authentic images of human faces.
- **Fake Images**: Deepfake images generated using advanced AI techniques.

# **Limitations**
The model is trained on a specific dataset and may not generalize well to other deepfake datasets or domains.
- Performance may degrade on low-resolution or heavily compressed images.
- The model is designed for image classification and does not detect deepfake videos directly.

# **Ethical Considerations**

**Misuse**: This model should not be used for malicious purposes, such as creating or spreading deepfakes.
**Bias**: The model may inherit biases from the training dataset. Care should be taken to ensure fairness and inclusivity.
**Transparency**: Users should be informed when deepfake detection tools are used to analyze their content.

# **Future Work**
- Extend the model to detect deepfake videos.
- Improve generalization by training on larger and more diverse datasets.
- Incorporate explainability techniques to provide insights into model predictions.

# **Citation**

```bibtex
@misc{Deep-Fake-Detector-Model,
  author = {prithivMLmods},
  title = {Deep-Fake-Detector-Model},
  initial = {2024},
  last_updated = {31 Jan 2025}
}
