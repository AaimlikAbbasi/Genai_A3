import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from PIL import Image
import gradio as gr

# Define Model Architectures
class VisionTransformer(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 embed_dim=128, 
                 num_layers=6, 
                 num_heads=4, 
                 mlp_dim=256, 
                 image_size=32, 
                 patch_size=4, 
                 num_classes=10, 
                 dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, 
                                     stride=patch_size)

        # Positional embedding
        self.position_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=mlp_dim, 
                                                   dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, embed_dim, num_patches_sqrt, num_patches_sqrt]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]

        # Add positional embedding
        x = x + self.position_embed  # [batch_size, num_patches, embed_dim]

        # Transformer expects [sequence_length, batch_size, embed_dim]
        x = x.transpose(0, 1)  # [num_patches, batch_size, embed_dim]

        # Transformer encoder
        x = self.transformer(x)  # [num_patches, batch_size, embed_dim]

        # Aggregate features (mean pooling)
        x = x.mean(dim=0)  # [batch_size, embed_dim]

        # Classification head
        logits = self.cls_head(x)  # [batch_size, num_classes]

        return logits

class HybridCNNMLP(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=10):
        super(HybridCNNMLP, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  # [batch, 64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [batch, 64, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [batch, 128, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [batch, 128, 8, 8]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # [batch, 256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [batch, 256, 4, 4]
        )
        
        # MLP for classification
        self.mlp = nn.Sequential(
            nn.Flatten(),  # [batch, 256*4*4]
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.cnn(x)
        logits = self.mlp(features)
        return logits

class ResNetTransferLearning(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetTransferLearning, self).__init__()
        # Load pretrained ResNet50 with updated weights parameter
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Modify the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        logits = self.resnet(x)
        return logits

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the test dataset and calculates metrics.
    
    Parameters:
    - model: The trained model to evaluate.
    - dataloader: DataLoader for the test dataset.
    - device: Device to perform computations on.
    
    Returns:
    - metrics: Dictionary containing accuracy, precision, recall, and F1-score.
    - all_preds: List of all predicted labels.
    - all_labels: List of all true labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    return metrics, all_preds, all_labels

def plot_confusion_matrix(true_labels, preds, classes, model_name):
    """
    Plots and displays the confusion matrix.
    
    Parameters:
    - true_labels: List of true class labels.
    - preds: List of predicted class labels.
    - classes: Tuple of class names.
    - model_name: Name of the model (for title).
    """
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()

def display_sample_predictions(model, dataloader, classes, num_samples=16):
    """
    Displays a grid of sample predictions with true and predicted labels.
    
    Parameters:
    - model: The trained model for inference.
    - dataloader: DataLoader for the test dataset.
    - classes: Tuple of class names.
    - num_samples: Number of samples to display.
    """
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 12))
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                if images_shown >= num_samples:
                    break
                img = images[i].cpu()
                img = img * 0.5 + 0.5  # Unnormalize
                npimg = img.numpy()
                plt.subplot(int(np.sqrt(num_samples)), int(np.sqrt(num_samples)), images_shown+1)
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                true_label = classes[labels[i]]
                pred_label = classes[preds[i]]
                color = 'green' if pred_label == true_label else 'red'
                plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
                plt.axis('off')
                images_shown += 1
            if images_shown >= num_samples:
                break
    plt.tight_layout()
    plt.show()

def predict_image(image):
    """
    Predicts the class of an input image using all three models.
    
    Parameters:
    - image: Input image in PIL format.
    
    Returns:
    - Dictionary containing predictions from ResNet, ViT, and Hybrid CNN + MLP.
    """
    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Get predictions from ResNet
    with torch.no_grad():
        resnet_output = resnet_model(image)
        resnet_pred = torch.argmax(resnet_output, dim=1).item()
        resnet_confidence = torch.softmax(resnet_output, dim=1)[0][resnet_pred].item()
    
    # Get predictions from ViT
    with torch.no_grad():
        vit_output = vit_model(image)
        vit_pred = torch.argmax(vit_output, dim=1).item()
        vit_confidence = torch.softmax(vit_output, dim=1)[0][vit_pred].item()
    
    # Get predictions from Hybrid CNN + MLP
    with torch.no_grad():
        hybrid_output = hybrid_model(image)
        hybrid_pred = torch.argmax(hybrid_output, dim=1).item()
        hybrid_confidence = torch.softmax(hybrid_output, dim=1)[0][hybrid_pred].item()
    
    # Prepare results
    results = {
        "ResNet Transfer Learning": f"{classes[resnet_pred]} ({resnet_confidence*100:.2f}%)",
        "Vision Transformer (ViT)": f"{classes[vit_pred]} ({vit_confidence*100:.2f}%)",
        "Hybrid CNN + MLP": f"{classes[hybrid_pred]} ({hybrid_confidence*100:.2f}%)"
    }
    
    return results

if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the models
    resnet_model = ResNetTransferLearning().to(device)
    vit_model = VisionTransformer().to(device)
    hybrid_model = HybridCNNMLP().to(device)

    # Paths to the saved model files
    resnet_model_path = 'D:\\Genai_a2\\models\\resnet-transfer-cifar10-best.pt'
    vit_model_path = 'D:\\Genai_a2\\models\\vit-layer6-32-cifar10-best.pt'
    hybrid_model_path = 'D:\\Genai_a2\\models\\hybrid-cnn-mlp-cifar10-best.pt'

    # Load the saved state dictionaries
    resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
    vit_model.load_state_dict(torch.load(vit_model_path, map_location=device))
    hybrid_model.load_state_dict(torch.load(hybrid_model_path, map_location=device))

    # Set models to evaluation mode
    resnet_model.eval()
    vit_model.eval()
    hybrid_model.eval()

    print("All models loaded and set to evaluation mode.")

    # Define transformations for test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Ensure same normalization as training
    ])

    # Load CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    print(f"Test Dataset Size: {len(test_loader.dataset)}")

    # Define CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Evaluate models
    resnet_metrics, resnet_preds, resnet_labels = evaluate_model(resnet_model, test_loader, device)
    vit_metrics, vit_preds, vit_labels = evaluate_model(vit_model, test_loader, device)
    hybrid_metrics, hybrid_preds, hybrid_labels = evaluate_model(hybrid_model, test_loader, device)

    # Print metrics
    print("ResNet with Transfer Learning Metrics:")
    for metric, value in resnet_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\n")

    print("Vision Transformer (ViT) Metrics:")
    for metric, value in vit_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\n")

    print("Hybrid CNN + MLP Metrics:")
    for metric, value in hybrid_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\n")

    # Plot Confusion Matrices for All Models
    plot_confusion_matrix(resnet_labels, resnet_preds, classes, "ResNet with Transfer Learning")
    plot_confusion_matrix(vit_labels, vit_preds, classes, "Vision Transformer (ViT)")
    plot_confusion_matrix(hybrid_labels, hybrid_preds, classes, "Hybrid CNN + MLP")

    # Display Sample Predictions for All Models
    print("ResNet with Transfer Learning - Sample Predictions:")
    display_sample_predictions(resnet_model, test_loader, classes, num_samples=16)

    print("Vision Transformer (ViT) - Sample Predictions:")
    display_sample_predictions(vit_model, test_loader, classes, num_samples=16)

    print("Hybrid CNN + MLP - Sample Predictions:")
    display_sample_predictions(hybrid_model, test_loader, classes, num_samples=16)

    # Define the Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# CIFAR-10 Image Classification")
        gr.Markdown("### Upload an image from the CIFAR-10 test set to see predictions from ResNet, ViT, and Hybrid CNN + MLP models.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                predict_btn = gr.Button("Predict")
            with gr.Column():
                resnet_output = gr.Textbox(label="ResNet Transfer Learning Prediction")
                vit_output = gr.Textbox(label="Vision Transformer (ViT) Prediction")
                hybrid_output = gr.Textbox(label="Hybrid CNN + MLP Prediction")
        
        predict_btn.click(fn=predict_image, inputs=image_input, outputs=[resnet_output, vit_output, hybrid_output])
    
    # Launch the Gradio app
    demo.launch()
