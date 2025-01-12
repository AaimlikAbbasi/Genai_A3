# train.py
import torch
import torch.optim as optim
from models.vit import VisionTransformer
from models.hybrid import HybridCNNMLP
from models.resnet import ResNetTransferLearning
from preprocessing import load_data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import torch.nn as nn


# Load Data
trainloader, testloader = load_data()

# Model Initialization
model = VisionTransformer()  # Change to HybridCNNMLP() or ResNetTransferLearning() for other models
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training Loop
def train_model():
    model.train()
    for epoch in range(10):  # You can adjust the number of epochs
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/10], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        print(f"Time taken for epoch: {time.time() - start_time:.2f} seconds")
        # Save model after training or after each epoch
        torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(epoch+1))


train_model()
