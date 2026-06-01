import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10MLP(nn.Module):
    """
    Rede de neurónios densa para isolar o efeito da poda em ligações totalmente 
    conectadas (Fully Connected), afetando diretamente as matrizes de pesos.
    """
    def __init__(self, input_dim=3 * 32 * 32, num_classes=10):
        super(CIFAR10MLP, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class CIFAR10VGG(nn.Module):
    """
    Rede convolucional profunda (baseada na VGG-11) ideal para testar a poda 
    estruturada de filtros e canais devido à sua elevada densidade de parâmetros.
    """
    def __init__(self, num_classes=10):
        super(CIFAR10VGG, self).__init__()
        
        # Configuração da VGG-11: números representam canais de saída, 'M' é MaxPool
        # Ajustada para CIFAR-10 para manter mapas de características viáveis (32x32 -> 1x1)
        self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        
        self.features = self._make_layers(self.cfg)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # Classificador linear simplificado para o CIFAR-10 pós-convoluções
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3  # Entrada RGB do CIFAR-10
        
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x), # Crucial para estabilidade antes/pós poda
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
                
        return nn.Sequential(*layers)

def train_model(model, trainloader, valloader, epochs=10, lr=1e-3):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):

        # -----------------
        # TRAIN
        # -----------------
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(trainloader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)

        # -----------------
        # VALIDATION
        # -----------------
        model.eval()

        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in valloader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(valloader)
        accuracy = 100 * correct / total

        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {accuracy:.2f}%"
        )

    return model, train_losses, val_losses, val_accuracies

def evaluate_model(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    y_probs = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # predicted class
            _, predicted = torch.max(outputs, 1)

            # probabilities
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    return y_true, y_pred, y_probs

def compute_metrics(y_true, y_pred):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "balanced_accuracy":
            balanced_accuracy_score(y_true, y_pred),

        "precision_macro":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "precision_weighted":
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "recall_macro":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "recall_weighted":
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "f1_macro":
            f1_score(
                y_true,
                y_pred,
                average="macro"
            ),

        "f1_weighted":
            f1_score(
                y_true,
                y_pred,
                average="weighted"
            ),

        "cohen_kappa":
            cohen_kappa_score(y_true, y_pred),

        "mcc":
            matthews_corrcoef(y_true, y_pred)
    }

    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )
    disp.plot(
        cmap="Blues",
        ax=ax,
        xticks_rotation=45,
        colorbar=False
    )
    plt.title(title)
    plt.show()

def show_predictions(model, loader, n_images=8):
    model.eval()
    images, labels = next(iter(loader))
    images = images[:n_images].to(device)
    labels = labels[:n_images]
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images = images.cpu()
    fig, axes = plt.subplots(1, n_images, figsize=(16,4))
    for i in range(n_images):
        img = images[i]

        # unnormalize
        img = img * 0.5 + 0.5
        img = img.permute(1,2,0)

        axes[i].imshow(img)
        axes[i].axis("off")

        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]

        color = "green" if labels[i] == predicted[i] else "red"

        axes[i].set_title(
            f"T:{true_label}\nP:{pred_label}",
            color=color,
            fontsize=9
        )

    plt.show()