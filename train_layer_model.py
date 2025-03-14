import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader.data_loaders import get_layered_data_loaders
from model import LayeredInstrumentClassifier
from sklearn.metrics import f1_score

def train_one_epoch_layered(model, train_loader, criterion, optimizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_layered(model, loader, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).int()  # threshold at 0.5
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics = {'f1_micro': f1_micro, 'f1_macro': f1_macro}
    return epoch_loss, metrics

def train_model_layered(epochs=10, lr=1e-3, save_path='best_model_layered.pth', n_layers=2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader, val_loader, num_classes = get_layered_data_loaders('C:\\Users\\trema\\Data Science\\DSCI 410 project\\dataset', batch_size=32, duration=12, n_layers=n_layers)
    
    model = LayeredInstrumentClassifier(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch_layered(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = evaluate_layered(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1-micro: {val_metrics['f1_micro']:.4f} | F1-macro: {val_metrics['f1_macro']:.4f}")
        if val_metrics['f1_micro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_micro']
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved to {save_path} with F1-micro: {best_val_f1:.4f}")

    return model

if __name__ == "__main__":
    train_model_layered(epochs=10, lr=1e-3, save_path='best_model_layered.pth', n_layers=2)
