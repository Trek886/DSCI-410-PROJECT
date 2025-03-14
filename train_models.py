import torch
import torch.nn as nn
import torch.optim as optim
from data_loader.data_loaders import get_data_loaders
from model.model import InstrumentClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_one_epoch(model, train_loader, criterion, optimizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate(model, loader, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average=None, labels=[0,1,2,3])
    rec = recall_score(all_labels, all_preds, average=None, labels=[0,1,2,3])
    f1 = f1_score(all_labels, all_preds, average=None, labels=[0,1,2,3])

    metrics = {
        'accuracy': acc,
        'precision_per_class': prec,
        'recall_per_class': rec,
        'f1_per_class': f1
    }
    return epoch_loss, metrics

def train_model(epochs=10, lr=1e-3, save_path='best_model.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader, val_loader = get_data_loaders(base_path='C:\\Users\\trema\\Data Science\\DSCI 410 project\\dataset', batch_size=32, duration=12)

    model = InstrumentClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = evaluate(model, val_loader, criterion)

        val_acc = val_metrics['accuracy']
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        print(f"  Val Precision: {val_metrics['precision_per_class']}")
        print(f"  Val Recall:    {val_metrics['recall_per_class']}")
        print(f"  Val F1:        {val_metrics['f1_per_class']}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved to {save_path} with val_acc={best_val_acc:.4f}")

    return model

if __name__ == "__main__":
    train_model(epochs=10, lr=1e-3, save_path='best_model.pth')
