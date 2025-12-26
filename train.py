import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.utils.data import DataLoader as SeqDataLoader
from dataset import MeltingPointDataset, SmilesTokenizer
from models import GATModel, SmilesTransformer
import argparse
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import csv

def train_epoch(model, loader, optimizer, criterion, device, mode):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        if mode == 'graph':
            batch = batch.to(device)
            target = batch.y
            pred = model(batch).squeeze()
        else:
            inputs, target = batch
            inputs, target = inputs.to(device), target.to(device)
            pred = model(inputs)
            
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * target.size(0)
        
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device, mode):
    model.eval()
    total_loss = 0
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if mode == 'graph':
                batch = batch.to(device)
                target = batch.y
                pred = model(batch).squeeze()
            else:
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)
                pred = model(inputs)
                
            loss = criterion(pred, target)
            total_loss += loss.item() * target.size(0)
            
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
    rmse = np.sqrt(np.mean((np.array(preds) - np.array(targets))**2))
    return total_loss / len(loader.dataset), rmse

def plot_metrics(history, model_type):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{model_type} - Loss Curve')
    plt.legend()
    
    # RMSE Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_rmse'], label='Val RMSE', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title(f'{model_type} - Validation RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_type}_learning_curves.png')
    print(f"Plots saved to {model_type}_learning_curves.png")

def main():
    parser = argparse.ArgumentParser(description='Train Melting Point Prediction Model')
    parser.add_argument('--model_type', type=str, required=True, choices=['graph', 'sequence'], help='Model type: graph or sequence')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    train_path = os.path.join(args.data_dir, 'train.csv')
    
    full_dataset = None
    tokenizer = None
    
    if args.model_type == 'sequence':
        tokenizer = SmilesTokenizer()
    
    full_dataset = MeltingPointDataset(train_path, tokenizer=tokenizer, mode=args.model_type)
    
    # Split Data
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    
    if args.model_type == 'graph':
        train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = SeqDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=None)
        val_loader = SeqDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = SeqDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
    # Model Setup
    model = None
    if args.model_type == 'graph':
        sample_data = full_dataset[0]
        node_in_dim = sample_data.x.shape[1]
        edge_in_dim = sample_data.edge_attr.shape[1]
        model = GATModel(node_in_dim, edge_in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    else:
        model = SmilesTransformer(vocab_size=len(tokenizer), d_model=128, num_layers=args.num_layers, max_len=128)
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    
    # Logging Setup
    log_file = f'{args.model_type}_training_log.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val RMSE'])
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.model_type)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, device, args.model_type)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")
        
        # Determine whether to save
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), f"best_{args.model_type}_model.pt")
        
        # Update History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        
        # Write to log file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_rmse])
            
    print("Training Complete.")
    
    # Plotting
    plot_metrics(history, args.model_type)
    
    # Final Test
    model.load_state_dict(torch.load(f"best_{args.model_type}_model.pt", map_location=device))
    test_loss, test_rmse = evaluate(model, test_loader, criterion, device, args.model_type)
    print(f"Test RMSE: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
