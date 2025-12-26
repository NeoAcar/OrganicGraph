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
import json

def count_parameters(model):
    """
    Counts parameters of the model, broken down by main modules.
    Returns:
        total_params (int): Total number of trainable parameters.
        breakdown (dict): Dictionary mapping module names to parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = {}
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        breakdown[name] = params
        
    # Handle any parameters directly in the model (not in children modules) if any
    # This is rare for the current architectures but good for completeness
    children_params = sum(breakdown.values())
    if total_params > children_params:
        breakdown['other'] = total_params - children_params
        
    return total_params, breakdown

def plot_parameter_distribution(breakdown, total_params, model_type):
    """
    Plots a bar chart of the parameter distribution.
    """
    modules = list(breakdown.keys())
    counts = list(breakdown.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modules, counts, color='skyblue')
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:,}',
                 ha='center', va='bottom')
                 
    plt.ylabel('Number of Parameters')
    plt.title(f'{model_type} Model Parameter Breakdown (Total: {total_params:,})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{model_type}_param_dist.png')
    print(f"Parameter distribution plot saved to {model_type}_param_dist.png")

def train_epoch(model, loader, optimizer, criterion, device, mode):
    model.train()
    total_loss = 0
    total_mae = 0
    
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
        total_mae += torch.abs(pred - target).sum().item()
        
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)

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
    mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
    
    return total_loss / len(loader.dataset), rmse, mae

def plot_metrics(history, model_type):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{model_type} - Loss Curve')
    plt.legend()
    
    # RMSE Plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_rmse'], label='Val RMSE', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title(f'{model_type} - Validation RMSE')
    plt.legend()
    
    # MAE Plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_mae'], label='Train MAE')
    plt.plot(epochs, history['val_mae'], label='Val MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title(f'{model_type} - Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_type}_learning_curves.png')
    print(f"Plots saved to {model_type}_learning_curves.png")

def save_config(args, filename, extra_info=None):
    config = vars(args)
    if extra_info:
        config.update(extra_info)
        
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Train Melting Point Prediction Model')
    parser.add_argument('--model_type', type=str, required=True, choices=['graph', 'sequence'], help='Model type: graph or sequence')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_scheduler', action='store_true', help='Use Cosine Annealing Scheduler')
    
    # Model Architecture Hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension (d_model for Transformer)')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Transformer feedforward dimension (Sequence model only)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data (Moved before config save to ensure tokenizer is ready if needed, though mostly indep)
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
        model = GATModel(
            node_in_dim, 
            edge_in_dim, 
            hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout
        )
    else:
        model = SmilesTransformer(
            vocab_size=len(tokenizer), 
            d_model=args.hidden_dim, 
            num_layers=args.num_layers, 
            nhead=args.heads,
            dim_feedforward=args.dim_feedforward,
            max_len=128, # Assuming fixed max_len for now, could be arg
            dropout=args.dropout
        )
        
    model = model.to(device)
    
    # --- Parameter Counting & Logging ---
    total_params, param_breakdown = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print("Parameter Breakdown:")
    for name, count in param_breakdown.items():
        print(f"  - {name}: {count:,}")
        
    plot_parameter_distribution(param_breakdown, total_params, args.model_type)
    
    # Save Config with Params
    config_file = f'{args.model_type}_config.json'
    save_config(args, config_file, extra_info={'total_params': total_params, 'param_breakdown': param_breakdown})
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Scheduler
    scheduler = None
    if args.use_scheduler:
        # T_max is the number of epochs until the first restart. Here we just set it to total epochs.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print("Using Cosine Annealing Scheduler")
        
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []}
    
    # Logging Setup
    log_file = f'{args.model_type}_training_log.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train MAE', 'Val Loss', 'Val RMSE', 'Val MAE', 'LR'])
    
    for epoch in range(args.epochs):
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device, args.model_type)
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, device, args.model_type)
        
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | LR: {current_lr:.6f}")
        
        # Determine whether to save
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), f"best_{args.model_type}_model.pt")
        
        # Update History
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        
        # Write to log file
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_mae, val_loss, val_rmse, val_mae, current_lr])
            
    print("Training Complete.")
    
    # Plotting
    plot_metrics(history, args.model_type)
    
    # Final Test
    model.load_state_dict(torch.load(f"best_{args.model_type}_model.pt", map_location=device))
    test_loss, test_rmse, test_mae = evaluate(model, test_loader, criterion, device, args.model_type)
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

if __name__ == '__main__':
    main()
