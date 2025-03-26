import yaml
import torch
from torch import optim
from models.unet import NUCNet
from data.loader import get_loaders
from utils.metrics import MetricCalculator

def train(config_path='configs/default.yaml'):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    model = NUCNet(features=config['features']).to(config['device'])
    train_loader, val_loader = get_loaders(config)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    metric_calculator = MetricCalculator(config['device'])
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        val_metrics = validate(model, val_loader, metric_calculator)
        print(f"Epoch {epoch+1} | Val SSIM: {val_metrics['SSIM']:.4f}")