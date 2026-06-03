#!/usr/bin/env python3
"""ML Training Pipeline for AMD GPUs"""
import torch, torch.nn as nn, torch.optim as optim, argparse, time
from torchvision import models

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, args.classes)
    model = model.to(device)
    
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"Training {args.model} on {device} for {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.train()
        x = torch.randn(args.batch, 3, 224, 224, device=device)
        y = torch.randint(0, args.classes, (args.batch,), device=device)
        
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), f"{args.model}_final.pth")
    print(f"Saved to {args.model}_final.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--classes", type=int, default=1000)
    train(p.parse_args())
