import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from ukko.core import DualAttentionClassifier, visualize_predictions
from ukko.data import SineWaveDataset

def test_dual_attention_classifier(n_epochs = 10):
    # Parameters
    batch_size = 32
    n_features = 10
    time_steps = 50
    n_classes = 3  # For example, classifying into 3 categories
    d_model = 64
    n_heads = 4

    # Create smaller datasets
    train_dataset = SineWaveDataset(
        n_samples=64,
        sequence_length=time_steps,
        n_features=n_features
    )
    test_dataset = SineWaveDataset(
        n_samples=32,
        sequence_length=time_steps,
        n_features=n_features
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualAttentionClassifier(
        n_features=n_features,
        time_steps=time_steps,
        n_classes=n_classes,
        d_model=d_model,
        n_heads=n_heads
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_losses = []

    print("Starting training...")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Create dummy class labels (for testing purposes)
            # Randomly assign classes 0, 1, or 2 to each feature
            target = torch.randint(0, n_classes, (data.size(0), n_features)).to(device)
            
            # Only print every 5 batches
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx} - Data shape: {data.shape}, Target shape: {target.shape}")

            optimizer.zero_grad()
            output, _, _ = model(data)
            loss = criterion(output.view(-1, n_classes), target.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}')

    print("Training complete!")

    #Plot training curve
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.show()

    # Test the model
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        output, feat_attn, time_attn = model(data)
        
        # Visualize attention weights
        fig2 = plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        feat_attn_avg = feat_attn.mean(dim=(0, 1)).cpu()
        plt.imshow(feat_attn_avg, aspect='auto')
        plt.title('Feature Attention')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        time_attn_avg = time_attn.mean(dim=(0, 1)).cpu()
        plt.imshow(time_attn_avg, aspect='auto')
        plt.title('Time Attention')
        plt.colorbar()
        
        plt.tight_layout()
        #plt.show()

    return model, train_losses, fig1, fig2

#if __name__ == "__main__":
#    test_dual_attention_classifier()
