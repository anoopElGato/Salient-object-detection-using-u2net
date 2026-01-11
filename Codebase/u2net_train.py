import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2

from s_measure import SaliencyMetrics


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Dice Loss"""
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()

    def dice_loss(self, pred, target, smooth=1e-5):
        """Dice loss"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return 1 - dice

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)

        return self.bce_weight * bce + self.dice_weight * dice


class MultiScaleLoss(nn.Module):
    """Multi-scale loss for U2-Net"""
    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        self.criterion = BCEDiceLoss()

    def forward(self, outputs, target):
        loss = 0
        for output in outputs:
            loss += self.criterion(output, target)

        return loss


def validate_epoch_with_s_measure(model, val_loader, criterion, device):
    """
    Validate for one epoch
    """
    model.eval()

    running_loss = 0.0
    running_mae = 0.0
    running_f_measure = 0.0
    running_iou = 0.0
    running_s_measure = 0.0
    running_sr = 0.0
    running_so = 0.0

    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Compute metrics
            running_mae += SaliencyMetrics.mae(outputs[0], masks)

            f_measure, _, _ = SaliencyMetrics.f_measure(outputs[0], masks)
            running_f_measure += f_measure

            iou = SaliencyMetrics.iou(outputs[0], masks)
            running_iou += iou

            # Compute S-measure
            s_measure, sr, so = SaliencyMetrics.s_measure(outputs[0], masks)
            running_s_measure += s_measure
            running_sr += sr
            running_so += so

    avg_loss = running_loss / len(val_loader)
    avg_mae = running_mae / len(val_loader)
    avg_f_measure = running_f_measure / len(val_loader)
    avg_iou = running_iou / len(val_loader)
    avg_s_measure = running_s_measure / len(val_loader)
    avg_sr = running_sr / len(val_loader)
    avg_so = running_so / len(val_loader)

    return {
        'loss': avg_loss,
        'mae': avg_mae,
        'f_measure': avg_f_measure,
        'iou': avg_iou,
        's_measure': avg_s_measure,
        'sr': avg_sr,
        'so': avg_so,
    }


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    running_mae = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        mae = SaliencyMetrics.mae(outputs[0], masks)
        running_mae += mae

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'mae': running_mae / (batch_idx + 1)
        })

    avg_loss = running_loss / len(train_loader)
    avg_mae = running_mae / len(train_loader)

    return avg_loss, avg_mae


def train_u2net(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda', save_dir='checkpoints'):
    """
    Main training pipeline
    """
    os.makedirs(save_dir, exist_ok=True)

    # Loss and optimizer
    criterion = MultiScaleLoss()
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training history
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_f_measure': [],
        'val_iou': [],
        'val_s_measure': [],
        'val_sr': [],
        'val_so': [],
    }

    best_f_measure = 0.0

    print("Starting training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print("-" * 70)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate_epoch_with_s_measure(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Save history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_f_measure'].append(val_metrics['f_measure'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_s_measure'].append(val_metrics['s_measure'])
        history['val_sr'].append(val_metrics['sr'])
        history['val_so'].append(val_metrics['so'])

        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
        print(f"Val F-measure: {val_metrics['f_measure']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
        print(f"Val S-measure: {val_metrics['s_measure']:.4f} (Sr={val_metrics['sr']:.4f}, So={val_metrics['so']:.4f})")

        # Save best model based on F-measure
        if val_metrics['f_measure'] > best_f_measure:
            best_f_measure = val_metrics['f_measure']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                's_measure': val_metrics['s_measure'],
                'f_measure': val_metrics['f_measure'],
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model (F-measure: {val_metrics['f_measure']:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

        print("-" * 70)

    return history


def plot_training_history(history, save_path='training_history_extended.png'):
    """Plot training history"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MAE
    axes[0, 1].plot(history['train_mae'], label='Train MAE')
    axes[0, 1].plot(history['val_mae'], label='Val MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F-measure
    axes[1, 0].plot(history['val_f_measure'], label='Val F-measure', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F-measure')
    axes[1, 0].set_title('Validation F-measure')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # IoU
    axes[1, 1].plot(history['val_iou'], label='Val IoU', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].set_title('Validation IoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # S-measure
    axes[2, 0].plot(history['val_s_measure'], label='S-measure', color='purple', linewidth=2)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('S-measure')
    axes[2, 0].set_title('Structure-Measure')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Sr and So components
    axes[2, 1].plot(history['val_sr'], label='Sr (region-aware)', linewidth=2)
    axes[2, 1].plot(history['val_so'], label='So (object-aware)', linewidth=2)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Score')
    axes[2, 1].set_title('S-measure Components')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


def test_model(model, test_loader, device, save_dir='results'):
    """Test the model"""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    metrics_list = []

    with torch.no_grad():
        for batch_idx, (images, masks, img_names) in enumerate(tqdm(test_loader, desc='Testing')):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            pred = outputs[0]  # Use fusion output

            # Compute metrics for each image
            for i in range(len(images)):
                # Standard metrics
                mae = SaliencyMetrics.mae(pred[i:i+1], masks[i:i+1])
                f_measure, precision, recall = SaliencyMetrics.f_measure(pred[i:i+1], masks[i:i+1])
                iou = SaliencyMetrics.iou(pred[i:i+1], masks[i:i+1])

                # S-measure
                s_measure, sr, so = SaliencyMetrics.s_measure(pred[i:i+1], masks[i:i+1])

                metrics_list.append({
                    'image': img_names[i],
                    'mae': mae,
                    'f_measure': f_measure,
                    'precision': precision,
                    'recall': recall,
                    'iou': iou,
                    's_measure': s_measure,
                    'sr': sr,
                    'so': so,
                })

                # Save prediction
                pred_np = pred[i, 0].cpu().numpy()
                pred_np = (pred_np * 255).astype(np.uint8)

                save_path = os.path.join(save_dir, f"pred_{img_names[i]}")
                cv2.imwrite(save_path, pred_np)

    # Compute average metrics
    avg_metrics = {
        'mae': np.mean([m['mae'] for m in metrics_list]),
        'f_measure': np.mean([m['f_measure'] for m in metrics_list]),
        'precision': np.mean([m['precision'] for m in metrics_list]),
        'recall': np.mean([m['recall'] for m in metrics_list]),
        'iou': np.mean([m['iou'] for m in metrics_list]),
        's_measure': np.mean([m['s_measure'] for m in metrics_list]),
        'sr': np.mean([m['sr'] for m in metrics_list]),
        'so': np.mean([m['so'] for m in metrics_list]),
    }

    print("\nTest Results:")
    print("-" * 60)
    print(f"MAE: {avg_metrics['mae']:.4f}")
    print(f"F-measure: {avg_metrics['f_measure']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"IoU: {avg_metrics['iou']:.4f}")
    print(f"S-measure: {avg_metrics['s_measure']:.4f}")
    print(f"  Sr (region-aware): {avg_metrics['sr']:.4f}")
    print(f"  So (object-aware): {avg_metrics['so']:.4f}")

    return avg_metrics, metrics_list
