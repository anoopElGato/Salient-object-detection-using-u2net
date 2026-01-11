import torch
import os
from pathlib import Path

from u2net_model import U2NET
from u2net_lightweight import U2NETLITE, count_parameters, get_model_size_mb
from SOD_dataset import create_dataloaders, SaliencyDataset
from u2net_train import train_u2net, test_model, plot_training_history
from compare_methods import MethodComparator


def main():
    CONFIG = {
        'train_img_dir': r'DUTS-TR/DUTS-TR-Image',
        'train_mask_dir': r'DUTS-TR/DUTS-TR-Mask',
        'val_img_dir': r'DUTS-TE/DUTS-TE-Image',
        'val_mask_dir': r'DUTS-TE/DUTS-TE-Mask',
        'test_img_dir': r'DUTS-TE/DUTS-TE-Image',
        'test_mask_dir': r'DUTS-TE/DUTS-TE-Mask',

        'img_size': 320,
        'batch_size': 8,
        'num_workers': 4,
        'num_epochs': 30,
        'learning_rate': 1e-3,

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',
    }

    print("\nPipeline Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Step 1: Create the model
    print("\n" + "=" * 60)
    print("1/5: Initializing the model")
    print("-" * 60)

    model = U2NETLITE(in_ch=3, out_ch=1)
    model = model.to(CONFIG['device'])

    print(f"Model: U2-Net-lite")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model Size: {get_model_size_mb(model):,}")
    print(f"Device: {CONFIG['device']}")

    # Step 2: Load data
    print("\n" + "=" * 60)
    print("2/5: Loading the dataset")
    print("-" * 60)

    if not Path(CONFIG['train_img_dir']).exists():
        print("ERROR: Dataset not found!")
        return

    train_loader, val_loader = create_dataloaders(
        CONFIG['train_img_dir'],
        CONFIG['train_mask_dir'],
        CONFIG['val_img_dir'],
        CONFIG['val_mask_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        img_size=CONFIG['img_size']
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Step 3: Train the model
    print("\n" + "=" * 60)
    print("3/5: Training the model")
    print("-" * 60)

    history = train_u2net(
        model,
        train_loader,
        val_loader,
        num_epochs=CONFIG['num_epochs'],
        lr=CONFIG['learning_rate'],
        device=CONFIG['device'],
        save_dir=CONFIG['checkpoint_dir']
    )

    # Plot training performance over the epochs
    plot_training_history(history, save_path='training_history.png')

    # Step 4: Test the model
    print("\n" + "=" * 60)
    print("4/5: Testing the model")
    print("-" * 60)

    # Load best model
    best_model_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")

    # Create test loader
    from torch.utils.data import DataLoader
    test_dataset = SaliencyDataset(
        CONFIG['test_img_dir'],
        CONFIG['test_mask_dir'],
        img_size=CONFIG['img_size']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )

    # Test
    avg_metrics, detailed_metrics = test_model(
        model,
        test_loader,
        CONFIG['device'],
        save_dir=CONFIG['results_dir']
    )

    # Step 5: Compare with paper's method
    print("\n" + "=" * 60)
    print("5/5: Comparing with Saliency Filters")
    print("-" * 60)

    comparator = MethodComparator(
        u2net_model_path=best_model_path,
        device=CONFIG['device']
    )

    comparison_df = comparator.compare_on_dataset(
        CONFIG['test_img_dir'],
        CONFIG['test_mask_dir'],
        save_dir='comparison_results'
    )

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("-" * 60)
    print("\nGenerated files:")
    print("  - checkpoints/best_model.pth: Best trained model")
    print("  - training_history.png: Training curves")
    print("  - results/: U2-Net predictions")
    print("  - comparison_results/: Method comparison")
    print("  - comparison_results/comparison_results.csv: Quantitative results")
    print("  - comparison_results/comparison_summary.png: Visual comparison")


if __name__ == "__main__":
    main()
