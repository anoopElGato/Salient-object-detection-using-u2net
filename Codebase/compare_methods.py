"""
Fuctions for comparing U2-Net with Saliency Filters
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time

from u2net_model import U2NET
from u2net_lightweight import U2NETLITE
from saliency_filters import SaliencyFilters
from s_measure import SaliencyMetrics


class MethodComparator:
    """
    Main comparator class to comapare U2-Net with Saliency Filters
    """
    def __init__(self, u2net_model_path=None, device='cuda'):
        self.device = device

        # Initialize U2-Net
        self.u2net = U2NETLITE(in_ch=3, out_ch=1).to(device)
        # Load the model
        if u2net_model_path and Path(u2net_model_path).exists():
            checkpoint = torch.load(u2net_model_path, map_location=device, weights_only= False)
            self.u2net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded U2-Net from {u2net_model_path}")
        else:
            print("ERROR: No pretrained U2-Net model found at given path")

        self.u2net.eval()

        # Initialize Saliency Filters
        self.saliency_filters = SaliencyFilters(num_segments=200)
        print("Initialized Saliency Filters")

    def predict_u2net(self, image):
        """Get U2-Net output resized back to input image size"""
        h, w = image.shape[:2]

        # Preprocess
        img_tensor = cv2.resize(image, (320, 320))
        img_tensor = img_tensor.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.u2net(img_tensor)
            pred = outputs[0][0, 0].cpu().numpy()

        # Resize back to original size
        pred = cv2.resize(pred, (w, h))

        return pred

    def predict_saliency_filters(self, image):
        """Get saliency map using Saliency Filters"""
        img_normalized = image.astype(np.float32) / 255.0
        saliency_map = self.saliency_filters.compute_saliency(img_normalized)
        return saliency_map

    def compare_on_image(self, image, ground_truth=None):
        """Compare both methods on a single image"""
        results = {}

        # U2-Net prediction
        start_time = time.time()
        pred_u2net = self.predict_u2net(image)
        u2net_time = time.time() - start_time
        results['u2net_pred'] = pred_u2net
        results['u2net_time'] = u2net_time

        # Saliency Filters prediction
        start_time = time.time()
        pred_sf = self.predict_saliency_filters(image)
        sf_time = time.time() - start_time
        results['sf_pred'] = pred_sf
        results['sf_time'] = sf_time

        # If ground truth is available, compute metrics
        if ground_truth is not None:
            # Resize ground truth if necessary
            if ground_truth.shape != pred_u2net.shape:
                gt = cv2.resize(ground_truth, (pred_u2net.shape[1], pred_u2net.shape[0]))
            else:
                gt = ground_truth

            # Calculate U2-Net Metrics
            u2net_mae = SaliencyMetrics.mae(pred_u2net, gt)
            u2net_f_measure, u2net_precision, u2net_recall = SaliencyMetrics.f_measure(pred_u2net, gt)
            u2net_iou = SaliencyMetrics.iou(pred_u2net, gt)
            u2net_s_measure, u2net_sr, u2net_so = SaliencyMetrics.s_measure(pred_u2net, gt)

            results['u2net_metrics'] = {
                'mae': u2net_mae,
                'f_measure': u2net_f_measure,
                'precision': u2net_precision,
                'recall': u2net_recall,
                'iou': u2net_iou,
                's_measure': u2net_s_measure,
                'sr': u2net_sr,
                'so': u2net_so,
            }

            # Calculate Saliency Filters Metrics
            sf_mae = SaliencyMetrics.mae(pred_sf, gt)
            sf_f_measure, sf_precision, sf_recall = SaliencyMetrics.f_measure(pred_sf, gt)
            sf_iou = SaliencyMetrics.iou(pred_sf, gt)
            sf_s_measure, sf_sr, sf_so = SaliencyMetrics.s_measure(pred_sf, gt)

            results['sf_metrics'] = {
                'mae': sf_mae,
                'f_measure': sf_f_measure,
                'precision': sf_precision,
                'recall': sf_recall,
                'iou': sf_iou,
                's_measure': sf_s_measure,
                'sr': sf_sr,
                'so': sf_so,
            }

        return results

    def compare_on_dataset(self, image_dir, mask_dir, save_dir='comparison_results_with_smeasure', max_images= 30):
        """Compare methods on entire dataset"""
        Path(save_dir).mkdir(exist_ok=True)

        # Get image files
        image_files = sorted([f for f in Path(image_dir).iterdir() 
                            if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])[:max_images]

        all_results = []

        for img_file in tqdm(image_files, desc='Processing images'):
            # Load image
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load ground truth
            mask_file = Path(mask_dir) / f"{img_file.stem}.png"
            if not mask_file.exists():
                mask_file = Path(mask_dir) / f"{img_file.stem}.jpg"

            if mask_file.exists():
                gt = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                gt = gt.astype(np.float32) / 255.0
            else:
                gt = None
                print(f"Warning: No ground truth for {img_file.name}")

            # Compare methods
            results = self.compare_on_image(image, gt)

            # Store results
            result_dict = {
                'image': img_file.name,
                'u2net_time': results['u2net_time'],
                'sf_time': results['sf_time'],
            }

            if gt is not None:
                # U2-Net metrics
                result_dict.update({
                    'u2net_mae': results['u2net_metrics']['mae'],
                    'u2net_precision': results['u2net_metrics']['precision'],
                    'u2net_recall': results['u2net_metrics']['recall'],
                    'u2net_f_measure': results['u2net_metrics']['f_measure'],
                    'u2net_iou': results['u2net_metrics']['iou'],
                    'u2net_s_measure': results['u2net_metrics']['s_measure'],
                    'u2net_sr': results['u2net_metrics']['sr'],
                    'u2net_so': results['u2net_metrics']['so'],
                })

                # Saliency Filters metrics
                result_dict.update({
                    'sf_mae': results['sf_metrics']['mae'],
                    'sf_precision': results['sf_metrics']['precision'],
                    'sf_recall': results['sf_metrics']['recall'],
                    'sf_f_measure': results['sf_metrics']['f_measure'],
                    'sf_iou': results['sf_metrics']['iou'],
                    'sf_s_measure': results['sf_metrics']['s_measure'],
                    'sf_sr': results['sf_metrics']['sr'],
                    'sf_so': results['sf_metrics']['so'],
                })

            all_results.append(result_dict)

            # Save visualizations
            self.visualize_comparison(
                image, 
                results['u2net_pred'], 
                results['sf_pred'],
                gt,
                save_path=Path(save_dir) / f"comparison_{img_file.stem}.png"
            )

        # Create summary DataFrame
        df = pd.DataFrame(all_results)

        # Print summary
        print("\n" + "-" * 60)
        print("COMPARISON RESULTS")
        print("-" * 60)

        print("\nSpeed Comparison:")
        print(f"  U2-Net average time: {df['u2net_time'].mean():.4f}s")
        print(f"  Saliency Filters average time: {df['sf_time'].mean():.4f}s")
        print(f"  Speedup: {df['sf_time'].mean() / df['u2net_time'].mean():.2f}x")

        if 'u2net_mae' in df.columns:
            print("\nAccuracy Comparison (U2-Net):")
            print(f"  MAE: {df['u2net_mae'].mean():.4f}")
            print(f"  Precision: {df['u2net_precision'].mean():.4f}")
            print(f"  Recall: {df['u2net_recall'].mean():.4f}")
            print(f"  F-measure: {df['u2net_f_measure'].mean():.4f}")

            print("\nAccuracy Comparison (Saliency Filters):")
            print(f"  MAE: {df['sf_mae'].mean():.4f}")
            print(f"  Precision: {df['sf_precision'].mean():.4f}")
            print(f"  Recall: {df['sf_recall'].mean():.4f}")
            print(f"  F-measure: {df['sf_f_measure'].mean():.4f}")

            print("\nImprovement (U2-Net vs Saliency Filters):")
            print(f"  MAE: {((df['sf_mae'].mean() - df['u2net_mae'].mean()) / df['sf_mae'].mean() * 100):.2f}% better")
            print(f"  F-measure: {((df['u2net_f_measure'].mean() - df['sf_f_measure'].mean()) / df['sf_f_measure'].mean() * 100):.2f}% better")

        if 'u2net_s_measure' in df.columns:
            print("\n" + "-" * 70)
            print("S-MEASURE COMPARISON:")
            print("-" * 70)
            print(f"\nU2-Net:")
            print(f"  S-measure: {df['u2net_s_measure'].mean():.4f}")
            print(f"  Sr (region-aware): {df['u2net_sr'].mean():.4f}")
            print(f"  So (object-aware): {df['u2net_so'].mean():.4f}")

            print(f"\nSaliency Filters:")
            print(f"  S-measure: {df['sf_s_measure'].mean():.4f}")
            print(f"  Sr (region-aware): {df['sf_sr'].mean():.4f}")
            print(f"  So (object-aware): {df['sf_so'].mean():.4f}")

            s_measure_improvement = ((df['u2net_s_measure'].mean() - df['sf_s_measure'].mean()) / 
                                   df['sf_s_measure'].mean() * 100)
            print(f"\nS-measure Improvement (U2-Net vs Saliency Filters):")
            print(f"  {s_measure_improvement:.2f}% better")

        # Save results to CSV
        csv_path = Path(save_dir) / 'comparison_results_with_smeasure.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # Plot comparison
        self.plot_comparison_summary(df, save_path=Path(save_dir) / 'comparison_summary_with_smeasure.png')

        return df

    def visualize_comparison(self, image, pred_u2net, pred_sf, ground_truth=None, 
                           save_path=None):
        """Visualize comparison between methods"""
        if ground_truth is not None:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            axes[1].imshow(ground_truth, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(pred_u2net, cmap='gray')
            axes[2].set_title('U2-Net Prediction')
            axes[2].axis('off')

            axes[3].imshow(pred_sf, cmap='gray')
            axes[3].set_title('Saliency Filters')
            axes[3].axis('off')
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            axes[1].imshow(pred_u2net, cmap='gray')
            axes[1].set_title('U2-Net Prediction')
            axes[1].axis('off')

            axes[2].imshow(pred_sf, cmap='gray')
            axes[2].set_title('Saliency Filters')
            axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_comparison_summary(self, df, save_path):
        """Plot summary comparison"""
        if 'u2net_s_measure' not in df.columns:
            print("No metrics to plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        methods = ['U2-Net', 'Saliency Filters']

        # S-measure
        s_measure_values = [df['u2net_s_measure'].mean(), df['sf_s_measure'].mean()]
        axes[0, 0].bar(methods, s_measure_values, color=['#2E86AB', '#A23B72'])
        axes[0, 0].set_ylabel('S-measure (higher is better)')
        axes[0, 0].set_title('Structure-Measure')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1])

        # Sr and So components
        sr_values = [df['u2net_sr'].mean(), df['sf_sr'].mean()]
        so_values = [df['u2net_so'].mean(), df['sf_so'].mean()]

        x = np.arange(len(methods))
        width = 0.35

        axes[0, 1].bar(x - width/2, sr_values, width, label='Sr (region-aware)', color='#2E86AB')
        axes[0, 1].bar(x + width/2, so_values, width, label='So (object-aware)', color='#A23B72')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('S-measure Components')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(methods)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)

        # MAE
        mae_values = [df['u2net_mae'].mean(), df['sf_mae'].mean()]
        axes[0, 2].bar(methods, mae_values, color=['#2E86AB', '#A23B72'])
        axes[0, 2].set_ylabel('MAE (lower is better)')
        axes[0, 2].set_title('Mean Absolute Error')
        axes[0, 2].grid(axis='y', alpha=0.3)

        # F-measure
        f_measure_values = [df['u2net_f_measure'].mean(), df['sf_f_measure'].mean()]
        axes[1, 0].bar(methods, f_measure_values, color=['#2E86AB', '#A23B72'])
        axes[1, 0].set_ylabel('F-measure (higher is better)')
        axes[1, 0].set_title('F-measure')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Precision-Recall
        precision_values = [df['u2net_precision'].mean(), df['sf_precision'].mean()]
        recall_values = [df['u2net_recall'].mean(), df['sf_recall'].mean()]

        x = np.arange(len(methods))
        width = 0.35

        axes[1, 1].bar(x - width/2, precision_values, width, label='Precision', color='#2E86AB')
        axes[1, 1].bar(x + width/2, recall_values, width, label='Recall', color='#A23B72')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(methods)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)

        # Speed comparison
        speed_values = [df['u2net_time'].mean(), df['sf_time'].mean()]
        axes[1, 2].bar(methods, speed_values, color=['#2E86AB', '#A23B72'])
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].set_title('Processing Speed')
        axes[1, 2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison summary saved to {save_path}")
