import numpy as np
import cv2
import torch
from scipy import ndimage


class StructureMeasure:
    """
    Compute S-measure (Structure-measure) for saliency detection evaluation
    S = (1 - α)·Sr + α·So
    """

    @staticmethod
    def to_numpy(x):
        """Convert to numpy array"""
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        return np.asarray(x)

    @staticmethod
    def ensure_2d(x):
        """Ensure array is 2D (H, W)"""
        if len(x.shape) == 4:
            x = x[0, 0]  # (B, C, H, W) -> (H, W)
        elif len(x.shape) == 3:
            x = x[0] if x.shape[0] == 1 else x[:, :, 0]  # (1,H,W) or (H,W,C) -> (H,W)
        return x

    @staticmethod
    def normalize(x):
        """Normalize to [0, 1]"""
        if x.max() > 1.0:
            x = x / 255.0
        return x.astype(np.float32)

    @staticmethod
    def ssim(x, y):
        """Compute structural similarity index"""
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        # Ensure 2D
        if len(x.shape) != 2:
            x = x.reshape(-1, 1)
        if len(y.shape) != 2:
            y = y.reshape(-1, 1)

        # Mean
        ux = cv2.blur(x, (7, 7))
        uy = cv2.blur(y, (7, 7))

        # Variance and covariance
        uxx = cv2.blur(x * x, (7, 7)) - ux * ux
        uyy = cv2.blur(y * y, (7, 7)) - uy * uy
        uxy = cv2.blur(x * y, (7, 7)) - ux * uy

        # SSIM computation
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        ssim_map = ((2 * ux * uy + c1) * (2 * uxy + c2)) / \
                   ((ux ** 2 + uy ** 2 + c1) * (uxx + uyy + c2) + 1e-8)

        return ssim_map

    @staticmethod
    def compute_sr(pred, gt):
        """Compute region-aware structural similarity (Sr)"""
        try:
            # Convert and prepare
            pred = StructureMeasure.to_numpy(pred)
            gt = StructureMeasure.to_numpy(gt)
            pred = StructureMeasure.ensure_2d(pred)
            gt = StructureMeasure.ensure_2d(gt)
            pred = StructureMeasure.normalize(pred)
            gt = StructureMeasure.normalize(gt)

            h, w = pred.shape

            # Divide into regions
            region_size = 16
            n_regions_h = max(1, h // region_size)
            n_regions_w = max(1, w // region_size)

            sr_values = []

            # Compute SSIM for each region
            for i in range(n_regions_h):
                for j in range(n_regions_w):
                    r1 = i * region_size
                    r2 = min((i + 1) * region_size, h)
                    c1 = j * region_size
                    c2 = min((j + 1) * region_size, w)

                    pred_region = pred[r1:r2, c1:c2]
                    gt_region = gt[r1:r2, c1:c2]

                    if pred_region.size > 0 and gt_region.size > 0:
                        ssim_map = StructureMeasure.ssim(pred_region, gt_region)
                        sr_values.append(float(np.nanmean(ssim_map)))

            sr = np.mean(sr_values) if sr_values else 0.0
            return float(np.clip(sr, 0, 1))
        except Exception as e:
            print(f"Error in compute_sr: {e}")
            return 0.0

    @staticmethod
    def compute_so(pred, gt):
        """Compute object-aware structural similarity (So)"""
        try:
            # Convert and prepare
            pred = StructureMeasure.to_numpy(pred)
            gt = StructureMeasure.to_numpy(gt)
            pred = StructureMeasure.ensure_2d(pred)
            gt = StructureMeasure.ensure_2d(gt)
            pred = StructureMeasure.normalize(pred)
            gt = StructureMeasure.normalize(gt)

            h, w = gt.shape

            # Binarize
            mean_gt = gt.mean()
            gt_bin = (gt >= mean_gt).astype(np.float32)

            # Foreground and background
            fg_mask = gt_bin > 0.5
            bg_mask = gt_bin <= 0.5

            # Compute for foreground
            so_fg = 0.0
            if fg_mask.sum() > 0:
                try:
                    fg_pred = pred[fg_mask].reshape(-1, 1)
                    fg_gt = gt[fg_mask].reshape(-1, 1)
                    if fg_pred.size > 0:
                        ssim_fg = StructureMeasure.ssim(fg_pred, fg_gt)
                        so_fg = float(np.nanmean(ssim_fg))
                except:
                    so_fg = 0.0

            # Compute for background
            so_bg = 0.0
            if bg_mask.sum() > 0:
                try:
                    bg_pred = pred[bg_mask].reshape(-1, 1)
                    bg_gt = gt[bg_mask].reshape(-1, 1)
                    if bg_pred.size > 0:
                        ssim_bg = StructureMeasure.ssim(bg_pred, bg_gt)
                        so_bg = float(np.nanmean(ssim_bg))
                except:
                    so_bg = 0.0

            # Combine
            fg_weight = float(fg_mask.sum()) / (h * w) if (h * w) > 0 else 0
            bg_weight = float(bg_mask.sum()) / (h * w) if (h * w) > 0 else 0

            so = fg_weight * so_fg + bg_weight * so_bg
            return float(np.clip(so, 0, 1))
        except Exception as e:
            print(f"Error in compute_so: {e}")
            return 0.0

    @staticmethod
    def compute_s_measure(pred, gt, alpha=0.5):
        """
        Compute S-measure: S = (1 - α)·Sr + α·So
        """
        try:
            sr = StructureMeasure.compute_sr(pred, gt)
            so = StructureMeasure.compute_so(pred, gt)

            s_measure = (1 - alpha) * sr + alpha * so

            return (
                float(np.clip(s_measure, 0, 1)),
                float(np.clip(sr, 0, 1)),
                float(np.clip(so, 0, 1))
            )
        except Exception as e:
            print(f"Error in compute_s_measure: {e}")
            return 0.0, 0.0, 0.0


class SaliencyMetrics:
    """Extended metrics including S-measure"""

    @staticmethod
    def to_numpy(x):
        """Convert to numpy array"""
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        return np.asarray(x)

    @staticmethod
    def ensure_2d(x):
        """Ensure 2D array"""
        if len(x.shape) == 4:
            x = x[0, 0]
        elif len(x.shape) == 3:
            x = x[0] if x.shape[0] == 1 else x[:, :, 0]
        return x

    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error"""
        try:
            pred = SaliencyMetrics.to_numpy(pred)
            target = SaliencyMetrics.to_numpy(target)
            pred = SaliencyMetrics.ensure_2d(pred).astype(np.float32)
            target = SaliencyMetrics.ensure_2d(target).astype(np.float32)

            return float(np.abs(pred - target).mean())
        except Exception as e:
            print(f"Error in mae: {e}")
            return 0.0

    @staticmethod
    def f_measure(pred, target, beta_square=0.3, threshold=0.5):
        """F-measure (F-beta score)"""
        try:
            # Convert to numpy
            pred = SaliencyMetrics.to_numpy(pred)
            target = SaliencyMetrics.to_numpy(target)

            # Ensure 2D
            pred = SaliencyMetrics.ensure_2d(pred).astype(np.float32)
            target = SaliencyMetrics.ensure_2d(target).astype(np.float32)

            # Normalize
            if pred.max() > 1:
                pred = pred / 255.0
            if target.max() > 1:
                target = target / 255.0

            # Binarize using numpy (not torch!)
            pred_binary = (pred > threshold).astype(np.float32)
            target_binary = (target > 0.5).astype(np.float32)

            # Compute metrics
            tp = float((pred_binary * target_binary).sum())
            fp = float((pred_binary * (1 - target_binary)).sum())
            fn = float(((1 - pred_binary) * target_binary).sum())

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            f_measure = ((1 + beta_square) * precision * recall) / \
                        (beta_square * precision + recall + 1e-8)

            return float(f_measure), float(precision), float(recall)
        except Exception as e:
            print(f"Error in f_measure: {e}")
            return 0.0, 0.0, 0.0

    @staticmethod
    def iou(pred, target, threshold=0.5):
        """Intersection over Union"""
        try:
            # Convert to numpy
            pred = SaliencyMetrics.to_numpy(pred)
            target = SaliencyMetrics.to_numpy(target)

            # Ensure 2D
            pred = SaliencyMetrics.ensure_2d(pred).astype(np.float32)
            target = SaliencyMetrics.ensure_2d(target).astype(np.float32)

            # Normalize
            if pred.max() > 1:
                pred = pred / 255.0
            if target.max() > 1:
                target = target / 255.0

            # Binarize using numpy (not torch!)
            pred_binary = (pred > threshold).astype(np.float32)
            target_binary = (target > 0.5).astype(np.float32)

            # Compute
            intersection = float((pred_binary * target_binary).sum())
            union = float(pred_binary.sum() + target_binary.sum() - intersection)

            return float(intersection / (union + 1e-8))
        except Exception as e:
            print(f"Error in iou: {e}")
            return 0.0

    @staticmethod
    def s_measure(pred, target, alpha=0.5):
        """Structure-measure"""
        try:
            s_measure, sr, so = StructureMeasure.compute_s_measure(pred, target, alpha=alpha)
            return float(s_measure), float(sr), float(so)
        except Exception as e:
            print(f"Error in s_measure: {e}")
            return 0.0, 0.0, 0.0

    @staticmethod
    def compute_all_metrics(pred, target, threshold=0.5, alpha=0.5):
        """Compute all metrics at once"""
        try:
            mae = SaliencyMetrics.mae(pred, target)
            f_measure, precision, recall = SaliencyMetrics.f_measure(
                pred, target, threshold=threshold
            )
            iou = SaliencyMetrics.iou(pred, target, threshold=threshold)
            s_measure, sr, so = SaliencyMetrics.s_measure(pred, target, alpha=alpha)

            return {
                'mae': mae,
                'f_measure': f_measure,
                'precision': precision,
                'recall': recall,
                'iou': iou,
                's_measure': s_measure,
                'sr': sr,
                'so': so,
            }
        except Exception as e:
            print(f"Error in compute_all_metrics: {e}")
            return {
                'mae': 0.0,
                'f_measure': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'iou': 0.0,
                's_measure': 0.0,
                'sr': 0.0,
                'so': 0.0,
            }
