import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt


class SaliencyFilters:
    """
    Saliency Filters implementation based on Perazzi et al. 2012
    """

    def __init__(self, num_segments=200, sigma_p=0.25, sigma_c=20, k=6, 
                 alpha=1/30, beta=1/30):
        """
        Initialize Saliency Filters

        Args:
            num_segments : Number of superpixels for image abstraction
            sigma_p : Spatial parameter for uniqueness (controls local vs global)
            sigma_c : Color parameter for distribution
            k : Scaling factor for exponential in saliency combination
            alpha : Color sensitivity for upsampling
            beta : Position sensitivity for upsampling
        """
        self.num_segments = num_segments
        self.sigma_p = sigma_p
        self.sigma_c = sigma_c
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def compute_saliency(self, image):
        """
        Compute saliency map for input image

        Args:
            image : Input RGB image (numpy ndarray) (H x W x 3)
        Returns:
            saliency_map : Per-pixel saliency map (numpy ndarray) (H x W)
        """
        # Step 1: Image Abstraction
        segments, segment_colors, segment_positions = self.abstract_image(image)

        # Step 2: Element Uniqueness
        uniqueness = self.compute_uniqueness(segment_colors, segment_positions)

        # Step 3: Element Distribution
        distribution = self.compute_distribution(segment_colors, segment_positions)

        # Step 4: Combine to get element saliency
        element_saliency = self.combine_saliency(uniqueness, distribution)

        # Step 5: Saliency assignment (upsampling to pixel level)
        saliency_map = self.assign_saliency(image, segments, element_saliency, 
                                             segment_colors, segment_positions)

        return saliency_map

    def abstract_image(self, image):
        """
        Abstract image into superpixels using SLIC

        Returns:
        segments : (numpy array) Segment labels for each pixel
        segment_colors : (numpy array) Mean LAB color for each segment
        segment_positions : (numpy array) Mean position for each segment
        """
        # Convert to LAB color space
        lab_image = rgb2lab(image)

        # Apply SLIC superpixel segmentation
        segments = slic(image, n_segments=self.num_segments, compactness=10, 
                       sigma=1, start_label=0)

        # Compute segment properties
        num_segments = segments.max() + 1
        segment_colors = np.zeros((num_segments, 3))
        segment_positions = np.zeros((num_segments, 2))

        h, w = segments.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        for i in range(num_segments):
            mask = segments == i
            segment_colors[i] = lab_image[mask].mean(axis=0)
            segment_positions[i] = [x_coords[mask].mean(), y_coords[mask].mean()]

        return segments, segment_colors, segment_positions

    def compute_uniqueness(self, colors, positions):
        """
        Compute element uniqueness using Gaussian filtering

        Args:
        colors : (numpy array) (N x 3) LAB colors of segments
        positions : (numpy array) (N x 2) XY positions of segments

        Returns:
        uniqueness : (numpy array) (N,) Uniqueness score for each segment
        """
        N = len(colors)
        uniqueness = np.zeros(N)

        # Normalize positions to [0, 1] range
        pos_normalized = positions.copy()
        pos_normalized[:, 0] /= positions[:, 0].max()
        pos_normalized[:, 1] /= positions[:, 1].max()

        for i in range(N):
            # Compute spatial weights
            spatial_dist = np.sum((pos_normalized - pos_normalized[i])**2, axis=1)
            w_p = np.exp(-spatial_dist / (2 * self.sigma_p**2))

            # Compute color distances
            color_dist = np.sum((colors - colors[i])**2, axis=1)

            # Weighted uniqueness
            Z_i = w_p.sum()
            uniqueness[i] = (color_dist * w_p).sum() / Z_i

        # Normalize to [0, 1]
        uniqueness = (uniqueness - uniqueness.min()) / (uniqueness.max() - uniqueness.min() + 1e-8)

        return uniqueness

    def compute_distribution(self, colors, positions):
        """
        Compute spatial distribution of elements

        Args:
            colors : (numpy array) (N x 3) LAB colors of segments
            positions : (numpy array) (N x 2) XY positions of segments
        Returns:
            distribution : (numpy array) (N,) Distribution score for each segment
        """
        N = len(colors)
        distribution = np.zeros(N)

        # Normalize positions
        pos_normalized = positions.copy()
        pos_normalized[:, 0] /= positions[:, 0].max()
        pos_normalized[:, 1] /= positions[:, 1].max()

        for i in range(N):
            # Compute color similarity weights
            color_dist = np.sum((colors - colors[i])**2, axis=1)
            w_c = np.exp(-color_dist / (2 * self.sigma_c**2))

            # Compute weighted mean position
            Z_i = w_c.sum()
            mu_i = (pos_normalized.T @ w_c) / Z_i

            # Compute spatial variance
            pos_diff = pos_normalized - mu_i
            spatial_var = np.sum(pos_diff**2, axis=1)
            distribution[i] = (spatial_var * w_c).sum() / Z_i

        # Normalize to [0, 1]
        distribution = (distribution - distribution.min()) / (distribution.max() - distribution.min() + 1e-8)

        return distribution

    def combine_saliency(self, uniqueness, distribution):
        """
        Combine uniqueness and distribution to compute element saliency

        Args:
            uniqueness : (numpy array) (N,) Uniqueness scores
            distribution : (numpy array) (N,) Distribution scores
        Returns:
            saliency : (numpy array) (N,) Combined saliency scores
        """
        # S_i = U_i * exp(-k * D_i)
        saliency = uniqueness * np.exp(-self.k * distribution)

        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        return saliency

    def assign_saliency(self, image, segments, element_saliency, 
                        segment_colors, segment_positions):
        """
        Assign saliency to pixels using Gaussian filtering in RGBXY space

        Args:
            image : (numpy array) (H x W x 3) Original RGB image
            segments : (numpy array) (H x W) Segment labels
            element_saliency : (numpy array) (N,) Saliency scores for each segment
            segment_colors : (numpy array) (N x 3) Segment colors in LAB
            segment_positions : (numpy array) (N x 2) Segment positions
        Returns:
            saliency_map : (numpy array) (H x W) Per-pixel saliency map
        """
        h, w = segments.shape
        saliency_map = np.zeros((h, w))

        # For each pixel, compute weighted average of nearby segment saliencies
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Convert image to LAB for color comparison
        lab_image = rgb2lab(image)

        # Normalize coordinates
        x_norm = x_coords / w
        y_norm = y_coords / h

        # For efficiency, use a simple approach: assign based on nearby segments
        for i in range(len(element_saliency)):
            mask = segments == i

            # Get pixels in this segment
            y_pix, x_pix = np.where(mask)

            if len(y_pix) == 0:
                continue

            # For pixels in this segment, consider neighboring segments
            for yi, xi in zip(y_pix, x_pix):
                pixel_color = lab_image[yi, xi]
                pixel_pos = np.array([xi / w, yi / h])

                # Compute weights for all segments
                color_dist = np.sum((segment_colors - pixel_color)**2, axis=1)
                pos_dist_x = (segment_positions[:, 0] / w - pixel_pos[0])**2
                pos_dist_y = (segment_positions[:, 1] / h - pixel_pos[1])**2
                pos_dist = pos_dist_x + pos_dist_y

                weights = np.exp(-self.alpha * color_dist - self.beta * pos_dist)
                weights /= weights.sum()

                saliency_map[yi, xi] = (weights @ element_saliency)

        # Normalize to [0, 1]
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

        return saliency_map


class SaliencyEvaluator:
    """
    Evaluate saliency detection performance
    """

    @staticmethod
    def adaptive_threshold(saliency_map):
        """
        Compute adaptive threshold as 2 * mean saliency

        Args:
            saliency_map : (numpy array) Saliency map
        Returns:
            threshold : (float) Adaptive threshold value
        """
        return 2 * saliency_map.mean()

    @staticmethod
    def compute_metrics(saliency_map, ground_truth, threshold=None):
        """
        Compute precision, recall, and F-measure

        Args:
            saliency_map : (numpy array) Saliency map (values in [0, 1])
            ground_truth : (numpy array) Binary ground truth (0 or 1)
            threshold : (float or None) Threshold for binarization. If None, use adaptive threshold
        Returns:
            metrics : (dict) Dictionary containing precision, recall, and f_measure
        """
        if threshold is None:
            threshold = SaliencyEvaluator.adaptive_threshold(saliency_map)

        # Binarize saliency map
        binary_map = (saliency_map >= threshold).astype(np.uint8)

        # Ensure ground truth is binary
        gt_binary = (ground_truth > 0.5).astype(np.uint8)

        # Compute metrics
        intersection = np.logical_and(binary_map, gt_binary).sum()

        precision = intersection / (binary_map.sum() + 1e-8)
        recall = intersection / (gt_binary.sum() + 1e-8)

        # F-measure with beta^2 = 0.3
        beta_squared = 0.3
        f_measure = ((1 + beta_squared) * precision * recall) / (beta_squared * precision + recall + 1e-8)

        return {
            'precision': precision,
            'recall': recall,
            'f_measure': f_measure,
            'threshold': threshold
        }

    @staticmethod
    def compute_pr_curve(saliency_map, ground_truth, num_thresholds=255):
        """
        Compute precision-recall curve

        Args:
            saliency_map : (numpy array) Saliency map (values in [0, 1])
            ground_truth : (numpy array) Binary ground truth
            num_thresholds : (int) Number of threshold values to evaluate
        Returns:
            precisions : (list) Precision values
            recalls : (list) Recall values
            thresholds : (list) Threshold values
        """
        gt_binary = (ground_truth > 0.5).astype(np.uint8).flatten()
        saliency_flat = saliency_map.flatten()

        precisions = []
        recalls = []
        thresholds = np.linspace(0, 1, num_thresholds)

        for threshold in thresholds:
            binary_map = (saliency_flat >= threshold).astype(np.uint8)

            intersection = np.logical_and(binary_map, gt_binary).sum()

            precision = intersection / (binary_map.sum() + 1e-8)
            recall = intersection / (gt_binary.sum() + 1e-8)

            precisions.append(precision)
            recalls.append(recall)

        return precisions, recalls, thresholds

    @staticmethod
    def compute_mae(saliency_map, ground_truth):
        """
        Compute Mean Absolute Error

        Args:
            saliency_map : (numpy array) Saliency map (values in [0, 1])
            ground_truth : (numpy array) Binary ground truth (0 or 1)
        Returns:
            mae : (float) Mean absolute error
        """
        gt_normalized = (ground_truth > 0.5).astype(np.float32)
        mae = np.abs(saliency_map - gt_normalized).mean()
        return mae
