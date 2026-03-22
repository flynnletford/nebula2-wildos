#!/usr/bin/env python3
"""
Test script for ExploRFM inference on a JPG image.
Outputs frontier detection visualization.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add explorfm to path
sys.path.insert(0, str(Path(__file__).parent))

from explorfm.explorfm_model import ExploRFMInference


def normalize_output(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor output to normalized numpy array (0-255)."""
    output = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    return output


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Convert grayscale heatmap to color using viridis colormap."""
    # Normalize to 0-1
    heatmap_norm = heatmap.astype(np.float32) / 255.0
    
    # Apply colormap
    cmap = plt.colormaps['viridis']
    colored = cmap(heatmap_norm)
    
    # Convert to BGR (OpenCV format)
    colored_bgr = cv2.cvtColor((colored[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return colored_bgr


def overlay_traversable_frontiers(image: np.ndarray, frontier_map: np.ndarray, traversability_map: np.ndarray, frontier_threshold: float = 0.5, traversability_threshold: float = 0.5) -> np.ndarray:
    """Overlay frontier heatmap on the original image, only where traversable.
    
    Args:
        image: Original BGR image
        frontier_map: Grayscale frontier heatmap (0-255)
        traversability_map: Grayscale traversability map (0-255)
        frontier_threshold: Confidence threshold for frontiers (0-1)
        traversability_threshold: Confidence threshold for traversability (0-1)
    
    Returns:
        Image with frontiers highlighted in red only in traversable regions
    """
    # Normalize maps to 0-1
    frontier_norm = frontier_map.astype(np.float32) / 255.0
    traversability_norm = traversability_map.astype(np.float32) / 255.0
    
    # Create combined mask: frontier must be high confidence AND area must be traversable
    frontier_mask = frontier_norm > frontier_threshold
    traversability_mask = traversability_norm > traversability_threshold
    combined_mask = frontier_mask & traversability_mask
    
    # Get intensity from frontier map
    frontier_intensity = frontier_norm * combined_mask.astype(np.float32)
    
    # Convert image to float
    result = image.astype(np.float32)
    
    # Create red overlay
    overlay_color = np.array([0, 0, 255], dtype=np.float32)  # BGR format (red)
    
    # Blend only in valid frontier areas
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - frontier_intensity * 0.3) + overlay_color[c] * (frontier_intensity * 0.7)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def overlay_traversability_on_image(image: np.ndarray, traversability_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Overlay traversability heatmap on the original image with green highlighting.
    
    Args:
        image: Original BGR image
        traversability_map: Grayscale traversability heatmap (0-255)
        threshold: Confidence threshold (0-1) for highlighting traversable areas
    
    Returns:
        Image with traversable areas highlighted in green
    """
    # Normalize traversability map to 0-1
    traversability_norm = traversability_map.astype(np.float32) / 255.0
    
    # Create mask for high-confidence traversable areas
    traversability_mask = traversability_norm > threshold
    
    # Get intensity of traversability
    traversability_intensity = traversability_norm * traversability_mask.astype(np.float32)
    
    # Convert image to float
    result = image.astype(np.float32)
    
    # Create green overlay (BGR format = [0, 255, 0])
    overlay_color = np.array([0, 255, 0], dtype=np.float32)  # BGR format (green)
    
    # Blend only in traversable areas
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - traversability_intensity * 0.3) + overlay_color[c] * (traversability_intensity * 0.7)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def overlay_frontiers_unfiltered(image: np.ndarray, frontier_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Overlay frontier heatmap on the original image without traversability filtering.
    
    Args:
        image: Original BGR image
        frontier_map: Grayscale frontier heatmap (0-255)
        threshold: Confidence threshold (0-1) for highlighting frontiers
    
    Returns:
        Image with frontiers highlighted in red
    """
    # Normalize frontier map to 0-1
    frontier_norm = frontier_map.astype(np.float32) / 255.0
    
    # Create mask for high-confidence frontiers
    frontier_mask = frontier_norm > threshold
    
    # Get intensity of frontier
    frontier_intensity = frontier_norm * frontier_mask.astype(np.float32)
    
    # Convert image to float
    result = image.astype(np.float32)
    
    # Create red overlay
    overlay_color = np.array([0, 0, 255], dtype=np.float32)  # BGR format (red)
    
    # Blend only in frontier areas
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - frontier_intensity * 0.3) + overlay_color[c] * (frontier_intensity * 0.7)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result



def test_explorfm(image_path: str, output_dir: str = "explorfm_outputs", frontier_confidence: float = 0.6, traversability_confidence: float = 0.5):
    """
    Run ExploRFM inference on an image and save frontier visualization.
    
    Args:
        image_path: Path to input JPG image
        output_dir: Directory to save output image
        frontier_confidence: Minimum confidence threshold for frontier detection (default: 0.6 from paper)
        traversability_confidence: Minimum confidence threshold for traversability (default: 0.5)
    """
    # Extract image name without extension and create subdirectory
    image_name = Path(image_path).stem  # Get filename without extension
    output_subdir = Path(output_dir) / image_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # Downsample to 540 x 960
    image = cv2.resize(image, (960, 540))
    
    # Convert BGR to RGB for model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]
    print(f"Image shape: {original_height} x {original_width}")
    
    # Initialize model (load both frontier and traversability heads)
    print("Initializing ExploRFM model...")
    model = ExploRFMInference(
        frontier_ckpt="ckpts/frontier_head.ckpt",
        traversability_ckpt="ckpts/trav_head.ckpt",  # Include traversability head
        model_version="c-radio_v3-b",
        adaptor_version=None,  # No text features needed
        radio_dim=768,
        static_scale_factor=1.0,
        model_precision="FP32",
    )
    print(f"Model loaded on device: {model.device}")
    
    # Run inference
    print("Running inference...")
    inference_start_time = time.time()
    with torch.no_grad():
        traversability, frontiers, _ = model.forward_on_numpy(image_rgb)
    inference_time = time.time() - inference_start_time
    
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Frontiers shape: {frontiers.shape}")
    print(f"Traversability shape: {traversability.shape}")
    
    # Process outputs
    frontiers_map = normalize_output(frontiers)
    traversability_map = normalize_output(traversability)
    
    # Save original image
    original_output_path = os.path.join(str(output_subdir), "00_original_image.jpg")
    cv2.imwrite(original_output_path, image)
    print(f"Saved original image to {original_output_path}")
    
    # Save and colorize frontiers
    frontiers_colored = colorize_heatmap(frontiers_map)
    frontiers_output_path = os.path.join(str(output_subdir), "01_frontiers.jpg")
    cv2.imwrite(frontiers_output_path, frontiers_colored)
    print(f"Saved frontiers map to {frontiers_output_path}")
    
    # Save and colorize traversability
    traversability_colored = colorize_heatmap(traversability_map)
    traversability_output_path = os.path.join(str(output_subdir), "02_traversability.jpg")
    cv2.imwrite(traversability_output_path, traversability_colored)
    print(f"Saved traversability map to {traversability_output_path}")
    
    # Create and save overlay of traversable frontiers on original image
    frontiers_overlay = overlay_traversable_frontiers(image, frontiers_map, traversability_map, 
                                                      frontier_threshold=frontier_confidence, 
                                                      traversability_threshold=traversability_confidence)
    overlay_output_path = os.path.join(str(output_subdir), "03_traversable_frontier_overlay.jpg")
    cv2.imwrite(overlay_output_path, frontiers_overlay)
    print(f"Saved traversable frontier overlay to {overlay_output_path}")
    
    # Create and save overlay of traversability on original image
    traversability_overlay = overlay_traversability_on_image(image, traversability_map, threshold=traversability_confidence)
    traversability_overlay_path = os.path.join(str(output_subdir), "04_traversability_overlay.jpg")
    cv2.imwrite(traversability_overlay_path, traversability_overlay)
    print(f"Saved traversability overlay to {traversability_overlay_path}")
    
    # Create and save overlay of raw frontiers on original image (unfiltered by traversability)
    frontiers_unfiltered_overlay = overlay_frontiers_unfiltered(image, frontiers_map, threshold=frontier_confidence)
    frontiers_unfiltered_overlay_path = os.path.join(str(output_subdir), "05_frontiers_unfiltered_overlay.jpg")
    cv2.imwrite(frontiers_unfiltered_overlay_path, frontiers_unfiltered_overlay)
    print(f"Saved unfiltered frontiers overlay to {frontiers_unfiltered_overlay_path}")
    
    print(f"\n✓ Inference complete! Check the output directory for results.")
    print(f"  Frontier confidence threshold: {frontier_confidence}")
    print(f"  Traversability confidence threshold: {traversability_confidence}")
    return {
        'original_path': original_output_path,
        'frontiers_path': frontiers_output_path,
        'traversability_path': traversability_output_path,
        'overlay_path': overlay_output_path,
        'traversability_overlay_path': traversability_overlay_path,
        'frontiers_unfiltered_overlay_path': frontiers_unfiltered_overlay_path,
    }


if __name__ == "__main__":
   
    
    input_path = Path("test_images")
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_path.glob(ext)))

    if not image_files:
        print(f"No images found in {input_path}")
    
    
    for image_file in image_files:
        print(f"\nProcessing image: {image_file}")
        test_explorfm(str(image_file), "test_results", frontier_confidence=0.6, traversability_confidence=0.9)
