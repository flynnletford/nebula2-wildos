# Quick Inference Reference for 3-Class Traversability

## Basic Inference Example

```python
import torch
import torch.nn.functional as F
from explorfm.explorfm_model import ExploRFM

# Load model with 3-class traversability head
model = ExploRFM(
    traversability_ckpt="ckpts/3class_traversability.ckpt",
    model_version="c-radio_v3-b",
    use_naclip=True,
)
model.eval()

# Prepare image (H, W, 3) -> (1, 3, H, W)
image = torch.rand(1, 3, 480, 640)  # Batch size 1

# Forward pass
with torch.no_grad():
    traversability_probs, frontiers, _ = model(image)

# traversability_probs shape: (1, 3, H, W) - softmax probabilities for 3 classes
# Class 0: Unsafe/obstacles
# Class 1: Safe terrain
# Class 2: Mildly risky (grass)
```

## Getting Predictions

```python
# Option 1: Class predictions (argmax)
pred_classes = torch.argmax(traversability_probs, dim=1)  # Shape: (1, H, W)
pred_classes = pred_classes.squeeze(0).cpu().numpy()

# Option 2: Per-class probabilities
unsafe_prob = traversability_probs[0, 0].cpu().numpy()      # (H, W)
safe_prob = traversability_probs[0, 1].cpu().numpy()        # (H, W)
risky_prob = traversability_probs[0, 2].cpu().numpy()       # (H, W)

# Option 3: Confidence
confidence = torch.max(traversability_probs, dim=1)[0]      # (1, H, W)
confidence = confidence.squeeze(0).cpu().numpy()
```

## Creating Navigation Costs

```python
import numpy as np

# Method 1: Simple cost map (0-1)
cost_map = np.ones_like(pred_classes, dtype=np.float32)
cost_map[pred_classes == 0] = 1.0   # Completely blocked
cost_map[pred_classes == 1] = 0.0   # Free to traverse
cost_map[pred_classes == 2] = 0.5   # Avoid but passable (grass)

# Method 2: Cost from probability (better)
cost_map = np.ones_like(safe_prob, dtype=np.float32)
cost_map = 1.0 - safe_prob  # High cost where not safe

# Increase cost for grass regions
cost_map[pred_classes == 2] *= 1.5

# Decrease cost for confident safe regions
confident_safe = (pred_classes == 1) & (confidence > 0.8)
cost_map[confident_safe] = 0.0

# Method 3: Multi-factor cost
cost_map = (
    0.5 * (1.0 - safe_prob) +           # Probability-based
    0.3 * (risky_prob > 0.5).astype(float) +  # Grass presence
    0.2 * (1.0 - confidence)                   # Confidence penalty
)
```

## For Navigation Planning

```python
# Get traversable regions (safe + passable grass)
traversable = pred_classes != 0

# Get preferred regions (safe only)
preferred = pred_classes == 1

# Get grass regions for logging/analysis
grass_regions = pred_classes == 2

# Create binary masks
safe_mask = (pred_classes == 1).astype(np.uint8)
grass_mask = (pred_classes == 2).astype(np.uint8)
unsafe_mask = (pred_classes == 0).astype(np.uint8)

# Path planning with avoidance
# Pass cost_map to your path planner (A*, RRT*, etc.)
# Path planner will prefer low-cost (safe) regions over high-cost (grass) regions
```

## Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_traversability(image, pred_classes, probs):
    """Create a comprehensive visualization."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Class predictions (color-coded)
    color_pred = np.zeros((*pred_classes.shape, 3), dtype=np.uint8)
    color_pred[pred_classes == 0] = [255, 0, 0]      # Red: unsafe
    color_pred[pred_classes == 1] = [0, 255, 0]      # Green: safe
    color_pred[pred_classes == 2] = [255, 255, 0]    # Yellow: grass
    
    axes[0, 1].imshow(color_pred)
    axes[0, 1].set_title("Class Predictions")
    
    # Create legend
    red_patch = mpatches.Patch(color=[1, 0, 0], label='Unsafe')
    green_patch = mpatches.Patch(color=[0, 1, 0], label='Safe')
    yellow_patch = mpatches.Patch(color=[1, 1, 0], label='Grass')
    axes[0, 1].legend(handles=[red_patch, green_patch, yellow_patch], loc='upper right')
    axes[0, 1].axis('off')
    
    # Unsafe probability
    im0 = axes[0, 2].imshow(probs[0], cmap='Reds')
    axes[0, 2].set_title("Unsafe Probability")
    plt.colorbar(im0, ax=axes[0, 2])
    axes[0, 2].axis('off')
    
    # Safe probability
    im1 = axes[1, 0].imshow(probs[1], cmap='Greens')
    axes[1, 0].set_title("Safe Probability")
    plt.colorbar(im1, ax=axes[1, 0])
    axes[1, 0].axis('off')
    
    # Grass probability
    im2 = axes[1, 1].imshow(probs[2], cmap='YlOrBr')
    axes[1, 1].set_title("Grass Probability")
    plt.colorbar(im2, ax=axes[1, 1])
    axes[1, 1].axis('off')
    
    # Overlay on image
    overlay = image.copy()
    grass_overlay = color_pred.copy()
    grass_overlay[pred_classes != 2] = 0
    
    axes[1, 2].imshow(image)
    axes[1, 2].imshow(grass_overlay, alpha=0.4)
    axes[1, 2].set_title("Grass Overlay on Image")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig

# Usage
image_np = torch.rand(3, 480, 640).permute(1, 2, 0).numpy()
probs_np = traversability_probs[0].cpu().numpy()
pred_classes_np = pred_classes[0].cpu().numpy()

fig = visualize_traversability(image_np, pred_classes_np, probs_np)
plt.savefig('traversability_prediction.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Real-time Processing

```python
def process_camera_stream(camera_feed, model, device='cuda'):
    """Process camera stream for real-time traversability estimation."""
    
    model = model.to(device).eval()
    
    for frame in camera_feed:
        # Preprocess
        image = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Inference
        with torch.no_grad():
            traversability_probs, _, _ = model(image)
        
        # Post-process
        pred_classes = torch.argmax(traversability_probs, dim=1)[0].cpu().numpy()
        confidence = torch.max(traversability_probs, dim=1)[0][0].cpu().numpy()
        
        # Create cost map
        cost_map = np.ones_like(pred_classes, dtype=np.float32)
        cost_map[pred_classes == 0] = 1.0
        cost_map[pred_classes == 1] = 0.0
        cost_map[pred_classes == 2] = 0.5
        
        # Send to navigator/planner
        yield {
            'class_predictions': pred_classes,
            'confidence': confidence,
            'cost_map': cost_map,
            'grass_mask': pred_classes == 2,
        }
```

## Key Points for Robot Testing

1. **Cost Map**: Use the cost_map for path planning - the planner will naturally prefer safe regions
2. **Grass Regions**: Track where grass was detected for post-run analysis
3. **Confidence**: Ignore predictions with low confidence (< 0.6) for critical decisions
4. **Real-time**: Use GPU for real-time inference on robot
5. **Fallback**: Have a conservative fallback (avoid everything) if confidence is very low

## Troubleshooting

- **Model outputs NaN**: Check input range (should be 0-1 normalized)
- **All predictions are "safe"**: Model may not have learned grass, check training
- **Very slow inference**: Use batch processing or reduce image resolution
- **Robot still hits grass**: Increase cost for grass class (adjust weights) or retrain
