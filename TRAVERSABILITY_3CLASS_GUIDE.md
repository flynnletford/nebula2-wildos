# 3-Class Traversability Training Guide

## Overview
The traversability head has been modified to support 3-class classification instead of binary classification:
- **Class 0**: Unsafe/Non-traversable terrain (obstacles, water, etc.)
- **Class 1**: Safe/Traversable terrain (dirt, sand, asphalt, concrete, etc.)
- **Class 2**: Mildly Risky terrain (grass) - **Avoidable but passable**

This allows the robot to prefer concrete over grass while still being able to traverse grass if necessary.

## Changes Made

### 1. Data Modules
**Files Modified:**
- `explorfm_trainer/src/data/rugd_traversability_datamodule.py`
- `explorfm_trainer/src/data/goose-ex_traversability_datamodule.py`

**Changes:**
- Split `safe_labels` into two lists:
  - `safe_labels`: All traversable terrain except grass
  - `risky_labels`: Contains "grass" (for RUGD) or "low_grass" (for GooseEx)
- Modified `get_traversability()` method to output 3-class labels instead of binary
- Updated `__getitem__()` to return labels as `torch.long` type (required for CrossEntropyLoss)
- Updated validation assertion to check for values in {0, 1, 2}

### 2. Training Module
**New File:**
- `explorfm_trainer/src/models/multiclass_segmentation_module.py`

**Features:**
- Inherits from LightningModule and supports 3-class classification
- Uses `CrossEntropyLoss` instead of binary cross-entropy loss
- Supports optional class weights for imbalanced datasets
- Uses `torch.argmax()` for predictions instead of sigmoid thresholding
- Automatically converts to multi-class metrics (Accuracy, MeanIoU)

### 3. Model Head
**File Modified:**
- `explorfm/explorfm_model.py`

**Changes:**
- Last layer of traversability head: `Conv2d(..., out_channels=3)` (was 1)
- Forward pass uses `F.softmax(dim=1)` instead of `F.sigmoid()`
- Added graceful fallback for loading old checkpoints (partial weight loading)
- Updated docstring to explain the 3 classes

## Training Configuration

The existing experiment configs have been minimally updated to support 3-class training:

- `explorfm_trainer/configs/experiment/rugd_radio_cnn.yaml` - Added `num_classes: 3` and `class_weights: [1.0, 1.0, 2.0]`
- `explorfm_trainer/configs/experiment/gooseex_radio_cnn.yaml` - Added `num_classes: 3` and `class_weights: [1.0, 1.0, 2.0]`

No new config files needed! The training module conditionally handles both binary and multiclass based on `num_classes` parameter.

## Training

### Start Training
```bash
cd explorfm_trainer

# Train 3-class traversability head on RUGD
python train.py experiment=rugd_radio_cnn

# Or train on GooseEx
python train.py experiment=gooseex_radio_cnn

# Or with SLURM
sbatch scripts/slurm_trainer.sh experiment=rugd_radio_cnn
```

## Inference and Post-Processing

### Getting Class Predictions
```python
import torch
import torch.nn.functional as F

# After forward pass, you get logits of shape (batch, 3, H, W)
logits = model.traversability_head(features)
logits = F.interpolate(logits, size=input_shape, mode='bilinear')
probs = F.softmax(logits, dim=1)  # Shape: (batch, 3, H, W)

# Get class predictions
pred_classes = torch.argmax(probs, dim=1)  # Shape: (batch, H, W) with values {0, 1, 2}

# Get confidence scores
confidence, _ = torch.max(probs, dim=1)  # Shape: (batch, H, W)

# Separate into binary masks for easier handling
safe_mask = (pred_classes == 1).float()           # Safe terrain
risky_mask = (pred_classes == 2).float()          # Grass
unsafe_mask = (pred_classes == 0).float()         # Obstacles
```

### Cost Map for Navigation
```python
import numpy as np

# Create a traversability cost map (0-1, lower is better)
# You can adjust these values based on your robot's capabilities
safe_cost = 0.0           # Safe terrain
risky_cost = 0.5          # Grass - moderate cost, prefer to avoid
unsafe_cost = 1.0         # Obstacles - completely blocked

# Create cost map
cost_map = np.ones_like(pred_classes, dtype=np.float32)
cost_map[pred_classes == 1] = safe_cost
cost_map[pred_classes == 2] = risky_cost
cost_map[pred_classes == 0] = unsafe_cost

# Weight by confidence (optional)
cost_map = cost_map * (1.0 - confidence)  # Lower confidence = higher cost

# Or use the risky mask for grass avoidance in your planner
risky_regions = pred_classes == 2
```

### Visualization
```python
import matplotlib.pyplot as plt

def visualize_3class_traversability(image, pred_classes, probs):
    """Visualize the 3-class predictions."""
    
    # Create color map: 0=red (unsafe), 1=green (safe), 2=yellow (risky)
    color_map = np.zeros((*pred_classes.shape, 3), dtype=np.uint8)
    color_map[pred_classes == 0] = [255, 0, 0]      # Red for unsafe
    color_map[pred_classes == 1] = [0, 255, 0]      # Green for safe
    color_map[pred_classes == 2] = [255, 255, 0]    # Yellow for grass/risky
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Class predictions
    axes[0, 1].imshow(color_map)
    axes[0, 1].set_title("Class Predictions\n(Green=Safe, Yellow=Risky, Red=Unsafe)")
    axes[0, 1].axis('off')
    
    # Safe probability
    axes[1, 0].imshow(probs[1], cmap='viridis')
    axes[1, 0].set_title("Safe Probability")
    axes[1, 0].colorbar()
    axes[1, 0].axis('off')
    
    # Risky (grass) probability
    im = axes[1, 1].imshow(probs[2], cmap='RdYlGn_r')
    axes[1, 1].set_title("Risky/Grass Probability")
    plt.colorbar(im, ax=axes[1, 1])
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig
```

## Testing with Different Weight Configurations

### Experiment 1: Prefer Concrete (High Grass Cost)
```yaml
class_weights: [1.0, 1.0, 3.0]  # High weight on grass makes model avoid it
```

### Experiment 2: Balanced
```yaml
class_weights: [1.0, 1.0, 1.0]  # Equal weight for all classes
```

### Experiment 3: More Tolerant
```yaml
class_weights: [1.0, 1.0, 1.5]  # Lower grass weight if you have limited traversable options
```

## Loading and Using Pre-trained 3-Class Head

### In Python
```python
from explorfm.explorfm_model import ExploRFM

# Load with 3-class traversability head
model = ExploRFM(
    traversability_ckpt="path/to/3class_checkpoint.ckpt",
    model_version="c-radio_v3-b",
    use_naclip=True,
)

# Inference
traversability_3class, frontiers, _ = model(image_tensor)
# traversability_3class shape: (batch, 3, H, W) with softmax probabilities
```

### Backward Compatibility
The model can load old binary (1-channel) checkpoints with the following code:

```python
# The model will attempt to load the checkpoint
# If the last layer shape doesn't match, it will:
# 1. Load weights for layers 0-8
# 2. Reinitialize layer 9 (the output layer) with 3 channels
# This allows transfer learning from binary-trained models
```

## Robot Testing

Once trained, test on your robot:

1. **Export to ONNX** (if needed):
   ```bash
   python -m explorfm.export_onnx \
     --traversability_ckpt path/to/checkpoint.ckpt \
     --output_path model.onnx
   ```

2. **Collect traversability measurements**:
   - Document grass avoidance effectiveness
   - Measure success rate on concrete vs grass routes
   - Collect feedback for model improvement

3. **Iterate**:
   - Adjust `class_weights` based on performance
   - Retrain with more diverse data if needed
   - Experiment with cost map values in navigation planner

## Key References

- **Data Modules**: `explorfm_trainer/src/data/rugd_traversability_datamodule.py`
- **Training Module**: `explorfm_trainer/src/models/multiclass_segmentation_module.py`
- **Model Head**: `explorfm/explorfm_model.py`
- **Configurations**: `explorfm_trainer/configs/experiment/` and `explorfm_trainer/configs/model/`

## Troubleshooting

### Issue: "Shape mismatch when loading checkpoint"
**Solution**: The model automatically handles this. It will load all weights except the last layer and reinitialize it.

### Issue: "Memory issues with 3-channel output"
**Solution**: Reduce batch size or use gradient accumulation.

### Issue: "Model not learning grass distinction"
**Solution**: 
- Increase `class_weights[2]` (grass weight)
- Ensure your dataset has sufficient grass samples
- Use data augmentation to increase grass variety

### Issue: "ONNX export fails"
**Solution**: Ensure you're exporting from a trained 3-class checkpoint, not a binary one.
