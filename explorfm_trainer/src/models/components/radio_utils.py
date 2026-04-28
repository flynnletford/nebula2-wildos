import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

RADIO_MODEL_VERSIONS = [
    "radio_v2.5-g", # for RADIOv2.5-g model (ViT-H/14)
    "radio_v2.5-h", # for RADIOv2.5-H model (ViT-H/16)
    "radio_v2.5-l", # for RADIOv2.5-L model (ViT-L/16)
    "radio_v2.5-b", # for RADIOv2.5-B model (ViT-B/16)
    "c-radio_v3-b", # for C_RADIOv3-B model (ViT-B/16)
    "c-radio_v3-l", # for C_RADIOv3-L model (ViT-L/16)
    "e-radio_v2", # for E-RADIO
]
RADIO_ADAPTOR_VERSIONS = {
    "none": 768,  # No adaptor
    "clip": 1280,  # CLIP adaptor
    "siglip": None,  # SigLIP adaptor
    "siglip2": 1152,  # SigLIP2 adaptor
    "dino_v2": 1536,  # DINO adaptor
    "sam": 1280,  # SAM adaptor
}

def gen_logging_image(
        batch_data: dict,
        seg_colormap: dict,
        num_log_imgs: int,
        cmap: str = "inferno",
        vmax: float = 1,
        num_classes: int = 1,
    ) -> np.ndarray:
    """
    Visualize the predictions. 
    Displays the original image, ground truth, the heatmap of predictions, and the predicted mask.
    Also display the legend for the segmentation categories and the heatmap.
    
    :param num_classes: Number of output classes. 1 for binary, 3 for traversability.
    """
    B = batch_data["raw_img"].shape[0]
    idxs = np.random.choice(
        range(B), min(num_log_imgs, B), replace=False
    )  # randomly select indices for logging

    log_imgs = []
    for i in idxs:
        img_rgb = batch_data["raw_img"][i].cpu().numpy().transpose(1, 2, 0)
        img_rgb = (img_rgb * 255).astype(np.uint8)  # convert to uint8
        gt_seg = batch_data["gt_segmentation"][i].cpu().numpy()
        gt_traversability = batch_data["gt_traversability"][i].cpu().numpy()
        preds = batch_data["preds"][i].cpu().numpy()
        probs = batch_data["probs"][i].cpu().numpy()

        # Handle multi-class case
        if num_classes > 1:
            # For 3-class: probs is (3, H, W), extract max probability
            probs_viz = np.max(probs, axis=0)  # Max probability for any class
            # Create colored class map for predictions
            preds_colored = np.zeros((*preds.shape, 3), dtype=np.float32)
            # Class 0 (safe) = green
            preds_colored[preds == 0] = [0, 1, 0]
            # Class 1 (mildly_dangerous) = yellow
            preds_colored[preds == 1] = [1, 1, 0]
            # Class 2 (untraversable) = red
            preds_colored[preds == 2] = [1, 0, 0]
            
            # Create colored ground truth map
            gt_colored = np.zeros((*gt_traversability.shape, 3), dtype=np.float32)
            gt_colored[gt_traversability == 0] = [0, 1, 0]
            gt_colored[gt_traversability == 1] = [1, 1, 0]
            gt_colored[gt_traversability == 2] = [1, 0, 0]
        else:
            # Binary case
            probs_viz = probs[0]  # Single channel
            preds_colored = None

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes[0, 2].axis('off')
        
        # 1. Original RGB Image
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title(f'Original RGB: {batch_data['img_path'][i]}')
        axes[0, 0].axis('off')
        
        # 2. Ground Truth Segmentation with Color Mapping
        axes[0, 1].imshow(img_rgb)
        axes[0, 1].imshow(gt_seg, alpha=0.5)
        axes[0, 1].set_title('Ground Truth Segmentation')
        axes[0, 1].axis('off')

        # 3. Heatmap (probability scores)
        axes[1, 0].imshow(img_rgb)
        heatmap = axes[1, 0].imshow(probs_viz, cmap=cmap, vmin=0, vmax=vmax, alpha=0.5)
        axes[1, 0].set_title('Predicted Probabilities')
        axes[1, 0].axis('off')
        plt.colorbar(heatmap, ax=axes[1,0], fraction=0.046, pad=0.04)

        # 4. Predicted mask/class map
        axes[1, 1].imshow(img_rgb)
        if num_classes > 1:
            axes[1, 1].imshow(preds_colored, alpha=0.5)
            axes[1, 1].set_title('Predicted 3-Class Map (G=safe, Y=mild, R=untrav)')
        else:
            axes[1, 1].imshow(preds, cmap='gray', alpha=0.5)
            axes[1, 1].set_title('Predicted Binary Mask')
        axes[1, 1].axis('off')

        # 5. Ground Truth Traversability
        axes[1, 2].imshow(img_rgb)
        if num_classes > 1:
            axes[1, 2].imshow(gt_colored, alpha=0.5)
            axes[1, 2].set_title('Ground Truth 3-Class Map (G=safe, Y=mild, R=untrav)')
        else:
            axes[1, 2].imshow(gt_traversability, cmap='gray', alpha=0.5)
            axes[1, 2].set_title('Ground Truth Safe Mask')
        axes[1, 2].axis('off')

        # Add segmentation legend below all subplots
        handles = [
            mpatches.Patch(color=np.array(color)/255.0, label=label)
            for label, color in seg_colormap.items()
        ]
        fig.legend(handles=handles, loc='upper right', ncol=4, fontsize='small', frameon=False)

        plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for the legend
        
        # Convert plot to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]

        plt.close(fig)  # Close the figure to free memory
        log_imgs.append(data)

    return log_imgs