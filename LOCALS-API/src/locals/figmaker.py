import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

import os

from .constants import NUM_GRID_CELLS
from .math import smooth_curve, closest_factors

image_size = 448

def save_loss_fig(train_loss_ot: list, val_loss_ot: list):
    os.makedirs('figures', exist_ok=True)
    
    # smooth the losses
    smooth_train_loss = smooth_curve(train_loss_ot)
    smooth_val_loss = smooth_curve(val_loss_ot) if val_loss_ot else None

    # plot the smoothed losses
    epochs = range(1, len(train_loss_ot) + 1)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, smooth_train_loss, color='blue', linewidth=2, label='Train Loss')
    if val_loss_ot:
        plt.plot(epochs, smooth_val_loss, color='red', linewidth=2, label='Validation Loss')
    plt.title(r'Smoothed Training Loss Over Epochs', fontsize=14)
    plt.xlabel(r'Epoch', fontsize=12)
    plt.ylabel(r'Loss', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.legend()
    plt.savefig("figures/smoothed_training_loss.png", dpi=300)
    
def visualise_dataset(data, num_images, plot_title=None):
    os.makedirs('figures', exist_ok=True)
    
    if not isinstance(data, DataLoader):
        data = DataLoader(
            data,
            batch_size=1,
            shuffle=False
        )
    
    rows, cols = closest_factors(num_images)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, max(3 * rows, 4)), squeeze=False)
    fig.suptitle('Visualising Dataset of LOCALS for Pencils') if not plot_title else fig.suptitle(plot_title)
    image_idx = 0
    axes = axes.flatten()
    
    for images, labels in data:
        if image_idx >= num_images: break
        for i, image in enumerate(images):
            if image_idx >= num_images: break
            image_plt = image.permute(1, 2, 0)
            ax = axes[image_idx]
            
            ax.axis('off')
            label = labels[i]
            ax.set_title(f'Sample {image_idx + 1}')
            ax.imshow(image_plt)
            
            grid_size = image_size / NUM_GRID_CELLS
            for g in range(NUM_GRID_CELLS + 1):
                pos = g * grid_size
                
                # vertical lines
                ax.axvline(
                    x=pos,
                    color='white',
                    linewidth=0.5,
                    alpha=0.5
                )

                # horizontal lines
                ax.axhline(
                    y=pos,
                    color='white',
                    linewidth=0.5,
                    alpha=0.5
                )
            
            for j in range(NUM_GRID_CELLS):
                for k in range(NUM_GRID_CELLS):
                    xnb, ynb, c = label[j][k]
                    if c < 0.5:
                        continue
                    x = ((k / NUM_GRID_CELLS) + (xnb * (1 / NUM_GRID_CELLS))) * image_size
                    y = ((j / NUM_GRID_CELLS) + (ynb * (1 / NUM_GRID_CELLS))) * image_size
                    
                    ax.scatter(x, y, color='red', marker='x', s=40)
            image_idx += 1
    
    plt.savefig('figures/visualised_dataset.png', dpi=300, bbox_inches='tight') if not plot_title else \
        plt.savefig(f"figures/visualised_dataset_{plot_title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def visualise_predictions(model, data, num_images, plot_title=None):
    os.makedirs('figures', exist_ok=True)
    
    if not isinstance(data, DataLoader):
        data = DataLoader(
            data,
            batch_size=1,
            shuffle=False
        )
    model.eval()

    with torch.no_grad():
        rows, cols = closest_factors(num_images)
        fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 3 * rows + 1),
        squeeze=False
    )
        fig.suptitle('Visualising Predictions of LOCALS for Pencils') if not plot_title else fig.suptitle(plot_title)
        axes = axes.flatten()
        image_idx = 0
        
        for images, labels in data:
            if image_idx >= num_images: break
            outputs = model(images.to(model.device))
            for i, image in enumerate(images):
                image_plt = image.permute(1, 2, 0)
                label = labels[i]
                output = outputs[i].detach().cpu()
                ax = axes[image_idx]
                ax.axis('off')
                ax.imshow(image_plt.detach().cpu())
                ax.set_title(f'Prediction {image_idx + 1}')
                grid_size = image_size / NUM_GRID_CELLS
                
                for g in range(NUM_GRID_CELLS + 1):
                    pos = g * grid_size
                    
                    # vertical lines
                    ax.axvline(
                        x=pos,
                        color='white',
                        linewidth=0.5,
                        alpha=0.5
                    )

                    # horizontal lines
                    ax.axhline(
                        y=pos,
                        color='white',
                        linewidth=0.5,
                        alpha=0.5
                    )

                for j in range(NUM_GRID_CELLS):
                    for k in range(NUM_GRID_CELLS):
                        
                        # ground truth
                        xnb, ynb, c = label[j][k]

                        if c.item() > 0.5:

                            x = ((k / NUM_GRID_CELLS) + (xnb * (1 / NUM_GRID_CELLS))) * image_size
                            y = ((j / NUM_GRID_CELLS) + (ynb * (1 / NUM_GRID_CELLS))) * image_size
                            ax.scatter(
                                x.detach().cpu(),
                                y.detach().cpu(),
                                color='yellow',
                                marker='x',
                                s=40,
                                alpha=1
                            )

                        # prediction
                        pxnb, pynb, pc = output[j][k]

                        if pc.item() >= 0.5:

                            x = ((k / NUM_GRID_CELLS) + (pxnb * (1 / NUM_GRID_CELLS))) * image_size
                            y = ((j / NUM_GRID_CELLS) + (pynb * (1 / NUM_GRID_CELLS))) * image_size

                            ax.scatter(
                                x.detach().cpu(),
                                y.detach().cpu(),
                                color='red',
                                marker='x',
                                s=40,
                                alpha=0.5,
                            )
                            
                            ax.text(
                                x.detach().cpu() + 5, # x offset
                                y.detach().cpu() + 5, # y offset
                                f'{pc.detach().cpu():.2f}',
                                color='white',
                                fontsize=10
                            )
                
                image_idx += 1
                
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        legend_elements = [
            Line2D([0], [0], marker='x', color='yellow',
                linestyle='None', markersize=8, label='Ground Truth'),
            Line2D([0], [0], marker='x', color='red',
                linestyle='None', markersize=8, label='Prediction')
        ]

        fig.legend(
            handles=legend_elements,
            loc='lower left',
            ncol=2,
            bbox_to_anchor=(0.0, 0.0)
        )
        
        plt.savefig(f"figures/visualised_predictions_{plot_title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight') if plot_title else \
            plt.savefig(f'figures/visualised_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
def save_recall_precision_f1_score(recall, precision, f1_score):
    os.makedirs('figures', exist_ok=True)
    
    metrics = ['Recall', 'Precision', 'F1 Score']
    values = [recall, precision, f1_score]

    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1.1)
    plt.title('Model Classification Performance Metrics')
    plt.ylabel('Score')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("figures/recall-precision-f1score.png", dpi=300)