import numpy as np
import matplotlib.pyplot as plt
from QBVAE_utils import bwr_imshow

def visualize_mnist_images(
    images, 
    titles=None, 
    nw=8, 
    figsize=None, 
    cmap='gray', 
    suptitle=None,
    use_bwr=True):
    "Visualize multiple MNIST images in a grid format."
    # Convert flattened images to 2D if needed
    if len(images.shape) == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    
    n = len(images)
    nh = int(np.ceil(n / nw))  # Number of rows needed
    
    if figsize is None:
        figsize = (nw * 2, nh * 2)
    
    fig, axes = plt.subplots(nh, nw, figsize=figsize, squeeze=False)
    
    # Plot each image
    for i in range(n):
        row, col = i // nw, i % nw
        ax = axes[row, col]
        
        if use_bwr:
            bwr_imshow(ax, images[i])
        else:
            ax.imshow(images[i], cmap=cmap, vmin=0, vmax=1)
        
        ax.axis('off')
        if titles is not None and i < len(titles):
            ax.set_title(str(titles[i]), fontsize=10)
    
    # Hide unused subplots
    for i in range(n, nh * nw):
        row, col = i // nw, i % nw
        axes[row, col].axis('off')
    
    if suptitle: fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig, axes