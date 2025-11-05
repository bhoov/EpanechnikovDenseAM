"""
Interactively check the quality of the MNIST training
"""
#%%
import torch
from QBVAE_utils import data_transform, load_bvae, load_data, batch_encode_data
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

#%% Loading the model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "expresults/QBVAE--beta-vae-mnist10.pt"
model = load_bvae(path)
Xtrain, Xtest = load_data()

model = model.to(DEVICE)
Xtrain = Xtrain.to(DEVICE)
Xtest = Xtest.to(DEVICE)

latents, mus, logvars = batch_encode_data(model, Xtrain)

# %% Can I generate meaningful samples from the model?
n_samples = 100
model.eval()
with torch.no_grad():
    # Sample from prior N(0,1)
    z = torch.randn(n_samples, model.latent_dim).to(DEVICE)
    samples = model.decode(z)

data_transform.decode(samples)

plt.imshow(data_transform.decode(samples)[85].cpu())

#%% Is the latent space semantically coherent?
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
real_img_idx = 76 # Nice 2
real_img = Xtrain[real_img_idx][None].to(DEVICE)

anchor_idxs = [74, 141] 
fig, axs = plt.subplots(1, 3)
axs[0].imshow(Xtrain[anchor_idxs[0]].cpu())
axs[0].set_title("Anchor 1")
axs[1].imshow(Xtrain[anchor_idxs[1]].cpu())
axs[1].set_title("Anchor 2")
axs[2].imshow(Xtrain[real_img_idx].cpu())
axs[2].set_title("Real")

model = model.to(DEVICE)
model.eval()

with torch.no_grad():
    mu, log_var = model.encode(data_transform(real_img))
    z_start = model.reparameterize(mu, log_var)

with torch.no_grad():
    ims = Xtrain[anchor_idxs].to(torch.float32).to(DEVICE)
    mus, log_vars = model.encode(data_transform(ims))
    mu1, mu2 = mus[0], mus[1]

# Create a 15x15 grid of latent vectors
grid_size = 20
alpha_range = np.linspace(0, 1, grid_size)  # Range of variation

# Initialize grid to store decoded images
decoded_grid = np.zeros((grid_size, grid_size, 28, 28))  # Assuming MNIST 28x28

# Generate the grid of latent vectors and decode them
with torch.no_grad():
    for i, alpha1 in enumerate(alpha_range):
        for j, alpha2 in enumerate(alpha_range):
            # Create a modified latent vector by moving in the two directions
            z_modified = z_start.clone()
            alpha0 = 1 - alpha1 - alpha2
            z_modified = alpha0 * z_start + alpha1 * mu1.unsqueeze(0) + alpha2 * mu2.unsqueeze(0)
            
            # Decode the modified latent vector
            decoded = model.decode(z_modified)
            img = data_transform.decode(decoded)[0]
            decoded_grid[i, j] = img.cpu().numpy()

# Plot the grid
fig2 = plt.figure(figsize=(12, 12))
for i in range(grid_size):
    for j in range(grid_size):
        plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
        plt.imshow(decoded_grid[i, j], cmap='gray')
        plt.axis('off')
        
        # Add red border around the original image
        if i == 0 and j == 0:
            plt.gca().add_patch(Rectangle((0, 0), 27, 27, linewidth=4, edgecolor='r', facecolor='none'))

plt.suptitle(f'Latent Space Traversal', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

fig2.savefig("figures/QBVAE--latent-space-traversal.png", dpi=300)
print("Saved to figures/QBVAE--latent-space-traversal.png")