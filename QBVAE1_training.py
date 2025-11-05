"""
Run this training script with

accelerate launch --mixed_precision="fp16" QBVAE1_training.py mnist10
"""
# %% Train on MNIST
import numpy as np
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from QBVAE_utils import BetaVAE, data_transform
from dataclasses import dataclass
import tyro
import jax_utils as ju

#%%
def train_vae(model, data, epochs=50, batch_size=128, lr=1e-3):
    # Prepare optimizer
    accelerator = Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Prepare dataloader
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Prepare model, optimizer, and dataloader with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop
    model.train()

    accelerator.print("\nModel:")
    accelerator.print(model)
    accelerator.print("\n\n")
    
    # Main training loop using tqdm for progress bars
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            x = batch[0]
            
            # Forward pass
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)

            # Access loss_function through .module if it's a DDP model
            if hasattr(model, 'module'):
                loss = model.module.loss_function(x, x_recon, mu, logvar)
            else:
                loss = model.loss_function(x, x_recon, mu, logvar)

            # loss = loss_function(model, x, x_recon, mu, logvar)
            
            # Backward pass with accelerator
            accelerator.backward(loss)
            optimizer.step()
            
            # Update progress bar and accumulate loss
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Print epoch summary
        avg_loss = total_loss / len(dataloader)
        accelerator.print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Unwrap model before returning
    return accelerator.unwrap_model(model), accelerator


@dataclass
class Config:
    outfile: str = "expresults/QBVAE--beta-vae-mnist.pt" # Where to save final checkpoint
    epochs: int = 50 # Number of epochs
    latent_dim: int = 10 # Dimension of final latent space
    hidden_dim: int = 512 # Dimension of hidden layer
    batch_size: int = 128 # Batch size
    lr: float = 1e-3 # Learning rate
    beta: float = 4.0 # scalar weight for KL divergence

default_configs = {
    "mnist10": (
        "MNIST with VAE encoding in 10 dimensions",
        Config(
            outfile="expresults/QBVAE--beta-vae-mnist10.pt", 
            epochs=50, 
            latent_dim=10, 
            hidden_dim=512, 
            batch_size=128, 
            lr=1e-3, 
            beta=4.0, 
        ),
    ),
    "mnist3": (
        "MNIST with VAE encoding in 3 dimensions",
        Config(
            outfile="expresults/QBVAE--beta-vae-mnist3.pt", 
            epochs=50, 
            latent_dim=10, 
            hidden_dim=512, 
            batch_size=128, 
            lr=1e-3, 
            beta=4.0, 
        ),
    ),
}

# Example usage:
if __name__ == "__main__":
    if ju.is_interactive():
        config = default_configs["mnist10"][1]
    else:
        config = tyro.extras.overridable_config_cli(default_configs)
    print(config)

    Xtrain, Xtest = torch.from_numpy(np.load("data/mnist/Xtrain.npy")), torch.from_numpy(np.load("data/mnist/Xtest.npy"))
    Xtrain = Xtrain.float()
    Xtest = Xtest.float()
    input_dim = 28*28  # MNIST images
    model = BetaVAE(input_dim=input_dim, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim, beta=config.beta)
    trained_model, accelerator = train_vae(model, data_transform(Xtrain), epochs=config.epochs, batch_size=config.batch_size, lr=config.lr)

    # Save the model
    if accelerator.is_main_process:
        print(f"Saving model to {config.outfile}")
        torch.save(trained_model.state_dict(), config.outfile)
