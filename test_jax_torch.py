#%%
import torch
import jax
import jax.numpy as jnp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

def test_pytorch():
    print("\n=== PyTorch Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Test tensor creation and operations
    x = torch.randn(3, 3)
    if torch.cuda.is_available():
        x = x.cuda()
    y = x * 2 + 1
    print("\nTensor operations test:")
    print(f"Original tensor:\n{x}")
    print(f"Transformed tensor (x * 2 + 1):\n{y}")

    # Multi-GPU tests for PyTorch
    if torch.cuda.device_count() > 1:
        print("\nMulti-GPU PyTorch tests:")
        # Test data transfer between devices
        tensor1 = torch.randn(1000, 1000).cuda(0)
        tensor2 = tensor1.cuda(1)
        print(f"Data transfer test - tensors on different devices: {tensor1.device} -> {tensor2.device}")
        
        # Test distributed computation
        input_tensor = torch.randn(100, 1000)
        # Split computation across GPUs
        splits = torch.chunk(input_tensor, 2)
        results = []
        for i, split in enumerate(splits):
            # Move both model and data to the same device
            device = f'cuda:{i}'
            model = torch.nn.Linear(1000, 1000).to(device)
            results.append(model(split.to(device)))
        
        # Move all results to the same device before concatenating
        results = [r.to('cuda:0') for r in results]
        result = torch.cat(results, dim=0)
        print(f"Distributed computation test - Result shape: {result.shape}")

def test_jax():
    print("\n=== JAX Test ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    
    # Test array creation and operations
    x = jnp.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    y = x * 2 + 1
    print("\nArray operations test:")
    print(f"Original array:\n{x}")
    print(f"Transformed array (x * 2 + 1):\n{y}")

    # Multi-GPU tests for JAX
    if len(jax.devices()) > 1:
        print("\nMulti-GPU JAX tests:")
        # Test pmap for parallel computation
        def parallel_computation(x):
            return jnp.sum(x ** 2)
        
        parallel_computation = jax.pmap(parallel_computation)
        data = jnp.arange(8).reshape((2, 4))  # 2 devices, 4 elements each
        result = parallel_computation(data)
        print(f"Parallel computation test - Input shape: {data.shape}, Result: {result}")

        # Test device memory transfer
        x = jax.device_put(jnp.ones((1000, 1000)), jax.devices()[0])
        y = jax.device_put(x, jax.devices()[1])
        print(f"Data transfer test - Arrays on different devices:")
        print(f"Source: {x.devices()}")
        print(f"Destination: {y.devices()}")

if __name__ == "__main__":
    print("Testing GPU availability and basic operations for PyTorch and JAX")
    test_pytorch()
    test_jax()
# %%
