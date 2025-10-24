#%%
from datasets import load_dataset
import numpy as np
import path_fixes as pf

def prep_tinyimnet_data():
    # Check if Xtrain.npy exists
    if not (pf.TINYIMGNET / "Xtrain.npy").exists():
        print("Prepping TinyImgnet data...")
        ds = load_dataset("zh-plus/tiny-imagenet").with_format("jax")
        desired_shape = (64, 64, 3)
        Xtrain = np.stack([im for im in ds['train']['image'] if im.shape == desired_shape])
        Xtest = np.stack([im for im in ds['valid']['image'] if im.shape == desired_shape])
        Xtrain = Xtrain / 255.
        Xtest = Xtest / 255.
        np.save(pf.TINYIMGNET/"Xtrain.npy", Xtrain)
        np.save(pf.TINYIMGNET/"Xtest.npy", Xtest)
    else:
        print("TinyImgnet data already exists")

def get_tiny_imagenet_traindata():
    Xtrain = np.load(pf.TINYIMGNET / "Xtrain.npy")
    Ytrain = None
    return Xtrain, Ytrain

def get_tiny_imagenet_testdata():
    Xtest = np.load(pf.TINYIMGNET / "Xtest.npy")
    Ytest = None
    return Xtest, Ytest

def prep_mnist_data():
    # Check if Xtrain.npy exists
    if not (pf.MNIST / "Xtrain.npy").exists():
        print("Prepping MNIST data...")
        ds = load_dataset('mnist').with_format('np')
        Xtrain = np.array(ds['train']['image'])  # Convert to numpy array
        Ytrain = np.array(ds['train']['label'])  # Convert to numpy array
        Xtrain = Xtrain / 255.

        Xtest = np.array(ds['test']['image'])   # Convert to numpy array
        Ytest = np.array(ds['test']['label'])   # Convert to numpy array
        Xtest = Xtest / 255.

        np.save(pf.MNIST / "Xtrain.npy", Xtrain)
        np.save(pf.MNIST / "Ytrain.npy", Ytrain)
        np.save(pf.MNIST / "Xtest.npy", Xtest)
        np.save(pf.MNIST / "Ytest.npy", Ytest)
    else:
        print("MNIST data already exists")


def get_mnist_traindata():
    Xtrain, Ytrain = np.load(pf.MNIST / "Xtrain.npy"), np.load(pf.MNIST / "Ytrain.npy")
    return Xtrain, Ytrain

def get_mnist_testdata():
    Xtest, Ytest = np.load(pf.MNIST / "Xtest.npy"), np.load(pf.MNIST / "Ytest.npy")
    return Xtest, Ytest

def prep_cifar_data():
    # Check if Xtrain.npy exists
    if not (pf.CIFAR10 / "Xtrain.npy").exists():
        print("Prepping CIFAR data...")
        ds = load_dataset('cifar10').with_format('np')
        Xtrain = np.array(ds['train']['img'])  # Convert to numpy array
        Ytrain = np.array(ds['train']['label'])
        Xtrain = Xtrain / 255.

        Xtest = np.array(ds['test']['img'])  # Convert to numpy array
        Ytest = np.array(ds['test']['label'])
        Xtest = Xtest / 255.

        np.save(pf.CIFAR10 / "Xtrain.npy", Xtrain)
        np.save(pf.CIFAR10 / "Ytrain.npy", Ytrain)
        np.save(pf.CIFAR10 / "Xtest.npy", Xtest)
        np.save(pf.CIFAR10 / "Ytest.npy", Ytest)
    else:
        print("CIFAR data already exists")

def get_cifar_traindata():
    Xtrain, Ytrain = np.load(pf.CIFAR10 / "Xtrain.npy"), np.load(pf.CIFAR10 / "Ytrain.npy")
    return Xtrain, Ytrain

def get_cifar_testdata():
    Xtest, Ytest = np.load(pf.CIFAR10 / "Xtest.npy"), np.load(pf.CIFAR10 / "Ytest.npy")
    return Xtest, Ytest

def get_tiny_imagenet_traindata():
    Xtrain = np.load(pf.TINYIMGNET / "Xtrain.npy")
    Ytrain = None
    return Xtrain, Ytrain

def get_tiny_imagenet_testdata():
    Xtest = np.load(pf.TINYIMGNET / "Xtest.npy")
    Ytest = None
    return Xtest, Ytest


if __name__ == "__main__":
    prep_tinyimnet_data()
    prep_mnist_data()
    prep_cifar_data()

# %%
