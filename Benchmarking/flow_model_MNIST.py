import torch 
import matplotlib.pyplot as plt 
import torch.optim as optim
from torch import nn
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import ssl

import sys
import os

# Adds the parent directory ('the folder before it') to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Flow_Model_NN import FlowFieldNetwork
from Flow_Model_Training import instantiate_training_func
from Flow_model_sampler import guided_flow_sampling, render_image

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

if __name__ == "__main__":
    # Fix for macOS SSL certificate verification error during download
    ssl._create_default_https_context = ssl._create_unverified_context

    print("Loading real MNIST dataset...")
    
    # 1. Setup dimensions for MNIST
    DATA_DIM = 28 * 28  # 784 features for flattened 28x28 grayscale image
    LABEL_DIM = 10      # 10 classes for digits 0 through 9
    BATCH_SIZE = 512

    # CONTINUE_PREV_TRAINING = True # Set this to True if you want to overwrite the saved model
    # # 2. Instantiate model on the device
    MODEL_WEIGHTS_PATH = "TRAINED_MODEL/MNIST_FLOW_MODEL_3_layer.pth"
    model = FlowFieldNetwork(DATA_DIM, LABEL_DIM).to(device)

    # if CONTINUE_PREV_TRAINING and os.path.exists(MODEL_WEIGHTS_PATH):
    #     model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))

    # 3. Setup transforms for flattening images to 784 vectors and one-hot encoding labels
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    target_transform = transforms.Lambda(
        lambda y: torch.nn.functional.one_hot(torch.tensor(y), num_classes=LABEL_DIM).float()
    )
    
    # 4. Download and load real MNIST dataset
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform,
        target_transform=target_transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 5. Instantiate training closure
    train_func = instantiate_training_func(
        model = model, 
        dataloader = dataloader, 
        input_vector_dim = DATA_DIM, 
        model_name = "MNIST_FLOW_MODEL",
        epochs= 1200,
        learning_rate= 1 * 10 **(-3)
    )

    FORCE_RETRAIN = False # Set this to True if you want to overwrite the saved model

    if os.path.exists(MODEL_WEIGHTS_PATH) and not FORCE_RETRAIN:

        print(f"Found saved weights at {MODEL_WEIGHTS_PATH}. Loading model...")
        # Load the saved state dictionary into the model
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
        print("Model loaded successfully! Skipping training.")

    else:
        # 6. Execute training test
        print("Starting training run on MNIST data...")
        train_func()
        print("MNIST training completed successfully!")

    # Use trained model to output an image 
    input_noise = torch.randn([1, DATA_DIM], device = device)

    #choose a label for no.6
    input_label = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], device= device)
    null_label = -1 * torch.ones_like(input_label)

    #display this output image
    output_image = guided_flow_sampling(input_label, model, 
                                        DATA_DIM, null_label, 
                                         guidance_weight= 3,
                                          sampling_step= 7000)

    render_image(output_image)