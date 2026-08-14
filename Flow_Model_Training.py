import torch 
import matplotlib.pyplot as plt 
import torch.optim as optim
from torch import nn
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import ssl
import math 

from Flow_Model_NN import FourierEmbedding

# Specifically uses the GPU if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

#================================================================================
# Defines a function that instantiates a neural network and initializes training
#================================================================================

def instantiate_training_func(model, dataloader,
                              input_vector_dim,
                              model_name, 
                         learning_rate = 10 **(-2), 
                         epochs = 100,
                         guidance_prob = 0.1,
                         mixed_qubit_set = False) : 
    
    """
    Instantiates a training function that depends on the 
    hyperparameters defined prior to training
    """
    # TODO : instantiate a batch sampling function (simply call dataloader
    # from pyTorch)
    # time -> sample batch with a uniform distribution 
    # label and data -> take all data and break into batches
    
    # instantiate gradient descent algorithm
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    criterion = nn.MSELoss()

    def train_flow_model() : 
        """
        Performs gradient descent and training to obtain the 
        desired result 
        """
        #set the model to training mode 
        model.train()

        #loop through the epoch_count
        for epoch in range(epochs) : 
            
            epoch_loss = 0
            #divide data into batches

            for batch_idx, (batch_data, batch_label) in enumerate(dataloader) : 
                
                batch_data = model.flatten(batch_data.to(device))
                batch_label = batch_label.to(device)
                batch_label = model.flatten(batch_label)

                mini_batch = batch_data.size(0)

                #sample the drop according to the guidance prob
                drop_bool = np.random.choice([True, False]
                                             , p=[guidance_prob, (1 - guidance_prob)])

                #sample noise according to the gaussian distribution in batches 
                gaussian_noise = torch.randn([mini_batch, input_vector_dim], device = device) / math.sqrt(2)

                #sample uniform time 
                time = torch.rand([mini_batch, 1], device = device)
                # time_data =  fourier_embedder_time(time) if not fourier_embedder_time is None else time
                time_data = time

                ones = torch.ones([mini_batch,1], device = device)

                #compute vector flow
                parametrized_input = time * batch_data + (ones - time) * gaussian_noise
                
                #actual vector flow is defined as z - epsilon
                target_vector_field = batch_data - gaussian_noise

                #zero out previous gradients 
                optimizer.zero_grad()

                #pass the data manually
                predicted_vector_field = model(parametrized_input, 
                                               batch_label, time_data, 
                                               drop_bool,
                                               mixed_qubit=mixed_qubit_set)

                #compute loss 
                loss = criterion(predicted_vector_field, target_vector_field)

                #computes gradient
                loss.backward()

                #update the model parameters
                optimizer.step()

                epoch_loss = epoch_loss + loss.item()
            
            #print training progress every 1 step
            if not ((epoch + 1) % 1) : 
                avg_loss = epoch_loss / (len(dataloader))
                print(f"The average cost for the {epoch}st iteration is computed at: {avg_loss}")

        save_path = f"Trained_Model/{model_name}.pth"
        print(f"Saving model weights to {save_path}...")
        torch.save(model.state_dict(), save_path)
        print("Model saved successfully!")

    return train_flow_model


if __name__ == "__main__" : 

    label_1 = torch.tensor([1.0])
    label_2 = torch.tensor([0.1])
    label_3 = torch.tensor([0.2])
    label_4 = torch.tensor([0.3])

    res = torch.stack([label_1, label_2, label_3, label_4])
    
    print(res.shape)
    fourier_model = FourierEmbedding(half_dim= 10)

    print(fourier_model(res))

