import torch 
from torch import nn
import matplotlib.pyplot as plt 
import torch.optim as optim
import math
import torch.nn.functional as F

#set the seed for the fourier embedding (so it doesn't create different hyperparameters everytime)
torch.manual_seed(52)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#================================================================================
# Instantiates a neural network to learn the vector field for the Flow Model
#================================================================================

class FlowFieldNetwork (nn.Module) : 
    """
    Instantiates a neural network that computes the flow vector field at any
    given quantum noise (x) and time (t), guided by the label (y)
    """
    def __init__(self, lin_input_dim, lin_input_label_dim, 
                 hot_label_dim = None, 
                 hidden_dim = 2048, 
                 fourier_half_dim_time= 5) : 
        #inherit from nn.module

        super().__init__() 
        self.flatten = nn.Flatten()

        self.input_dim = lin_input_dim 
        self.input_label_dim = lin_input_label_dim 

        time_dim = 2 * fourier_half_dim_time

        input_layer_dim = 0 
        if hot_label_dim is None : 
            input_layer_dim = lin_input_dim + lin_input_label_dim + time_dim
        else : 
            input_layer_dim = lin_input_dim + lin_input_label_dim + time_dim + hot_label_dim


        #initialize the layers of the neural network 
        self.linear_ReLU_stack = nn.Sequential(
            nn.Linear(input_layer_dim, hidden_dim, device = device),
            nn.SiLU(), 
            # ========= ADD MORE HIDDEN LAYERS HERE (IF NECESSARY) =========
            nn.Linear(hidden_dim, hidden_dim, device = device), 
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim , device = device), 
            nn.SiLU(), 
            # nn.Linear(hidden_dim//4 * 3, hidden_dim // 4 * 3, device = device), 
            # nn.SiLU(), 
            #===============================================================

            #vector field dimension is the same as the space the noise is in 
            nn.Linear(hidden_dim , self.input_dim, device = device), 
        )

        #initialize the embedder within the neural network instance 
        self.fourier_label_embedder = FourierEmbedding(lin_input_label_dim // 2)
        self.fourier_time_embedder = FourierEmbedding(fourier_half_dim_time)


        self.hot_label_embedder = HotLabelEmbedder(class_count = hot_label_dim) if hot_label_dim is not None else None
        
    def forward(self, batched_data, batched_label, batched_time, drop_bool = False,
                mixed_qubit = False) : 
        """
        Performs forward pass on the input data along with the label 
        """
        #mini-batch dimension allows for parallel computation
        embedded_time = self.fourier_time_embedder(batched_time)

        embedded_label = None

        if not mixed_qubit : 
            embedded_label = self.fourier_label_embedder(batched_label, drop = drop_bool)

        else : 
            sre_values = torch.reshape(batched_label[:, 0], shape = [-1, 1])
            qubit_values = torch.reshape(batched_label[:, 1], shape = [-1,1]).long()

            #pass this into the embedding
            embedded_label = self.fourier_label_embedder(sre_values, drop = drop_bool)

            #prepare a hot label embedder for the qubit classes
            qubit_labels = self.hot_label_embedder(qubit_values, 
                                                   drop_bool = drop_bool)

            #concatenate these two encding
            embedded_label = torch.cat([embedded_label, qubit_labels], dim = 1)

        neural_inputs = torch.cat([batched_data, embedded_label, embedded_time], dim = 1)

        #flatten the neural_inputs such that we have a vector after the batch dim
        neural_inputs = self.flatten(neural_inputs)

        #the neural network's dimensions
        vec_field = self.linear_ReLU_stack(neural_inputs)

        return vec_field

class FourierEmbedding (nn.Module):
    """
    Performs fourier embedding on the labels for better details
    scalar -> high dimensional labels
    """
    def __init__ (self, half_dim): 
        """
        Instantiate the weights for the fourier embedding

        batch of inputs -> batch of ([cos 2pi omega * input .... sin 2pi omega * input ...])
        """
        super().__init__()
        self.dim = half_dim

        #initialize the frequency weights
        self.omega = torch.randn(self.dim, device= device)

    def forward(self, input_batch, drop = False) :
        """
        Computes the fourier change
        """
        if drop : 
            res = -1 * torch.ones_like(input_batch) * torch.ones_like(self.omega)
            
            return torch.cat([res, res], dim = 1)
        
        weights = 2 * torch.pi * self.omega * input_batch
        sin_term, cos_term = torch.sin(weights), torch.cos(weights)

        #normalization factor sqrt(2/d)
        res = torch.sqrt(2/ torch.tensor([self.dim]
                                         , device= device)) * torch.cat([cos_term, sin_term], 
                                                                        dim = 1)

        return res 

class HotLabelEmbedder (nn.Module) : 
    """
    Performs a hot label embedding on distinct classes to get better dimensionality
    for model training
    """
    def __init__ (self, class_count) : 
        super().__init__()
        self.flatten = nn.Flatten()

        self.class_count = class_count 
    
    def forward(self, input_batch, drop_bool = False) : 
        """
        given an input batch of N scalars, return the associated embedding that
        corresponds to the class. Input must be shape (N,1)

        """
        if drop_bool : 
            res = -1 * torch.ones_like(input_batch) * torch.ones(self.class_count,
                                                                 device = device)
            
            return res
        
        #otherwise creates a hot label encoding for each class 
        input_batch = input_batch[:, 0]
        res = F.one_hot(input_batch, 
                        num_classes= self.class_count)
        # identity_mat = torch.diag(torch.ones(self.class_count)) #shape (class_count, class_count)

        return res



if __name__ == "__main__" : 
    
    # # simple testing 
    # model = FlowFieldNetwork(32, 1).to(device)

    # print(model)

    # X = torch.rand(1, 33, device=device)
    # output = model(X)

    # print(output)

    #test hotlabelembedder
    input_tensor = torch.reshape(torch.tensor([5, 4, 3, 2, 1], device = device), shape = [5,1],
                                 )
    hotlabel_embed= HotLabelEmbedder(6)

    out = hotlabel_embed(input_tensor)
    out_dropped = hotlabel_embed(input_tensor, drop_bool = True)
    print(out_dropped)
    print(out_dropped.shape)

    print(out)
    print(out.shape)

