import torch 
from torch.utils.data import DataLoader 

import os 
import sys 

root_dir = os.path.join(os.getcwd(), "..")
dataset_dir = os.path.join(root_dir, "QuantumStateSREData")

TRAIN_MODE = "train"
TEST_MODE = "test"

#add root and dataset directory to main file 
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(dataset_dir))

from Flow_Model_NN import FlowFieldNetwork, FourierEmbedding
from Flow_model_sampler import test_sampling_SRE_mixed_dataset, PSamplingQubitDependent, eff_test_sampling_SRE_mixed_dataset
from Flow_Model_Training import instantiate_training_func
from quantum_state_data_loader import QuantumStateSREData
from utils import concatenated_to_complex

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

#============ Perform neural net training on Quantum State SRE Data ============

if __name__ == "__main__" : 

    #1. Initialize the Qubit count for the available dataset (mixed qubit sizes)  
    max_qubit, min_qubit = 6, 2
    input_dim =  2 ** (max_qubit + 1)                                              # quantum state dimension (2 ^ (n + 1) in [X Y] representation)
    label_half_dim = 30                                                               # fourier SRE magic as label
    time_half_dim = 40                                                                #time as label 
    hot_label_dim = 5
    hidden_dim = (input_dim + 2 * label_half_dim + 2 * time_half_dim +hot_label_dim) * 4 // 3       # hidden neural network vector dim

    # For a mixed qubit system, the input dimension is 
    #2. Instantiate the neural network
    model = FlowFieldNetwork(lin_input_dim = input_dim,
                             hot_label_dim= hot_label_dim,
                             lin_input_label_dim = 2 * label_half_dim,
                             hidden_dim = hidden_dim,
                             fourier_half_dim_time= time_half_dim,
                            )
    
    #3. Prepare the dataset 
    data_filename = "2-3_qubit_Quantum_State_SRE_dirichlet_dataset.h5"

    dataset = QuantumStateSREData(setname = TRAIN_MODE, 
                                  dataset_folder = dataset_dir,
                                  filename= data_filename)
    
    # use pytorch's wrapper to instantiate batched data loader
    dataLoader = DataLoader(dataset= dataset,
                            batch_size = 128, 
                            shuffle = True,
                            num_workers = 0)
    
    #4. Instantiate training function
    model_name = f"SRE_FLOW_MODEL_{min_qubit}-{max_qubit}_Qubit_dirichlet_gaussian_2.pth"
    training_func = instantiate_training_func(model = model,
                                              dataloader= dataLoader, 
                                              input_vector_dim= input_dim, 
                                              learning_rate= 10 ** (-3),
                                              model_name = model_name, 
                                              epochs = 100,
                                              mixed_qubit_set= True)
    
    FORCE_RETRAIN = False
    SRE_FLOW_MODEL_DIR = os.path.join(
        os.path.join(os.getcwd(), "Trained_Model"), model_name)

    if not FORCE_RETRAIN and os.path.exists(SRE_FLOW_MODEL_DIR) : 
        #skip training, load the model
        print(f"Found saved weights at {SRE_FLOW_MODEL_DIR}. Loading model...")
        # Load the saved state dictionary into the model
        model.load_state_dict(torch.load(SRE_FLOW_MODEL_DIR, map_location=device, weights_only=True))
        print("Model loaded successfully! Skipping training.")
    
    else : 
        # 6. Execute training test
        print(f"Starting training run on SRE data for {min_qubit}-{max_qubit}_Qubits...")
        training_func()
        print("SRE training completed successfully!")

    #call the test sre function 
    save_dir = os.path.join(os.path.join(os.getcwd(), "SRE_test"), "sre_sampling_error_data_mixed.csv")

    # test_sampling_SRE_mixed_dataset(model, 
    #                   iterations = 100, 
    #                   guidance_weight=1.5, 
    #                   sampling_step = 400,
    #                   output_csv_path= save_dir)

    eff_test_sampling_SRE_mixed_dataset(model,
                                        iterations= 500,
                                        guidance_weight=1.3,
                                        sampling_step=400,
                                        output_csv_path= save_dir)
    
    # #Test the flow_model
    # qubit_batch = torch.stack ([torch.tensor([1]), torch.tensor([2]),
    #                             torch.tensor([3]),
    #                             torch.tensor([4]), torch.tensor([5])])
    
    # print(qubit_batch, '\n', qubit_batch.shape)

    # #Initialize the parallel sampler 
    # sampler = PSamplingQubitDependent(1.2353, model= model,
    #                                   sampling_step = 400, 
    #                                   guidance_weight= 1.3)
     
    # res = sampler(qubit_batch)
    # print(sampler(res))
    # complex_res = concatenated_to_complex(res)
    # print(complex_res)

    # magnitude= torch.sqrt(torch.sum(complex_res * complex_res, dim = 1))
    # print(magnitude)