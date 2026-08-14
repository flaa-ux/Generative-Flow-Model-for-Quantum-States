from Flow_Model_NN import FlowFieldNetwork
from Flow_Model_Training import instantiate_training_func

import torch 
import matplotlib.pyplot as plt 
import torch.optim as optim
from torch import nn
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import csv
import math

import os 
import sys
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.manual_seed(10)

QUANTUMSTATESREDIR = os.path.abspath(os.path.join(os.getcwd(), "QuantumStateSREData"))
sys.path.append(QUANTUMSTATESREDIR)

from utils import concatenated_to_complex, ParallelSREMagic

#=========================== Parallel Sampling ==================================
class PSamplingQubitDependent (nn.Module) : 
    """
    Wraps the guided_flow_sampling_SRE function with the pytorch 
    nn module to support parallel batching for more efficient compute
    """
    def __init__ (self, SRE_val, model, 
                  sampling_step = 150, 
                  guidance_weight = 1,
                  max_input_dim = 0) : 
        
        super().__init__()
        #instantiate instance attributes
        self.SRE_val = SRE_val
        self.model = model
        self.sampling_step = sampling_step
        self.guidance_weight = guidance_weight
        self.max_input_dim = max_input_dim

        #flatten dim (N *) -> (N, -1)
        self.flatten = nn.Flatten()
    
    def forward (self, qubit_batch) : 
        """
        Given the qubit batch value, returns a batch of sampled quantum states
        from the input model.

        We assume that the qubit batch shape is (N, 1)
        """
        model = self.model 
        qubit_batch_count = qubit_batch.shape[0]

        data_dim = self.max_input_dim

        #initialize a batch of images based on random errors 
        current_image = torch.randn([qubit_batch_count, data_dim], device = device) / math.sqrt(2)

        #Current step can be a scalar because it's not passed into the model
        current_step, current_time = 0, torch.zeros(qubit_batch_count, 1,
                                   dtype = torch.float32, device = device)
        
        SRE_values_batched = self.SRE_val * torch.ones(qubit_batch_count, 1,
                                                       dtype = torch.float32,
                                                       device = device)
        
        #ensure that qubit_batch is in device to prevent errors
        qubit_batch = qubit_batch.to(device)

        #perform concatenation with the input qubit 
        SRE_Qubit_label = torch.cat([SRE_values_batched, qubit_batch], dim = 1)

        #set model to evaluation mode
        model.eval()
        with torch.no_grad(): 
            while current_step < self.sampling_step :

                current_guided_field = model(current_image, SRE_Qubit_label, current_time,
                                            mixed_qubit = True)

                unguided_field = model(current_image, SRE_Qubit_label, current_time, drop_bool = True, mixed_qubit = True)

                res_field = self.guidance_weight  * current_guided_field + (1 - self.guidance_weight) * unguided_field
                current_image = current_image + res_field * 1 / self.sampling_step

                #update the current_time value 
                current_time = current_time + 1/ self.sampling_step

                current_step = current_step + 1

            return current_image
        

#================================================================================

#================================================================================
# Instantiates sampling of an image given a valid neural network
#================================================================================

def guided_flow_sampling (label_encoding, model,
                          data_dim,
                          null_label,
                          sampling_step = 300,
                          guidance_weight = 2,):
    """
    Performs incremental flows to output a valid image
    """
    current_image = torch.randn([1, data_dim], device = device)

    current_step, current_time = 0 , torch.tensor([[0]], dtype = torch.float32, device = device)

    #set model to evaluation mode
    model.eval()
    with torch.no_grad(): 
        while current_step < sampling_step :

            current_input = torch.cat([current_image, label_encoding, current_time], dim = 1)
            current_guided_field = model(current_input)

            unguided_input = torch.cat([current_image, null_label, current_time ], dim = 1)
            unguided_field = model(unguided_input)

            res_field = guidance_weight  * current_guided_field + (1 - guidance_weight) * unguided_field
            current_image = current_image + res_field * 1 / sampling_step

            #update the current_time value 
            current_time = current_time + 1/ sampling_step

            current_step = current_step + 1

        return current_image

def guided_flow_sampling_SRE (SRE_val, model,
                          sampling_step = 150,
                          guidance_weight = 2,):
    """
    Performs incremental flows to output a valid image
    """
    data_dim = model.input_dim
    current_image = torch.randn([1, data_dim], device = device) / math.sqrt(2)

    current_step, current_time = 0 , torch.tensor([[0]], dtype = torch.float32, device = device)
    SRE_val = torch.tensor([[SRE_val]], dtype = torch.float32, device = device)

    #set model to evaluation mode
    model.eval()
    with torch.no_grad(): 
        while current_step < sampling_step :

            current_guided_field = model(current_image,SRE_val, current_time)

            unguided_field = model(current_image, SRE_val, current_time, drop_bool = True)

            res_field = guidance_weight  * current_guided_field + (1 - guidance_weight) * unguided_field
            current_image = current_image + res_field * 1 / sampling_step

            #update the current_time value 
            current_time = current_time + 1/ sampling_step

            current_step = current_step + 1

        return current_image


def guided_flow_sampling_SRE_with_qubit (SRE_val, qubit, model,
                          sampling_step = 150,
                          guidance_weight = 2,):
    """
    Performs incremental flows to output a valid image
    """
    data_dim = model.input_dim
    current_image = torch.randn([1, data_dim], device = device) / math.sqrt(2)

    current_step, current_time = 0 , torch.tensor([[0]], dtype = torch.float32, device = device)
    SRE_Qubit_label = torch.tensor([[SRE_val, qubit]], dtype = torch.float32, device = device)

    #set model to evaluation mode
    model.eval()
    with torch.no_grad(): 
        while current_step < sampling_step :

            current_guided_field = model(current_image, SRE_Qubit_label, current_time,
                                         mixed_qubit = True)

            unguided_field = model(current_image, SRE_Qubit_label, current_time, drop_bool = True, mixed_qubit = True)

            res_field = guidance_weight  * current_guided_field + (1 - guidance_weight) * unguided_field
            current_image = current_image + res_field * 1 / sampling_step

            #update the current_time value 
            current_time = current_time + 1/ sampling_step

            current_step = current_step + 1

        return current_image

# ================================================ Sampling Wrapper ================================================
def render_image(neural_net_out):
    """
    Renders the image using matplotlib
    """
    image = neural_net_out.cpu().numpy().reshape(28,28)

    # Plot using matplotlib
    plt.figure(figsize=(4, 4))
    
    # vmin and vmax clip the output for optimal grayscale rendering
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title("Generated Digit from model")
    plt.axis('off') # Hides the axes and tick marks
    
    # Display the plot window
    plt.show()

def test_sampling_SRE(
    model, 
    iterations=100,
    guidance_weight=2, 
    sampling_step= 300, 
    output_csv_path= "sre_sampling_results.csv"
):
    """
    Generate random SRE (magic) target values, sample quantum states using a guided flow model,
    evaluate the actual SRE of the generated states, and export the error metrics to a CSV file.

    Parameters:
        model: Trained guided flow model.
        iterations (int): Number of test samples to evaluate.
        guidance_weight (float/int): Classifier-free guidance weight.
        sampling_step (int): Number of integration steps for flow sampling.
        output_csv_path (str): File path to save the generated CSV report.

    Returns:
        tuple: (target_sre_vals, actual_sre_vals, sre_errors)
    """
    magic_calculator = ParallelSREMagic()

    sre_vals = []
    actual_sres = []
    sre_errors = []

    pre_projection_norms = []
    # Disable gradient tracking for inference to reduce memory and boost performance
    with torch.no_grad():
        # Fix: use range(iterations) instead of iterating over the integer directly
        for i in range(iterations):
            # Sample a target SRE value (uniform in [0, 4))
            sre_val = (4 * torch.rand(1)).item()
            sre_vals.append(sre_val)

            # Generate the concatenated state representation [X Y]
            gen_quantum_state = guided_flow_sampling_SRE(
                sre_val,
                model,
                sampling_step=sampling_step,
                guidance_weight=guidance_weight
            )

            print(gen_quantum_state.shape)

            magnitude =torch.sqrt(torch.sum(gen_quantum_state * gen_quantum_state, dim = 1))

            pre_projection_norms.append(magnitude.item())

            gen_quantum_state = gen_quantum_state / magnitude
            # Convert from [X Y] concatenated format to complex tensor [X + iY]
            gen_quantum_state_complex = concatenated_to_complex(gen_quantum_state)

            gen_quantum_state_complex = torch.reshape(gen_quantum_state_complex,
                                                      shape = gen_quantum_state_complex.shape + (1,))
            
            #normalize the quantum vector 
            

            # Compute actual SRE magic value
            # print(gen_quantum_state_complex)
            actual_sre = magic_calculator(gen_quantum_state_complex)[0].item()
            actual_sres.append(actual_sre)
            # print(actual_sre, sre_val)
            # Compute absolute error
            abs_err = abs(actual_sre - sre_val)
            sre_errors.append(abs_err)

            if (i + 1) % 10 == 0 or (i + 1) == iterations:
                print(f"Sample [{i + 1}/{iterations}] - Target SRE: {sre_val:.4f} | Actual SRE: {actual_sre:.4f} | Error: {abs_err:.4f}")

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    # Export results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_index", "target_sre", "actual_sre", "absolute_error"])
        for idx, (target, actual, err) in enumerate(zip(sre_vals, actual_sres, sre_errors)):
            writer.writerow([float(idx), target, actual, err])

    mae = sum(sre_errors) / len(sre_errors)
    print(f"\nResults successfully saved to '{output_csv_path}'.")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean magnitude: {np.mean(pre_projection_norms)}")

    return sre_vals, actual_sres, sre_errors
        

def test_sampling_SRE_mixed_dataset(
    model, 
    iterations=100,
    guidance_weight=2, 
    sampling_step= 300, 
    output_csv_path= "sre_sampling_results.csv"
):
    """
    Generate random SRE (magic) target values, sample quantum states using a guided flow model,
    evaluate the actual SRE of the generated states, and export the error metrics to a CSV file.

    Parameters:
        model: Trained guided flow model.
        iterations (int): Number of test samples to evaluate.
        guidance_weight (float/int): Classifier-free guidance weight.
        sampling_step (int): Number of integration steps for flow sampling.
        output_csv_path (str): File path to save the generated CSV report.

    Returns:
        tuple: (target_sre_vals, actual_sre_vals, sre_errors)
    """
    #TODO : Make some necessary modifications to this.
    # claims : There should be no need to cut the hilbert space by the correct dimension,
    # the sampler can simply use the state out of the box

    # during training, we should try all qubits in the dataset to get the best match with the desired SRE val
    # the user will not know how many qubits can give good results. 

    magic_calculator = ParallelSREMagic()

    sre_vals = []
    qubit_list= []
    actual_sres = []
    sre_errors = []

    pre_projection_norms = []

    # Disable gradient tracking for inference to reduce memory and boost performance
    with torch.no_grad():
        # Fix: use range(iterations) instead of iterating over the integer directly
        for i in range(iterations):
            # Sample a target SRE value (uniform in [0, 4))
            sre_val = (4 * torch.rand(1)).item()
            sre_vals.append(sre_val)
            #initialize the state for comparison (SRE value, norm value, qubit_count)
            best_state = (float('inf'), None, None)

            for selected_qubit in range(2, 7) : 

                print (f"Trial {selected_qubit - 1} :")
                print (f"Target SRE val : {sre_val}, Associated Qubit Count: {selected_qubit}" )
                print ("=======================================================")

                # Generate the concatenated state representation [X Y]
                gen_quantum_state = guided_flow_sampling_SRE_with_qubit(
                    sre_val,
                    selected_qubit,
                    model,
                    sampling_step=sampling_step,
                    guidance_weight=guidance_weight
                )

                magnitude =torch.sqrt(torch.sum(gen_quantum_state * gen_quantum_state, dim = 1))

                gen_quantum_state = gen_quantum_state / magnitude
                # Convert from [X Y] concatenated format to complex tensor [X + iY]
                gen_quantum_state_complex = concatenated_to_complex(gen_quantum_state)

                gen_quantum_state_complex = torch.reshape(gen_quantum_state_complex,
                                                        shape = gen_quantum_state_complex.shape + (1,))
                
                #normalize the quantum vector 
                
                # Compute actual SRE magic value
                # print(gen_quantum_state_complex)
                actual_sre = magic_calculator(gen_quantum_state_complex)[0].item()
                # actual_sres.append(actual_sre)
                # print(actual_sre, sre_val)
                # Compute absolute error
                abs_err = abs(actual_sre - sre_val)

                best_state = min(best_state, (abs_err, magnitude.item(), selected_qubit))

            #append our best results 
            sre_errors.append(best_state[0])
            pre_projection_norms.append(best_state[1])
            qubit_list.append(best_state[2])

            if (i + 1) % 10 == 0 or (i + 1) == iterations:
                print(f"Sample [{i + 1}/{iterations}] - Target SRE: {sre_val:.4f} | Error: {sre_errors[-1]:.4f}")

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    # Export results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_index", "target_sre", "absolute_error"])
        for idx, (target, err) in enumerate(zip(sre_vals, sre_errors)):
            writer.writerow([float(idx), target, err])

    mae = sum(sre_errors) / len(sre_errors)
    print(f"\nResults successfully saved to '{output_csv_path}'.")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean magnitude: {np.mean(pre_projection_norms)}")

    return sre_vals, sre_errors

def eff_test_sampling_SRE_mixed_dataset(
    model, max_input_dim,
    iterations=100,
    guidance_weight=2, 
    sampling_step= 300, 
    output_csv_path= "sre_sampling_results.csv"
):
    """
    Generate random SRE (magic) target values, sample quantum states using a guided flow model,
    evaluate the actual SRE of the generated states, and export the error metrics to a CSV file.

    Parameters:
        model: Trained guided flow model.
        iterations (int): Number of test samples to evaluate.
        guidance_weight (float/int): Classifier-free guidance weight.
        sampling_step (int): Number of integration steps for flow sampling.
        output_csv_path (str): File path to save the generated CSV report.

    Returns:
        tuple: (target_sre_vals, actual_sre_vals, sre_errors)
    """
    #TODO : Make some necessary modifications to this.
    # claims : There should be no need to cut the hilbert space by the correct dimension,
    # the sampler can simply use the state out of the box

    # during training, we should try all qubits in the dataset to get the best match with the desired SRE val
    # the user will not know how many qubits can give good results. 

    magic_calculator = ParallelSREMagic()

    sre_vals = []
    qubit_list= []
    sre_errors = []

    pre_projection_norms = []

    # Disable gradient tracking for inference to reduce memory and boost performance
    with torch.no_grad():
        # Fix: use range(iterations) instead of iterating over the integer directly
        for i in range(iterations):

            # Sample a target SRE value (uniform in [0, 4))
            sre_val = (4.0 * torch.rand(1)).item()
            sre_vals.append(sre_val)

            #Initialize the sampler instance
            parallel_sampler = PSamplingQubitDependent(SRE_val = sre_val,
                                                       model = model,
                                                       sampling_step = sampling_step,
                                                       guidance_weight= guidance_weight,
                                                       max_input_dim = max_input_dim)

            selected_qubit_batch = torch.unsqueeze(torch.arange(2, 9), 1)

            print (f"Target SRE val : {sre_val}, Computing qubit batches 2-8" )
            print ("=======================================================")

            # Generate the concatenated state representation [X Y]
            gen_quantum_state = parallel_sampler(selected_qubit_batch)
            
            magnitude = torch.sqrt(torch.sum(gen_quantum_state * gen_quantum_state, dim = 1)).unsqueeze(dim = 1)

            #normalize the quantum vector 
            gen_quantum_state = gen_quantum_state / magnitude

            # Convert from [X Y] concatenated format to complex tensor [X + iY]
            gen_quantum_state_complex = concatenated_to_complex(gen_quantum_state)

            #quantum state must be reconverted into a column vector 
            gen_quantum_state_complex = torch.reshape(gen_quantum_state_complex,
                                                    shape = gen_quantum_state_complex.shape + (1,))
            
            
            # Compute actual SRE magic value
            # print(gen_quantum_state_complex)
            actual_sre = magic_calculator(gen_quantum_state_complex)        #this should be shaped (N, 1)

            abs_err = torch.abs(actual_sre - sre_val)

            min_idx = torch.argmin(abs_err)

            min_associated_qubit = selected_qubit_batch[min_idx].item()
            min_associated_error = abs_err[min_idx].item()
            min_associated_mag = magnitude[min_idx].item()

            #append our best results 
            sre_errors.append(min_associated_error)
            pre_projection_norms.append(min_associated_mag)
            qubit_list.append(min_associated_qubit)

            if (i + 1) % 10 == 0 or (i + 1) == iterations:
                print(f"Sample [{i + 1}/{iterations}] - Target SRE: {sre_val:.4f} | Error: {sre_errors[-1]:.4f} | Qubit : {qubit_list[-1]}")

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    # Export results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_index", "target_sre", "absolute_error", "chosen_qubit"])
        for idx, (target, err, qubit) in enumerate(zip(sre_vals, sre_errors, qubit_list)):
            writer.writerow([float(idx), target, err, qubit])

    mae = sum(sre_errors) / len(sre_errors)
    print(f"\nResults successfully saved to '{output_csv_path}'.")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean magnitude: {np.mean(pre_projection_norms)}")

    return sre_vals, sre_errors