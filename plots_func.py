import numpy as np
import matplotlib.pyplot as plt 
# from directory_manager import * 
import os 
from pathlib import Path


def plot_x_y (y_values, x_values, plot_name, 
              line_style = None, marker = False, starting_dir = "",
              v_lines = False, x_axis = None, y_axis = None,
              semi_log = None, log = False): 

    y_vals = y_values
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Add vertical lines in case we do a marker
    if v_lines :
        plt.vlines(x_values, ymin=0, ymax=y_vals, colors='r', linestyles='-')

    if semi_log is None and not log : 
        plt.plot(x_values, y_vals, color='b', linewidth=2, marker = "o" if marker else None, 
             linestyle = "-" if line_style is None else line_style)
    
    else: 
        semilog_dict = {"x": plt.semilogx, "y": plt.semilogy}
        if not (semi_log is None) : 
            semilog_dict[semi_log](x_values, y_vals, color='b', linewidth=2, marker = "o" if marker else None, 
                linestyle = "-" if line_style is None else line_style)
        
        else : 
            plt.loglog(x_values, y_vals, color='b', linewidth=2, marker = "o" if marker else None, 
                linestyle = "-" if line_style is None else line_style)

    # Add labels and title
    plt.title(plot_name, fontsize=20)

    x_axis = "Input (x)" if x_axis is None else x_axis
    y_axis = "Output (y)" if y_axis is None else y_axis

    plt.xlabel(x_axis, fontsize=18)
    plt.ylabel(y_axis, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
        
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    save_dir = starting_dir + f"{plot_name}.png"

    if starting_dir : 
        #check if we have the directory created 
        os.makedirs(starting_dir, exist_ok= True)

    plt.savefig(save_dir, dpi= 300)
    # # # Display the plot
    plt.show()


def plot(func, input) : 
    """
    Given an input array and a function that converts that input into another 
    array value, show a plot using matplotlib 
    """
    vectorized_func = np.vectorize(func)

    y_vals = vectorized_func(input)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(input, y_vals, label=f'Function: {func.__name__}', color='b', linewidth=2)
    
    # Add labels and title
    plt.title(f"Plot of {func.__name__}", fontsize=14)
    plt.xlabel("Input (x)", fontsize=12)
    plt.ylabel("Output (y)", fontsize=12)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"func : {func.__name__}.png", dpi= 300)
    # Display the plot
    plt.show()

    return y_vals

def plot_log_log(func, input) : 
    """
    Given an input array and a function that converts that input into another 
    array value, show a plot using matplotlib 
    """
    vectorized_func = np.vectorize(func)

    y_vals = np.log(vectorized_func(input))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(input, y_vals, label=f'Function: {func.__name__}', color='b', linewidth=2)
    
    # Add labels and title
    plt.title(f"Log Plot of {func.__name__}", fontsize=14)
    plt.xlabel("Input (x)", fontsize=12)
    plt.ylabel("Output (y)", fontsize=12)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"func_log : {func.__name__}.png", dpi= 300)
    # Display the plot
    plt.show() 
   

def plot2(funcs, input, title) : 
    """
    Given an input array and a function that converts that input into another 
    array value, show a plot using matplotlib 
    """
    vectorized_funcs = [np.vectorize(func) for func in funcs]

    # Create the plot
    plt.figure(figsize=(10, 6))

    for vectorized_func in vectorized_funcs :
        y_vals = vectorized_func(input)
        plt.plot(input, y_vals, label=f'Function: {vectorized_func.__name__}' , linewidth=2)
        
    # Add labels and title
    plt.title(f"{title}.png", fontsize=14)
    plt.xlabel("Voltage (V)", fontsize=12)
    plt.ylabel("Current (A)", fontsize=12)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{title}.png", dpi= 300)
    # Display the plot
    plt.show()

def plot_folder_data(plot_name, folder_directory, file_ending = "csv", 
                     line_style = None, marker = False, v_lines = False,
                     x_axis = None, y_axis = None) : 
    """
    Given a directory to the designated folder, plot all files ending with 
    the designated ending (defaults to a csv file)
    """
    # Create the plot
    plt.figure(figsize=(10, 6))

    cmap = plt.colormaps['inferno']

    # Define the path to your folder
    folder_path = Path(folder_directory)
    ending = f"*.{file_ending}"
    file_list = list(folder_path.glob(ending))

    # Loop through all files ending with the specific ending in the folder

    for counter, file_path in enumerate(file_list):

        print(f"Opening file: {file_path.name}")

        # Open the file safely using numpy's reader
        color_norm = (counter) / len(file_list)

        #extract the data
        data = np.genfromtxt(file_path, dtype = complex, delimiter = ",", skip_header= 1)

        x_axis_data = data[:, 1].real       #phase angle
        y_axis_data = np.abs(data[:, 0])       #magic quantity

        # Add vertical lines in case we do a marker
        if v_lines :
            plt.vlines(x_axis_data, ymin=0, ymax=y_axis_data, colors=cmap(color_norm), linestyles='-')

        plt.plot(x_axis_data, y_axis_data, linewidth=2, 
                 label=file_path.name,  marker = "o" if marker else None, 
                 linestyle = "-" if line_style is None else line_style)

    # Add labels and title
    plt.title(plot_name, fontsize=14)
    x_axis = "Input (x)" if x_axis is None else x_axis
    y_axis = "Output (y)" if y_axis is None else y_axis

    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel(y_axis, fontsize=12)

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{plot_name}.png", dpi= 300)
    # Display the plot
    plt.show()


def plot_multiple_data(y_values_list, x_values_list, plot_name, 
                       line_style=None, marker=False, starting_dir="", 
                       labels=None, x_axis=None, y_axis=None, 
                       y_errors_list=None): # Added y_errors_list

    if starting_dir:
        """
        make the starting directory to save the plot
        """
        os.makedirs(starting_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))

    labels = [i for i in range(len(y_values_list))] if labels is None else labels
    
    # If no errors are provided, create a list of None to zip cleanly
    if y_errors_list is None:
        y_errors_list = [None] * len(y_values_list)

    for y_values, x_values, label, y_errors in zip(y_values_list, x_values_list, labels, y_errors_list): 

        # Use errorbar instead of plot to support standard deviations
        plt.errorbar(x_values, y_values, yerr=y_errors, label=label, 
                     linewidth=2, marker="o" if marker else None, 
                     linestyle="-" if line_style is None else line_style,
                     capsize=4, elinewidth=1.5) # capsize adds the horizontal ticks to error bars
    
    # Add labels and title
    plt.title(plot_name, fontsize=17)
    x_axis = "Input (x)" if x_axis is None else x_axis
    y_axis = "Output (y)" if y_axis is None else y_axis

    plt.xlabel(x_axis, fontsize=15)
    plt.ylabel(y_axis, fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
    plt.tight_layout() # Prevents the legend from being clipped

    # Using os.path.join is safer than string concatenation for directories
    save_dir = os.path.join(starting_dir, f"{plot_name}.png") if starting_dir else f"{plot_name}.png"

    plt.savefig(save_dir, dpi=300)
    # Display the plot
    plt.show()

# def plot_files_in_folder(directory, line_style = None, 
#                          marker = False, v_line = False, 
#                          ending = "*.csv", starting_dir = "",
#                          x_axis = None, y_axis = None):
#     """
#     Plot files with a specific file ending from a specific folder into 
#     seperate image files
#     """
#     path_list = access_files_in_folder(directory, ending)

#     for file_path in path_list : 

#         file_data = np.genfromtxt(file_path, skip_header = 1 , 
#                                   delimiter = ",", dtype = complex)
        
#         x_values = file_data[:, 0].real 
#         y_values = file_data[:, 1].real 

#         plot_x_y(y_values, x_values, file_path.name[:-14], line_style= line_style,
#                  marker = marker, v_lines = v_line,
#                  starting_dir = starting_dir,
#                  x_axis= x_axis, y_axis= y_axis)
    
#     return None 

# def plot_files_in_folder(directory, line_style = None, 
#                          marker = False, v_line = False, 
#                          ending = "*.csv", starting_dir = "",
#                          x_axis = None, y_axis = None):
#     """
#     Plot files with a specific file ending from a specific folder into 
#     seperate image files
#     """
#     path_list = access_files_in_folder(directory, ending)

#     for file_path in path_list : 

#         file_data = np.genfromtxt(file_path, skip_header = 1 , 
#                                   delimiter = ",", dtype = complex)
        
#         x_values = file_data[:, 0].real 
#         y_values = file_data[:, 1].real 

#         plot_x_y(y_values, x_values, file_path.name[:-14], line_style= line_style,
#                  marker = marker, v_lines = v_line,
#                  starting_dir = starting_dir,
#                  x_axis= x_axis, y_axis= y_axis)
    
#     return None 


# def plot_files_in_folder_long_csv(directory, line_style=None, 
#                                   marker=False, v_line=False, 
#                                   ending="*.csv", starting_dir="",
#                                   x_axis=None, y_axis=None,
#                                   plot_title = None,
#                                   semi_log = None, loglog = None):
#     """
#     Given a multi column result in the csv file, plot the columns 
#     of data into a single figure, where the y axis of the figure are all
#     columns following the first and the x axis is the first column 
#     in the csv file. The axis is labeled according to the input 
#     x_axis and y_axis of the function. Save the file into the starting_dir
#     if given.

#     if marker is marked true, points indicating the data should appear
#     """
    
#     # Get all files matching the ending
#     path_list = access_files_in_folder(directory, ending)

#     if not path_list:
#         print(f"No files found matching {ending} in {directory}")
#         return

#     for file_path in path_list: 
#         # Read the file data using numpy, extracting complex data 
#         # as in the provided style example
#         file_data = np.genfromtxt(file_path, skip_header=1, 
#                                   delimiter=",", dtype=complex)
        
#         with open(file_path, "r") as file : 
#             header = file.readline().strip()

#             header_lst = header.split(",")

#         # Ensure we are working with a 2D array (in case of single row files)
#         if file_data.ndim == 1:
#             file_data = file_data.reshape(1, -1)
            
#         # First check the number of columns the csv file has
#         column_count = file_data.shape[1]
        
#         if column_count < 2:
#             print(f"File {file_path.name} doesn't have enough columns. Skipping.")
#             continue
            
#         # The first column is our X axis
#         x_values = file_data[:, 0].real 
        
#         # Extract the plot name cleanly using pathlib (.stem removes extension)
#         plot_name = file_path.stem if plot_title is None else plot_title
        
#         # Create the plot
#         plt.figure(figsize=(10, 6))
        
#         # Loop column_count - 1 times to plot all y columns against x
#         for i in range(1, column_count):
#             y_values = file_data[:, i].real  * 1000           #convert to ms 
            
#             # Add vertical lines if requested
#             if v_line:
#                 plt.vlines(x_values, ymin=0, ymax=y_values, colors='r', linestyles='-', alpha=0.6)

#             if semi_log is None and loglog is None : 
#                 plt.plot(x_values, y_values, linewidth=2, marker = "o" if marker else None, 
#                     linestyle = "-" if line_style is None else line_style,
#                     label= header_lst[i])
            
#             semilog_dict = {"x": plt.semilogx, "y": plt.semilogy}
#             if not (semi_log is None) : 
#                 semilog_dict[semi_log](x_values, y_values, linewidth=2, marker = "o" if marker else None, 
#                     linestyle = "-" if line_style is None else line_style,
#                     label= header_lst[i])
            
#             if not (loglog is None) : 
#                 plt.loglog(x_values, y_values, linewidth=2, marker = "o" if marker else None, 
#                     linestyle = "-" if line_style is None else line_style,
#                     label= header_lst[i])
                
#             # plt.plot(x_values, y_values, linewidth=2, 
#             #          marker="o" if marker else None, 
#             #          linestyle="-" if line_style is None else line_style,
#             #          label= header_lst[i])
        
#         # Add labels and title mapping to your style
#         plt.title(plot_name, fontsize=20)

#         x_label = "Input (x)" if x_axis is None else x_axis
#         y_label = "Output (y)" if y_axis is None else y_axis

#         plt.xlabel(x_label, fontsize=18)
#         plt.ylabel(y_label, fontsize=18)
#         plt.xticks(fontsize=14)
#         plt.yticks(fontsize=14)
            
#         # Add a grid for better readability
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.legend()

#         if starting_dir: 
#             # Check if we have the directory created 
#             os.makedirs(starting_dir, exist_ok=True)
#             # Use os.path.join to safely format the directory path
#             save_dir = os.path.join(starting_dir, f"{plot_name}.png")
#             plt.savefig(save_dir, dpi=300)
#             print(f"Saved plot to: {save_dir}")
#         else: 
#             plt.savefig(f"{plot_name}.png", dpi = 300)
        
#         # Display the plot
#         plt.show()

if __name__ == "__main__" : 
    print("hello_world")