# %%
# Library Imports
import argparse
from pprint import *
import pandas as pd
import scipy as sp
from scipy.sparse.linalg import spsolve
from scipy import interpolate
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# Import Solver script
import ir_solver as ir_solver
# import sys
# log_file = open("LOG.txt", "w")
# sys.stdout = log_file
# sys.stderr = log_file

import sys

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)  # Print to console
        self.log.write(message)       # Write to file

    def flush(self):
        pass  # Required for compatibility with sys.stdout
    
    def close(self):
        self.log.close()  # Close the log file

log_file = os.path.join(os.path.dirname(__file__), "FEATURE_EXTRACTION_LOG.log")
sys.stdout = Logger(log_file)

PLOT = 0

# %%
# Extract each line as pandas dataframe
def parse_lines(FILE):
    print(f"\nProcessing file : {os.path.basename(FILE)}\n")
    with open(FILE, 'r') as f:

        file_content = []
        for line in f :
            components = line.split()
            if len(components) < 4 :
                continue

            electrical_component = components[0]
            node1 = components[1]
            node2 = components[2]
            component_value = components[3]

            # Store every component as a dict in a list.
            file_content.append({
                'electrical_component' : electrical_component,
                'node1' : node1,
                'node2' : node2,
                'value' : component_value
            })

        file_contents = pd.DataFrame(file_content)
        
        print("Parsed file successfully...")
        return file_contents

def get_node_location(node):
    """Returns Net_name, Metal_layer, X, Y for each node"""
    if node == 0:
        return 0, 0, 0, 0
    
    node_components = node.split('_')
    
    net_name     = node_components[0]
    metal_layer  = node_components[1]
    x_coord      = node_components[2]
    y_coord      = node_components[3]

    return net_name, metal_layer, x_coord, y_coord
    
def get_manhattan_dist( node1, node2) -> float : 
    """ Returns Manhattan distance between two nodes"""
    _ ,_, x1, y1 = get_node_location(node1)
    _, _, x2, y2 = get_node_location(node2)

    # Scale by factor of 2000 DBU.
    x1 = float(x1) / 2000
    y1 = float(y1) / 2000
    x2 = float(x2) / 2000
    y2 = float(y2) / 2000

    distance = abs(x1 - x2) + abs(y1 - y2)
    return distance

def calc_avg_dist( voltage_sources : pd.DataFrame, node) -> float :
    """Calculates avg distance of node to all voltage sources """

    volt_dist = [] 

    for i in range(len(voltage_sources)):
        volt_dist.append(get_manhattan_dist(
                            voltage_sources.iloc[i,0],
                            node        
                        ))       

    # non_zero_distances = [v for v in volt_dist if v > 0]
    # if not non_zero_distances:
    #     return float('inf')


    inverse_sum = sum( (1 / v) if v != 0 else 0 for v in volt_dist)
    effective_dist = 1/inverse_sum

    return effective_dist

def parse_voltage_file(DOT_VOTLAGE_FILE:str):
    """Reads .voltage file for IR_drop_map() function."""

    with open(DOT_VOTLAGE_FILE, 'r') as f:
        file_content = []   
        for line in f: 
            components = line.split()

            node = components[0]
            voltage = components[1]

            x_coord = float(node.split('_')[2]) / 2000
            y_coord = float(node.split('_')[3]) / 2000
            voltage = 1.1 - float(voltage)
            file_content.append({
                # 'node': node,
                'x': x_coord,
                'y': y_coord,
                'voltage': voltage,
            })
        
        voltage_df = pd.DataFrame(file_content)
        return voltage_df

def get_pitch_for_layer( node1, node2 ):
    """Returns pitch between nodes"""

    _ ,_, x1, y1 = get_node_location(node1)
    _, _, x2, y2 = get_node_location(node2)

    # Scale by factor of 2000 DBU.
    x1 = float(x1) 
    y1 = float(y1) 
    x2 = float(x2) 
    y2 = float(y2) 

    pitch = int(abs(x1 - x2) + abs(y1 - y2)) 
    return pitch

def plot_metal_Dataframe(DATAFRAME, m1_dataframe, m4_dataframe, m7_dataframe, m8_dataframe, m9_dataframe):
    """Plots m1 to m9 dataframes in separate subplots within the same image."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # Create a 3x2 grid of subplots
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Define dataframes and labels for each subplot
    dataframes = [
        (m1_dataframe, 'M1', 'blue'),
        (m4_dataframe, 'M4', 'green'),
        (m7_dataframe, 'M7', 'red'),
        (m8_dataframe, 'M8', 'purple'),
        (m9_dataframe, 'M9', 'orange')
    ]

    # Plot each dataframe in a separate subplot
    for i, (df, label, color) in enumerate(dataframes):
        ax = axes[i]
        ax.scatter(df['y'], df['x'], c=color, s=5, label=label)
        ax.set_title(f'{label} Distribution')
        ax.set_xlabel('Y Coordinate')
        ax.set_ylabel('X Coordinate')
        ax.legend()
        ax.invert_yaxis()  # Invert y-axis if required

    # Hide the last subplot if there are fewer than 6 subplots
    if len(dataframes) < len(axes):
        for j in range(len(dataframes), len(axes)):
            fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Code in the PDN_plot:
    m1_dataframe = DATAFRAME[DATAFRAME['node'].str.contains('_m1_')]
    m4_dataframe = DATAFRAME[DATAFRAME['node'].str.contains('_m4_')]
    m7_dataframe = DATAFRAME[DATAFRAME['node'].str.contains('_m7_')]
    m8_dataframe = DATAFRAME[DATAFRAME['node'].str.contains('_m8_')]
    m9_dataframe = DATAFRAME[DATAFRAME['node'].str.contains('_m9_')]
    
    max_x = (DATAFRAME['x'].max() / 100 ) * 100
    max_y = (DATAFRAME['y'].max() / 100 ) * 100 

    print(f"Max X : {max_x}, Max Y : {max_y}")
    print(" - Plotting metal layer distribution...")
    plot_metal_Dataframe(m1_dataframe, m4_dataframe, m7_dataframe, m8_dataframe, m9_dataframe)


# %%
# INPUT and OUTPUT file locations :

INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase1.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase2.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase3.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/training_data/data_point00.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/training_data/data_point10.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/training_data/data_point25.sp'

# OUTPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/output.voltage'
# OUTPUT_FILE_WO_EXT = os.path.splitext(os.path.basename(INPUT_FILE))[0]

# FEATURE_DIR = os.path.join(os.path.dirname(__file__), "FEATURE_DIR")
# OUTPUT_FILE_WO_EXT = os.path.join(FEATURE_DIR, os.path.splitext(os.path.basename(INPUT_FILE))[0])
# print(OUTPUT_FILE_WO_EXT)

# file_content = parse_lines(INPUT_FILE)


# %% 

# Get Current Map features.
def create_current_map(file_dataframe:pd.DataFrame, OUTPUT_FILE_WO_EXT ):
    """Create current map from INPUT file.

    For every node in the ckt, i.e the first column 
    of dataframe, Set current_value param to zero at 
    specified node location. 

    Then filter out nodes that have current sources.
    Add value of these current sources to location of nodes. 

    Save (x,y, current_value) to a new panda dataframe and plot this dataframe. 

    Arguments 
    ---------
    file_dataframe: 
        Dataframe made from the contents of the file.
    OUTPUT_FILE_WO_EXT:
        Path to save the output .imap file
    """

    print("   -----------------------")
    print(" - Creating current map...")
    # Dataframe containing nodes only. 
    # nodes = | node | current_value | x | y |
    nodes_df = pd.DataFrame(file_dataframe['node1'].unique(), columns=['node'])
    nodes_df['Current'] = 0 # Initialize current_value to 0
    nodes_df[['x', 'y']] = nodes_df['node'].apply(lambda n: pd.Series([
        float(n.split('_')[2]) / 2000,
        float(n.split('_')[3]) / 2000
    ]))

    # print(" - Updating values for current sources...")

    current_sources = file_dataframe.loc[file_dataframe['electrical_component'].str.startswith('I'), ['node1', 'value']].rename(columns={'node1': 'node'})
    nodes_df = nodes_df.merge(current_sources, on='node', how='left') #merging on the 'node' column
    nodes_df['Current'] = nodes_df['value'].fillna(0) 
    nodes_df = nodes_df.drop(columns=['value']) 

    # print(nodes_df.head(10))
    nodes_df['Current'] = pd.to_numeric(nodes_df['Current'], errors='coerce')
    nodes_df = nodes_df.drop(columns=['node'])[['x', 'y', 'Current']]
    
    current_map_file = OUTPUT_FILE_WO_EXT + ".imap"
    nodes_df.to_csv(current_map_file, index=False)

    # print(nodes_df.head(10))
    print(f" - Saved current map to file : {os.path.basename(current_map_file)}")

    if PLOT: 
        print("Plotting current map...")
        plt.figure(figsize=(8, 6))
        plt.scatter(nodes_df['y'], 
                    nodes_df['x'], 
                    c=nodes_df['Current'], 
                    # cmap='coolwarm', s=5) # Color by current_value
                    cmap='viridis', s=5) # Color by current_value
        plt.colorbar(label='Current Value')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Current Map')
        # plt.grid(True)
        plt.gca().invert_yaxis()
        plt.show()



# %% 
# Get effective distance to voltage sources.
def distance_to_voltage_source(
        file_dataframe : pd.DataFrame,
        OUTPUT_FILE_WO_EXT:str
):
    
    """Calculates effective distance to voltage sources

    For each entry :
    1/d_eff = 1/d1 + 1/d2 + ... 1/d_n
    
    Arguments 
    ---------
    file_dataframe: 
        Dataframe made from the contents of the file.
    OUTPUT_FILE_WO_EXT:
        Path to save the output .vmap file
    """
    print("   -----------------------")
    print(" - Creating effective distance to voltage source map...")

    voltage_sources = file_dataframe.loc[
        file_dataframe['electrical_component'].str.startswith('V'),
        ['node1', 'value']
    ].rename(columns={'node1': 'node'})

    # Drop unused columns to save memory
    file_dataframe = file_dataframe.drop(columns=['node2', 'value'])

    # Efficient vectorized extraction of x and y
    node_parts = file_dataframe['node1'].str.split('_', expand=True)
    file_dataframe['x'] = node_parts[2].astype(float) / 2000
    file_dataframe['y'] = node_parts[3].astype(float) / 2000

    # Compute average Manhattan distance to all voltage sources
    file_dataframe['Dist_to_Voltage'] = file_dataframe['node1'].apply(
        lambda node: 0 if node in voltage_sources['node'].values else calc_avg_dist(voltage_sources, node)
    )

    effective_voltage_map_file = OUTPUT_FILE_WO_EXT + ".vmap"
    # Dropping the `electrical_component`,`node1` columns.
    file_dataframe = file_dataframe.drop(columns=['electrical_component', 'node1'])

    file_dataframe.to_csv(effective_voltage_map_file, index=False)
    print(f" - Saved effective distance to voltage source map to file : {os.path.basename(effective_voltage_map_file)}")

    if PLOT: 
        print("Plotting voltage map...")
        plt.figure(figsize=(8, 6))
        plt.scatter(file_dataframe['y'],  
                    file_dataframe['x'], 
                    c=file_dataframe['Dist_to_Voltage'], 
                    # cmap='coolwarm', s=5) # Color by current_value
                    cmap='viridis', s=5) # Color by current_value
        plt.colorbar(label='Voltage Value')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Effective Distance to Voltage Source map')
        # plt.grid(True)
        plt.gca().invert_yaxis()
        plt.show()



# %%

# Get .voltage file  using ir_solver module.

def IR_drop_map(
        INPUT_FILE:str,
        OUTPUT_FILE_WO_EXT:str
        # OUTPUT_FILE:str
):
    """Run IR drop solver to get voltage map.
    
    Arguments 
    ---------
    INPUT_FILE: 
        Path to input .sp file.
    OUTPUT_FILE_WO_EXT:
        Path to save the generated output .irdop file
    """

    print("   -----------------------")
    print(" - Running IR drop solver...")
    
    OUTPUT_FILE = OUTPUT_FILE_WO_EXT + ".voltage"

    ir_solver.run_solver( INPUT_FILE, OUTPUT_FILE )

    dot_voltage_df = parse_voltage_file(OUTPUT_FILE)
    dot_voltage_df['voltage'] = pd.to_numeric(dot_voltage_df['voltage'], errors='coerce')
    # print(dot_voltage_df.head(10))  

    DOT_OUT_FILE = OUTPUT_FILE_WO_EXT + ".irdrop"
    dot_voltage_df.to_csv(DOT_OUT_FILE, index=False)
    print(f" - Saved IR Drop distribution to file : {os.path.basename(DOT_OUT_FILE)}")

    if PLOT : 
        plt.figure(figsize=(8, 6))
        plt.scatter(dot_voltage_df['y'], 
                    dot_voltage_df['x'], 
                    c=dot_voltage_df['voltage'], 
                    cmap='viridis', s=5)
                    # cmap='coolwarm', s=5) # Color by current_value
        plt.colorbar(label='Voltage Value')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Voltage Map')
        # plt.grid(True)
        plt.gca().invert_yaxis()  # Invert y-axis if required
        plt.show()



# %% 
# Create PDN plot 
def PDN_plot(DATAFRAME:pd.DataFrame, OUTPUT_FILE_WO_EXT:str):
    """Create PDN distribution from .sp file
    
    Arguments 
    ---------
    DATAFRAME: 
        Dataframe made from the contents of the file.
    OUTPUT_FILE_WO_EXT:
        Path to save the output .imap file

    """

    print("   -----------------------")
    print(" - Creating PDN distribution plot...")

    DATAFRAME = DATAFRAME.drop(
                    columns=['value', 'node2', 'electrical_component']
                ).rename(columns={'node1': 'node'})
    
    print(" - Creating X and Y columns...")
    DATAFRAME[['x', 'y']] = DATAFRAME['node'].apply(lambda n: pd.Series([
        float(n.split('_')[2]) / 2000,
        float(n.split('_')[3]) / 2000
    ]))

    max_x_DF = math.ceil(DATAFRAME['x'].max() / 100 ) * 100
    max_y_DF = math.ceil(DATAFRAME['y'].max() / 100 ) * 100
    num_regions_x_axis = int( max_x_DF / 100 )
    num_regions_y_axis = int( max_y_DF / 100 )

    # print(" - Filtering metal layer...")
    # m4_dataframe = DATAFRAME[DATAFRAME['node'].str.contains('_m4_')]

    # max_x_m4 = m4_dataframe['x'].max()
    # max_y_m4 = m4_dataframe['y'].max()

    # print(f"  Max X DF : {max_x_DF}, Max Y DF: {max_y_DF}")
    # print(f"  Max X: {max_x_m4}, Max Y: {max_y_m4}")
    # print(f"  Num x regs: {num_regions_x_axis}, Num y regs : {num_regions_y_axis}")


    DATAFRAME['pitch'] = np.nan
    printed_indices = set()  

    print(" - Calculating pitch for each region...")
    for j in range(num_regions_y_axis):
        # Filter rows where y < ((j + 1) * 100)
        y_limited_df = DATAFRAME[DATAFRAME['y'] < ((j + 1) * 100)]

        for i in range(num_regions_x_axis):
            # Further filter rows where x < ((i + 1) * 100)
            xy_limited_df = y_limited_df[y_limited_df['x'] < ((i + 1) * 100)]
            xy_limited_df = xy_limited_df[~xy_limited_df.index.isin(printed_indices)]
            printed_indices.update(xy_limited_df.index)
            xy_limited_df.sort_values(by=['y', 'x'], inplace=True)

            # Filter `_m4` nodes within the current x and y limits
            m4_limited_df = xy_limited_df[xy_limited_df['node'].str.contains('_m4_')]

            # Get the first two `_m4` nodes
            first_two_rows = m4_limited_df.iloc[:2]

            # Ensure there are at least two `_m4` nodes to calculate the pitch
            if len(first_two_rows) == 2:
                node1 = first_two_rows.iloc[0]['node']
                node2 = first_two_rows.iloc[1]['node']

                # Calculate the Manhattan distance
                pitch = get_pitch_for_layer(node1, node2)

                # Assign the pitch value to all nodes within the current x and y limits
                DATAFRAME.loc[xy_limited_df.index, 'pitch'] = pitch



    print(" - Pitch values calculated for all regions.")
    
    if PLOT :
        plt.figure(figsize=(8, 6))
        plt.scatter(DATAFRAME['y'], 
                    DATAFRAME['x'],
                    c=DATAFRAME['pitch'],
                    cmap='viridis', s=10)
                    # cmap='coolwarm', s=5) # Color by current_value
        plt.colorbar(label='PDN Map')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Voltage Map')
        # plt.gca().invert_yaxis()  # Invert y-axis if required
        plt.show()

    PDN_plot_file = OUTPUT_FILE_WO_EXT + ".pdn"
    DATAFRAME.to_csv(PDN_plot_file, index=False)
    print(f" - Saved PDN distribution to file : {os.path.basename(PDN_plot_file)}")


# %%

def generate_features(
        INPUT_FILE:str
):
    """Generates the following four features per file for model training
    
    1. Current map (.imap file)
    2. Effective distance to voltage source (.vmap file)
    3. PDN distribution (.pdn file)
    4. IR drop (.irdrop file)

    Arguments
    ---------
    INPUT_FILE : 
        Path to input file. 

    Returns 
    -------
    None
        No values returned, but above four feature files are 
        created per data point for model training
    """

    # Creating output directory :
    FEATURE_DIR = os.path.join(os.path.dirname(__file__), "FEATURE_DIR")
    if not os.path.exists(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)
    
    OUTPUT_FILE_WO_EXT = os.path.join(FEATURE_DIR, os.path.splitext(os.path.basename(INPUT_FILE))[0])

    FILE_DATAFRAME = parse_lines(INPUT_FILE)

    create_current_map(FILE_DATAFRAME, OUTPUT_FILE_WO_EXT)
    distance_to_voltage_source(FILE_DATAFRAME, OUTPUT_FILE_WO_EXT)
    IR_drop_map(INPUT_FILE, OUTPUT_FILE_WO_EXT)
    PDN_plot(FILE_DATAFRAME,OUTPUT_FILE_WO_EXT)

    print(f"\nGenerated features for {os.path.basename(INPUT_FILE)}")


generate_features(INPUT_FILE)

# %%

# # For logging print statements to log file
# import sys
# log_file = os.path.join(os.path.dirname(__file__), "feature_extraction.log")
# sys.stdout = open(log_file, "w")
# # To stop logging 
# sys.stdout.close()
# sys.stdout = original_stdout

def generate_training_dataset(DATASET_FOLDER):
    """
        Processes all files in the given folder and generates features for each file.

        Arguments
        ---------
        DATASET_FOLDER : str
            Path to the folder containing the dataset files.

        Returns
        -------
        None
            Features are generated and saved for each file in the folder.
    """

    print(f"Processing dataset folder: {os.path.basename(DATASET_FOLDER)}")
    for file_name in os.listdir(DATASET_FOLDER):
        file_path = os.path.join(DATASET_FOLDER, file_name)

        generate_features(file_path)  # Call the generate_features() function for each file
        # Check if the current item is a file
        # if os.path.isfile(file_path):
    print("-----------------------------------------------------------------")
    print("Feature generation completed for all files in the dataset folder.")

SMOL_DATA = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/smol_train_data'
start_time = time.time()
generate_training_dataset(SMOL_DATA)
end_time = time.time()

print(f"Time to run : {(end_time - start_time):.6f} seconds")

sys.stdout.close()
sys.stdout = sys.__stdout__



# %%
