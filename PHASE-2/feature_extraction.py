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



# %%
# Extract each line as pandas dataframe
def parse_lines(FILE):
    print("\nParsing SPICE netlist into Pandas Dataframe...")
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
        
        print(" Parsed file successfully...")
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


    

# %%
# INPUT and OUTPUT file locations :

# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/simple_circuit1.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/simple_circuit2.sp'
INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase1.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase2.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase3.sp'

# OUTPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/output.voltage'
OUTPUT_FILE_WO_EXT = os.path.splitext(os.path.basename(INPUT_FILE))[0]

file_content = parse_lines(INPUT_FILE)


# %% 

# Get node location information from nodes 
def create_current_map(file_dataframe:pd.DataFrame):
    """Create current map from INPUT file.

    For every node in the ckt, i.e the first column 
    of dataframe, Set current_value param to zero at 
    specified node location. 

    Then filter out nodes that have current sources.
    Add value of these current sources to location of nodes. 

    Save (x,y, current_value) to a new panda dataframe and plot this dataframe. 
    """

    print("   -----------------------")
    print(" - Creating current map...")
    # Dataframe containing nodes only. 
    # nodes = | node | current_value | x | y |
    nodes_df = pd.DataFrame(file_dataframe['node1'].unique(), columns=['node'])
    nodes_df['current_value'] = 0 # Initialize current_value to 0
    nodes_df[['x', 'y']] = nodes_df['node'].apply(lambda n: pd.Series([
        float(n.split('_')[2]) / 2000,
        float(n.split('_')[3]) / 2000
    ]))

    print(" - Updating values for current sources...")

    current_sources = file_dataframe.loc[file_dataframe['electrical_component'].str.startswith('I'), ['node1', 'value']].rename(columns={'node1': 'node'})
    nodes_df = nodes_df.merge(current_sources, on='node', how='left') #merging on the 'node' column
    nodes_df['current_value'] = nodes_df['value'].fillna(0) 
    nodes_df = nodes_df.drop(columns=['value']) 

    # print(nodes_df.head(10))

    current_map_file = OUTPUT_FILE_WO_EXT + ".imap"
    nodes_df.to_csv(current_map_file, index=False)

    # print(nodes_df.head(10))
    print(f" Saved current map to file : {current_map_file}")

    # print("Plotting current map...")
    # plt.figure(figsize=(8, 6))
    # plt.scatter(nodes_df['x'], nodes_df['y'], c=nodes_df['current_value'], cmap='viridis', s=50) # Color by current_value
    # plt.colorbar(label='Current Value')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Current Map')
    # plt.grid(True)

    # plt.gca().invert_yaxis()

    # plt.show()
create_current_map(file_content)

# %% 

def distance_to_voltage_source(
        file_dataframe : pd.DataFrame
):
    
    """Calculate distance to Voltage source
    
    Takes the average of MANHATTAN distance to all voltage sources. 
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
    file_dataframe['v_distance'] = file_dataframe['node1'].apply(
        lambda node: 0 if node in voltage_sources['node'].values else calc_avg_dist(voltage_sources, node)
    )

    effective_voltage_map_file = OUTPUT_FILE_WO_EXT + ".vmap"
    
    # Dropping the `electrical_component`,`node1` columns.
    file_dataframe = file_dataframe.drop(columns=['electrical_component', 'node1'])

    file_dataframe.to_csv(effective_voltage_map_file, index=False)
    print(f" Saved effective distance to voltage source map to file : {effective_voltage_map_file}")

    # print("Plotting voltage map...")
    # plt.figure(figsize=(8, 6))
    # plt.scatter(file_dataframe['x'],  
    #             file_dataframe['y'], 
    #             c=file_dataframe['v_distance'], 
    #             # cmap='coolwarm', s=50) # Color by current_value
    #             cmap='viridis', s=50) # Color by current_value
    # plt.colorbar(label='Voltage Value')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Effective Distance to Voltage Source map')
    # plt.grid(True)

    # plt.gca().invert_yaxis()

    # plt.show()

distance_to_voltage_source(file_content)

# %%
