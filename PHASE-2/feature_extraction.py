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

# Import Solver script
import ir_solver as ir_solver

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
            file_content.append({
                # 'node': node,
                'x': x_coord,
                'y': y_coord,
                'voltage': voltage,
            })
        
        voltage_df = pd.DataFrame(file_content)
        return voltage_df

# %%
# INPUT and OUTPUT file locations :

INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase1.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase2.sp'
# INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase3.sp'

# OUTPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/output.voltage'
OUTPUT_FILE_WO_EXT = os.path.splitext(os.path.basename(INPUT_FILE))[0]

file_content = parse_lines(INPUT_FILE)


# %% 

# Get Current Map features.
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
    nodes_df = nodes_df.drop(columns=['node'])[['x', 'y', 'Current']]

    current_map_file = OUTPUT_FILE_WO_EXT + ".imap"
    nodes_df.to_csv(current_map_file, index=False)

    # print(nodes_df.head(10))
    print(f"Saved current map to file : {current_map_file}")

    print("Plotting current map...")
    plt.figure(figsize=(8, 6))
    plt.scatter(nodes_df['x'], 
                nodes_df['y'], 
                c=nodes_df['Current'], 
                cmap='coolwarm', s=5) # Color by current_value
                # cmap='viridis', s=10) # Color by current_value
    plt.colorbar(label='Current Value')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Current Map')
    # plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()

# create_current_map(file_content)

# %% 

# Get effective distance to voltage sources.
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
    file_dataframe['Dist_to_Voltage'] = file_dataframe['node1'].apply(
        lambda node: 0 if node in voltage_sources['node'].values else calc_avg_dist(voltage_sources, node)
    )

    effective_voltage_map_file = OUTPUT_FILE_WO_EXT + ".vmap"
    
    # Dropping the `electrical_component`,`node1` columns.
    file_dataframe = file_dataframe.drop(columns=['electrical_component', 'node1'])

    file_dataframe.to_csv(effective_voltage_map_file, index=False)
    print(f"Saved effective distance to voltage source map to file : {effective_voltage_map_file}")

    # print("Plotting voltage map...")
    # plt.figure(figsize=(8, 6))
    # plt.scatter(file_dataframe['y'],  
    #             file_dataframe['x'], 
    #             c=file_dataframe['Dist_to_Voltage'], 
    #             cmap='coolwarm', s=5) # Color by current_value
    #             # cmap='viridis', s=10) # Color by current_value
    # plt.colorbar(label='Voltage Value')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Effective Distance to Voltage Source map')
    # # plt.grid(True)
    # plt.gca().invert_yaxis()
    # plt.show()

distance_to_voltage_source(file_content)

# %%

# Get .voltage file  using ir_solver module.

# OUTPUT_FILE = OUTPUT_FILE_WO_EXT + ".voltage"
def IR_drop_map(
        INPUT_FILE:str,
        # OUTPUT_FILE:str
):
    """Run IR drop solver to get voltage map."""

    print("   -----------------------")
    print(" - Running IR drop solver...")
    
    OUTPUT_FILE = OUTPUT_FILE_WO_EXT + ".voltage"
    
    ir_solver.run_solver( INPUT_FILE, OUTPUT_FILE )

    dot_voltage_df = parse_voltage_file(OUTPUT_FILE)
    dot_voltage_df['voltage'] = pd.to_numeric(dot_voltage_df['voltage'], errors='coerce')
    # print(dot_voltage_df.head(10))  

    DOT_OUT_FILE = OUTPUT_FILE_WO_EXT + ".irdrop"
    dot_voltage_df.to_csv(DOT_OUT_FILE, index=False)

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


IR_drop_map(INPUT_FILE)
