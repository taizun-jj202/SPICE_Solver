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

    print(nodes_df.head(10))

    current_map_file = OUTPUT_FILE_WO_EXT + ".imap"
    nodes_df.to_csv(current_map_file, index=False)
    print(f" Saved current mape to file : {current_map_file}\n")

    # print("Plotting current map...")
    # plt.figure(figsize=(8, 6))
    # plt.scatter(nodes_df['x'], nodes_df['y'], c=nodes_df['current_value'], cmap='viridis', s=50) # Color by current_value
    # plt.colorbar(label='Current Value')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Current Distribution Across Nodes')
    # plt.grid(True)
    # plt.show()
create_current_map(file_content)

# %%



def create_PDN_map():
    pass




