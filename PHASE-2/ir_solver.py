"""

Mini-Project-2 : IR Drop Solver, Phase-1 

Static IR Drop Solver . 
Uses GV=J method to solve for voltage at each node in the circuit.


@author: Taizun J, jafri.taizun.s@gmail.com
@date  : Apr 5 11:16:29 2025

"""

import argparse
from pprint import *
import pandas as pd
import scipy as sp
from scipy.sparse.linalg import spsolve
import os
import time

def parse_lines(FILE):
    # print("Parsing SPICE netlist into datastructure...")
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
        
        # print("Parsed file successfully...")
        return file_contents
        

def get_unique_nodes(file_contents):
    """
    Get unique nodes from the parsed file contents.
    Returns sorted set of unique nodes excluding the ground node '0'.
    """
    unique_nodes = file_contents[['node1', 'node2']].values.flatten()
    unique_nodes = set(unique_nodes) 
    unique_nodes.discard('0') # Discard ground node
    unique_nodes = sorted(unique_nodes) 
    return unique_nodes


def construct_G_and_J_matrix(
        file_contents
):
    # print("Constructing G and J matrices from parsed file contents...")
    unique_nodes = get_unique_nodes(file_contents)
    node_indexes = {node: idx for idx, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)

    # voltage_sources = file_contents[file_contents['electrical_component'].str.startswith('V')]
    # # print(voltage_sources)
    # num_voltage_sources = len(voltage_sources)
    # extra_V_row_index = 1  

    N = num_nodes 

    # Sparse G matrix 
    G = sp.sparse.lil_matrix((N, N))
    # Sparse J vector
    J = sp.sparse.lil_matrix((N, 1))

    for index, row in file_contents.iterrows():
        electrical_component = row['electrical_component']
        node1 = row['node1']
        node2 = row['node2']
        value = row['value']

        if electrical_component.startswith('R'):
            g = 1 / float(value)  # Conductance for resistors

            if node1 != '0' and node2 != '0':
                
                i = node_indexes[node1]
                j = node_indexes[node2]

                G[i, i] += g
                G[j, j] += g
                G[i, j] -= g
                G[j, i] -= g

            # Only node1 is not 0, node2 is zero, i.e a conenction to GND
            elif node1 != '0': 
                i = node_indexes[node1]
                G[i, i] += g  # Add conductance to the node connected to ground
            
            # Node2 is not '0' and node1 is '0', i.e a connection from GND to a node
            elif node2 != '0':
                j = node_indexes[node2]
                G[j, j] += g

        # Update J-vector for every current element.
        elif electrical_component.startswith('I'):
            
            if node1 != '0':
                i = node_indexes[node1]
                J[i,0] -= float(value)
            
            if node2 != '0':
                j = node_indexes[node2]
                J[j,0] += float(value)

        # # Updating G matrix for voltage : 
        elif electrical_component.startswith('V'):

            # Voltage source is attached to node1
            if node1 != '0' :             
                i = node_indexes[node1]  # Get the index of node1
                G[i, :] = 0  
                G[i, i] = 1
                J[i,0] = value

            elif node2 != '0':
                # Voltage source is attached to node2
                j = node_indexes[node2]  
                G[j, :] = 0  
                G[j, j] = 1
                J[j,0] = -value
    
    
    G = G.tocsr()
    J = J.tocsr()
    # print("Constructed G and J matrices and converted them into CSR format...")
    return G, J


def solve_G_J(
        G,
        J
):
    """
    Solve the linear system Gx = J to find x.
    """

    # print("Solving GV=J equation...")
    V = spsolve(G, J)
    return V


def format_to_output_file(
        OUTPUT_FILE ,
        unique_nodes,
        solved_voltage_vector
        ):
    
    # print("Saving to output file...")
    with open(OUTPUT_FILE, 'w') as f:
        for unique_node, votlage in zip(unique_nodes, solved_voltage_vector):
            f.write(f"{unique_node} {votlage:.6f}\n")
    


def run_solver(
        INPUT_FILE,
        OUTPUT_FILE
):
    
    # print(f"\nProcessing file: {os.path.basename(INPUT_FILE)}\n")


    file_contents = parse_lines(INPUT_FILE)
    G, J = construct_G_and_J_matrix( file_contents )
    V = solve_G_J(G,J)
    format_to_output_file(
        OUTPUT_FILE, 
        unique_nodes=get_unique_nodes(file_contents),
        solved_voltage_vector=V
    )
    # print(f'\nGenerated output file: {os.path.basename(OUTPUT_FILE)}\n')
    print(f' - Generated output file: {os.path.basename(OUTPUT_FILE)}')





if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='IR Drop Solver')
    # parser.add_argument(
    #     '--input_file', 
    #     required=True, 
    #     help='Input SPICE netlist file'
    # )
    # parser.add_argument(
    #     '--output_file', 
    #     required=True, 
    #     help='Output voltage file'
    # )
    # args = parser.parse_args()

    # INPUT_FILE = args.input_file 
    # OUTPUT_FILE = args.output_file
    
    # INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/simple_circuit1.sp'
    # INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/simple_circuit2.sp'
    INPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/SPICE_Netlists/testcase1.sp'
    
    OUTPUT_FILE = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-2/output.voltage'

    start = time.time()
    run_solver(
        INPUT_FILE=INPUT_FILE,
        OUTPUT_FILE=OUTPUT_FILE
    )
    end = time.time()
    print(f"\n Solution in {(end - start):.2f} seconds ")