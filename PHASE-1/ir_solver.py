"""
Mini-Project 2 Phase 1

This file reads .sp file and fills in objects with correct data.

Each line in sp file is written in following format from which data 
must be parsed correctly.

<electrical_component> <node1> <node2> <value>
R1 n1 n2 1

However, each node might be in the format below :
<netname>_<layer-idx>_<x-coordinate>_<y-coordinate> 
n1_m1_4800_0 


@author Taizun J
@date 24 Mar 10:39:21 2025
"""

# Global imports
import re 
import numpy as np
import scipy as sp

def parse_node_name(node_str):
    """
    Extracts netname, layer index, x, y coordinates from a node string.
    
    Each node is defined as follows : 
        <netname>_<layer-idx>_<x-coordinate>_<y-coordinate>
        
        E.g : n1_m1_9600_196800
        
        Here, n1 = node name
              m1 = metal layer
            9600 = x-coordinate
            196800 = y-coordinate
    """
    
    parts = node_str.split('_')
    if len(parts) == 4:
        netname, layer_idx, x, y = parts
        return netname, layer_idx, float(x), float(y)
    
    return node_str, None, None, None


def parse_netlist(file_path):

    """

    Reads .sp SPICE netlist and extracts elements.
    Stores each element in their corresponding lists.

    Returns :
    --------- 

    R: 
        Resistive node list.
        A list of all nodes that have resistive component
    I: 
        Current node list.
        A list of all nodes that have current sources
    V:
        Voltage source list.
        A list of all nodes that have voltage sources
    
    nodes_set :
        Set of all nodes.
        Set is used to store all nodes so no duplicates are stored.
    
    node_metdata :
        Python dict containing,
            1. net name
            2. metal layer on which the node is.
            3. X-Coordinate of node.
            4. Y-Coordinate of node.   

    """

    nodes_set = set()  # Track unique nodes
    node_metadata = {}  # Store node properties
    R = []  # (node1, node2, resistance)
    I = []  # (node, current)
    V = []  # (node, voltage)

    with open(file_path, "r") as file:
        for line in file:
           
            tokens = line.split()
            element = tokens[0]

            if element.startswith("R"):  # Resistor
                _, n1, n2, resistance = tokens
                R.append((n1, n2, float(resistance)))
                
                for node in (n1, n2):
                    if node not in node_metadata:
                        parsed = parse_node_name(node)
                        if parsed:
                            node_metadata[node] = parsed
                    nodes_set.add(node)

            elif element.startswith("I"):  # Current Source
                _, n1, n2, current = tokens
                I.append((n1, n2, float(current)))
                
                if node not in node_metadata:
                    parsed = parse_node_name(node)
                    if parsed:
                        node_metadata[node] = parsed
                nodes_set.add(node)

            elif element.startswith("V"):  # Voltage Source
                _, n1, n2, voltage = tokens
                V.append((n1, n2, float(voltage)))

                if node not in node_metadata:
                    parsed = parse_node_name(node)
                    if parsed:
                        node_metadata[node] = parsed
                nodes_set.add(node)

    return R, I, V, nodes_set, node_metadata


def construct_sparse_G(resistors_list:list, node_index:dict, N:int):
    """Builds the sparse G matrix 
    
    For every element at position (i,i), conductance is +g. 
    For every element at position (i,j) or (j,i), conductance is -g.
    For every other element, the conductance is not updated. i.e conductance is 0.

    PARAMETERS
    ----------
    resistors_list :
        List of elements with voltage drop.
        In our case, this is only Resistive elements.
    node_index :
        Dict containing index of node.
        Any element in this dict is stored in a set 
        Accessing those elements is O(1). Hence why this is 
            implemented as a dict/set combination.
    N : 
        Size of the sparse matrix is N x N.
        

    """
    G = sp.sparse.lil_matrix((N, N))  

    for n1, n2, R in resistors_list:
        
        g = 1 / R
        
        if n1 != 0:  
            i = node_index[n1]
            G[i, i] += g  
        
        if n2 != 0:  
            j = node_index[n2]
            G[j, j] += g  
        
        if n1 != 0 and n2 != 0:  
            G[i, j] -= g  
            G[j, i] -= g  

    return G.tocsr()  


# SPNETLIST = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/simple_circuit1.sp'
# SPNETLIST = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/simple_citcuit2.sp'
SPNETLIST = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/testcase3.sp'


def main():
    file_path = SPNETLIST  # Change to your file path
    R, I, V, nodes_set, node_metadata = parse_netlist(file_path)

    # Remove GND nodes, as their +/-g are 0, hence unncecssary calculations.
    nodes = sorted(nodes_set - {"0"})
    node_index = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)
    G = construct_sparse_G(R, node_index, N)

    print("\nSparse G matrix in CSR format :\n", G)

if __name__ == "__main__":
    main()




