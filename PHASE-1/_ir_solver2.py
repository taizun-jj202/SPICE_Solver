
import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve

FILE_PATH = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/simple_circuit1.sp'



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

def construct_sparse_G(
        resistors_list:list, 
        R_set_index:dict,
        node_index:dict, 
        N:int
    ):

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
        
        if n2 not in R_set_index:
            continue

        if n2 != 0:  
            j = node_index[n2]
            G[j, j] += g  
        
        if n1 != 0 and n2 != 0:  
            G[i, j] -= g  
            G[j, i] -= g  

    return G.tocsr()

def solve_voltage(G, I):
    """
    Solves the linear system G * V = I to compute the voltage vector V.

    PARAMETERS
    ----------
    G : scipy.sparse.csr_matrix
        Sparse conductance matrix in CSR format.
    I : numpy.ndarray
        Current vector.

    RETURNS
    -------
    V : numpy.ndarray
        Voltage vector.
    """
    
    return spsolve(G, I)

def parse_netlist(file_path):

    nodes_set = set()  # Track all unique nodes
    R_set = set()   # Track all unique resistances
                    # Used to find the size of G-Matrix. 

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
                    
                    R_set.add(n1)

                    if node not in node_metadata:
                        parsed = parse_node_name(node)
                        node_metadata[node] = parsed
            
                    nodes_set.add(node)

            elif element.startswith("I"):  # Current Source
                _, n1, n2, current = tokens
                I.append((n1, n2, float(current)))
                
                if node not in node_metadata:
                    parsed = parse_node_name(node)
                    node_metadata[node] = parsed
                
                nodes_set.add(node)


            elif element.startswith("V"):  # Voltage Source
                _, n1, n2, voltage = tokens
                V.append((n1, n2, float(voltage)))

                if node not in node_metadata:
                    parsed = parse_node_name(node)
                    node_metadata[node] = parsed

                nodes_set.add(node)

    return R, I, V, nodes_set, node_metadata, R_set


def main():

    file_path = FILE_PATH
    R, I, V, nodes_set, node_metadata, R_set = parse_netlist(file_path)

    nodes = sorted(R_set - {"0"})
    # nodes = sorted(nodes_set - {"0"})
    node_index = {node: i for i, node in enumerate(nodes_set - {"0"})}
    R_set_index = {node: i for i, node in enumerate(nodes)}
    

    N = len(nodes)
    G = construct_sparse_G(R, R_set_index, node_index, N)

    # Make the Current 1D vector
    I_vector = np.zeros(N)
    for n1, n2, current in I:
        if n1 != "0":
            I_vector[node_index[n1]] += current
        if n2 != "0":
            I_vector[node_index[n2]] -= current

    V_vector = solve_voltage(G, I_vector)
    print("Sparse G matrix in CSR format:\n", G)
    print("\nCurrent vector I:", I_vector)
    print("Voltage vector V:", V_vector)


if __name__ == "__main__":
    main()


