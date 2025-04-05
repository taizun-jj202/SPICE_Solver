
import numpy as np
from collections import defaultdict
import scipy as sp
from scipy.sparse.linalg import spsolve



def sparse_vector_from_defaultdict(input_default_dict, N: int):
    """
    Converts a defaultdict to a sparse vector.
    :param input_default_dict: defaultdict to convert
    :param N: Size of the vector
    :return: scipy.sparse.lil_matrix (1D sparse vector)
    """
    sparse_vector = sp.sparse.lil_matrix((N, 1))  # Create a sparse vector of size N x 1
    for index, value in input_default_dict.items():
        sparse_vector[index, 0] = value  # Only set non-zero values
    return sparse_vector.tocsr()  # Convert to CSR format for efficient operations

def parse_nodes(FILE):

    """ 
    Reads input file and extracts all unique nodes 
    and all voltage nodes.

    Returns: 
        - dict of { nodes: (value, index) }
        - N -> Size of G matrix
    """

    nodes = {} # {node_name: (value, index)}
    voltage_vector = defaultdict(float) # This 1D vector will be appended below 
                                        #   and to right of G matrix of size size(nodes)
                                        #    i.e G =( n+V+I) * (n+V+I), where n is size of nodes{} dict.
                                        #  Convert this later to a numpy array.
    voltage_rows_and_cols_vector = [] 
    current_vector = defaultdict(float) # 1D vector used in the equation GV=I
                                        #   Will convert this to a numpy array later along with voltage_vector.
    index = 0

    with open(FILE, 'r') as f:
        for line in f :
            
            line = line.strip()
            if line.startswith('.') or not line:
                continue
            parts = line.split()
            if len(parts) < 4 :
                continue

            electrical_component, node1, node2, value = parts[0], parts[1], parts[2], float(parts[3])

            if node1 != 0 and node1 not in nodes:
                nodes[node1] = value, index
                index = index + 1


            if node2 != 0 and node2 not in nodes:
                nodes[node2] = value, index
                index = index + 1


            if electrical_component.startswith('V'):
                _, voltage_index = nodes[node1] # Get the index of node1 
                voltage_vector[voltage_index] += value 
                
                # Create a 1D vector for this voltage source
                # voltage_row = np.zeros(len(nodes) + len(voltage_rows_and_cols_vector) + 1)
                voltage_row = sparse_vector_from_defaultdict(
                                                             voltage_vector,
                                                             len(nodes) + len(voltage_rows_and_cols_vector) 
                                                             )
                # voltage_row[voltage_index] += value
                voltage_row[voltage_index, 0] = voltage_row[voltage_index, 0] + value
                voltage_rows_and_cols_vector.append(voltage_row)



            if electrical_component.startswith('I'):
                _, current_index = nodes[node1] # Get the index of node1
                current_vector[current_index] += value 

        N = len(nodes)  
        # voltage_vector = sparse_vector_from_defaultdict(voltage_vector, N) # Gives a 1D numpy array from voltage_dict
        current_vector = sparse_vector_from_defaultdict(current_vector, N) # Gives a 1D numpy array from current_dict


    return nodes, voltage_rows_and_cols_vector, current_vector, N




def construct_sparse_G(unique_nodes_dict,
                       voltage_rows_and_cols_vector:list,
                       current_vector,
                       N:int
                       ):
    
    G_matrix = sp.sparse.lil_matrix((N, N))  

    num_voltage_sources = len(voltage_rows_and_cols_vector)
    new_size = N + num_voltage_sources

    voltage_rows = sp.sparse.lil_matrix( (num_voltage_sources, N) )
    voltage_columns = sp.sparse.lil_matrix( (new_size, num_voltage_sources) )

    # Get index and values of nodes from dict
    for node in unique_nodes_dict:

        resistance_value, index = unique_nodes_dict[node]
        conductance_value = 1 / resistance_value if resistance_value != 0 else 0 


        # Stamp G matrix on each node that exists in the sparse matrix
        # Stamp for G :
        #   g = conductance_value = 1 / resistance_value
        #   [ +g  -g ]
        #   [ -g  +g ]
        
        G_matrix[index, index]         += conductance_value  
        if index + 1 < N:
            G_matrix[index, index + 1]     -= conductance_value
            G_matrix[index + 1, index]     += conductance_value
            G_matrix[index + 1, index + 1] -= conductance_value
    

    voltage_rows = sp.sparse.lil_matrix((num_voltage_sources, N))
    for i, vector in enumerate(voltage_rows_and_cols_vector):
        voltage_rows[i, :] = vector.toarray().flatten()

    voltage_columns = sp.sparse.lil_matrix((new_size, num_voltage_sources))
    for i, vector in enumerate(voltage_rows_and_cols_vector):
        voltage_columns[:N, i] = vector
    

    # Convert G_matrix to CSR format for efficient operations
    G_matrix = G_matrix.tocsr()
    # Add voltage_rows to the bottom of G_matrix and voltage columns to right of expanded matrix
    expanded_G_matrix = sp.sparse.vstack([G_matrix, voltage_rows])
    expanded_G_matrix = sp.sparse.hstack([expanded_G_matrix, voltage_columns])
    expanded_G_matrix = expanded_G_matrix.tocsr()


    # Also return expanded current vector padded with zeros 
    expanded_currentnt_vector = expand_current_vector(current_vector, new_size)
    
    return expanded_G_matrix, expanded_currentnt_vector


        
def expand_current_vector (current_vector, new_size) :

    expanded_vector = sp.sparse.lil_matrix((new_size, 1))
    expanded_vector[:N, 0] = current_vector[:N, 0]

    return expanded_vector.tocsr()


def solve_voltage(G, I) :

    return spsolve(G, I)


if __name__ == "__main__":

    FILE_PATH = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/simple_circuit1.sp'
    # FILE_PATH = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/simple_citcuit2.sp'
    # FILE_PATH = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/testcase1.sp'
    # FILE_PATH = '/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/benchmarks/test.sp'
    
    
    nodes, voltage_rows_and_cols_vector, current_vector, N = parse_nodes(FILE_PATH)
    G, I = construct_sparse_G( nodes, voltage_rows_and_cols_vector, current_vector, N)
    V = solve_voltage(G, I)

    print(V)
    print()


#     When I run this file, I get the following error :
# Traceback (most recent call last):
#   File "/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-1/IR_SOLVER_V3.py", line 182, in <module>
#     nodes, voltage_rows_and_cols_vector, current_vector, N = parse_nodes(FILE_PATH)
#                                                              ~~~~~~~~~~~^^^^^^^^^^^
#   File "/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/PHASE-1/IR_SOLVER_V3.py", line 85, in parse_nodes
#     voltage_row[voltage_index] += value
#   File "/Users/taizunj/Documents/Masters_2024/ASU/Student_Docs/SEM2/EEE598_VLSI_Design_Automation/Mini_Project-2/EEE598/lib/python3.13/site-packages/scipy/sparse/_base.py", line 554, in __add__
#     raise NotImplementedError('adding a nonzero scalar to a '
#                               'sparse array is not supported')
# NotImplementedError: adding a nonzero scalar to a sparse array is not supported

# How do I fix this ? I want to add the number `value` at the index voltage_index of the sparse array. 