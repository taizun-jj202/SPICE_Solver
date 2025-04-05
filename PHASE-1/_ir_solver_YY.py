#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static IR Drop Solver using Modified Nodal Analysis

@author: yogeshyadav

Static IR Drop Solver 
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time

def parse_netlist(input_file):
    """Parse the SPICE netlist file into a DataFrame"""
    print("\n=== Parsing Netlist ===")
    with open(input_file, 'r') as f:
        content = [line.strip() for line in f if line.strip() and not line.startswith('*')]
    
    data = []
    for i, line in enumerate(content):
        parts = line.split()
        if len(parts) < 4:
            continue
            
        element = parts[0]
        node1 = parts[1]
        node2 = parts[2]
        value = parts[3]
        
        #print(f"Line {i+1}: {element} {node1} {node2} {value}")
        data.append({
            'element': element,
            'node1': node1,
            'node2': node2,
            'value': value
        })
    
    df = pd.DataFrame(data)
    #print("\nParsed DataFrame:")
    print(df)
    return df

def parse_component_value(value):
    """Parse component values that might be complex"""
    try:
        val = complex(value.replace('j', 'j').replace('J', 'j'))
        #print(f"Parsed {value} as complex: {val}")
        return val
    except:
        try:
            val = float(value)
            #print(f"Parsed {value} as float: {val}")
            return val
        except:
            raise ValueError(f"Cannot parse value: {value}")

def build_matrices(df):
    """Build the G matrix and J vector with validation output"""
    print("\n=== Building Matrices ===")
    
    # Get all unique nodes (excluding ground)
    nodes = sorted(set(df['node1'].tolist() + df['node2'].tolist()) - {'0'})
    node_index = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    #print(f"\nNodes (excluding ground): {nodes}")
    #print(f"Node index mapping: {node_index}")
    
    # Initialize matrices
    G = np.zeros((num_nodes, num_nodes), dtype=complex)
    J = np.zeros(num_nodes, dtype=complex)
    
    #print("\nInitial G matrix:")
    #print(G)
    #print("\nInitial J vector:")
    #print(J)
    
    # Process each component
    for idx, row in df.iterrows():
        element = row['element']
        n1 = row['node1']
        n2 = row['node2']
        value = parse_component_value(row['value'])
        
        #print(f"\nProcessing {element} between {n1} and {n2} with value {value}")
        
        if element.startswith('R'):
            conductance = 1.0 / value
            #print(f"  Conductance: {conductance}")
            
            if n1 != '0' and n2 != '0':
                i = node_index[n1]
                j = node_index[n2]
                G[i,i] += conductance
                G[j,j] += conductance
                G[i,j] -= conductance
                G[j,i] -= conductance
                #print(f"  Updated G[{i},{i}] += {conductance}")
                #print(f"  Updated G[{j},{j}] += {conductance}")
                #print(f"  Updated G[{i},{j}] -= {conductance}")
                #print(f"  Updated G[{j},{i}] -= {conductance}")
            elif n1 != '0':
                i = node_index[n1]
                G[i,i] += conductance
                #print(f"  Updated G[{i},{i}] += {conductance}")
            elif n2 != '0':
                j = node_index[n2]
                G[j,j] += conductance
                #print(f"  Updated G[{j},{j}] += {conductance}")
                
        elif element.startswith('I'):
            # Corrected current source handling (SPICE convention)
            # I1 nA nB val means current flows from nA to nB
            if n1 != '0':
                i = node_index[n1]
                J[i] -= value  # Current LEAVING node n1
                #print(f"  Updated J[{i}] -= {value} (current leaving node {n1})")
            if n2 != '0':
                j = node_index[n2]
                J[j] += value  # Current ENTERING node n2
                #print(f"  Updated J[{j}] += {value} (current entering node {n2})")
                
        elif element.startswith('V'):
            if n1 != '0' and n2 == '0':
                i = node_index[n1]
                G[i,:] = 0
                G[i,i] = 1
                J[i] = value
                #print(f"  Fixed node {n1} voltage to {value}")
                #print(f"  Set G[{i},:] = 0")
                #print(f"  Set G[{i},{i}] = 1")
                #print(f"  Set J[{i}] = {value}")
            elif n1 == '0' and n2 != '0':
                j = node_index[n2]
                G[j,:] = 0
                G[j,j] = 1
                J[j] = -value
                #print(f"  Fixed node {n2} voltage to {-value} (inverted polarity)")
            else:
                raise ValueError("Voltage sources must be connected to ground")
    
    #print("\nFinal G matrix:")
    #print(G)
    #print("\nFinal J vector:")
    #print(J)
    
    return G, J, nodes

def solve_ir_drop(G, J):
    """Solve the system with validation output"""
    print("\n=== Solving System ===")
    print("Equation: G * V = J")
    
    G_sparse = csr_matrix(G)
    #print("\nSparse G matrix:")
    #print(G_sparse)
    
    V = spsolve(G_sparse, J)
    #print("\nSolution vector V:")
    #print(V)
    
    return V

def write_output(output_file, nodes, voltages):
    """Write results with validation"""
    print("\n=== Writing Results ===")
    with open(output_file, 'w') as f:
        for node, voltage in zip(nodes, voltages):
            magnitude = abs(voltage)
            #phase = np.angle(voltage, deg=True)
            line = f"{node} {magnitude:.6f}"
            #print(f"Writing: {line}")
            f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description='Static IR Drop Solver with Validation')
    parser.add_argument('--input_file', required=True, help='Input SPICE netlist file')
    parser.add_argument('--output_file', required=True, help='Output voltage file')
    args = parser.parse_args()
    start_time = time.time()
    
    
    try:
        print("=== Starting IR Drop Solver ===")
        df = parse_netlist(args.input_file)
        G, J, nodes = build_matrices(df)
        voltages = solve_ir_drop(G, J)
        write_output(args.output_file, nodes, voltages)
        
        #print("\n=== Final Results ===")
        for node, voltage in zip(nodes, voltages):
            print(f"Node {node}: {voltage:.6f} V")
            
        print(f"\nSuccessfully wrote results to {args.output_file}")
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
    
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
    

if __name__ == "__main__":
    main()