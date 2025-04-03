import numpy as np
from matplotlib import pyplot as plt

import argparse
import matplotlib

def main():
    # Create a parser object
    parser = argparse.ArgumentParser(description="VLSI Design Automation report project metrics")

    # Add arguments for the two file names
    parser.add_argument('spice_file', type=str, help='Path to the Spice file')
    parser.add_argument('output_file', type=str, help='Path to the Output file')

    # Parse the arguments
    args = parser.parse_args()

    spice_file = args.spice_file
    output_file = args.output_file

    print(f"SPICE file: {spice_file}")
    print(f"Output file: {output_file}")
    golden_reference = extract_nodes_and_voltages(spice_file)
    output_values = read_output_file(output_file)
    analyze_and_print_results(golden_reference, output_values)
    plot_voltage_comparison(golden_reference, output_values)

def extract_nodes_and_voltages(file_name):
    nodes_and_voltages = {}  # Dictionary to store node names and their corresponding voltages
    start_parsing = False  # Flag to start parsing when "Node   Voltage" is encountered

    with open(file_name, 'r') as file:
        for line in file:
            # Check for the starting phrase
            if "Node" in line and "Voltage" in line:
                start_parsing = True
                continue
            
            # Check for the stopping phrase
            if "Source" in line and "Current" in line and start_parsing:
                break  # Stop parsing when this line is encountered
            
            # Parse the node-voltage data after the start and before the stop
            if start_parsing:
                if (line := line.strip()) and not line.startswith(("Node", "-")):  # Skip headers and blank lines
                    tokens = line.split()  # Split the line into tokens
                    if len(tokens) == 2:  # Ensure it contains exactly 2 tokens
                        node = tokens[0].strip()
                        voltage = float(tokens[1].strip())  # Convert voltage to a floating-point number
                        # Extract subfields from the node name
                        net_name, metal_layer, x_coordinate, y_coordinate = node.split('_')
                        # Create dictionary structure
                        nodes_and_voltages[node] = {
                            "net_name": net_name,
                            "metal_layer": metal_layer,
                            "x": int(x_coordinate),
                            "y": int(y_coordinate),
                            "voltage": voltage
                        }
                        
    return nodes_and_voltages

def read_output_file(file_name):
    nodes_and_voltages = {}  # Dictionary to store node names and voltages

    with open(file_name, 'r') as file:
        for line in file:
            if (line := line.strip()):  # Skip empty lines
                tokens = line.split()  # Split the line by spaces or tabs
                # Check if the line has exactly 2 tokens, and the second token is a valid number
                if len(tokens) == 2 and is_float(tokens[1]):
                    node = tokens[0].strip()
                    voltage = float(tokens[1].strip())  # Convert voltage to a floating-point number
                    nodes_and_voltages[node] = voltage
                else:
                    # Log or print ignored lines for debugging purposes (optional)
                    print(f"Ignored line: {line.strip()}")

    return nodes_and_voltages

def is_float(value):
    try:
        np.float64(value)  # Attempt to convert the string to a numpy float
        return True
    except ValueError:
        return False

def plot_voltage_comparison(dict1, dict2):
    # Extract shared keys (common nodes) to compare voltages
    common_keys = set(dict1.keys()).intersection(dict2.keys())

    # Extract voltages for the scatter plot
    voltages_1 = [dict1[key]["voltage"] for key in common_keys]
    voltages_2 = [dict2[key] for key in common_keys]

    # Create the scatter plot
    plt.scatter(voltages_1, voltages_2, color='blue', label="Node Voltages")
    
    # Add the y = x line
    plt.plot([min(voltages_1 + voltages_2), max(voltages_1 + voltages_2)],
             [min(voltages_1 + voltages_2), max(voltages_1 + voltages_2)],
             color='red', linestyle='--', label="y = x")

    # Add labels and title
    plt.title("Voltage Comparison")
    plt.xlabel("Voltage from SPICE reference")
    plt.ylabel("Voltage from Output file")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def analyze_and_print_results(dict1, dict2):
    # Find missing and extra nodes
    keys_dict1 = set(dict1.keys())
    keys_dict2 = set(dict2.keys())
    extra_nodes = keys_dict2 - keys_dict1
    missing_nodes = keys_dict1 - keys_dict2

    # Compute errors for common nodes
    common_keys = keys_dict1 & keys_dict2  # Intersection of both dictionaries
    errors = []
    for key in common_keys:
        voltage1 = dict1[key]["voltage"]
        voltage2 = dict2[key]
        errors.append(abs(voltage1 - voltage2))  # Absolute error

    mean_error = np.mean(errors) if errors else 0
    max_error = np.max(errors) if errors else 0

    # Print the results in a formatted template
    print("===== Analysis Results =====")
    print(f"Mean Error: {mean_error:.6f} V")
    print(f"Max Error: {max_error:.6f} V")
    
    if missing_nodes:
        print("\nMissing Nodes in Output file:")
        for node in missing_nodes:
            print(f"  - {node}")
    else:
        print("\nNo missing nodes in Output file.")

    if extra_nodes:
        print("\nExtra Nodes in Output file:")
        for node in extra_nodes:
            print(f"  - {node}")
    else:
        print("\nNo extra nodes in Output file.")
    print("=============================")


if __name__ == "__main__":
    main()
