
# Phase-1

## IR Drop Solver


This phase of the project is aimed at developing a 'Golden' Static IR drop solver using nodal analysis. We develop this solver using techniques of Modified Nodal Analysis(MNA) and handles SPICE netlists with resistance, voltage and current sources. 

This solver is intended to be a golden script to generate training data for the ML model in Phase-2. 


## Usage 

Run the following command to generate a `.voltage` output file.

```bash 

    python3 ir_solver.py --input_file <input_file> --output_file <output_file>

```

Please note here the `<input_file>` must be a `.sp` file and the output file must be named with a `.voltage` extension for clarity. 

An example use of this solver on  `testcase3.sp` has been given below along with the console logs that should be printed when we run the solver.

```bash 

    python3 ir_solver.py --input_file testcase3.sp --output_file testcase3.voltage 

```

Console log when running this solver for `testcase3.sp` input file: 

```bash 


Processing file: testcase3.sp

Parsing SPICE netlist into datastructure...
Parsed file successfully...
Constructing G and J matrices from parsed file contents...
Constructed G and J matrices and converted them into CSR format...
Solving GV=J equation...
Saving to output file...

Generated output file: testcase3.voltage


 Solution in 7.82 seconds 

```
