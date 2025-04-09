
# Static IR Drop Solver for VLSI Netlists

This project is a Static IR drop solver developed as part of my graduate coursework in VLSI Design Automation. This project serves as a introduction to how Machine Learning is used in EDA for VLSI design. 

## Project Overview 

This project is split into three phases, namely :
1. [IR Drop Solver](#ir-drop-solver)
2. [ML model to predict Static IR drop](#ml-model-for-static-ir-drop-prediction)
3. [Verification of model using OpenROAD](#ml-model-verification-using-openroad)



### IR Drop Solver

This phase of the project is aimed at developing a 'Golden' Static IR drop solver using nodal analysis. We develop this solver using techniques of Modified Nodal Analysis(MNA) and handles SPICE netlists with resistance, voltage and current sources. 

This solver is intended to be a golden script to generate training data for the ML model in Phase-2. 

Usage instructutions for this phase can be found [here](/PHASE-1/README.md)

### ML Model for Static IR-drop prediction

This phase includes the development of an ML model to predict static IR drop. The goal is to run the previously created solver on a hundred different testcases to 
create a training dataset. We then use this training dataset to train our choice of a ML model. We evalute this model on 10 hidden testcases.

### ML Model Verification using OpenROAD

In this phase, we familiarize ourselves with open-source EDA tool (OpenROAD). We develop scripts to interact with OpenROAD to generate SPICE netlist representation of a power grid, run a IR drop simulation, and visualize IR drop within OpenROAD GUI. We noe export the SPICE netlist and run the previously created ML model to predict IR drop and compare this to OpenROAD. 


## Usage 

Files related to each of the above three phases are in their own branch. Currently, I have developed Phase-1. As each phase builds on top of the other, usage instructions will be given in the corresponding folder for each phase. Please look at individual folders if you would like to run each phase separately. 

