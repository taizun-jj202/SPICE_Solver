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



class Node:
    """
    Each node has the following data : 
     <netname>_<layer-idx>_<x-coordinate>_<y-coordinate> 
     For e.g :
        n1_m1_4800_0     
    """

    def __init__(self):
        self.netname   : str = None
        self.layer_idx : int = None
        self.x_coord        = None
        self.y_coord        = None
