simple_circuit2
R1 n1 n2 1
R2 n2 n3 1
R3 n3 n4 1
R4 n4 n5 1
R5 n5 n6 1
R6 n6 n7 1
R7 n7 n8 1
R8 n8 n9 1
R9 n9 n10 1
V1 n1 0 7
V2 n6 0 7
I1 n10 0 1
I2 n4 0 1
.print dc v(*)
.op
.end