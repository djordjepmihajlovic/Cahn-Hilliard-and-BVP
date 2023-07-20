READ ME

Cahn Hilliard

For the Cahn Hilliard simulator three different files are available to set up the 
update rules, the visualization and the calculation and plotting of the free energy density.

BVP

Boundary value problem simulators (electric , magnetic and testing of over relaxation). 
Run code in the BVP file and follow prompts the desired simulation be it magnetic (wire in center) or Electric (choice of point charge
or arbitrary distribution). The code allows the user to choose the dimension of the lattice and the
accuracy of the simulation and generates the required graphs upon running. 
Graphs for both the electric and magnetic case are included in the graphs file and the corresponding 
data in the data file. 

One thing to note is that all data (fields x, y, z) have all already been normalized.

All the simulations were run on a 100x100x100 lattice with plots concerning the values at the cut [z:50]
i.e. the midplane of the lattice.

SOR

SOR problem code for the time taken to convergence for w values between 1 and 2 to be calculated
A tolerance of 0.001 was used. The optimum w value can be seen to be ~1.94

nb. The BVP code is quite long however its easy to follow; I've just included a number of 'if' statements in the 
main() so that the user only needs to interact with the console when running the program.

