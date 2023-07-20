READ ME

Cahn Hilliard

For the Cahn Hilliard part of the checkpoint I've set up three different files to set up the 
update rules, the visualization and the calculation and plotting of the free energy density.
This was done using classes as thats what I figured would better organize my code.
The data and the resultant graphs are in the folders data and graphs respectively. I ran the simulation
for both a 100x100 lattice and a 50x50 lattice; data and graphs for both of which have been provided.

BVP

For the boundary value problems (electric , magnetic and testing of over relaxation)
I decided to not use classes however looking back most techniques were repeated often
so maybe it wouldve been best to have set up classes. The code in the BVP file runs and prompts
the user for the desired simulation be it magnetic (wire in center) or Electric (choice of point charge
or arbitrary distribution). The code allows the user to choose the dimension of the lattice and the
accuracy of the simulation and generates the required graphs upon running. 
The graphs for both the electric and magnetic case are included in the graphs file and the corresponding 
data in the data file. I've already plotted the log relations of distance vs potential and field as well
as the un'log'-ed versions.

One thing to note is that in my files all my data (fields x, y, z) have all already been normalized.

All the simulations were run on a 100x100x100 lattice with plots concerning the values at the cut [z:50]
i.e. the midplane of the lattice.

I've also included the vector potential graphs of a simulation run of 50x50 lattices so that the vector plots
could be better seen.

SOR

For the SOR problem I've coded for the time taken to convergence for w values between 1 and 2 to be calculated
A tolerance of 0.001 was used. The optimum w value can be seen to be ~1.94

The corresponding graphs and data files are included in the corresponding folders.


nb. The BVP code is quite long however its easy to follow; I've just included a number of 'if' statements in the 
main() so that the user only needs to interact with the console when running the program.

