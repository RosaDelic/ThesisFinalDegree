Biologically inspired network model simulating V1 cortical activity with Short-Term Synaptic Depression and Facilitation

We provide four implementations of the network:

(1) Using Python with Numba and Functional Programming approach.

(2) Using Matlab.

(3) Using Julia.

(4) Using Python with Numba and Object Oriented Programming approach.

We consider the first 3 options as efficient enough for the users to execute them. Each of them contains a Main file in which the user can set the following parameters simulation parameters:

    * Dnumber: Depression factor (f_D).

    * Fnumber: Facilitation factor (f_F).

    * P0_stf: Probability of release for the facilitation .

    * t0: Initial time of the simulation. The program expectes 0 to be the initial time .

    * tf: Final time of the simulation in ms.

    * h: Discretization step used to integrate the network.

After the simulation is completed, a file containig all the simulation results will be saved in the current folder in which the user is working. All implementations contain four distinguished files:

    (1) NetworkSTDall: Represents the initialization method of the implementation where the necessary simulation tools and packages are loaded and the simulation parameters are initialized.

    (2) rk45Network: Main method of the implementation, where the network is integrated.

    (3) NetworkField: Method used to compute the vector field defined by the network at the next time step from values in the previous time step which are passed as parameters.
    
    (4) Prerelease: Parameters involved in the probability of release equations are initialized. The probability of release of each neuron is computed and updated according with their activity.

The Matlab and Julia implementations have no specific requirements other than having a Matlab licence and being able to install the integrated development environments to work with. 

Regarding the Python implementation, an environment with the following requirements needs to be created and used to run the network code:
    * Numpy: version 1.24.3

    * Pandas: version 2.0.3

    * Matplotlib: version 3.7.2

    * Seaborn: version 0.13.0

    * Numba: version 0.57.1

    * Nitime: version 0.10.2
