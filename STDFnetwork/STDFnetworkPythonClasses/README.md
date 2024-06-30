STDFnetwork implemented in Python using Object Oriented Programming.

We provide 5 distinguished classes:
(1) Neuron: Class to create objects representing a general neuron. 
    * Attributes:
        ** Name of the neuron
        ** Type of neuron
    * Methods:
        ** f_presyn: Calculates f(V_presyn) needed to add synaptic equations to the neuron
(2/3) PyramidalNeuron/Interneuron: Class that inherits from Neuron and provides specific methods for the excitatory/inhibitory neurons
    * Attributes:
        ** Number of equations describing the neuronal model
        ** Specific parameters involved in the excitatory neuronal model
        ** Random parameters
    * Methods:
        ** Update random parameters
        ** Compute neuronal model
        ** Connect neuronal model with other neurons

(4) Parameters: Class to create a parameter object. Contains all the parameters involved in the network simulation, that is, parameters of both Pyramidal neurons and Interneurons, and parameters of the Synaptic connections.
    * Attributes: All paramters
    * Methods: Get specific parameters either from PyramidalNeurons, Interneurons or SynapticConnections

(5) SynapticConnections: Class to create Synapsis objects. Contains all methods involved in computing the probability of release and updating it. 
    * Attributes: Probability of release arrays and indexs containing positions of strength of connection in the general network field
    * Methods: 
        ** Compute the synaptic conductances of a neuron at a given integration step of the network
        ** Compute and update the probability of release
