import numpy as np
from Params import Parameters
from SynapticConn import SynapticConn
import time
from numba import jit

@jit(nopython=True)
def NetworkField(t0, x0, neq, nNeurons, nvar, synaptic_params, Pyramneuron, Interneuron, Synapsis, ExcInh, randomvL, randomgL, randomgsd):
    dx = np.zeros(nvar)
    sAMPA, sNMDA, sGABA, synAMPA, synNMDA, synGABA = Synapsis.take_synapticvariables(x0)
    
    #network loop
    for postsyn_neuron in range(nNeurons):

        index = postsyn_neuron*neq
        #postsyn_neuron identifies the current neuron, we calculate the Isyn by looking all the presyn_neurons
        #calculate the s_i from this neuron to the other postsyn_neurons
        
        #Calculate the synaptic factors Isyn = fact_syn*(v-vsyn) of this neuron
        fact_AMPA, fact_NMDA, fact_GABA = Synapsis.calculate_synapticfactors(postsyn_neuron, x0, sAMPA, sNMDA, sGABA, synAMPA, synNMDA, synGABA)
        
        x0_actualneuron = x0[index:index+neq]

        #Update neuron field
        if not ExcInh[postsyn_neuron]:
            dx[index:index+neq] = Pyramneuron.neuronalmodel_connected(t0,x0_actualneuron,neq,synaptic_params,fact_AMPA, fact_NMDA, fact_GABA, randomvL, randomgL, randomgsd, postsyn_neuron)
            
        else:
            dx[index:index+neq] = Interneuron.neuronalmodel_connected(t0,x0_actualneuron,neq,synaptic_params,fact_AMPA, fact_NMDA, fact_GABA, randomvL, randomgL, postsyn_neuron)
    
    return dx 