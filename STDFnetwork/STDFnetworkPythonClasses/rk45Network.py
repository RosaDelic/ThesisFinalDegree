import numpy as np
import time
from Params import Parameters
from SynapticConn import SynapticConn
from PyramidalNeuron import PyramidalNeuron 
from InterNeuron import InterNeuron
from numba import jit


@jit(nopython=True)
def rk45Network(RHS, t0, tf, x0, N, h, neq, nNeurons, nvar, P, ExcInh, fD_AMPA, fD_NMDA, fD_GABA, fF_AMPA, fF_NMDA, fF_GABA, randomvL, randomgL, randomgsd):
    #RHS: differential equations system to solve with rk45 method
    #t0: initial time of the simulation
    #tf: final time of the simulation
    #h: Runge-kutta step 
    #N: total number of iterations in the RK-45 
    #neq: number of equations for each neuron in the network
    #P: 320X320 table where (i,j) = 0 if neurons i and j are connected; (i,j) = 0 otherwise /// i: presyn_neuron, j: postsyn_neuron
    #ExcInh: 1x320 vector where i = 1 identifies inhibitory neurons, i = 0 identifies excitatory neurons
    #fD_X: Depression factor for the specific neurotransmitter X = AMPA, NMDA, GABA
    #fF_X: Facilitation factor for the specific neurotransmitter X = AMPA, NMDA, GABA
    

    #--------------------------------------  Parameters class  ------------------------------------
    #create a Parameters object with the specified rk45 step: h
    Params = Parameters(h)
    #initialize all the parameters
    Params.initialize_allparams()
    #get the synaptic_params, pyramneuron_params, interneuron_params
    synaptic_params = Params.get_synaptic_params()
    pyramneuron_params = Params.get_pyramneuron_params()
    interneuron_params = Params.get_interneuron_params()
    
    #-------------------------------------  Neuron class  -----------------------------------------
    #create a pyramidal_neuron object
    Pyramneuron = PyramidalNeuron('Pyramidal two compartment neuron','Excitatory',pyramneuron_params)
    #create an interneuron object
    Interneuron = InterNeuron('Interneuron single compartment','Inhibitory',interneuron_params)
    
    #--------------------------------------  SynapticConn class  ----------------------------------
    #create a SynapticConnection object
    Synapsis = SynapticConn(ExcInh, P,synaptic_params,neq)
    
    #------------------------------------ Initial values to rk45 Outputs  -------------------------
    #initialize time vector (ti), system solution vector (wi), all probability of Release vectors (pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA)
    
    ti=np.empty(N+1)
    wi = np.empty((nvar,N+1))
    pRelAMPA = np.empty((nNeurons,N+1))
    pRelNMDA = np.empty((nNeurons,N+1))
    pRelGABA = np.empty((nNeurons,N+1))
    pRel_stfAMPA = np.empty((nNeurons,N+1))
    pRel_stfNMDA = np.empty((nNeurons,N+1))
    pRel_stfGABA = np.empty((nNeurons,N+1))

    #save initial conditions in the vectors
    ti[0] = t0
    wi[0:nvar, 0] = x0
    pRelAMPA[0:nNeurons,0], pRelNMDA[0:nNeurons,0], pRelGABA[0:nNeurons,0], pRel_stfAMPA[0:nNeurons,0], pRel_stfNMDA[0:nNeurons,0], pRel_stfGABA[0:nNeurons,0] = Synapsis.get_pRel()

    #---------------------------------  Loop to integrate the system ----------------------------------

    i = 1
    
    while(t0+h < tf):
        if i%100 == 0:
            print('Actual i: ', i)

        #-------------------------  RK45-Field integrator -----------------------------
        pre = x0;
        k1 = h * RHS(t0, x0, neq, nNeurons, nvar, synaptic_params, Pyramneuron, Interneuron, Synapsis, ExcInh, randomvL, randomgL, randomgsd)
        k2 = h * RHS(t0 + h/2, x0 + k1/2, neq, nNeurons, nvar, synaptic_params, Pyramneuron, Interneuron, Synapsis, ExcInh, randomvL, randomgL, randomgsd)
        k3 = h * RHS(t0 + h/2, x0 + k2/2, neq, nNeurons, nvar, synaptic_params, Pyramneuron, Interneuron, Synapsis, ExcInh, randomvL, randomgL, randomgsd)
        k4 = h * RHS(t0 + h, x0 + k3, neq, nNeurons, nvar, synaptic_params, Pyramneuron, Interneuron, Synapsis, ExcInh, randomvL, randomgL, randomgsd)

        t0 = t0 + h
        x0 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6

        ti[i] = t0
        wi[0:nvar, i] = x0
        pRelAMPA[0:nNeurons,i], pRelNMDA[0:nNeurons,i], pRelGABA[0:nNeurons,i], pRel_stfAMPA[0:nNeurons,i], pRel_stfNMDA[0:nNeurons,i], pRel_stfGABA[0:nNeurons,i] = Synapsis.Prerelease(x0,pre,fD_AMPA,fD_NMDA,fD_GABA,fF_AMPA, fF_NMDA,fF_GABA)
        
        i = i + 1
        
    return ti, wi, pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA 


#def save_variables(wi,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA):
    