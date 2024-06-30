import numpy as np
import time
from Prerelease import Prerelease
import numba as nb
from numba import jit, njit

@njit
def rk45Network(RHS, t0, tf, x0, N, h, neq, nNeurons, nvar, P, ExcInh, fD_AMPA, fD_NMDA, fD_GABA, fF_AMPA, fF_NMDA, fF_GABA, randomvL, randomgL, randomgsd, p0_stf):
    #RHS: differential equations system to solve with rk45 method
    #t0: initial time of the simulation
    #tf: final time of the simulation
    #h: Runge-kutta step 
    #N: total number of iterations in the rk45 
    #neq: number of equations for each neuron in the network
    #P: 320X320 table where (i,j) = 0 if neurons i and j are connected; (i,j) = 0 otherwise /// i: presyn_neuron, j: postsyn_neuron
    #ExcInh: 1x320 vector where i = 1 identifies inhibitory neurons, i = 0 identifies excitatory neurons
    #fD_X: Depression factor for the specific neurotransmitter X = AMPA, NMDA, GABA
    #fF_X: Facilitation factor for the specific neurotransmitter X = AMPA, NMDA, GABA
    
    #Prel vectors we will need during all the simulation --> they are initialized at all ones
    #For depression
    pRelAMPA = np.ones((nNeurons,N))
    pRelNMDA = np.ones((nNeurons,N))
    pRelGABA = np.ones((nNeurons,N))
    #For facilitation
    pRel_stfAMPA = np.ones((nNeurons,N))
    pRel_stfNMDA = np.ones((nNeurons,N))
    pRel_stfGABA = np.ones((nNeurons,N))


    #indexes of x0 where the variables AMPA, NMDA and GABA for pyramidal neurons are located
    indexAMPA = 9+np.arange(nNeurons)*neq #vectors of length 320
    indexNMDA = 10+np.arange(nNeurons)*neq
    indexGABA = 12+np.arange(nNeurons)*neq


    #indexes of x0 where the variables AMPA, NMDA and GABA for interneurons are located
    indexsynAMPA = 16+np.arange(nNeurons)*neq #vectors of length 320
    indexsynNMDA = 17+np.arange(nNeurons)*neq
    indexsynGABA = 19+np.arange(nNeurons)*neq

    #indexes of x0 and pre where the variables of voltage of the pyramidal neurons (vs) and interneurons (v) are located --> used to update pRel 
    indexvspyramneuron = 1+np.arange(nNeurons)*neq #vectors of length 320
    indexvinterneuron = 13+np.arange(nNeurons)*neq
    
    #------------------------------------ Initial values to rk45 Outputs  -------------------------
    #initialize time vector (ti), system solution vector (wi)
    ti=np.empty(N)
    wi = np.empty((nvar,N))

    #save initial conditions in the vectors
    ti[0] = t0
    wi[0:nvar, 0] = x0

    #---------------------------------  Loop to integrate the system ----------------------------------

    i = 1
    t0 = h
    
    while(t0+h < tf):
        if i%1000 == 0:
            print('Actual i: ', i)

        #-------------------------  RK45-Field integrator -----------------------------
        pre = x0;
        k1 = h * RHS(t0, x0, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1])
        k2 = h * RHS(t0 + h/2, x0 + k1/2, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1])
        k3 = h * RHS(t0 + h/2, x0 + k2/2, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1])
        k4 = h * RHS(t0 + h, x0 + k3, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1])

        t0 = t0 + h
        x0 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6

        #------------------------  Update time and wi vectors --------------------------
        ti[i] = t0
        wi[0:nvar, i] = x0
        
        i=i+1
        
        #------------------------  Update pRel vectors ---------------------------------
        pRelAMPA[0:nNeurons,i], pRelNMDA[0:nNeurons,i], pRelGABA[0:nNeurons,i], pRel_stfAMPA[0:nNeurons,i], pRel_stfNMDA[0:nNeurons,i], pRel_stfGABA[0:nNeurons,i] = Prerelease(x0,pre,nNeurons,neq,h,fD_AMPA,fD_NMDA,fD_GABA,fF_AMPA,fF_NMDA,fF_GABA,p0_stf,indexvspyramneuron,indexvinterneuron,ExcInh,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1])
        
    
    return ti, wi, pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA 