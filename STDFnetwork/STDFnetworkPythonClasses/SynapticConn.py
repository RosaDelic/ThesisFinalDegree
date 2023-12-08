import numpy as np
import numba as nb
from numba.types import unicode_type,DictType,float64,int64
#from numba.types import int64[:],int64[:,:],float64[:]
from numba.typed import Dict
from numba.experimental import jitclass

spec = [
    #General external params
    ('__ExcInh',nb.types.int64[::]),
    ('__P', nb.types.int64[::,::]),
    ('__params', nb.types.DictType(unicode_type, float64)),
    ('__nNeurons', nb.types.int64),
    ('__neq', nb.types.int64),
    ('__nvar', nb.types.int64),
    #Prel vectors for depression
    ('__pRelAMPA',nb.types.float64[::]),
    ('__pRelNMDA',nb.types.float64[::]),
    ('__pRelGABA',nb.types.float64[::]),
    #Prel vectors for facilitation
    ('__pRel_stfAMPA',nb.types.float64[::]),
    ('__pRel_stfNMDA',nb.types.float64[::]),
    ('__pRel_stfGABA',nb.types.float64[::]),
    #indexes pyrammidal neuron
    ('__indexAMPA',nb.types.int64[::]),
    ('__indexNMDA',nb.types.int64[::]),
    ('__indexGABA',nb.types.int64[::]),
    #indexes interneuron
    ('__indexsynAMPA',nb.types.int64[::]),
    ('__indexsynNMDA',nb.types.int64[::]),
    ('__indexsynGABA',nb.types.int64[::]),
    #voltage indexes
    ('__indexvspyramneuron',nb.types.int64[::]),
    ('__indexvinterneuron',nb.types.int64[::]),
]

@jitclass(spec)
class SynapticConn():
    
    #Inputs:
    #ExcInh: 1xm vector such that in each position i: i=1 if the neuron is inhibitory; i=0 if the neuron is excitatory
    #P: mxm matrix such that for position (i,j): (i,j)=1 if neurons i and j are connected
    #synaptic_params: dictionary containing all synaptic params
    #neq: number of equations of each neuron (pyramidal + interneuron --> 20 equations)
    
    #InstanceAttributes:
    #self.__nNeurons: number of neurons in the network
    #self.__nvar: total number of variables in the network field nvar=neq*nNeurons=20*320
    #self.__pRelX: 1xnNeurons vector containing pRelease for the depression case (fD) where X is AMPA, NMDA, GABA 
    #self.__pRel__stfX: 1xnNeurons vector containing pRelease for the facilitation case (fF) where X is AMPA, NMDA, GABA 
    
    def __init__(self,ExcInh,P,synaptic_params,neq):
        #General external params
        self.__ExcInh = ExcInh
        self.__P = P
        self.__params = synaptic_params
        self.__nNeurons = len(ExcInh) #nNeurons=320
        self.__neq = neq #neq = 20
        self.__nvar = self.__neq*self.__nNeurons
        
        #Prel vectors we will need during all the simulation --> they are initialized at all ones
        #For depression
        self.__pRelAMPA = np.ones(self.__nNeurons)
        self.__pRelNMDA = np.ones(self.__nNeurons)
        self.__pRelGABA = np.ones(self.__nNeurons)
        #For facilitation
        self.__pRel_stfAMPA = np.ones(self.__nNeurons)
        self.__pRel_stfNMDA = np.ones(self.__nNeurons)
        self.__pRel_stfGABA = np.ones(self.__nNeurons)
        
        #indexes of x0 where the variables AMPA, NMDA and GABA for pyramidal neurons are located
        self.__indexAMPA = 9+np.arange(self.__nNeurons)*self.__neq #vectors of length 320
        self.__indexNMDA = 10+np.arange(self.__nNeurons)*self.__neq
        self.__indexGABA = 11+np.arange(self.__nNeurons)*self.__neq
        
                
        #indexes of x0 where the variables AMPA, NMDA and GABA for interneurons are located
        self.__indexsynAMPA = 16+np.arange(self.__nNeurons)*self.__neq #vectors of length 320
        self.__indexsynNMDA = 17+np.arange(self.__nNeurons)*self.__neq
        self.__indexsynGABA = 19+np.arange(self.__nNeurons)*self.__neq
        
        #indexes of x0 and pre where the variables of voltage of the pyramidal neurons (vs) and interneurons (v) are located --> used to update pRel 
        self.__indexvspyramneuron = 1+np.arange(self.__nNeurons)*self.__neq #vectors of length 320
        self.__indexvinterneuron = 13+np.arange(self.__nNeurons)*self.__neq
      
    
    def get_pRel(self):
        #Public method: Return preRelease probability of AMPA, NMDA, GABA for the depression case (fD) and facilitation case (fF)
        return self.__pRelAMPA, self.__pRelNMDA, self.__pRelGABA, self.__pRel_stfAMPA, self.__pRel_stfNMDA, self.__pRel_stfGABA
    

    def take_synapticvariables(self,x0):
            #public method: called at the beginning of each NetworkField, takes the synaptic variables for the current evaluation of x0
            #x0: 1xself.__neq*self.__nNeurons vector containing the variables of the network field

            #synaptic variables in pyramidal neurons (AMPA, NMDA, GABA)
            #sAMPA = np.empty(self.__nNeurons)
            #sNMDA = np.empty(self.__nNeurons)
            #sGABA = np.empty(self.__nNeurons) #this one should be zero, the excitatory neurons do not release GABA inhibitory neurotransmiters

            #synaptic variables in interneurons (AMPA, NMDA, GABA)
            #synAMPA = np.empty(self.__nNeurons)
            #synNMDA = np.empty(self.__nNeurons)
            #synGABA = np.empty(self.__nNeurons)

            #defining vectors of synaptic variables AMPA, NMDA and GABA for pyramidal neurons
            sAMPA = x0[self.__indexAMPA]
            sNMDA = x0[self.__indexNMDA]
            sGABA = x0[self.__indexGABA]

            #defining vectors of synaptic variables AMPA, NMDA and GABA for interneurons
            synAMPA = x0[self.__indexsynAMPA]
            synNMDA = x0[self.__indexsynNMDA]
            synGABA = x0[self.__indexsynGABA]

            return sAMPA, sNMDA, sGABA, synAMPA, synNMDA, synGABA
    

    def calculate_synapticfactors(self, postsyn_neuron, x0, sAMPA, sNMDA, sGABA, synAMPA, synNMDA, synGABA):
        #public method: given a postsyn_neuron, calculates the factX that are needed to compute IsynX=factX*(v-Vsyn) of the neuron for X = AMPA, NMDA, GABA
        #postsyn_neuron: index in [0,319] of the vector ExcInh that identifies the current neuron in the network field loop
        #x0: 1xself.__neq*self.__nNeurons vector containing the variables of the network field
        #sX: 1xself.__nNeurons vector containing the synaptic variables for the pyramidal neurons for X = AMPA, NMDA, GABA
        #synX: 1xself.__nNeurons vector containing the synaptic variables for the interneurons for X = AMPA, NMDA, GABA
        
        #neuron_type: 0 if postsyn_neuron is pyramidal neuron (excitatory), 1 if postsyn_neuron is interneuron (inhibitory)
        neuron_type = self.__ExcInh[postsyn_neuron]
        
        #initialize AMPA, NMDA, GABA factors
        fact_AMPA = fact_NMDA = fact_GABA = 0
        
        #define gAMPA, gNMDA, gGABA depending on the postsyn_neuron neuron_type
        #if neuron_type == 0 -->  gAMPA = self.__params['gEE_AMPA']; gNMDA = self.__params['gEE_NMDA']; gGABA = self.__params['gIE_GABA']
        #if neuron_type == 1 -->  gAMPA = self.__params['gEI_AMPA']; gNMDA = self.__params['gEI_NMDA']; gGABA = self.__params['gII_GABA']
        
        gAMPA = self.__params['gEE_AMPA']*(1-neuron_type)+self.__params['gEI_AMPA']*neuron_type
        gNMDA = self.__params['gEE_NMDA']*(1-neuron_type)+self.__params['gEI_NMDA']*neuron_type
        gGABA = self.__params['gIE_GABA']*(1-neuron_type)+self.__params['gII_GABA']*neuron_type
        
        #iterate over all presyn neurons (matrix P by rows) for AMPA
        facts_AMPA = gAMPA*sAMPA*self.__P[postsyn_neuron,:]*self.__pRelAMPA*self.__pRel_stfAMPA
        
        #sum of factors of all presynaptic neurons AMPA contributions
        fact_AMPA = facts_AMPA.sum()
        
        #iterate over all presyn neurons (matrix P by rows) for NMDA
        facts_NMDA = gNMDA*sNMDA*self.__P[postsyn_neuron, :]*self.__pRelNMDA*self.__pRel_stfNMDA
        
        #sum of factors of all presynaptic neurons NMDA contributions
        fact_NMDA = facts_NMDA.sum()
            
        #iterate over all presyn neurons (matrix P by rows) for GABA    
        facts_GABA = gGABA*sGABA*self.__P[postsyn_neuron, :]*self.__pRelGABA*self.__pRel_stfGABA
        
        #sum of factors of all presynaptic neurons GABA contributions
        fact_GABA = facts_GABA.sum()
            
        return fact_AMPA, fact_NMDA, fact_GABA
    

    def Prerelease(self,x0,pre,fD_AMPA,fD_NMDA,fD_GABA,fF_AMPA,fF_NMDA,fF_GABA):
        #x0: actual vector for the network field
        #pre: previous vector for the network field
        #fD_AMPA = fD_NMDA = fD_GABA: depression factor in the network
        #fF_AMPA = fF_NMDA = fF_GABA: facilitation factor in the network
        
        #pRel for synaptic depression (fD) distinguishing AMPA, NMDA, GABA
        self.__pRelAMPA = self.__params['p0_AMPA']*(1-self.__params['factor_relAMPA'])+self.__pRelAMPA*self.__params['factor_relAMPA']
        self.__pRelNMDA = self.__params['p0_NMDA']*(1-self.__params['factor_relNMDA'])+self.__pRelNMDA*self.__params['factor_relNMDA']
        self.__pRelGABA = self.__params['p0_GABA']*(1-self.__params['factor_relGABA'])+self.__pRelGABA*self.__params['factor_relGABA']
    

        #pRel for synaptic facilitation (fF) distinguishing AMPA, NMDA, GABA
        self.__pRel_stfAMPA = self.__params['p0_stfAMPA']*(1-self.__params['factor_stfAMPA'])+self.__pRel_stfAMPA*self.__params['factor_stfAMPA']
        self.__pRel_stfNMDA = self.__params['p0_stfNMDA']*(1-self.__params['factor_stfNMDA'])+self.__pRel_stfNMDA*self.__params['factor_stfNMDA']
        self.__pRel_stfGABA = self.__params['p0_stfGABA']*(1-self.__params['factor_stfGABA'])+self.__pRel_stfGABA*self.__params['factor_stfGABA']
        
        #--------------------update pRel if the neuron elicits a spike---------------------
        
        #vector 1xself.__nNeurons that in each position contains True/False indicating wheather the neurons fires a spike or not     
        compare = (pre[self.__indexvspyramneuron]<self.__params['Vthre'])*(x0[self.__indexvspyramneuron]>self.__params['Vthre'])*(1-self.__ExcInh)+(pre[self.__indexvinterneuron]<self.__params['Vthre'])*(x0[self.__indexvinterneuron]>self.__params['Vthre'])*(self.__ExcInh)
        
        #updating the vector of pRelAMPA for depression (fD) <--> pyramidal_neurons only
        self.__pRelAMPA = self.__pRelAMPA*fD_AMPA*compare*(1-self.__ExcInh)+self.__pRelAMPA*(compare*self.__ExcInh+(1-compare))
        
        
        #updating the vector of pRelNMDA for depression (fD) <--> pyramidal_neurons only
        self.__pRelNMDA = self.__pRelNMDA*fD_NMDA*compare*(1-self.__ExcInh)+self.__pRelNMDA*(compare*self.__ExcInh+(1-compare))
        
        #updating the vector of pRelAMPA for facilitation (fF) <--> pyramidal_neurons only
        self.__pRel_stfAMPA = (self.__pRel_stfAMPA+(1-self.__pRel_stfAMPA)*fF_AMPA)*compare*(1-self.__ExcInh)+self.__pRel_stfAMPA*(compare*self.__ExcInh+(1-compare))
        
        #updating the vector of pRelNMDA for facilitation (fF) <--> pyramidal_neurons only
        self.__pRel_stfNMDA = (self.__pRel_stfNMDA+(1-self.__pRel_stfNMDA)*fF_NMDA)*compare*(1-self.__ExcInh)+self.__pRel_stfNMDA*(compare*self.__ExcInh+(1-compare))
        
        #updating the vector of pRelGABA for depression (fD) <--> interneurons only
        self.__pRelGABA = self.__pRelGABA*fD_GABA*compare*self.__ExcInh+self.__pRelGABA*(compare*(1-self.__ExcInh)+(1-compare))
        
        #updating the vector of pRelGABA for facilitation (fD) <--> interneurons only
        self.__pRel_stfGABA = (self.__pRel_stfGABA + (1-self.__pRel_stfGABA)*fF_GABA)*compare*self.__ExcInh+self.__pRel_stfGABA*(compare*(1-self.__ExcInh)+(1-compare))
        
        return self.__pRelAMPA, self.__pRelNMDA, self.__pRelGABA, self.__pRel_stfAMPA, self.__pRel_stfNMDA, self.__pRel_stfGABA         