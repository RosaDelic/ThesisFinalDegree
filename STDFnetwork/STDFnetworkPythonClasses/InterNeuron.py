import random
import numpy as np
import numba as nb
from numba.types import unicode_type,DictType,float64,int64
from numba.typed import Dict
from numba.experimental import jitclass
#from Tools import SDNumber
from Neuron import Neuron

spec = [
    ('__neq',nb.types.int64),
    ('__name', nb.types.unicode_type),
    ('__type', nb.types.unicode_type),
    ('__params', nb.types.DictType(unicode_type, float64))
]

@jitclass(spec)
class InterNeuron(Neuron):
    
    #InstanceInputs:
    #name, type: inherited from Neuron class
    #neuron_params: dictionary containing all pyramidal neurons parameters necessary to build the model
    
    #InstanceAttributes:
    #self.__name, self.__type: name & type
    #self.__params: neuron params + set randomly distributed params gL,vl, gsd
    
    def __init__(self,name,type_neuron,neuron_params):
        #neq: number of equations of the model
        self.__neq = 3
        
        #name and type of the neuron
        self.__name = name
        self.__type = type_neuron
        
        #parameters of InterNeuron
        self.__params = neuron_params 
        
        #initialize random params with value 0, they will be updated in the bigloop for each neuron with self.__update_randomparams
        self.__params['gl'] = 0
        self.__params['vL'] = 0
           
    def get_neuronparams(self):
        #public method: return the dictionary of neuronal parameters
        return self.__params
    
    def __update_randomparams(self,vL,gL):
        #public method: update the random parameters of the neuronal model for each specific neuron
        self.__params['vL'] = vL
        self.__params['gl'] = gL
    
    def neuronalmodel(self,t,x):
        #public method: build the equations that describe the interneuron model
        #x[0] = v; x[1] = h; x[2] = n
        
        #vector to store the field that describes the model
        dx = np.zeros(self.__neq)

        #voltage
        #v=x[0];
        # ionic channels kinetics variables
        #h=x[1]; 
        #n=x[2]; 
        
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  PREVIOUS CALCULUS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

        #leak current
        Il=self.__params['gl']*(x[0]-self.__params['vL'])

        #sodium (Na)
        am=0.5*(x[0]+35)/(1-np.exp(-(x[0]+35)/10))
        bm=20*np.exp(-(x[0]+60)/18)
        minf=am/(am+bm)
        Ina=self.__params['gna']*(minf**3)*x[1]*(x[0]-self.__params['vNa'])
        ah=0.35*np.exp(-(x[0]+58)/20)
        bh=5/(1+np.exp(-(x[0]+28)/10))

        #delayed rectifier potassium (K)
        Ik=self.__params['gk']*(x[2]**4)*(x[0]-self.__params['vK'])
        an=0.05*(x[0]+34)/(1-np.exp(-(x[0]+34)/10))
        bn=0.625*np.exp(-(x[0]+44)/80)

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  DIFFERENTIAL FIELD MODEL EQUATIONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        dx[0]=(-(Il+Ina+Ik))/self.__params['Cm'] 
        dx[1]=self.__params['phi']*(ah*(1-x[1])-bh*x[1])
        dx[2]=self.__params['phi']*(an*(1-x[2])-bn*x[2])

        return dx
    
    def neuronalmodel_connected(self,t,x,neq,synaptic_params,fact_AMPA, fact_NMDA, fact_GABA, randomvL, randomgL, postsyn_neuron):
        #public method: build the equations that describe the interneuron model and add the synaptic variables
        #x: 1xneq vector containing the equations of a Neuron
        #x[13] = v; x[14] = h; x[15] = n; x[16] = synAMPA; x[17] = synNMDA; x[18] = xNMDA; x[19] = synGABA
        
        #update the random neuronal params to current neuron
        self.__update_randomparams(randomvL[postsyn_neuron], randomgL[postsyn_neuron])
        
        #vectors to store the field that describes the model
        dx_neuron = np.zeros(neq)

        #synaptic current impinging in the neuron
        Isyn_AMPA = fact_AMPA*(x[13]-synaptic_params['VsynAMPA'])
        Isyn_NMDA = fact_NMDA*(x[13]-synaptic_params['VsynNMDA'])
        Isyn_GABA = fact_GABA*(x[13]-synaptic_params['VsynGABA'])

        dx_neuron[13:13+self.__neq] = self.neuronalmodel(t,x[13:13+self.__neq])
        dx_neuron[13] = dx_neuron[13] - (Isyn_AMPA + Isyn_NMDA + Isyn_GABA)/(self.__params['Cm']*self.__params['A'])
        
        dx_neuron[16] = synaptic_params['aAMPA']*self.f_presyn(x[13])-x[16]/synaptic_params['tauAMPA']
        dx_neuron[17] = synaptic_params['aNMDA']*x[18]*(1-x[17])-x[17]/synaptic_params['tauNMDA']
        dx_neuron[18] = synaptic_params['aX']*self.f_presyn(x[13])-x[18]/synaptic_params['tauX']
        dx_neuron[19] = synaptic_params['aGABA']*self.f_presyn(x[13])-x[19]/synaptic_params['tauGABA']

        return dx_neuron
