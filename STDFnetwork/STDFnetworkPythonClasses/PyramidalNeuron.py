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
class PyramidalNeuron(Neuron):
    
    #InstanceInputs:
    #name, type: inherited from Neuron class
    #neuron_params: dictionary containing all pyramidal neurons parameters necessary to build the model, the random ones are initialized to 0
    
    #InstanceAttributes:
    #self.__name, self.__type: name & type
    #self.__params: neuron params + set randomly distributed params gL,vl, gsd
    
    def __init__(self,name,type_neuron,neuron_params):
        #neq: number of equations of the model
        self.__neq = 8
        
        #name and type of the neuron
        self.__name = name
        self.__type = type_neuron
        
        #parameters of InterNeuron
        self.__params = neuron_params
        
        #initialize random params with value 0, they will be updated in the bigloop for each neuron with self.__update_randomparams
        self.__params['gl'] = 0
        self.__params['vL'] = 0
        self.__params['gsd'] = 0
        
    def get_neuronparams(self):
        #public method: return the dictionary of neuronal parameters
        return self.__params
    
    def __update_randomparams(self,vL, gL, gsd):
        #public method: update the random parameters of the neuronal model for each specific neuron
        self.__params['vL'] = vL
        self.__params['gl'] = gL
        self.__params['gsd'] = gsd
      
    def neuronalmodel(self,t,x):
        #public method: build the equations that describe the pyramidal neuron model
        #x[0] = vs; x[1] = vd; x[2] = h; x[3] = n; x[4] = ha; x[5] = mks; x[6] = Na; x[7] = Ca
        
        #vector to store the field that describes the model
        dx = np.zeros(self.__neq)


        #soma's voltage
        #vs=x[0];
        #dendrite's voltage
        #vd=x[1];
        #ionic channels kinetics variables
        #h=x[2];
        #n=x[3];
        #ha=x[4];
        #mks=x[5];
        #Na=x[6];
        #Ca=x[7]; 

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  PREVIOUS CALCULUS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        #--------------------------currents in soma (equations for Vs)-----------------------------

        #leak current
        Il=self.__params['gl']*(x[0]-self.__params['vL'])
        
        #sodium (Na)
        am=0.1*(x[0]+33)/(1-np.exp(-(x[0]+33)/10))
        bm=4*np.exp(-(x[0]+53.7)/12)
        minf=am/(am+bm)
        Ina=self.__params['gna']*(minf**3)*x[2]*(x[0]-self.__params['vNa'])
        ah=0.07*np.exp(-(x[0]+50)/10)
        bh=1/(1+np.exp(-(x[0]+20)/10))

        #delayed rectifier ~potassium (K)
        Ik=self.__params['gk']*(x[3]**4)*(x[0]-self.__params['vK'])
        an=0.01*(x[0]+34)/(1-np.exp(-(x[0]+34)/10))
        bn=0.125*np.exp(-(x[0]+44)/25)

        #fast A-type K channel
        haInf=1/(1+np.exp((x[0]+80)/6))
        maInf=1/(1+np.exp(-(x[0]+50)/20))
        Ia=self.__params['ga']*(maInf**3)*x[4]*(x[0]-self.__params['vK'])

        #non-inactivating K channel
        Iks=self.__params['gks']*x[5]*(x[0]-self.__params['vK'])
        mksinf=1/(1+np.exp(-(x[0]+34)/6.5))
        tks=8/(np.exp(-(x[0]+55)/30)+np.exp((x[0]+55)/30))

        #Na dependent K channel
        if x[6]<10**(-5):
            Ikna = 0
            wNa=0
        else:
            wNa=0.37/(1+(38.7/x[6])**3.5)
            Ikna=self.__params['gkna']*wNa*(x[0]-self.__params['vK'])
        
        #--------------------------currents in dendrite (equations for Vd)-----------------------------

        #calcium channel
        mCainf=1/(1+np.exp(-(x[1]+20)/9))
        Ica=self.__params['gca']*(mCainf**2)*(x[1]-self.__params['vCa'])

        #Ca dependent K channel
        Ikca=((self.__params['gkca']*x[7])/(x[7]+self.__params['Kd']))*(x[1]-self.__params['vK'])

        #persistent sodium channel (NaP)
        mNapinf=1/(1+np.exp(-(x[1]+55.7)/7.7))
        INap=self.__params['gnap']*(mNapinf**3)*(x[1]-self.__params['vNa'])


        #inward rectifying K channel
        hArinf=1/(1+np.exp((x[1]+75)/4.))
        Iar=self.__params['gar']*hArinf*(x[1]-self.__params['vK'])

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  MODEL EQUATIONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        dx[0]=(-(Il+Ina+Ik+Ia+Iks+Ikna)-self.__params['gsd']*(x[0]-x[1])/self.__params['As'])/self.__params['Cm']
        dx[1]=(-(Ica+Ikca+INap+Iar)-self.__params['gsd']*(x[1]-x[0])/self.__params['Ad'])/self.__params['Cm']
        dx[2]=self.__params['phi']*(ah*(1-x[2])-bh*x[2])
        dx[3]=self.__params['phi']*(an*(1-x[3])-bn*x[3])
        dx[4]=self.__params['phiHa']*(haInf-x[4])/self.__params['tauHa']
        dx[5]=self.__params['phiKs']*(mksinf-x[5])/tks
        dx[6]=-self.__params['aNa']*(self.__params['As']*Ina+self.__params['Ad']*INap)-self.__params['Rpump']*((x[6]**3)/((x[6]**3)+3375)-(self.__params['NaEq']**3)/((self.__params['NaEq']**3)+3375))
        dx[7]=-self.__params['aCa']*(self.__params['Ad']*Ica)-(x[7]/self.__params['tauCa'])
        
        
        return dx
    
    def neuronalmodel_connected(self,t,x,neq,synaptic_params,fact_AMPA, fact_NMDA, fact_GABA, randomvL, randomgL, randomgsd, postsyn_neuron):
        #public method: build the equations that describe the pyramidal neuron model and add the synaptic variables
        #x: 1xneq vector containing the equations of a Neuron
        #x[1] = vs; x[2] = vd; x[9] = sAMPA; x[10] = sNMDA; x[11] = XNMDAs; x[12] = sGABA
        
        #update the random neuronal params to current neuron
        self.__update_randomparams(randomvL[postsyn_neuron], randomgL[postsyn_neuron], randomgsd[postsyn_neuron])
        
        #simplify variable notation
        #vs = x[1]
        #vd = x[2]
        #sAMPA = x[9]
        #sNMDA = x[10]
        #xNMDAs = x[11]
        #sGABA = x[12]

        #vectors to store the field that describes the model
        dx_neuron = np.zeros(neq)

        #synaptic current impinging in the dendrite
        Isyn_AMPA = fact_AMPA*(x[2]-synaptic_params['VsynAMPA'])
        Isyn_NMDA = fact_NMDA*(x[2]-synaptic_params['VsynNMDA'])

        #synaptic current impinging in the soma
        Isyn_GABA = fact_GABA*(x[1]-synaptic_params['VsynGABA'])

        dx_neuron[1:1+self.__neq] = self.neuronalmodel(t,x[1:1+self.__neq])
        dx_neuron[1] = dx_neuron[1] - Isyn_GABA/(self.__params['Cm']*self.__params['As'])
        dx_neuron[2] = dx_neuron[2] - (Isyn_AMPA + Isyn_NMDA)/(self.__params['Cm']*self.__params['Ad'])

        dx_neuron[9] = synaptic_params['aAMPA']*self.f_presyn(x[1])-x[9]/synaptic_params['tauAMPA']
        dx_neuron[10] = synaptic_params['aNMDA']*x[11]*(1-x[10])-x[10]/synaptic_params['tauNMDA']
        dx_neuron[11] = synaptic_params['aX']*self.f_presyn(x[1])-x[11]/synaptic_params['tauX'] 
        dx_neuron[12] = synaptic_params['aGABA']*self.f_presyn(x[1])-x[12]/synaptic_params['tauGABA']
        
        return dx_neuron