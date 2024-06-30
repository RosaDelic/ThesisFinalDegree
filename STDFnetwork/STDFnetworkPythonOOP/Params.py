import numpy as np
import numba as nb
from numba.types import unicode_type,DictType,float64,int64
from numba.typed import Dict
from numba.experimental import jitclass
from Neuron import Neuron

spec = [
    ('__rk45_TOL',nb.types.float64),
    ('__pyramneuron_params', nb.types.DictType(unicode_type, float64)),
    ('__interneuron_params', nb.types.DictType(unicode_type, float64)),
    ('__synaptic_params', nb.types.DictType(unicode_type, float64)),
]

@jitclass(spec)
class Parameters():
    
    #Inputs:
    #rk45_TOL: step size in rk45 method
    
    #InstanceAttributes:
    #self.__pyramneuron_params: dictionary which will contain all pyramidalneuron_params
    #self.__interneuron_params: dictionary which will contain all interneuron_params
    #self.__synaptic_params: dictionary which will contain all synaptic_connections_params
    
    def __init__(self,rk45_TOL):
        self.__rk45_TOL = rk45_TOL
        self.__pyramneuron_params = Dict.empty(key_type=nb.types.unicode_type,value_type=nb.types.float64)
        self.__interneuron_params = Dict.empty(key_type=nb.types.unicode_type,value_type=nb.types.float64)
        self.__synaptic_params = Dict.empty(key_type=nb.types.unicode_type,value_type=nb.types.float64)
        
    def initialize_allparams(self):
        #Public method: Initialize all dictionaries
    
        self.__pyramneuron_params['Cm'] = 1
        self.__pyramneuron_params['gna'] = 50
        self.__pyramneuron_params['gk'] = 10.5
        self.__pyramneuron_params['ga'] = 1
        self.__pyramneuron_params['gks'] = 0.576
        self.__pyramneuron_params['gnap'] = 0.0686
        self.__pyramneuron_params['gar'] = 0.0257
        self.__pyramneuron_params['gca'] = 0.43
        self.__pyramneuron_params['gkca'] = 0.57
        self.__pyramneuron_params['gkna'] = 1.33
        self.__pyramneuron_params['vNa'] = 55
        self.__pyramneuron_params['vK'] = -100
        self.__pyramneuron_params['vCa'] = 120
        self.__pyramneuron_params['As'] = 0.015
        self.__pyramneuron_params['Ad'] = 0.035
        self.__pyramneuron_params['tauHa'] = 15
        self.__pyramneuron_params['tauCa'] = 150
        self.__pyramneuron_params['phi'] = 4
        self.__pyramneuron_params['phiHa'] = 1
        self.__pyramneuron_params['phiKs'] = 1
        self.__pyramneuron_params['aNa'] = 0.01*10
        self.__pyramneuron_params['aCa'] = 0.005*10
        self.__pyramneuron_params['Kd'] = 30
        self.__pyramneuron_params['Rpump'] = 0.018
        self.__pyramneuron_params['NaEq'] = 9.5
        
        self.__interneuron_params['Cm'] = 1
        self.__interneuron_params['gna'] = 35
        self.__interneuron_params['gk'] = 9
        self.__interneuron_params['vNa'] = 55
        self.__interneuron_params['vK'] = -90
        self.__interneuron_params['A'] = 0.02
        self.__interneuron_params['phi'] = 1
        
        self.__synaptic_params['aAMPA'] = 3.48
        self.__synaptic_params['aNMDA'] = 0.5
        self.__synaptic_params['aGABA'] = 1
        self.__synaptic_params['aX'] = 3.48
        self.__synaptic_params['tauAMPA'] = 2 
        self.__synaptic_params['tauNMDA'] = 100
        self.__synaptic_params['tauGABA'] = 10
        self.__synaptic_params['tauX'] = 2
        self.__synaptic_params['VsynAMPA'] = 0
        self.__synaptic_params['VsynNMDA'] = 0
        self.__synaptic_params['VsynGABA'] = -70
        self.__synaptic_params['gEE_AMPA'] = 5.4/10000
        self.__synaptic_params['gEE_NMDA'] = 0.9/10000
        self.__synaptic_params['gIE_GABA'] = 4.15/10000
        self.__synaptic_params['gEI_AMPA'] = 2.25/10000                        
        self.__synaptic_params['gEI_NMDA'] = 0.5/10000                       
        self.__synaptic_params['gII_GABA'] = 0.165/10000                      
        self.__synaptic_params['tau_relAMPA'] = 400  
        self.__synaptic_params['tau_relNMDA'] = 400
        self.__synaptic_params['tau_relGABA'] = 400
        self.__synaptic_params['tau_stfAMPA'] = 50
        self.__synaptic_params['tau_stfNMDA'] = 50                       
        self.__synaptic_params['tau_stfGABA'] = 50                          
        self.__synaptic_params['p0_AMPA'] = 1  #Initial probability of release
        self.__synaptic_params['p0_NMDA'] = 1
        self.__synaptic_params['p0_GABA'] = 1
        self.__synaptic_params['p0_stfAMPA'] = 0.7
        self.__synaptic_params['p0_stfNMDA'] = 0.7
        self.__synaptic_params['p0_stfGABA'] = 0.7
        self.__synaptic_params['Vthre'] = -50
                                  

        #Adding depression factors
        self.__synaptic_params['factor_relAMPA'] = np.exp(-self.__rk45_TOL/self.__synaptic_params['tau_relAMPA'])
        self.__synaptic_params['factor_relNMDA'] = np.exp(-self.__rk45_TOL/self.__synaptic_params['tau_relNMDA'])
        self.__synaptic_params['factor_relGABA'] = np.exp(-self.__rk45_TOL/self.__synaptic_params['tau_relGABA'])
        
        #Adding facilitation factors
        self.__synaptic_params['factor_stfAMPA'] = np.exp(-self.__rk45_TOL/self.__synaptic_params['tau_stfAMPA'])
        self.__synaptic_params['factor_stfNMDA'] = np.exp(-self.__rk45_TOL/self.__synaptic_params['tau_stfNMDA'])
        self.__synaptic_params['factor_stfGABA'] = np.exp(-self.__rk45_TOL/self.__synaptic_params['tau_stfGABA'])

    def get_pyramneuron_params(self):
        #Public method: return pyramidalneuron_params dictionary
        return self.__pyramneuron_params
    
    def get_interneuron_params(self):
        #Public method: return interneuron_params dictionary
        return self.__interneuron_params
    
    def get_synaptic_params(self):
        #Public method: return synaptic_params dictionary
        return self.__synaptic_params
    