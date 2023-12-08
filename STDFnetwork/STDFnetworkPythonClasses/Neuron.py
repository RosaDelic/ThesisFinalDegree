#import random
#from Tools import SDNumber
#from Neuron import Neuron
import numba as nb
from numba.types import unicode_type
from numba import jit
import numpy as np
#from numba.experimental import jitclass

#specification of the types of each attribute
#spec = [
#    ('__name', nb.types.unicode_type),
#    ('__type', nb.types.unicode_type),
#       ]

#@jitclass(spec)
class Neuron(object):
    
    #InstanceInputs:
    #name: name of the neuron-->pyramidal cells, interneurons
    #type_neuron: excitatory, inhibitory


    def __init__(self,name,type_neuron):
        self.__name = name
        self.__type = type_neuron
        
    def __str__(self):
        #Public method: overwrites print(neuron_object)
        return f'neuron name: {self.__name} \nneuron type: {self.__type}'

    def f_presyn(self,v):
        #Public method: takes the voltage of the current neuron and calculates f(V_presyn) needed to add synaptic equations
        return (1/(1+np.exp(-(v-20)/2)))