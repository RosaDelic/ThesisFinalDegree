Rate Model of the STDFnetwork

Minimal model describing the Network dynamics under the effects of short-term synaptic plasticity. We provide two files to do simulations:

(1) AdaptationDepression: In this file, we provide code to simulate the rate model only with adaptation and synaptic depression. The user can modify the depression factor with which the model is simulated $f_D \in [0,1]$. This parameter can be set in line 119 of this file. By running the cells in an orderly fashion in which they appear, the user can view images and videos of system dynamics.

(2) JointDepressionFacilitation: In this file, we provide code to simulate the rate model under the effects of both synaptic depression and facilitation. The adaptation is already included in the system. The user can modify 3 parameters to run the simulations: probability of release of the facilitation $(P_0^{STF})$,depression factor $(f_D)$, facilitation factor $(f_F)$. All of them must take values in the set $[0,1]$. These parameters can be set in lines 49, 51 and 52 of this file. By running the cells in an orderly fashion in which they appear, the user can view images and videos of system dynamics.

