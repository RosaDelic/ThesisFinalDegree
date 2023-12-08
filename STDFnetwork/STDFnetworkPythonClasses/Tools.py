import numpy as np
from numba import jit

@jit(nopython=True)
def SDNumber(len):
    SD = np.random.normal(0,1,len)
    SD = (SD-np.floor(SD))*(SD>1)+(SD-np.ceil(SD))*(SD<-1)+SD*(np.abs(SD)<=1)
    return SD

#@jit("(int64,float64,float64)", nopython=True)
@jit(nopython=True)
def initialize_randomparameters(ExcInh,SDvL, SDgL, SDgsd):
    randomvL = (-60.95 + 0.3*SDvL)*(1-ExcInh)+(-63.8 + 0.15*SDvL)*ExcInh
    randomgL = (0.0667 + 0.0067*SDgL)*(1-ExcInh)+(0.1025 + 0.0025*SDgL)*ExcInh
    randomgsd = ((1.75 + 0.1*SDgsd)*0.1)*(1-ExcInh)
    return randomvL, randomgL, randomgsd

@jit("float64[::](int64[::],float64[::],int64[::],int64[::],int64[::])",nopython=True,fastmath=True)
def initialize_randomvoltage(ExcInh,x0,excvs_positions,excvd_positions,inhvs_positions):
    excvs_random = np.random.normal(0,1,len(excvs_positions))
    excvd_random = np.random.normal(0,1,len(excvd_positions))
    inhvs_random = np.random.normal(0,1,len(inhvs_positions))
    #These random numbers no estan ben generats entre [-60,-55]
    x0[excvs_positions] = (-60+5*excvs_random)*(1-ExcInh)
    x0[excvd_positions] = (-60+5*excvd_random)*(1-ExcInh)
    x0[inhvs_positions] = (-60-5*inhvs_random)*ExcInh
    
    return x0

@jit(forceobj=True)
def save_files(wi,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA):
    with open('w_matrix.npy', 'wb') as f_w_matrix:
        np.save(f_w_matrix,wi)
    with open('prelAMPA_matrix.npy', 'wb') as f_prelAMPA_matrix:
        np.save(f_prelAMPA_matrix,pRelAMPA)
    with open('prelNMDA_matrix.npy', 'wb') as f_prelNMDA_matrix:
        np.save(f_prelNMDA_matrix,pRelNMDA)
    with open('prelGABA_matrix.npy', 'wb') as f_prelGABA_matrix:
        np.save(f_prelGABA_matrix,pRelGABA)
    with open('pRel_stfAMPA_matrix.npy', 'wb') as f_pRel_stfAMPA_matrix:
         np.save(f_pRel_stfAMPA_matrix,Rel_stfAMPA)
    with open('pRel_stfNMDA_matrix.npy', 'wb') as f_pRel_stfNMDA_matrix:
        np.save(f_pRel_stfNMDA_matrix,Rel_stfNMDA)
    with open('pRel_stfGABA_matrix.npy', 'wb') as f_pRel_stfGABA_matrix:
        np.save(f_pRel_stfGABA_matrix,Rel_stfGABA)                           
                                    
                                    
                                    
                                   
                                    
                                    