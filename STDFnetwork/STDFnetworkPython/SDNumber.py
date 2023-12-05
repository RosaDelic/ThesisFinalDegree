import numpy as np
def SDNumber(length):    
    SD = np.random.normal(0,1,length)
    SD = (SD-np.floor(SD))*(SD>1)+(SD-np.ceil(SD))*(SD<-1)+SD*(np.abs(SD)<=1)
    return SD