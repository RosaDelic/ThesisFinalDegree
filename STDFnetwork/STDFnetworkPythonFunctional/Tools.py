import numpy as np
from numba import jit, njit,prange
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from scipy.signal import find_peaks

def SDNumber(length):    
    SD = np.random.normal(0,1,length)
    SD = (SD-np.floor(SD))*(SD>1)+(SD-np.ceil(SD))*(SD<-1)+SD*(np.abs(SD)<=1)
    return SD

@njit()
def padding(array,num_zerosleft,num_zerosright):
    left = np.zeros(num_zerosleft)
    right = np.zeros(num_zerosright)
    padding_array = np.concatenate([left,array,right])
    
    return padding_array

@njit(parallel=True)
def IFRnetwork(window, window_width, firing_matrix, tol, nNeurons,ExcInh):
    #Function to calculate the firing rate of each neuron in the network and store it in a matrix firing_rate

    #calculate the length of the slidding window
    len_window = int(window_width/tol)
    
    #calculate the length of each neuron spiking trace    
    len_spikes = len(firing_matrix[0])

    #calculate the length difference between the window and the spiking trace
    difference = int((len_spikes-len_window)/2)

    #create product output
    prod = np.empty((nNeurons,len_spikes))

    #apply slidding window
    for shift in prange(2*difference):
        prod[:,shift] = np.sum(firing_matrix[:,shift:shift+len_window]*window,axis=1)
        


    firing_rate = np.divide(prod,window_width/1000) #divide betwee 1000 because we return units in spikes/second 
    
    #Define lists to store excitatory/inhibitory neurons firing rate
    Exc = []
    Inh = []
    
    #Group in different arrays excitatory and inhibitory neurons
    for neuron in range(nNeurons):
        if ExcInh[neuron]==1:
            Inh.append(firing_rate[neuron,:])
        else:
            Exc.append(firing_rate[neuron,:])

    return firing_rate,Exc,Inh   

def plot_IFRnetwork(file:str,time,ExcColor,InhColor,nNeurons,save_foldername,tol):
    #public method: Load file containing firing rate data and plot differentiating between excitatory/inhibitory population

    #-----------------------  IFR over time -------------------
    data = np.load(file)
    Exc_IFR = data["ExcIFR"]
    Inh_IFR = data["InhIFR"]
    firing_rate = data["firing_rate"]

    Exc_mean = np.mean(Exc_IFR,axis=0)
    Inh_mean = np.mean(Inh_IFR,axis=0)

    #-----------------------  IFR eventplot -------------------

    #color to use
    jet = mpl.colormaps['jet']

    #initialize vectors needed for ploting: to store peak positions, the amplitude of the peaks, the color corresponding to the normalized amplitud
    tk_neurons = []
    IFR_neurons = []
    normalized_IFR = []
    colors = []

    #calculate needed vectors for ploting: peaks_neuron and amplitude
    for i in range(nNeurons):
        tk_neuron_i = np.array(np.where(firing_rate[i,:]!=0))
        IFR_neuron_i = firing_rate[i,tk_neuron_i]

        tk_neuron_i = tk_neuron_i.reshape((tk_neuron_i.size,)).tolist()
        IFR_neuron_i = IFR_neuron_i.reshape((IFR_neuron_i.size,)).tolist()

        tk_neurons.append(tk_neuron_i)
        IFR_neurons.append(IFR_neuron_i)


    #take the minimum and maximum value of amplituds 
    IFR_min = min(min(IFR_neurons))
    IFR_max = max(max(IFR_neurons))
    IFR_norm = mpl.colors.Normalize(vmin=10.0, vmax=60.0)
    #print("IFR_min: ", IFR_min)
    #print("IFR_max: ", IFR_max)

    #plot
    for i in range(int(nNeurons)):
        normalized_IFR.append(IFR_norm(IFR_neurons[i]))

    fig = plt.figure(figsize=(20,8))
    ax = plt.axes()
    ax.set_facecolor("#00007F") #darker blue color in jet palette
    event_collection = plt.eventplot(tk_neurons,linewidths=4)
    for idx, col in enumerate(event_collection):
        col.set_colors(jet(normalized_IFR[idx]))

    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=IFR_norm, cmap=jet),location="right",pad=0.01)

    plt.ylim(0,int(nNeurons))
    plt.xlim(0,len(time))
    plt.ylabel("Neuron #",fontsize=40)
    xticks = np.arange(20000+1,step=2000/tol)
    plt.xticks(xticks,fontsize=40)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    new_labels = []
    for idx in range(len(labels)):
        element = labels[idx]
        new_labels.append(str(int((int(element)*tol)/1000)))
    ax.set_xticklabels(new_labels)
    #ax.set_xticks([])
    #ax.set_yticks([])
    plt.yticks(fontsize=40)
    tick_font_size = 30
    cbar.ax.tick_params(labelsize=tick_font_size)
    

    #plt.savefig(save_foldername+'/IFReventplot.png')

    plt.close()

    return fig
