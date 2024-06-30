import numpy as np
import scipy as sc
import pandas as pd
from scipy.signal import find_peaks, medfilt, welch, correlate, correlation_lags
from scipy.fft import fft, fftfreq,fftshift,rfft,ifft
import nitime
import nitime.algorithms as tsa
import nitime.algorithms.spectral as spectrum
from scipy.stats import chi2

import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
#from matplotlib.patches import Patch
import seaborn as sns
import matplotlib.colors as mcolors
from scipy import signal
#import dill
#import cv2
import pickle
#import multiprocessing
#from multiprocessing import Pool
from Tools import IFRnetwork

import warnings
warnings.filterwarnings('ignore')


class TreatNetworkData():
   
    #InstanceInputs:
    #file (str): Name of the file the user wants to work with, specific depresion and facilitation factors (fD/fF) 
    
    #InstanceAttributes:
    #self.__data: all the data contained in the file
    #self.__exec_time: execution time of the simulation
    #self.__ti: N length time vector of the simulation, where N=t_finalxh and h the rk45 discretization time
    #self.__wi: (nNeuronsxneq,N) matrix containing the time evolution of all the variables for every neuron
    #self.__pRelX: (nNeurons,N) matrix containing the time evolution of the prob.Release when fD considered for every neurotransmitter: AMPA, NMDA, GABA
    #self.__pRel_stfX: (nNeurons,N) matrix containing the time evolution of the prob.Release when fF considered for every neurotransmitter: AMPA, NMDA, GABA    
    #self.__voltage: (nNeurons,N) matrix containing the voltage for every neuron
    
    def __init__(self,file: str, Dnumber, Fnumber,h):
        #0. Load P and ExcInh data
        self.__BN = sc.io.loadmat('BuildNetwork320.mat')
        self.__P = self.__BN['P']
        ExcInh = self.__BN['ExcInh']
        self.__ExcInh = ExcInh.reshape((ExcInh.size,))
        self.__neq = 20
        self.__nNeurons = len(self.__ExcInh)
        self.__tol = h
        self.__numInh = int(np.sum(self.__ExcInh))
        self.__numExc = int(self.__nNeurons-self.__numInh)
        self.__fD = Dnumber
        self.__fF = Fnumber
        self.__foldername = "fD"+str(Dnumber)+"_fF"+str(Fnumber)
        
        #1. Load saved data from simulation
        self.__data = np.load(file)
        
        #2. Read data from the file
        
        #General data
        self.__exec_time = self.__data['exec_time']
        self.__ti = self.__data['ti']
        self.__wi = self.__data['wi']

        #Probability of release
        #Depression
        self.__pRelAMPA = self.__data['pRelAMPA']
        self.__pRelNMDA = self.__data['pRelNMDA']
        self.__pRelGABA = self.__data['pRelGABA']

        #Facilitation
        self.__pRel_stfAMPA = self.__data['pRel_stfAMPA']
        self.__pRel_stfNMDA = self.__data['pRel_stfNMDA']
        self.__pRel_stfGABA = self.__data['pRel_stfGABA']        
        
        #Conductances
        self.__gEE_AMPA = 5.4/10000
        self.__gEE_NMDA = 0.9/10000
        self.__gIE_GABA = 4.15/10000
        self.__gEI_AMPA = 2.25/10000                        
        self.__gEI_NMDA = 0.5/10000                       
        self.__gII_GABA = 0.165/10000 
        
        #3. Build voltage table
        self.__voltage = np.empty([self.__nNeurons,len(self.__ti)])
        voltage_positions = (1+np.arange(self.__nNeurons)*self.__neq)*(1-self.__ExcInh)+(13+np.arange(self.__nNeurons)*self.__neq)*self.__ExcInh
        self.__voltage = self.__wi[voltage_positions,:]
        
        #4. Build strength-of-connection tables
        #AMPA
        self.__sAMPA = np.empty([self.__nNeurons,len(self.__ti)])
        sAMPA_positions = (9+np.arange(self.__nNeurons)*self.__neq)*(1-self.__ExcInh)+(16+np.arange(self.__nNeurons)*self.__neq)*self.__ExcInh
        self.__sAMPA = self.__wi[sAMPA_positions,:]
        
        #NMDA
        self.__sNMDA = np.empty([self.__nNeurons,len(self.__ti)])
        sNMDA_positions = (10+np.arange(self.__nNeurons)*self.__neq)*(1-self.__ExcInh)+(17+np.arange(self.__nNeurons)*self.__neq)*self.__ExcInh
        self.__sNMDA = self.__wi[sNMDA_positions,:]
        
        #GABA
        self.__sGABA = np.empty([self.__nNeurons,len(self.__ti)])
        sGABA_positions = (12+np.arange(self.__nNeurons)*self.__neq)*(1-self.__ExcInh)+(19+np.arange(self.__nNeurons)*self.__neq)*self.__ExcInh
        self.__sGABA = self.__wi[sGABA_positions,:]
        
        #5. Build tables to separate ExcNeurons and InhNeurons voltage
        #Save exc and inh positions in two lists
        self.__exc_block = []
        self.__inh_block = []
        for neuron in range(self.__nNeurons):
            if self.__ExcInh[neuron]==0:
                self.__exc_block.append(neuron)
            else:
                self.__inh_block.append(neuron)
                
        #Initialize exc inh arrays
        self.__Exc_neurons = np.zeros([len(self.__exc_block),len(self.__ti)])
        self.__Inh_neurons = np.zeros([len(self.__inh_block),len(self.__ti)])
        
        #
        self.__Exc_neurons[0:len(self.__exc_block),:]=self.__voltage[self.__exc_block,:]
        self.__Inh_neurons[0:len(self.__inh_block),:]=self.__voltage[self.__inh_block,:]
        
        #4. Colors
        #dictionary of colors to work with
        color_dict = mcolors.CSS4_COLORS
        self.__ExcColor = color_dict["salmon"]
        self.__InhColor = color_dict["cornflowerblue"]
        self.__AMPAColor = color_dict["orange"]
        self.__NMDAColor = color_dict["royalblue"]
        self.__GABAColor = color_dict["purple"]
        self.__ExcFFT = color_dict["firebrick"]
        self.__InhFFT = color_dict["mediumblue"]
        
        del self.__data
                

    def get_time(self):
        #public method: return time vector
        
        return self.__ti
    
    def get_nNeurons(self):
        #public method: return number of neurons considered in the simulation
        
        return self.__nNeurons
    
    def get_ExcInhColors(self):
        #public method: return colors used to plot data of excitatory and inhibitory populations
        
        return self.__ExcColor,self.__InhColor
    
    def get_tol(self):
        #public method: return tolerance used in the simulation
        
        return self.__tol
    
    def get_voltage(self):
        #public method: return voltage matrix
        np.savez(self.__foldername+'/VoltageData',voltage = self.__voltage)
              
        return self.__voltage
    
    def get_variable_neuron(self,neuron_number, variable):
        #public method: return array containing the variable specified for the neuron_number introduced
        #neuron_number: integer (0-12) for excitatory neurons//(13-19) for inhibitory neurons
        return self.__wi[variable+neuron_number*20,:]
    
    def firing_amplitude_matrix(self):
        #public method: calculate firing matrix and peak matrix
        #firing_matrix: (nNeurons,N) matrix with (n,t)=1 if at time t, neuron n elicits a spike
        #peak_matrix: (nNeurons,N) matrix with the value of the amplitude of the spike fired at time t
        
        #Matrix to return
        firing_matrix = np.zeros([self.__nNeurons,len(self.__ti)])
        amplitude_matrix = np.zeros([self.__nNeurons,len(self.__ti)])
        
        #Matrix to store peaks positions for each neuron
        for neuron in range(self.__nNeurons):
            peaks, _ = find_peaks(self.__voltage[neuron,:], height=0) #height: required heigh for peaks to be considered spikes

            firing_matrix[neuron,peaks] = np.ones(len(peaks),)

            amplitude_matrix[neuron,peaks] = self.__voltage[neuron,peaks]

            
        return firing_matrix,amplitude_matrix
    
    def eventplotFiring(self,firing_matrix):
        #public method: given the firing_matrix, return the Eventplot using different colors for excitatory and inhibitory neurons
        
        #vector to store spike positions of neurons
        peaks_neurons = []
        
        #vector to store color corresponding to that spike (excitatory/inhibitory neuron)
        colors = []
        
        #calculate needed vectors for ploting: peaks_neuron and colors
        for i in range(320):
            peaks_neuron_i = np.array(np.where(firing_matrix[i,:]==1))
            peaks_neuron_i = peaks_neuron_i.reshape((peaks_neuron_i.size)).tolist()
            peaks_neurons.append(peaks_neuron_i)
            if self.__ExcInh[i]==0:
                colors.append(self.__ExcColor)
            else:
                colors.append(self.__InhColor)
                
        #plot
        fig = plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.eventplot(peaks_neurons,colors = colors,linewidths=4)
        plt.xlabel("Time(s)",fontsize=40)
        plt.ylabel("Neuron #",fontsize=40)
        xticks = np.arange(len(self.__ti)+1,step=2000/self.__tol)
        plt.xticks(xticks,fontsize=40)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        new_labels = []
        for idx in range(len(labels)):
            element = labels[idx]
            new_labels.append(str(int((int(element)*self.__tol)/1000)))
        ax.set_xticklabels(new_labels)
        plt.yticks(fontsize=40)
        plt.ylim(0, self.__nNeurons-1)
        plt.xlim(0, len(self.__ti))
        legend_elements = [Line2D([0],[0],color=self.__ExcColor,lw=4,label="Excitatory"),
                           Line2D([0],[0],color=self.__InhColor,lw=4,label="Inhibitory")]
        
        ax.legend(handles=legend_elements,bbox_to_anchor=(1,1.18),loc="upper right",fontsize=30,ncol=2)
        
        
        plt.savefig(self.__foldername+'/SpikesRasterPlot.png')
        plt.close()
                
        return fig
    
    def eventplotAmplitude(self,firing_matrix,amplitude_matrix):
        #public method: given the firing_matrix and amplitude_matrix return the Eventplot using different colors for the spike amplitude
        
        #colormap to use in the plot 
        jet = mpl.colormaps['jet']
    
        #initialize vectors needed for ploting: to store peak positions, the amplitude of the peaks, the color corresponding to the normalized amplitud
        peaks_neurons = []
        amplituds_neurons = []
        normalized_am = []
        colors = []

        #calculate needed vectors for ploting: peaks_neuron and amplitude
        for i in range(320):
            peaks_neuron_i = np.array(np.where(firing_matrix[i,:]==1))
            amplituds_neuron_i = amplitude_matrix[i,peaks_neuron_i]

            peaks_neuron_i = peaks_neuron_i.reshape((peaks_neuron_i.size,)).tolist()
            amplituds_neuron_i = amplituds_neuron_i.reshape((amplituds_neuron_i.size,)).tolist()

            peaks_neurons.append(peaks_neuron_i)
            amplituds_neurons.append(amplituds_neuron_i)

        
        #take the minimum and maximum value of amplituds 
        am_min = min(min(amplituds_neurons))
        am_max = max(max(amplituds_neurons))
        am_norm = mpl.colors.Normalize(vmin=am_min, vmax=27)
        #am_norm = mpl.colors.LogNorm(vmin=am_min, vmax=am_max)

        #plot
        for i in range(320):
            normalized_am.append(am_norm(amplituds_neurons[i]))

        fig = plt.figure(figsize=(20,8))
        ax = plt.axes()
        ax.set_facecolor("#00007F") #darker blue color in jet palette
        event_collection = plt.eventplot(peaks_neurons,linewidths=4)
        for idx, col in enumerate(event_collection):
            col.set_colors(jet(normalized_am[idx]))
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=am_norm, cmap=jet))
        tick_font_size = 30
        cbar.ax.tick_params(labelsize=tick_font_size)
        
        plt.ylim(0, self.__nNeurons-1)
        plt.xlim(0, len(self.__ti))
        plt.xlabel("Time(s)",fontsize=40)
        plt.ylabel("# Neuron",fontsize=40)
        xticks = np.arange(len(self.__ti)+1,step=2000/self.__tol)
        plt.xticks(xticks,fontsize=40)#,rotation=25)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        new_labels = []
        for idx in range(len(labels)):
            element = labels[idx]
            new_labels.append(str(int((int(element)*self.__tol)/1000)))
        ax.set_xticklabels(new_labels)
        plt.yticks(fontsize=40)

        plt.savefig(self.__foldername+'/AmplitudeRasterPlot.png')
            
        plt.close()
        
        return fig

    def count_spike_bins(self,firing_matrix,bin_length_ms):
        #public method: divide the voltage matrix in bins
        #count spikes and interspike interval duration within bin
        
        #length of bin vectors, considering the tolerance used in the simulation
        length_locs_bin = int(bin_length_ms/self.__tol)
        
        #number of bins created
        nBins = int(len(self.__ti)/length_locs_bin)
        
        #create reshaped firing_matrix into bins per neuron
        bin_firing_matrix = np.empty((self.__nNeurons,nBins,length_locs_bin))
        
        #create array to save numSpikes of each neuron in each bin
        nSpikes = np.empty((self.__nNeurons,nBins))

        #Lists for ISI df
        neuron_number_ISI = []
        type_neuron_ISI = []
        ISI_list = []
        
        #Lists for nSpikes df
        neuron_number_nSpikes = []
        type_neuron_nSpikes = []        
        bin_list = []
        nSpikes_list = []

        for neuron in range(self.__nNeurons):

            
            #-----------------------------------------------------  get neuron type  ------------------------------------------------
            if self.__ExcInh[neuron]==1:
                type_n = 'Inhibitory'
                 
            else:
                type_n = 'Excitatory'

            
            #------------------------------------------  store ISI info in df_ISI dataFrame  ----------------------------------------
            peaks, _ = find_peaks(self.__voltage[neuron,:], height=0) #height: required heigh for peaks to be considered spikes
            ISI_vector_neuron = np.diff(peaks)
            for ISI in ISI_vector_neuron:
                #append neuron_number to list
                neuron_number_ISI.append(neuron)
                
                #apend neuron_type to list
                type_neuron_ISI.append(type_n)
                
                #append ISI to list
                ISI_list.append(float(ISI*self.__tol))
                
                
            #----------------------------------------  store nSpikes info in df_nSpikes dataFrame  ------------------------------------                
            #get the spikes times of each neuron
            neuron_spikes = firing_matrix[neuron,:]
            
            #reshape the spike times of neuron into bins
            bin_neuron_vector = neuron_spikes.reshape((nBins,-1))
            
            #fill the bin firing matrix with the bins of current neuron
            bin_firing_matrix[neuron,:,:] = bin_neuron_vector
        
            #count number of spikes in each bin for current neuron
            nSpikes_neuron = np.sum(bin_neuron_vector,axis=1,dtype=np.int64)
            
            #fill nSpikes matrix with the number of spikes in each bin of current neuron 
            nSpikes[neuron,:] = nSpikes_neuron
            
            for bin_number in range(len(nSpikes[neuron,:])):
                #append neuron_number to list
                neuron_number_nSpikes.append(neuron)
                
                #apend neuron_type to list
                type_neuron_nSpikes.append(type_n)
                
                #append bin to list
                bin_list.append(str(int(bin_number*bin_length_ms/1000))+ '-' +str(int((bin_number+1)*bin_length_ms/1000)))
                
                #append nSpikes to list
                n_spikes_bin = nSpikes[neuron,bin_number]
                nSpikes_list.append(int(n_spikes_bin))
        #----------------------------------------  fill df_ISI and nSpikes_ISI ------------------------------------ 
      
        #define dicts to create pd DataFrames
        dict_ISI = {'Neuron #': neuron_number_ISI, 'Type': type_neuron_ISI, 'ISI (ms)': ISI_list}
        dict_nSpikes = {'Neuron #': neuron_number_nSpikes, 'Type': type_neuron_nSpikes,'Bin': bin_list, 'numSpikes': nSpikes_list}


        #create dataFrame to store InterSpikeIntervals (ISI) of each neuron
        df_ISI = pd.DataFrame(dict_ISI)
        
        #create dataFrame to store nSpikes of each neuron in each bin
        df_nSpikes = pd.DataFrame(dict_nSpikes)
        
        #----------------------------------------  calculate discrete-time firing rate of each neuron ------------------------------------ 
        #average number of spikes of each neuron and bin by the duration length of the bin in seconds
        neuron_discr_firing_rate = np.divide(nSpikes,bin_length_ms/1000)
        
        return df_ISI,df_nSpikes,neuron_discr_firing_rate
    
    
    def plot_spike_distribution(self,df_nSpikes):
        #public method: given the df_nSpikes dataset, plot the average of spikes per bin, differentiating between excitatory and inhibitory neurons

        sns.set_theme(style='ticks',context = 'talk',font_scale = 1.25)
        g = sns.catplot(data = df_nSpikes,x = 'Bin',y = 'numSpikes',hue = 'Type',kind = 'bar', errorbar = 'sd', palette = [self.__ExcColor,self.__InhColor], aspect=2)
        g.set_axis_labels('Time bin (s)','# Spikes fired')
        g.set_xticklabels(rotation = 45)
        g.legend.set_title('Neuron type')
        g.set(title ='Distribution of spikes across time bins')
        plt.savefig(self.__foldername+'/SpikeDistribution.png')
        
        return
    
    
    def plot_ISIdistribution(self,df_ISI):
        #public method: given the df_ISI fig1=plt.figure(figsize=(40,8))
        dfISI = df_ISI[df_ISI['Neuron #']==0]
        sns.set_theme(style='ticks',context = 'talk',font_scale = 1.2)
        g = sns.displot(data=df_ISI, x='ISI (ms)', hue='Type',stat='density',common_norm='False',bins=94,aspect=2, kde=True,log_scale=True, palette = [self.__ExcColor,self.__InhColor])#.set(xlim=(0,80))
        g.set_axis_labels('ISI (ms)','Density')
        g.legend.set_title('Neuron type')
        g.set(title ='ISI distribution')
        
        plt.savefig(self.__foldername+'/ISIdistribution.png')
        
        
        return
    
    def plot_firing_rate(self,network_fr):
        #Public method: given the network_fr per bins, plot differentiating between excitatory and inhibitory population
        
        #get number of bins
        nBins = network_fr.shape[1]
        
        #length of each bin according to the tolerance
        bin_len = int((len(self.__ti)*self.__tol)/nBins)
        #define edges
        edges = np.arange(len(self.__ti)*self.__tol+1,step=bin_len)/1000
        edgesxlabel = np.arange(len(self.__ti)*self.__tol+1,step=20*bin_len)/1000
        #initialize lists to store EI firing rate data
        InhibitorySpikes = [] 
        ExcitatorySpikes = []
        
        for neuron in range(self.__nNeurons):
            if self.__ExcInh[neuron]==0:
                ExcitatorySpikes.append(network_fr[neuron,:])
            else:
                InhibitorySpikes.append(network_fr[neuron,:])
            
        ExcitatorySpikes = np.array(ExcitatorySpikes)
        InhibitorySpikes = np.array(InhibitorySpikes)
        
        #average number of spikes over number of EI neurons respectively
        nSpikesExcitatory = np.sum(ExcitatorySpikes,axis=0)/self.__numExc
        nSpikesInhibitory = np.sum(InhibitorySpikes,axis=0)/self.__numInh
        
        fig=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.stairs(nSpikesExcitatory,edges,edgecolor = self.__ExcColor,linewidth=6)
        plt.stairs(nSpikesInhibitory,edges,edgecolor = self.__InhColor,linewidth=6)
        plt.xlabel("Time bins (s)",fontsize=40)
        plt.ylabel("spikes/s",fontsize=40)
        plt.xticks(ticks=edgesxlabel,fontsize=40)
        plt.yticks(fontsize=40)
        last_second = len(self.__ti)*self.__tol/1000
        plt.xlim(0, last_second)
        legend_elements = [Line2D([0],[0],color=self.__ExcColor,lw=4,label="Excitatory"),
                           Line2D([0],[0],color=self.__InhColor,lw=4,label="Inhibitory")]
        
        ax.legend(handles=legend_elements,bbox_to_anchor=(1,1.2),ncol=2,loc="upper right",fontsize=30)
        
        plt.savefig(self.__foldername+'/SpikeCount.png')
        
        return 
    
    def IFRnetwork(self, window_type, window_width, firing_matrix):
        #Define the type of sliding window to use to compute the firing rate of each neuron
        
        #Define lists to store excitatory/inhibitory neurons firing rate
        Exc = []
        Inh = []
        
        #define the window array, depending on the window_type
        if window_type=="rectangular":
            window = signal.windows.boxcar(int(window_width/self.__tol))
        elif window_type=="gaussian":
            window = signal.windows.gaussian(int(window_width/self.__tol), std=500) 

        firing_rate, Exc, Inh = IFRnetwork(window, window_width, firing_matrix, self.__tol, self.__nNeurons,self.__ExcInh)
        
        #Convert lists to array
        Exc = np.array(Exc)
        Inh = np.array(Inh)
        
        #Save info into file
        np.savez(self.__foldername+'NetworkIFR.npz', ExcIFR = Exc, InhIFR = Inh, firing_rate=firing_rate)  
        
        return firing_rate, Exc, Inh
        
    def plot_IFRnetwork(self,file:str):
        #public method: Load file containing firing rate data and plot differentiating between excitatory/inhibitory population

        data = np.load(file)
        Exc_IFR = data["ExcIFR"]
        Inh_IFR = data["InhIFR"]
        firing_rate = data["firing_rate"]
        
        Exc_mean = np.mean(Exc_IFR,axis=0)
        Inh_mean = np.mean(Inh_IFR,axis=0)
        
        #-----------------------  IFR over time -------------------

        fig1=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.plot(self.__ti/1000,Exc_mean,color=self.__ExcColor,linewidth=6)
        plt.plot(self.__ti/1000,Inh_mean,color=self.__InhColor,linewidth=6)
        plt.xlabel("Time (ms)",fontsize=40)
        plt.ylabel("IFR (sp/s)",fontsize=40)
        xticks = np.arange(len(self.__ti)*self.__tol/1000+1,step=2)
        plt.xticks(xticks,fontsize=40)
        plt.xticks(fontsize=40)#,rotation=45)
        plt.yticks(fontsize=40)
        legend_elements = [Line2D([0],[0],color=self.__ExcColor,lw=4,label="Excitatory"),
                           Line2D([0],[0],color=self.__InhColor,lw=4,label="Inhibitory")]

        ax.legend(handles=legend_elements,bbox_to_anchor=(1,1.18),loc="upper right",fontsize=30,ncol=2)
        last_second = len(self.__ti)*self.__tol/1000
        plt.xlim(0, last_second)
        plt.savefig(self.__foldername+'/IFRtimeplot.png')

        plt.close()

        #-----------------------  IFR phase space -------------------
        fig2=plt.figure(figsize=(7,7))
        ax=plt.axes()
        plt.plot(Exc_mean,Inh_mean,color="k",linewidth=6)
        plt.xlabel("IFR Exc",fontsize=24)
        plt.ylabel("IFR Inh",fontsize=24)
        plt.xticks(np.arange(200,step=50),fontsize=22)#,rotation=45)
        plt.yticks(np.arange(200,step=50),fontsize=22)
        plt.xlim(0,np.max(Exc_mean)+10)
        plt.ylim(0,np.max(Inh_mean)+10)
        plt.savefig(self.__foldername+'/IFRphaseplot.png')

        plt.close()
        
        return fig1,fig2,Exc_mean,Inh_mean
    
    def getAveragePrel(self):
        #public method: returns 6-(1xlen(time)) arrays, each one containing the averaged probability of release of the network.
        #3 arrays are for the fD case, the other 3 for the fF case
        
        #Note: 
        #Excitatory neurons only contribute to AMPA, NMDA
        #Inhibitory neurons only contribute to GABA
        
        avg_pRelAMPA = np.mean(self.__pRelAMPA,axis=0)
        avg_pRelNMDA = np.mean(self.__pRelNMDA,axis=0)
        avg_pRelGABA = np.mean(self.__pRelGABA,axis=0)
        
        avg_pRel_stfAMPA = np.mean(self.__pRel_stfAMPA,axis=0)
        avg_pRel_stfNMDA = np.mean(self.__pRel_stfNMDA,axis=0)
        avg_pRel_stfGABA = np.mean(self.__pRel_stfGABA,axis=0)
        
        return avg_pRelAMPA,avg_pRelNMDA,avg_pRelGABA,avg_pRel_stfAMPA,avg_pRel_stfNMDA,avg_pRel_stfGABA
    
    def plotAveragePrel(self, avg_pRelAMPA, avg_pRelNMDA, avg_pRelGABA,plastic_type):
        #public method: plot the probability of release separating between AMPA, NMDA, GABA given the 3 input arrays 
        #plastic_type: D/F
        fig=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.plot(self.__ti/1000,avg_pRelNMDA,color=self.__NMDAColor,linewidth=6)
        plt.plot(self.__ti/1000,avg_pRelAMPA,color=self.__AMPAColor,linewidth=6)
        plt.plot(self.__ti/1000,avg_pRelGABA,color=self.__GABAColor,linewidth=6)
        plt.xlabel("Time(s)",fontsize=40)
        plt.ylabel("Prel",fontsize=40)
        xticks = np.arange(len(self.__ti)*self.__tol/1000+1,step=2)
        plt.xticks(xticks,fontsize=40)
        plt.yticks(fontsize=40)
        legend_elements = [Line2D([0],[0],color=self.__NMDAColor,lw=4,label="NMDA"),
                           Line2D([0],[0],color=self.__AMPAColor,lw=4,label="AMPA"),
                           Line2D([0],[0],color=self.__GABAColor,lw=4,label="GABA")]
        
        ax.legend(handles=legend_elements,bbox_to_anchor=(0.98,1.2),loc="upper right",fontsize=30,ncol=3)
        last_second = len(self.__ti)*self.__tol/1000
        plt.xlim(0, last_second)
        plt.savefig(self.__foldername+"/Prel"+plastic_type+'.png')
        
        return fig
    
    def getSynapticConductances(self):
        #public method: returns 3-(1xlen(time)) arrays, each one containing the synaptic conductance of the network for AMPA, NMDA, GABA neurotransmitters
        
        #Note: 
        #Excitatory neurons only contribute to AMPA, NMDA
        #Inhibitory neurons only contribute to GABA
        
        gAMPA = (self.__gEE_AMPA*(1-self.__ExcInh)+self.__gEI_AMPA*self.__ExcInh).reshape((self.__nNeurons,1))*self.__sAMPA*self.__pRelAMPA*self.__pRel_stfAMPA

        gNMDA = (self.__gEE_NMDA*(1-self.__ExcInh)+self.__gEI_NMDA*self.__ExcInh).reshape((self.__nNeurons,1))*self.__sNMDA*self.__pRelNMDA*self.__pRel_stfNMDA

        gGABA = (self.__gIE_GABA*(1-self.__ExcInh)+self.__gII_GABA*self.__ExcInh).reshape((self.__nNeurons,1))*self.__sGABA*self.__pRelGABA*self.__pRel_stfGABA
        
        
        return gAMPA, gNMDA, gGABA
    
    
    def plotConductances(self, gAMPA, gNMDA, gGABA):
        #public method: plot the probability of release separating between AMPA, NMDA, GABA given the 3 input arrays 
        fig=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.plot(self.__ti/1000,np.mean(gNMDA,axis=0)*10**9,color=self.__NMDAColor,linewidth=6)
        plt.plot(self.__ti/1000,np.mean(gAMPA,axis=0)*10**9,color=self.__AMPAColor,linewidth=6)
        plt.plot(self.__ti/1000,np.mean(gGABA,axis=0)*10**9,color=self.__GABAColor,linewidth=6)
        plt.xlabel("Time(ms)",fontsize=40)
        plt.ylabel("Conductance (nS)",fontsize=40)
        xticks = np.arange(len(self.__ti)*self.__tol/1000+1,step=2)
        plt.xticks(xticks,fontsize=40)
        plt.yticks(fontsize=40)
        legend_elements = [Line2D([0],[0],color=self.__NMDAColor,lw=4,label="NMDA"),
                           Line2D([0],[0],color=self.__AMPAColor,lw=4,label="AMPA"),
                           Line2D([0],[0],color=self.__GABAColor,lw=4,label="GABA")]
        
        ax.legend(handles=legend_elements,bbox_to_anchor=(0.98,0.98),loc="upper right",fontsize=30,ncol=3)
        plt.xlim(0, 20)
        plt.yscale("log")
        plt.savefig(self.__foldername+'Conductances.png')
        
        return fig
    
    def getEIPools(self):
        #public method: returns arrays containing voltage of excitatory and inhibitory neurons
        #np.savez('EIPools_'+self.__foldername+'.npz',ExcPool = self.__Exc_neurons, InhPool = self.__Inh_neurons)
        
        return self.__Exc_neurons, self.__Inh_neurons
        
    
    def RawPeriodogramFFT(self,neuronExc, neuronInh):
        #public method: plot the PowerSpectralDensity of an excitatory and an inhibitory neuron using RawPeriodogram
        #In both the power (Y-axis) is shown in dB and normalized, the frequency (X-axis) is shown in logarithmic scale and cutted at 350Hz
        
        
        #neuronExc: 1xlen(self.__ti) array containing the voltage of an excitatory neuron
        #neuronInh: 1xlen(self.__ti) array containing the voltage of an inhibitory neuron
        
        #Some properties
        N=len(neuronExc) #length of the signal
        delta=self.__tol/10**3 #time discretization we divide by 10^3 because our units are in ms
        T = N*delta# 20000 total time duration of the signal
        df = 1/T #Frecuency resolution
        samFrec = 1/delta #Sampling frecuency
        nyqFrec = 1/(2*delta) #Nyquist frecuency
        
        print('Number of samples: ',N, ' [samples]')
        print('Total time of the recording',T, '[s]')
        print('Sampling rate: ',delta, '[s]')
        print('Frecuency resolution', df, '[Hz]')
        print('Sampling frecuency',samFrec, '[Hz]')
        print('Nyquist frecuency', nyqFrec, '[Hz]')
        
        #Number of points of the FFT
        Nfft = 2**(int(np.ceil(np.log2(N))))
        #Frequency[Hz] at which we cut the spectrum
        freqcut = 350 
        
        #Compute the frequency axis
        frequency_axis = fftfreq(Nfft, d=delta)
        frequency_axis = frequency_axis[0:int(freqcut/df)]
        
        #Compute the FFT for real-valued functions detrending the data
        xf_exc = rfft(neuronExc-np.mean(neuronExc),n=Nfft)
        xf_exc = xf_exc[0:int(freqcut/df)]
        Sexc = 10*np.log10(delta*(2/Nfft)*(abs(xf_exc)**2)/max(delta*(2/Nfft)*(abs(xf_exc)**2)))
        Sexc_filtered = medfilt(Sexc,7)
        #Sexc_filtered = 10*np.log10(filtered_exc/max((2/Nfft)*abs(xf_exc)))
        
        xf_inh = rfft(neuronInh-np.mean(neuronInh),n=Nfft)
        xf_inh = xf_inh[0:int(freqcut/df)]
        Sinh = 10*np.log10(delta*(2/Nfft)*(abs(xf_inh)**2)/max(delta*(2/Nfft)*(abs(xf_inh)**2)))
        Sinh_filtered = medfilt(Sinh,7)
        #Sinh_filtered = 10*np.log10(filtered_inh/max((2/Nfft)*abs(xf_inh)))
        
                
        fig1=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.plot(frequency_axis,Sexc,color=self.__ExcColor,linewidth=6,alpha=0.5)
        plt.plot(frequency_axis,Sexc_filtered,color=self.__ExcFFT,linewidth=4)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.xscale("log")
        plt.ylabel("PSD (dB)",fontsize=40)
        plt.ylim(-80,5)
        plt.xlim(0,270)
        plt.xlabel("Frequency (Hz)",fontsize=40)
        plt.savefig(self.__foldername+'/RawPeriodogramExcitatory.png')
        plt.close()
        
        
        fig2=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.plot(frequency_axis,Sinh,color=self.__InhColor,linewidth=6,alpha=0.5)
        plt.plot(frequency_axis,Sinh_filtered,color=self.__InhFFT,linewidth=4)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.xscale("log")
        plt.ylabel("PSD (dB)",fontsize=40)
        plt.ylim(-80,5)
        plt.xlim(0,270)
        plt.xlabel("Frequency (Hz)",fontsize=40)
        plt.savefig(self.__foldername+'/RawPeriodogramInhibitory.png')
        plt.close()
        
        return fig1,fig2
        
    
 
    def TapperedHanningPeriodogramFFT(self,neuronExc, neuronInh):
        #public method: plot the frequency spectrum of an excitatory and an inhibitory neuron
        #In both the power (Y-axis) is shown in dB and normalized, the frequency (X-axis) is shown in logarithmic scale and cutted at 350Hz
        #A Hanning taper is applied to reduce the side lobes' frequency influence due to having a cut signal in time
        
        #neuronExc: 1xlen(self.__ti) array containing the voltage of an excitatory neuron
        #neuronInh: 1xlen(self.__ti) array containing the voltage of an inhibitory neuron
        
        #Some properties
        N=len(neuronExc) #length of the signal
        delta=self.__tol/10**3 #time discretization dividim entre 10^3 perquè les nostres unitats estan en ms
        T = N*delta# 20000 total time duration of the signal
        df = 1/T #Frecuency resolution
        samFrec = 1/delta #Sampling frecuency
        nyqFrec = 1/(2*delta) #Nyquist frecuency
        
        print('Number of samples: ',N, ' [samples]')
        print('Total time of the recording',T, '[s]')
        print('Sampling rate: ',delta, '[s]')
        print('Frecuency resolution', df, '[Hz]')
        print('Sampling frecuency',samFrec, '[Hz]')
        print('Nyquist frecuency', nyqFrec, '[Hz]')
        
        #Number of points of the FFT
        Nfft = 2**(int(np.ceil(np.log2(N))))
        #Frequency[Hz] at which we cut the spectrum
        freqcut = 350 
        
        #Compute the frequency axis
        frequency_axis = fftfreq(Nfft, d=delta)
        frequency_axis = frequency_axis[0:int(freqcut/df)]
        
        #Apply a hanning tapper to the neuron voltage
        tappered_exc = signal.windows.hann(len(neuronExc-np.mean(neuronExc)))*(neuronExc-np.mean(neuronExc))
        tappered_inh = signal.windows.hann(len(neuronInh-np.mean(neuronInh)))*(neuronInh-np.mean(neuronInh))
        
        #Compute the FFT for real-valued functions and apply median filter
        xf_exc = rfft(tappered_exc,n=Nfft)
        xf_exc = xf_exc[0:int(freqcut/df)]
        scaling_window_exc = np.sum(signal.windows.hann(len(neuronExc-np.mean(neuronExc))))
        #Sexc = 10*np.log10(delta*(2/Nfft)*(abs(xf_exc)**2)/max(delta*(2/Nfft)*(abs(xf_exc)**2)))
        Sexc = 10*np.log10(delta*(2/scaling_window_exc)*(abs(xf_exc)**2)/max(delta*(2/scaling_window_exc)*(abs(xf_exc)**2)))
        Sexc_filtered = medfilt(Sexc,7)
        #Sexc_filtered = 10*np.log10(filtered_exc/max((2/Nfft)*abs(xf_exc)))
        
        xf_inh = rfft(tappered_inh,n=Nfft)
        xf_inh = xf_inh[0:int(freqcut/df)]
        scaling_window_inh = np.sum(signal.windows.hann(len(neuronInh-np.mean(neuronInh))))
        #Sinh = 10*np.log10(delta*(2/Nfft)*(abs(xf_inh)**2)/max(delta*(2/Nfft)*(abs(xf_inh)**2)))
        Sinh = 10*np.log10(delta*(2/scaling_window_inh)*(abs(xf_inh)**2)/max(delta*(2/scaling_window_inh)*(abs(xf_inh)**2)))
        Sinh_filtered = medfilt(Sinh,7)
        #Sinh_filtered = 10*np.log10(filtered_inh/max((2/Nfft)*abs(xf_inh)))
        
                
        fig1=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.plot(frequency_axis,Sexc,color=self.__ExcColor,linewidth=6,alpha=0.5)
        plt.plot(frequency_axis,Sexc_filtered,color=self.__ExcFFT,linewidth=4)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.xscale("log")
        plt.ylabel("PSD (dB)",fontsize=40)
        plt.ylim(-80,5)
        plt.xlim(0,270)
        plt.xlabel("Frequency (Hz)",fontsize=40)
        plt.savefig(self.__foldername+'/HanningTapperedPeriodogramExcitatory.png')
        plt.close()
        
        
        fig2=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.plot(frequency_axis,Sinh,color=self.__InhColor,linewidth=6,alpha=0.5)
        plt.plot(frequency_axis,Sinh_filtered,color=self.__InhFFT,linewidth=4)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.xscale("log")
        plt.ylabel("PSD (dB)",fontsize=40)
        plt.ylim(-80,5)
        plt.xlim(0,270)
        plt.xlabel("Frequency (Hz)",fontsize=40)
        plt.savefig(self.__foldername+'/HanningTapperedPeriodogramInhibitory.png')
        plt.close()
        
        
        return fig1, fig2
    
        
    def SpectrumMultitaper(self,neuronExc,neuronInh):
        #public method: Computes an estimated of the power density spectrum by using the multitaper method
        
        #neuronExc: 1xlen(self.__ti) array containing the voltage of an excitatory neuron
        #neuronInh: 1xlen(self.__ti) array containing the voltage of an inhibitory neuron
        
        
        #Some properties
        N=len(neuronExc) #length of the signal
        delta=self.__tol/10**3 #time discretization dividim entre 10^3 perquè les nostres unitats estan en ms
        T = N*delta# 20000 total time duration of the signal
        df = 1/T #Frecuency resolution
        samFrec = 1/delta #Sampling frecuency
        nyqFrec = 1/(2*delta) #Nyquist frecuency
        
        print('Number of samples: ',N, ' [samples]')
        print('Total time of the recording',T, '[s]')
        print('Sampling rate: ',delta, '[s]')
        print('Frecuency resolution', df, '[Hz]')
        print('Sampling frecuency',samFrec, '[Hz]')
        print('Nyquist frecuency', nyqFrec, '[Hz]')
        
        #Frequency[Hz] at which we cut the spectrum
        freqcut = 350 
        
        #Compute the frequency axis
        frequency_axis = fftfreq(N, d=delta)
        frequency_axis = frequency_axis[0:int(freqcut/df)]
        
        #Set the frequency resolution we are going to tolerate
        band_resolution = 0.5 #2W
        
        #Define the normalized time-bandwidth product TxW
        NW = T*(band_resolution/2)
        
        #Compute the multitaper spectrum
        #Excitatory neurons
        faxis_exc, Sexc, _ = spectrum.multi_taper_psd(neuronExc-np.mean(neuronExc), Fs=self.__tol, NW=NW)
        Sexc = Sexc[0:len(frequency_axis)]
        #Inhibitory neurons
        faxis_inh, Sinh, _ = spectrum.multi_taper_psd(neuronInh-np.mean(neuronInh), Fs=self.__tol, NW=NW)
        Sinh = Sinh[0:len(frequency_axis)]
        
        #Define the confidence interval
        #Number of tapers used
        K = 2*NW-1
        #Confidence level ([0.5, 1])
        ci = 0.95 
        #UpperLower bounds for confidence level
        #Excitatory neurons
        ub_exc = 2 * K / chi2.ppf(1 - ci, 2 * K) * Sexc
        lb_exc = 2 * K / chi2.ppf(    ci, 2 * K) * Sexc
        #Inhibitory neurons
        ub_inh = 2 * K / chi2.ppf(1 - ci, 2 * K) * Sinh
        lb_inh = 2 * K / chi2.ppf(    ci, 2 * K) * Sinh
        
        #Plots
        fig1=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.plot(frequency_axis,10*np.log10(Sexc/max(Sexc)),color=self.__ExcFFT,linewidth=4)
        plt.fill_between(frequency_axis, 10*np.log10(lb_exc/max(Sexc)), 10*np.log10(ub_exc/max(Sexc)),color=self.__ExcColor,linewidth=6,alpha=0.5)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.xscale("log")
        plt.ylabel("Power (dB)",fontsize=40)
        plt.ylim(-80,5)
        plt.xlim(0,270)
        plt.xlabel("Frequency (Hz)",fontsize=40)
        plt.savefig(self.__foldername+'/MultitapperExcitatory.png')
        plt.close()
        
        fig2=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.plot(frequency_axis,10*np.log10(Sinh/max(Sinh)),color=self.__InhFFT,linewidth=4)
        plt.fill_between(frequency_axis, 10*np.log10(lb_inh/max(Sinh)), 10*np.log10(ub_inh/max(Sinh)),color=self.__InhColor,linewidth=6,alpha=0.5)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.xscale("log")
        plt.ylabel("Power (dB)",fontsize=40)
        plt.ylim(-80,5)
        plt.xlim(0,270)
        plt.xlabel("Frequency (Hz)",fontsize=40)
        plt.savefig(self.__foldername+'/MultitapperInhibitory.png')
        plt.close()
        
        return fig1, fig2
    
    def NeuronSpectrogram(self,neuronExc,neuronInh):
        
        #Some properties
        N=len(neuronExc) #length of the signal
        delta=self.__tol/10**3 #time discretization dividim entre 10^3 perquè les nostres unitats estan en ms
        T = N*delta# 20000 total time duration of the signal
        df = 1/T #Frecuency resolution
        samFrec = 1/delta #Sampling frecuency
        nyqFrec = 1/(2*delta) #Nyquist frecuency
        
        print('Number of samples: ',N, ' [samples]')
        print('Total time of the recording',T, '[s]')
        print('Sampling rate: ',delta, '[s]')
        print('Frecuency resolution', df, '[Hz]')
        print('Sampling frecuency',samFrec, '[Hz]')
        print('Nyquist frecuency', nyqFrec, '[Hz]')
        
        #Frequency[Hz] at which we cut the spectrum
        freqcut = 350 
        
        fe, te, Sxx_e = signal.spectrogram(neuronExc, fs = samFrec,window='hann',nperseg=2000,noverlap=500,scaling='density')
        Sxx_e[np.where(Sxx_e<10**(-61))] = 10**(-6)
        minvalue_e=Sxx_e.min()
        print('minvalue_e: ',minvalue_e)
        maxvalue_e=Sxx_e.max()
        print('maxvalue_e: ',maxvalue_e)

        fi, ti, Sxx_i = signal.spectrogram(neuronInh, fs = samFrec,window='hann',nperseg=2000,noverlap=500,scaling='density')
        Sxx_i[np.where(Sxx_i<10**(-61))] = 10**(-6)
        minvalue_i=Sxx_i.min()
        print('minvalue_i: ',minvalue_i)
        maxvalue_i=Sxx_i.max()
        print('maxvalue_i: ',maxvalue_i)
        
        
        PSD_e = 10*np.log10(Sxx_e/Sxx_e.max())
        #PSD_e[np.where(PSD_e<10**(-8))] = 10**(-8)
        PSD_e[np.where(PSD_e<-70)] = -70
        
        
        PSD_i = 10*np.log10(Sxx_i/Sxx_i.max())
        #PSD_i[np.where(PSD_i<10**(-8))] = 10**(-8)
        PSD_i[np.where(PSD_i<-70)] = -70

        fig1=plt.figure(figsize=(20,8))
        ax=plt.axes()
        #plt.pcolormesh(te, fe, Sxx_e/Sxx_e.max(),norm=LogNorm(10**(-6),min(maxvalue_e,maxvalue_i)),cmap='inferno')
        plt.pcolormesh(te, fe, PSD_e,cmap='inferno')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        #plt.yscale("log")
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.xlabel('Time (s)',fontsize=40)
        plt.ylabel('Frequency (Hz)',fontsize=40)
        plt.ylim(0,200)
        plt.savefig(self.__foldername+'/SpectrogramExcitatory.png')
        #plt.show()

        fig2=plt.figure(figsize=(20,8))
        ax=plt.axes()
        #plt.pcolormesh(ti, fi, Sxx_i/Sxx_i.max(),norm=LogNorm(10**(-6),min(maxvalue_e,maxvalue_i)),cmap='inferno')
        plt.pcolormesh(ti, fi, PSD_i,cmap='inferno')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        #plt.yscale("log")
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.xlabel('Time (s)',fontsize=40)
        plt.ylabel('Frequency (Hz)',fontsize=40)
        plt.ylim(0,200)
        plt.savefig(self.__foldername+'/SpectrogramInhibitory.png')
        #plt.show()

        
        return fig1,fig2
    
    
    def SpectrogramAllNeurons(self):
        #public method: Compute the spectrogram of all neurons in the network using the multitapering method
        
        #Some properties
        tol = self.__tol
        N=len(self.__voltage[0,:]) #length of the signal
        delta=tol/10**3 #time discretization dividim entre 10^3 perquè les nostres unitats estan en ms
        T = N*delta# 20000 total time duration of the signal
        df = 1/T #Frecuency resolution
        samFrec = 1/delta #Sampling frecuency
        nyqFrec = 1/(2*delta) #Nyquist frecuency

        print('Number of samples: ',N, ' [samples]')
        print('Total time of the recording',T, '[s]')
        print('Sampling rate: ',delta, '[s]')
        print('Frecuency resolution', df, '[Hz]')
        print('Sampling frecuency',samFrec, '[Hz]')
        print('Nyquist frecuency', nyqFrec, '[Hz]')
        
        #Frequency[Hz] at which we cut the spectrum
        freqcut = 350 

        #Compute the frequency axis
        frequency_axis = fftfreq(N, d=delta)
        frequency_axis = frequency_axis[0:int(freqcut/df)]

        #Set the frequency resolution we are going to tolerate
        band_resolution = 0.5 #2W

        #Define the normalized time-bandwidth product TxW
        NW = T*(band_resolution/2)
        
        #Initialize arrays to contain spectrum of excitatory and inhibitory neurons
        #Number of neurons
        #Build arrays to save data
        Stotal = np.zeros((self.__nNeurons,len(frequency_axis)))

        #Compute the multitaper spectrum
        for neuron in range(self.__nNeurons):
            print('iter', neuron)
            #Excitatory neurons
            faxis_exc, S_iter, _ = spectrum.multi_taper_psd(self.__voltage[neuron,:]-np.mean(self.__voltage[neuron,:]), Fs=tol, NW=NW)
            Stotal[neuron,:] = S_iter[0:len(frequency_axis)]
            
        np.savez(self.__foldername+'/SpectrumTableTotal.npz',SpectrumTable=Stotal)
        
        for neuron in range(self.__nNeurons):
            Stotal[neuron,:] = 10*np.log10(Stotal[neuron,:]/max(Stotal[neuron,:]))
        
        Neurons = np.array(range(self.__nNeurons))
        logfrequency_axis=np.log10(frequency_axis)
        logfrequency_axis[0] = 0
        
        fig=plt.figure(figsize=(20,8))
        ax=plt.axes()
        plt.pcolormesh(logfrequency_axis,Neurons,Stotal,cmap='inferno',vmin=-50,vmax=0)
        yticks = np.arange(self.__nNeurons,step=50)
        plt.xticks([])
        plt.yticks(yticks,fontsize=40)
        plt.ylabel("# Neuron",fontsize=40)
        plt.xlabel("Frequency (Hz)",fontsize=40)
        cbar = plt.colorbar()
        tick_font_size = 30
        cbar.ax.tick_params(labelsize=tick_font_size)
        plt.savefig(self.__foldername+'/NetworkDFTLogScale.png')
        
        return fig