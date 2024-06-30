def NetworkSTDall(t0,tf,h,Dnumber,Fnumber):
    #==========================================================MAIN PROGRAM ALL FUNCTIONS TOGETHER=============================================================
    #delete the objects/variables specified
    #del variable_name

    #printing the directory varialbes
    #d=dir()
    #print(d)

    #printing all global variables
    #g = globals()
    #print(g)

    #0. Imports
    import time
    import numpy as np
    import scipy as sc
    import matplotlib.pyplot as plt
    import numba as nb
    from NetworkField import NetworkField
    from rk45Network import rk45Network
    from Tools import SDNumber,initialize_randomparameters,initialize_randomvoltage,save_files


    print('--------------------------  STD - NETWORK  ---------------------------')


    #1. load BuildNetwork file which contains: 
        # P: 320X320 table where (i,j) = 0 if neurons i and j are connected; (i,j) = 0 otherwise /// i: presyn_neuron, j: postsyn_neuron
        # ExcInh: 1x320 vector where i = 1 identifies inhibitory neurons, i = 0 identifies excitatory neurons

    BN = sc.io.loadmat('BuildNetwork320.mat')
    P = BN['P']
    ExcInh = BN['ExcInh']

    #print shape of read data
    print('----------------------------- Data loaded  ---------------------------')
    P = nb.types.int64(P)
    print('P shape: ', P.shape)
    print('ExcInh shape before reshape: ', ExcInh.shape)
    ExcInh = ExcInh.reshape((ExcInh.size,))
    ExcInh = nb.types.int64(ExcInh)
    print('ExcInh shape after reshape: ', ExcInh.shape)


    #2. Define the simulation parameters
        #Dnumber = 1: means no depression
        #Fnumber = 0: means no facilitation
        #fD_AMPA: depression factor in the AMPA neurotransmiters <--> Dnumber
        #fD_NMDA: depression factor in the NMDA neurotransmiters <--> Dnumber
        #fD_GABA: depression factor in the GABA neurotransmiters <--> Dnumber
        #fF_AMPA: facilitation factor in the AMPA neurotransmiters <--> Fnumber
        #fF_NMDA: facilitation factor in the NMDA neurotransmiters <--> Fnumber
        #fF_GABA: facilitation factor in the GABA neurotransmiters <--> Fnumber
        #t0: initial time of the simulation
        #tf: final time of the simulation
        #h: Runge-kutta step 
        #N: total number of iterations in the RK-45 
        #nNeurons: number of neurons in the network
        #neq: number of equations for each neuron in the network

    fD_AMPA = fD_NMDA = fD_GABA = Dnumber
    fF_AMPA = fF_NMDA = fF_GABA = Fnumber
    N = int((tf-t0)/h)
    nNeurons = ExcInh.size
    neq = 20
    nvar = nNeurons*neq

    #print simulation parameters
    print('----------------------------- Simulation parameters  ---------------------------')
    print('Depression factor, fD: ', Dnumber)
    print('Facilitation factor, fF: ', Fnumber)
    print('Initial time, t0: ', t0)
    print('Final time, tf: ', tf)
    print('Runge-kutta step, h: ', h)
    print('Number of rk45 iterations, N: ', N)
    print('Number of neurons in the network, Nneurons: ', nNeurons)
    print('Number of equations for each neuron, neq: ', neq)
    print('Number of total variables in the network, nvar = Nneuronsxneq: ',nvar)

    #3. Define the vector of initial conditions
    x0 = np.zeros(nvar)
    excvs_positions = 1+np.arange(nNeurons)*neq #these vectors have length 320 (already checked)
    excvd_positions = 2+np.arange(nNeurons)*neq
    inhvs_positions = 13+np.arange(nNeurons)*neq

    x0 = initialize_randomvoltage(ExcInh,x0,excvs_positions,excvd_positions,inhvs_positions)

    #4. Define the vectors for random parameters of the neurons
    SDvL = SDNumber(nNeurons)
    SDgL = SDNumber(nNeurons)
    SDgsd = SDNumber(nNeurons)

    randomvL, randomgL, randomgsd = initialize_randomparameters(ExcInh,SDvL, SDgL, SDgsd)


    #5. Execute rk45 to solve NetworkField
    start = time.time()
    [ti, wi, pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA] = rk45Network(NetworkField, t0, tf, x0, N, h, neq, nNeurons, nvar, P, ExcInh, fD_AMPA, fD_NMDA, fD_GABA, fF_AMPA, fF_NMDA, fF_GABA, randomvL, randomgL, randomgsd)
    fin = time.time()
    exec_time = fin-start


    #6. Save the outputs in file: ti, wi, pRelsX, pRels_stfX where X=AMPA, NMDA, GABA
    np.savez('fD'+str(Dnumber)+'_fF'+str(Fnumber)+'.npz', exec_time = exec_time,ti=ti, wi=wi, pRelAMPA=pRelAMPA, pRelNMDA=pRelNMDA, pRelGABA=pRelGABA, pRel_stfAMPA=pRel_stfAMPA, pRel_stfNMDA=pRel_stfNMDA, pRel_stfGABA=pRel_stfGABA)