function [t_final, ti, wi, pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA] = NetworkSTDall(Dnumber,Fnumber,t0,tf,h,p0_stf)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%  MAIN PROGRAM  ALL FUNCTIONS TOGETHER %%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_init = tic;
    disp('---------------------------------  STD - Network  ----------------------------------');
    
    %1. load BuildNetwork file which contains: 
        % P: 320X320 table where (i,j) = 0 if neurons i and j are connected; (i,j) = 0 otherwise /// i: presyn_neuron, j: postsyn_neuron
        % ExcInh: 1x320 vector where i = 1 identifies inhibitory neurons, i = 0 identifies excitatory neurons
        
    disp('Reading P');
    BN=load('BuildNetwork320.mat');
    P=BN.P;
    disp('Reading EI');
    ExcInh=BN.ExcInh;
    %print shape of read data
    disp('----------------------------- Data loaded  ---------------------------')
    disp(['P shape:', mat2str(size(P))]);
    disp(['ExcInh shape:', mat2str(size(ExcInh))]); 
           
    %2. Define the simulation parameters
        %Dnumber = 1: means no depression
        %Fnumber = 0: means no facilitation
        %fD_AMPA: depression factor in the AMPA neurotransmiters <--> Dnumber
        %fD_NMDA: depression factor in the NMDA neurotransmiters <--> Dnumber
        %fD_GABA: depression factor in the GABA neurotransmiters <--> Dnumber
        %fF_AMPA: facilitation factor in the AMPA neurotransmiters <--> Fnumber
        %fF_NMDA: facilitation factor in the NMDA neurotransmiters <--> Fnumber
        %fF_GABA: facilitation factor in the GABA neurotransmiters <--> Fnumber
        %t0: initial time of the simulation
        %tf: final time of the simulation
        %h: Runge-kutta step 
        %N: total number of iterations in the RK-45 
        %nNeurons: number of neurons in the network
        %neq: number of equations for each neuron in the network
    
    
    %Dnumber = 0.97;
    fD_AMPA = Dnumber;
    fD_NMDA = Dnumber;
    fD_GABA = Dnumber;
    
    %Fnumber = 1.00;
    fF_AMPA = Fnumber;
    fF_NMDA = Fnumber;
    fF_GABA = Fnumber;
    
    N = floor((tf-t0)/h);
    sizenNeurons = size(ExcInh);
    nNeurons = sizenNeurons(2);
    neq = 20;
    nvar = nNeurons*neq;
    
    %---------------------------  Define file name  ---------------------------
    filename = ['fD_', num2str(Dnumber) 'fF_' num2str(Fnumber) '.mat'];
    %print simulation parameters
    disp('----------------------------- Simulation parameters  ---------------------------')
    disp(['Depression factor, fD: ', mat2str(Dnumber)]);
    disp(['Facilitation factor, fF: ', mat2str(Fnumber)]);
    disp(['Initial time, t0: ', mat2str(t0)]);
    disp(['Final time, tf: ', mat2str(tf)]);
    disp(['Runge-kutta step, h: ', mat2str(h)]);
    disp(['Number of rk45 iterations, N: ', mat2str(N)]);
    disp(['Number of neurons in the network, Nneurons: ', mat2str(nNeurons)]);
    disp(['Number of equations for each neuron, neq: ', mat2str(neq)]);
    disp(['Number of total variables in the network, nvar = Nneuronsxneq: ',mat2str(nvar)]);
    
    %3. Define the vector of initial conditions
    x0 = zeros(1,nvar);
    excvs_positions = 1+(0:(nNeurons-1))*neq; %these vectors have length 320 (already checked)
    excvd_positions = 2+(0:(nNeurons-1))*neq;
    inhvs_positions = 13+(0:(nNeurons-1))*neq;
    
    excvs_random = normrnd(0,1,[1,length(excvs_positions)]);    
    excvd_random = normrnd(0,1,[1,length(excvd_positions)]);   
    inhvs_random = normrnd(0,1,[1,length(inhvs_positions)]);
    
    x0(excvs_positions) = (-60+5*excvs_random).*(1-ExcInh);
    x0(excvd_positions) = (-60+5*excvd_random).*(1-ExcInh);
    x0(inhvs_positions) = (-60-5*inhvs_random).*ExcInh;
    
    SDvL = SDNumber(nNeurons);
    SDgL = SDNumber(nNeurons);
    SDgsd = SDNumber(nNeurons);
    
    randomvL = (-60.95 + 0.3*SDvL).*(1-ExcInh)+(-63.8 + 0.15*SDvL).*ExcInh;
    randomgL = (0.0667 + 0.0067*SDgL).*(1-ExcInh)+(0.1025 + 0.0025*SDgL).*ExcInh;
    randomgsd = ((1.75 + 0.1*SDgsd)*0.1).*(1-ExcInh);
    
    [ti, wi, pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA] = rk45Network('NetworkField', t0, tf, x0, N, h, neq, nNeurons, nvar, ExcInh, P, fD_AMPA, fD_NMDA, fD_GABA, fF_AMPA, fF_NMDA, fF_GABA, randomvL, randomgL, randomgsd,p0_stf);
    t_final = toc(t_init);
    %data_w = load('w_matrix.mat','wi_matrix');
    %save(filename,"ti","wi", "pRelAMPA", "pRelNMDA", "pRelGABA", "pRel_stfAMPA", "pRel_stfNMDA", "pRel_stfGABA");
end