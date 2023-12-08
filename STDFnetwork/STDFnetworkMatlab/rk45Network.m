function [ti, wi, pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA] = rk45Network(RHS, t0, tf, x0, N, h, neq, nNeurons, nvar, ExcInh, P, fD_AMPA, fD_NMDA, fD_GABA, fF_AMPA, fF_NMDA, fF_GABA, randomvL, randomgL, randomgsd)
    %RHS: differential equations system to solve with rk45 method
    %t0: initial time of the simulation
    %tf: final time of the simulation
    %h: Runge-kutta step 
    %N: total number of iterations in the RK-45 
    %neq: number of equations for each neuron in the network
    %P: 320X320 table where (i,j) = 0 if neurons i and j are connected; (i,j) = 0 otherwise /// i: presyn_neuron, j: postsyn_neuron
    %ExcInh: 1x320 vector where i = 1 identifies inhibitory neurons, i = 0 identifies excitatory neurons
    %fD_X: Depression factor for the specific neurotransmitter X = AMPA, NMDA, GABA
    %fF_X: Facilitation factor for the specific neurotransmitter X = AMPA, NMDA, GABA

    %------------------   Open files to save data at each iteration  ----------------------
    %file_w_matrix
    filename_w_matrix = 'w_matrix.mat';
    mfw_w_matrix = dsp.MatFileWriter(filename_w_matrix,'VariableName','wi_matrix');

    %file_PrelAMPA_matrix
    filename_prelAMPA_matrix = 'prelAMPA_matrix.mat';
    mfw_prelAMPA_matrix = dsp.MatFileWriter(filename_prelAMPA_matrix,'VariableName','prelAMPA_matrix');

    %file_PrelNMDA_matrix
    filename_prelNMDA_matrix = 'prelNMDA_matrix.mat';
    mfw_prelNMDA_matrix = dsp.MatFileWriter(filename_prelNMDA_matrix,'VariableName','prelNMDA_matrix');

    %file_PrelGABA_matrix
    filename_prelGABA_matrix = 'prelGABA_matrix.mat';
    mfw_prelGABA_matrix = dsp.MatFileWriter(filename_prelGABA_matrix,'VariableName','prelGABA_matrix');

    %file_pRel_stfAMPA_matrix
    filename_pRel_stfAMPA_matrix = 'pRel_stfAMPA_matrix.mat';
    mfw_pRel_stfAMPA_matrix = dsp.MatFileWriter(filename_pRel_stfAMPA_matrix,'VariableName','pRel_stfAMPA_matrix');

    %file_pRel_stfNMDA_matrix
    filename_pRel_stfNMDA_matrix = 'pRel_stfNMDA_matrix.mat';
    mfw_pRel_stfNMDA_matrix = dsp.MatFileWriter(filename_pRel_stfNMDA_matrix,'VariableName','pRel_stfNMDA_matrix');

    %file_pRel_stfGABA_matrix
    filename_pRel_stfGABA_matrix = 'pRel_stfGABA_matrix.mat';
    mfw_pRel_stfGABA_matrix = dsp.MatFileWriter(filename_pRel_stfGABA_matrix,'VariableName','pRel_stfGABA_matrix');
    
    disp("---------------------------Inside rk45: -------------------------");
    %------------------------------------ Initial values to rk45 Outputs  -------------------------
    
    %initialize time vector (ti), system solution vector (wi), all probability of Release vectors (pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA)
    ti=zeros(1,N);
    wi = zeros(1,nvar);

    %save initial conditions in the vectors
    ti(1) = t0;
    wi(1,1:nvar) = x0;

    %Prel vectors we will need during all the simulation --> they are initialized at all ones
    %For depression
    pRelAMPA = ones(1,nNeurons);
    pRelNMDA = ones(1,nNeurons);
    pRelGABA = ones(1,nNeurons);
    %For facilitation
    pRel_stfAMPA = ones(1,nNeurons);
    pRel_stfNMDA = ones(1,nNeurons);
    pRel_stfGABA = ones(1,nNeurons);


    %indexes of x0 where the variables AMPA, NMDA and GABA for pyramidal neurons are located
    indexAMPA = 9+(0:(nNeurons-1))*neq; %vectors of length 320
    indexNMDA = 10+(0:(nNeurons-1))*neq;
    indexGABA = 12+(0:(nNeurons-1))*neq;
        
                
    %indexes of x0 where the variables AMPA, NMDA and GABA for interneurons are located
    indexsynAMPA = 16+(0:(nNeurons-1))*neq; %vectors of length 320
    indexsynNMDA = 17+(0:(nNeurons-1))*neq;
    indexsynGABA = 19+(0:(nNeurons-1))*neq;
    
    %Define index of x0 where the voltage of the Pyramidal neurons and Interneurons is located
    indexvspyramneuron = 1+(0:(nNeurons-1))*neq;
    indexvinterneuron = 13+(0:(nNeurons-1))*neq;


    %save initial conditions in the files
    mfw_w_matrix(wi);
    mfw_prelAMPA_matrix(pRelAMPA);
    mfw_prelNMDA_matrix(pRelNMDA);
    mfw_prelGABA_matrix(pRelGABA);    
    mfw_pRel_stfAMPA_matrix(pRel_stfAMPA);
    mfw_pRel_stfNMDA_matrix(pRel_stfNMDA);
    mfw_pRel_stfGABA_matrix(pRel_stfGABA);
    %---------------------------------  Loop to integrate the system ----------------------------------

    i = 2;
    
    while(t0+h < tf)
        if mod(i,100) == 0
            disp(['Actual i: ', mat2str(i)]);
        end
        %-------------------------  RK45-Field integrator -----------------------------
        
        pre = x0;
        k1 = h * feval(RHS, t0, x0, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA);
        k2 = h * feval(RHS, t0 + h/2, x0 + k1/2, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA);
        k3 = h * feval(RHS,t0 + h/2, x0 + k2/2, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA);
        k4 = h * feval(RHS, t0 + h, x0 + k3, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA);

        t0 = t0 + h;
        x0 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6;

        ti(i) = t0;
        wi(1,1:nvar) = x0;
        
        %------------------------  Update pRel vectors  -------------------------------
        [pRelAMPA(1,1:nNeurons),pRelNMDA(1,1:nNeurons),pRelGABA(1,1:nNeurons),pRel_stfAMPA(1,1:nNeurons),pRel_stfNMDA(1,1:nNeurons),pRel_stfGABA(1,1:nNeurons)] = Prerelease(x0,pre,h,fD_AMPA,fD_NMDA,fD_GABA,fF_AMPA,fF_NMDA,fF_GABA,indexvspyramneuron,indexvinterneuron,ExcInh,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA); 
        
        
        %------------------------ Save current iteration vectors in files  ------------
        mfw_w_matrix(wi);
        mfw_prelAMPA_matrix(pRelAMPA);
        mfw_prelNMDA_matrix(pRelNMDA);
        mfw_prelGABA_matrix(pRelGABA);    
        mfw_pRel_stfAMPA_matrix(pRel_stfAMPA);
        mfw_pRel_stfNMDA_matrix(pRel_stfNMDA);
        mfw_pRel_stfGABA_matrix(pRel_stfGABA);

        i = i + 1;
    end

    %--------------------  Finish saving in files  ------------------------
    release(mfw_w_matrix);
    release(mfw_prelAMPA_matrix);
    release(mfw_prelNMDA_matrix);
    release(mfw_prelGABA_matrix);    
    release(mfw_pRel_stfAMPA_matrix);
    release(mfw_pRel_stfNMDA_matrix);
    release(mfw_pRel_stfNMDA_matrix); 
    save('ti.mat','ti');
    
end