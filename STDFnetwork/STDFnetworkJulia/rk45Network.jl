import DelimitedFiles as dlm
include("NetworkField.jl")
include("Prerelease.jl")

function rk45Network(RHS, t0, tf, x0, N, h, neq, nNeurons, nvar, ExcInh, P, fD_AMPA, fD_NMDA, fD_GABA, fF_AMPA, fF_NMDA, fF_GABA, randomvL, randomgL, randomgsd, p0_stf)
    #RHS: differential equations system to solve with rk45 method
    #t0: initial time of the simulation
    #tf: final time of the simulation
    #h: Runge-kutta step 
    #N: total number of iterations in the RK-45 
    #neq: number of equations for each neuron in the network
    #P: 320X320 table where (i,j) = 0 if neurons i and j are connected; (i,j) = 0 otherwise /// i: presyn_neuron, j: postsyn_neuron
    #ExcInh: 1x320 vector where i = 1 identifies inhibitory neurons, i = 0 identifies excitatory neurons
    #fD_X: Depression factor for the specific neurotransmitter X = AMPA, NMDA, GABA
    #fF_X: Facilitation factor for the specific neurotransmitter X = AMPA, NMDA, GABA

    #------------------------------------ Initial values to rk45 Outputs  -------------------------
    
    #initialize time vector (ti), system solution vector (wi), all probability of Release vectors (pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA)
    ti=zeros(N);
    wi = zeros((nvar,N));
    #save initial conditions in the vectors
    ti[1] = t0;
    wi[:,1] = x0;

    #Prel vectors we will need during all the simulation --> they are initialized at all ones
    #For depression
    pRelAMPA = ones((nNeurons,N));
    pRelNMDA = ones((nNeurons,N));
    pRelGABA = ones((nNeurons,N));
    #For facilitation
    pRel_stfAMPA = ones((nNeurons,N));
    pRel_stfNMDA = ones((nNeurons,N));
    pRel_stfGABA = ones((nNeurons,N));


    #indexes of x0 where the variables AMPA, NMDA and GABA for pyramidal neurons are located
    indexAMPA = 9 .+ (0:(nNeurons-1)) .*neq; #vectors of length 320
    indexNMDA = 10 .+ (0:(nNeurons-1)) .*neq;
    indexGABA = 12 .+ (0:(nNeurons-1)) .*neq;
        
                
    #indexes of x0 where the variables AMPA, NMDA and GABA for interneurons are located
    indexsynAMPA = 16 .+ (0:(nNeurons-1)) .*neq; #vectors of length 320
    indexsynNMDA = 17 .+ (0:(nNeurons-1)) .*neq;
    indexsynGABA = 19 .+ (0:(nNeurons-1)) .*neq;
    
    #Define index of x0 where the voltage of the Pyramidal neurons and Interneurons is located
    indexvspyramneuron = 1 .+ (0:(nNeurons-1)) .*neq;
    indexvinterneuron = 13 .+ (0:(nNeurons-1)) .*neq;

    #---------------------------------  Loop to integrate the system ----------------------------------

    i = 2;
    t0 = h;

    while(t0+h < tf)
        if i%1000 == 0
            print("Actual i: ", i, "\n");
        end
        #-------------------------  RK45-Field integrator -----------------------------
        #@time begin
        pre = x0;
        k1 = h * RHS(t0, x0, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1]);
        k2 = h * RHS(t0 + h/2, x0 + k1/2, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1]);
        k3 = h * RHS(t0 + h/2, x0 + k2/2, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1]);
        k4 = h * RHS(t0 + h, x0 + k3, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1]);

        t0 = t0 + h;
        x0 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6;

        ti[i] = t0;
        wi[:,i] = x0;

        #------------------------  Update pRel vectors  -------------------------------
        pRelAMPA[:,i],pRelNMDA[:,i],pRelGABA[:,i],pRel_stfAMPA[:,i],pRel_stfNMDA[:,i],pRel_stfGABA[:,i] = Prerelease(x0,pre,nNeurons, neq,h,fD_AMPA,fD_NMDA,fD_GABA,fF_AMPA,fF_NMDA,fF_GABA,p0_stf,ExcInh,pRelAMPA[:,i-1],pRelNMDA[:,i-1],pRelGABA[:,i-1],pRel_stfAMPA[:,i-1],pRel_stfNMDA[:,i-1],pRel_stfGABA[:,i-1]); 
        i += 1;
        end

    return ti, wi, pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA
end