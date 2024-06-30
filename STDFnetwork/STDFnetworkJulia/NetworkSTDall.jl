#0. Necessary imports
import Random as rnd
import MAT
include("SDNumber.jl")
include("NetworkField.jl")
include("rk45Network.jl")
function NetworkSTDall(Dnumber,Fnumber,t0,tf,h,p0_stf)
    #######################################################################
    ################  MAIN PROGRAM  ALL FUNCTIONS TOGETHER ################
    #######################################################################
    @time begin
    print("---------------------------------  STD - Network  ---------------------------------- \n")



    #1. load BuildNetwork file which contains: 
        # P: 320X320 table where (i,j) = 0 if neurons i and j are connected; (i,j) = 0 otherwise /// i: presyn_neuron, j: postsyn_neuron
        # ExcInh: 1x320 vector where i = 1 identifies inhibitory neurons, i = 0 identifies excitatory neurons

    print("Reading P \n")
    BN=MAT.matopen("BuildNetwork320.mat")
    P=read(BN,"P");
    print("Reading EI \n")
    ExcInh=read(BN,"ExcInh");
    close(BN)
    sizeExcInh = size(ExcInh);
    ExcInh = reshape(ExcInh, (sizeExcInh[2],1))
    #print shape of read data
    print("----------------------------- Data loaded  --------------------------- \n")
    print("P shape: ", size(P), "\n")
    print("ExcInh shape: ", size(ExcInh), "\n")
        
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


    fD_AMPA = Dnumber;
    fD_NMDA = Dnumber;
    fD_GABA = Dnumber;

    fF_AMPA = Dnumber;
    fF_NMDA = Dnumber;
    fF_GABA = Dnumber;

    N = Int64(floor((tf-t0)/h));
    sizenNeurons = size(ExcInh);
    nNeurons = sizenNeurons[1];
    neq = 20;
    nvar = nNeurons*neq;

    #---------------------------  Define file name  ---------------------------
    #print simulation parameters
    print("----------------------------- Simulation parameters  --------------------------- \n");
    print("Depression factor, fD: ", Dnumber, "\n")
    print("Facilitation factor, fF: ", Fnumber, "\n")
    print("Initial time, t0: ", t0, "\n")
    print("Final time, tf: ", tf, "\n")
    print("Runge-kutta step, h: ", h, "\n");
    print("Number of rk45 iterations, N: ", N, "\n");
    print("Number of neurons in the network, Nneurons: ", nNeurons, "\n")
    print("Number of equations for each neuron, neq: ", neq, "\n")
    print("Number of total variables in the network, nvar = Nneuronsxneq: ", nvar, "\n")

    #3. Define the vector of initial conditions
    x0 = zeros(nvar);

    #-----------------------------  vectorized form  --------------------------    
    excvs_positions = 1 .+ (0:(nNeurons-1)) .*neq; #these vectors have length 320 (already checked)
    excvd_positions = 2 .+ (0:(nNeurons-1)) .*neq;
    inhvs_positions = 13 .+ (0:(nNeurons-1)) .*neq;

    excvs_random = rnd.randn(length(excvs_positions));    
    excvd_random = rnd.randn(length(excvd_positions));   
    inhvs_random = rnd.randn(length(inhvs_positions));

    x0[excvs_positions] = (-60 .+ 5 .*excvs_random) .*(1 .- ExcInh);
    x0[excvd_positions] = (-60 .+ 5 .*excvd_random) .*(1 .- ExcInh);
    x0[inhvs_positions] = (-60 .- 5 .*inhvs_random) .*ExcInh;

    SDvL = SDNumber(nNeurons);
    SDgL = SDNumber(nNeurons);
    SDgsd = SDNumber(nNeurons);

    randomvL = (-60.95 .+ 0.3 .*SDvL) .*(1 .- ExcInh) .+ (-63.8 .+ 0.15 .*SDvL) .*ExcInh;
    randomgL = (0.0667 .+ 0.0067 .*SDgL) .*(1 .- ExcInh) .+ (0.1025 .+ 0.0025 .*SDgL).*ExcInh;
    randomgsd = ((1.75 .+ 0.1 .*SDgsd) .*0.1) .*(1 .- ExcInh);

    ti, wi, pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA = rk45Network(NetworkField, t0, tf, x0, N, h, neq, nNeurons, nvar, ExcInh, P, fD_AMPA, fD_NMDA, fD_GABA, fF_AMPA, fF_NMDA, fF_GABA, randomvL, randomgL, randomgsd, p0_stf)
    end

    filename = string("fD",Dnumber,"_fF",Fnumber,".mat")
    MAT.matwrite(filename, Dict(
        "ti" => ti,
        "wi" => wi,
        "pRelAMPA" => pRelAMPA,
        "pRelNMDA" => pRelNMDA,
        "pRelGABA" => pRelGABA,
        "pRel_stfAMPA" => pRel_stfAMPA,
        "pRel_stfNMDA" => pRel_stfNMDA,
        "pRel_stfGABA" => pRel_stfGABA
        ))
end