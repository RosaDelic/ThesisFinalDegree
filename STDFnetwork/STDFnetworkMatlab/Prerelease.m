function [pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA] = Prerelease(x0,pre,rk45_TOL,fD_AMPA,fD_NMDA,fD_GABA,fF_AMPA,fF_NMDA,fF_GABA,indexvspyramneuron,indexvinterneuron,ExcInh,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA)
    %x0: actual vector for the network field
    %pre: previous vector for the network field
    %fD_AMPA = fD_NMDA = fD_GABA: depression factor in the network
    %fF_AMPA = fF_NMDA = fF_GABA: facilitation factor in the network
    tic
    %--------------  Define synaptic parameters we'll need  ---------------
    tau_relAMPA = 400;
    tau_relNMDA = 400;
    tau_relGABA = 400;
    tau_stfAMPA = 50;
    tau_stfNMDA = 50;
    tau_stfGABA = 50;

    %Adding depression factors
    factor_relAMPA = exp(-rk45_TOL/tau_relAMPA);
    factor_relNMDA = exp(-rk45_TOL/tau_relNMDA);
    factor_relGABA = exp(-rk45_TOL/tau_relGABA);
        
    %Adding facilitation factors
     factor_stfAMPA = exp(-rk45_TOL/tau_stfAMPA);
     factor_stfNMDA = exp(-rk45_TOL/tau_stfNMDA);
     factor_stfGABA = exp(-rk45_TOL/tau_stfGABA);

    %Initial probability of release 
    p0_AMPA = 1; 
    p0_NMDA = 1;
    p0_GABA = 1;
    p0_stfAMPA = 1; %I think these ones should be 0.1 
    p0_stfNMDA = 1;
    p0_stfGABA = 1;

    Vthre = -50;
        
    %pRel for synaptic depression (fD) distinguishing AMPA, NMDA, GABA
    pRelAMPA = p0_AMPA*(1-factor_relAMPA)+pRelAMPA*factor_relAMPA;
    pRelNMDA = p0_NMDA*(1-factor_relNMDA)+pRelNMDA*factor_relNMDA;
    pRelGABA = p0_GABA*(1-factor_relGABA)+pRelGABA*factor_relGABA;
    

    %pRel for synaptic facilitation (fF) distinguishing AMPA, NMDA, GABA
    pRel_stfAMPA = p0_stfAMPA*(1-factor_stfAMPA)+pRel_stfAMPA*factor_stfAMPA;
    pRel_stfNMDA = p0_stfNMDA*(1-factor_stfNMDA)+pRel_stfNMDA*factor_stfNMDA;
    pRel_stfGABA = p0_stfGABA*(1-factor_stfGABA)+pRel_stfGABA*factor_stfGABA;


    %--------------------update pRel if the neuron elicits a spike---------------------
        
    %vector 1xnNeurons that in each position contains True/False indicating wheather the neurons fires a spike or not     
    compare = (pre(indexvspyramneuron)<Vthre).*(x0(indexvspyramneuron)>Vthre).*(1-ExcInh)+(pre(indexvinterneuron)<Vthre).*(x0(indexvinterneuron)>Vthre).*(ExcInh);
        
    %updating the vector of pRelAMPA for depression (fD) <--> pyramidal_neurons only
    pRelAMPA = pRelAMPA.*fD_AMPA.*compare.*(1-ExcInh)+pRelAMPA.*(compare.*ExcInh+(1-compare));
        
    %updating the vector of pRelNMDA for depression (fD) <--> pyramidal_neurons only
    pRelNMDA = pRelNMDA.*fD_NMDA.*compare.*(1-ExcInh)+pRelNMDA.*(compare.*ExcInh+(1-compare));
        
    %updating the vector of pRelAMPA for facilitation (fF) <--> pyramidal_neurons only
    pRel_stfAMPA = (pRel_stfAMPA+(1-pRel_stfAMPA).*fF_AMPA).*compare.*(1-ExcInh)+pRel_stfAMPA.*(compare.*ExcInh+(1-compare));
        
    %updating the vector of pRelNMDA for facilitation (fF) <--> pyramidal_neurons only
    pRel_stfNMDA = (pRel_stfNMDA+(1-pRel_stfNMDA).*fF_NMDA).*compare.*(1-ExcInh)+pRel_stfNMDA.*(compare.*ExcInh+(1-compare));
        
    %updating the vector of pRelGABA for depression (fD) <--> interneurons only
    pRelGABA = pRelGABA.*fD_GABA.*compare.*ExcInh+pRelGABA.*(compare.*(1-ExcInh)+(1-compare));
        
    %updating the vector of pRelGABA for facilitation (fD) <--> interneurons only
    pRel_stfGABA = (pRel_stfGABA + (1-pRel_stfGABA).*fF_GABA).*compare.*ExcInh+pRel_stfGABA.*(compare.*(1-ExcInh)+(1-compare));
    toc
end