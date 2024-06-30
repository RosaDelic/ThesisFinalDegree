function Prerelease(x0,pre,nNeurons, neq, rk45_TOL,fD_AMPA,fD_NMDA,fD_GABA,fF_AMPA,fF_NMDA,fF_GABA,p0_stf,ExcInh,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA)
    #x0: actual vector for the network field
    #pre: previous vector for the network field
    #fD_AMPA = fD_NMDA = fD_GABA: depression factor in the network
    #fF_AMPA = fF_NMDA = fF_GABA: facilitation factor in the network

    #--------------  Define synaptic parameters we'll need  ---------------
    tau_relAMPA = 400;
    tau_relNMDA = 400;
    tau_relGABA = 400;
    tau_stfAMPA = 50;
    tau_stfNMDA = 50;
    tau_stfGABA = 50;

    #Adding depression factors
    factor_relAMPA = exp(-rk45_TOL/tau_relAMPA);
    factor_relNMDA = exp(-rk45_TOL/tau_relNMDA);
    factor_relGABA = exp(-rk45_TOL/tau_relGABA);
        
    #Adding facilitation factors
    factor_stfAMPA = exp(-rk45_TOL/tau_stfAMPA);
    factor_stfNMDA = exp(-rk45_TOL/tau_stfNMDA);
    factor_stfGABA = exp(-rk45_TOL/tau_stfGABA);

    #Initial probability of release 
    p0_AMPA = 1; 
    p0_NMDA = 1;
    p0_GABA = 1;
    p0_stfAMPA = p0_stf; 
    p0_stfNMDA = p0_stf;
    p0_stfGABA = p0_stf;

    Vthre = -50;

    for i in eachindex(1:nNeurons)
        #pRel for synaptic depression (fD) distinguishing AMPA, NMDA, GABA
        pRelAMPA[i] = p0_AMPA*(1-factor_relAMPA) + pRelAMPA[i]*factor_relAMPA;
        pRelNMDA[i] = p0_NMDA*(1-factor_relNMDA) + pRelNMDA[i]*factor_relNMDA;
        pRelGABA[i] = p0_GABA*(1-factor_relGABA) + pRelGABA[i]*factor_relGABA;
        

        #pRel for synaptic facilitation (fF) distinguishing AMPA, NMDA, GABA
        pRel_stfAMPA[i] = p0_stfAMPA*(1-factor_stfAMPA) + pRel_stfAMPA[i]*factor_stfAMPA;
        pRel_stfNMDA[i] = p0_stfNMDA*(1-factor_stfNMDA) + pRel_stfNMDA[i]*factor_stfNMDA;
        pRel_stfGABA[i] = p0_stfGABA*(1-factor_stfGABA) + pRel_stfGABA[i]*factor_stfGABA;

        #excitatory neuron
        if ExcInh[i]==0
            #neuron elicits a spike
            if pre[(i-1)*neq+1]<Vthre && x0[(i-1)*neq+1]>Vthre
                pRelAMPA[i] = pRelAMPA[i]*fD_AMPA;
                pRelNMDA[i] = pRelNMDA[i]*fD_NMDA;
                pRel_stfAMPA[i] = pRel_stfAMPA[i] + (1-pRel_stfAMPA[i])*fF_AMPA;
                pRel_stfNMDA[i] = pRel_stfNMDA[i] + (1-pRel_stfAMPA[i])*fF_NMDA;
            end
        #inhibitory neuron
        else
            #neuron elicits a spike
            if pre[(i-1)*neq+13]<Vthre && x0[(i-1)*neq+13]>Vthre
                pRelGABA[i] = pRelGABA[i]*fD_GABA;
                pRel_stfGABA[i] = pRel_stfGABA[i] + (1-pRel_stfGABA[i])*fF_GABA;
            end

        end
    end

    return pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA
end