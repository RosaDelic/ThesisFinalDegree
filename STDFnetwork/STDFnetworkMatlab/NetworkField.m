function dx = NetworkField(t0, x, neq, nNeurons, nvar, ExcInh, P, randomvL, randomgL, randomgsd,indexAMPA,indexNMDA,indexGABA,indexsynAMPA,indexsynNMDA,indexsynGABA,pRelAMPA,pRelNMDA,pRelGABA,pRel_stfAMPA,pRel_stfNMDA,pRel_stfGABA)

    %-------------------------Pyramidal neuron parameters----------------------
    Pyramneuron_Cm = 1;
    Pyramneuron_gna = 50;
    Pyramneuron_gk = 10.5;
    Pyramneuron_ga = 1;
    Pyramneuron_gks = 0.576;
    Pyramneuron_gnap = 0.0686; 
    Pyramneuron_gar = 0.0257;
    Pyramneuron_gca = 0.43;
    Pyramneuron_gkca = 0.57;
    Pyramneuron_gkna = 1.33;
    Pyramneuron_vNa = 55;
    Pyramneuron_vK = -100;
    Pyramneuron_vCa = 120;
    Pyramneuron_As = 0.015;
    Pyramneuron_Ad = 0.035;
    Pyramneuron_tauHa = 15;
    Pyramneuron_tauCa = 150;
    Pyramneuron_phi = 4;
    Pyramneuron_phiHa = 1;
    Pyramneuron_phiKs = 1;
    Pyramneuron_aNa = 0.01*10;
    Pyramneuron_aCa = 0.005*10;
    Pyramneuron_Kd = 30;
    Pyramneuron_Rpump = 0.018;
    Pyramneuron_NaEq = 9.5;
    
    %-------------------------Interneuron parameters----------------------
    Interneuron_Cm = 1;
    Interneuron_gna = 35;
    Interneuron_gk = 9;
    Interneuron_vNa = 55;
    Interneuron_vK = -90;
    Interneuron_A = 0.02;
    Interneuron_phi = 1;
    
    %-------------------------SynapticConnection parameters----------------
    aAMPA = 3.48;
    aNMDA = 0.5;
    aGABA = 1;
    aX = 3.48;
    tauAMPA = 2; 
    tauNMDA = 100;
    tauGABA = 10;
    tauX = 2;
    VsynAMPA = 0;
    VsynNMDA = 0;
    VsynGABA = -70;
    gEE_AMPA = 5.4/10000;
    gEE_NMDA = 0.9/10000;
    gIE_GABA = 4.15/10000;
    gEI_AMPA = 2.25/10000;                        
    gEI_NMDA = 0.5/10000;                       
    gII_GABA = 0.165/10000; 
    
    % ===========================================  NetworkField evaluations  ==================================================
    % initialize vector field to all zeros
    
    dx = zeros(1,nvar);
    
    %defining vectors of synaptic variables AMPA, NMDA and GABA for pyramidal neurons
    sAMPA_vector = x(indexAMPA);
    sNMDA_vector = x(indexNMDA);
    sGABA_vector = x(indexGABA);
    
    %defining vectors of synaptic variables AMPA, NMDA and GABA for interneurons
    synAMPA_vector = x(indexsynAMPA);
    synNMDA_vector = x(indexsynNMDA);
    synGABA_vector = x(indexsynGABA);
        
    
    % network loop
    for postsyn_neuron=0:(nNeurons-1)
        
        index = postsyn_neuron*neq;
        %postsyn_neuron identifies the current neuron, we calculate the Isyn by looking all the presyn_neurons
        %calculate the s_i from this neuron to the other postsyn_neurons
        
        %calculate the synaptic factors Isyn = fact_syn*(v-vsyn) of this neuron
        neuron_type = ExcInh(postsyn_neuron+1);
        
        %define gAMPA, gNMDA, gGABA depending on the postsyn_neuron neuron_type
        %if neuron_type == 0 -->  gAMPA = gEE_AMPA; gNMDA = gEE_NMDA; gGABA = gIE_GABA
        %if neuron_type == 1 -->  gAMPA = gEI_AMPA; gNMDA = gEI_NMDA; gGABA = gII_GABA
        
        gAMPA = gEE_AMPA*(1-neuron_type)+gEI_AMPA*neuron_type;
        gNMDA = gEE_NMDA*(1-neuron_type)+gEI_NMDA*neuron_type;
        gGABA = gIE_GABA*(1-neuron_type)+gII_GABA*neuron_type;
        
        %iterate over all presyn neurons (matrix P by rows) for AMPA and sum all presynaptic neurons contributions
        fact_AMPA = sum(gAMPA*sAMPA_vector.*P(postsyn_neuron+1,:).*pRelAMPA.*pRel_stfAMPA);
        
        %iterate over all presyn neurons (matrix P by rows) for NMDA and sum all presynaptic neurons contributions
        fact_NMDA = sum(gNMDA*sNMDA_vector.*P(postsyn_neuron+1, :).*pRelNMDA.*pRel_stfNMDA);
    
        %iterate over all presyn neurons (matrix P by rows) for GABA and sum all presynaptic neurons contributions    
        fact_GABA = sum(gGABA*sGABA_vector.*P(postsyn_neuron+1, :).*pRelGABA.*pRel_stfGABA);
        
    
        if ~ExcInh(postsyn_neuron+1)
            %pyramidal_neuron
            %soma's voltage
            vs = x(1+index);
            %dendrite's voltage
            vd = x(2+index);
            %ionic channels kinetics variables
            h=x(3+index);
            n=x(4+index);
            ha=x(5+index);
            mks=x(6+index);
            Na=x(7+index);
            Ca=x(8+index);
            %synaptic variables
            sAMPA = x(9+index);
            sNMDA = x(10+index);
            xNMDAs = x(11+index);
            sGABA = x(12+index);
    
            %update random params to current neuron
            gl = randomgL(postsyn_neuron+1);
            vL = randomvL(postsyn_neuron+1);
            gsd = randomgsd(postsyn_neuron+1);
    
            %synaptic current impinging in the soma
            Isyn_GABA = fact_GABA*(vs-VsynGABA);
    
            %synaptic current impinging in the dendrite
            Isyn_AMPA = fact_AMPA*(vd-VsynAMPA);
            Isyn_NMDA = fact_NMDA*(vd-VsynNMDA);
    
    
    
            %----------------------  f(V_presyn)  -----------------------
            f_presyn = (1/(1+exp(-(vs-20)/2)));
    
    
    
            %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  PREVIOUS CALCULUS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
            %--------------------------currents in soma (equations for Vs)-----------------------------
    
            %leak current
            Il=gl*(vs-vL);
            
            %sodium (Na)
            am=0.1*(vs+33)/(1-exp(-(vs+33)/10));
            bm=4*exp(-(vs+53.7)/12);
            minf=am/(am+bm);
            Ina=Pyramneuron_gna*(minf^3)*h*(vs-Pyramneuron_vNa);
            ah=0.07*exp(-(vs+50)/10);
            bh=1/(1+exp(-(vs+20)/10));
    
            %delayed rectifier ~potassium (K)
            Ik=Pyramneuron_gk*(n^4)*(vs-Pyramneuron_vK);
            an=0.01*(vs+34)/(1-exp(-(vs+34)/10));
            bn=0.125*exp(-(vs+44)/25);
    
            %fast A-type K channel
            haInf=1/(1+exp((vs+80)/6));
            maInf=1/(1+exp(-(vs+50)/20));
            Ia=Pyramneuron_ga*(maInf^3)*ha*(vs-Pyramneuron_vK);
    
            %non-inactivating K channel
            Iks=Pyramneuron_gks*mks*(vs-Pyramneuron_vK);
            mksinf=1/(1+exp(-(vs+34)/6.5));
            tks=8/(exp(-(vs+55)/30)+exp((vs+55)/30));
    
            %Na dependent K channel
            if Na<1e-7
                Ikna = 0;
                wNa=0;
            else
                wNa=0.37/(1+(38.7/Na)^3.5); 
                Ikna=Pyramneuron_gkna*wNa*(vs-Pyramneuron_vK);
            end
            %--------------------------currents in dendrite (equations for Vd)-----------------------------
    
            %calcium channel
            mCainf=1/(1+exp(-(vd+20)/9));
            Ica=Pyramneuron_gca*(mCainf^2)*(vd-Pyramneuron_vCa);
    
            %Ca dependent K channel
            Ikca=((Pyramneuron_gkca*Ca)/(Ca+Pyramneuron_Kd))*(vd-Pyramneuron_vK);
    
            %persistent sodium channel (NaP)
            mNapinf=1/(1+exp(-(vd+55.7)/7.7));
            INap=Pyramneuron_gnap*(mNapinf^3)*(vd-Pyramneuron_vNa);
    
    
            %inward rectifying K channel
            hArinf=1/(1+exp((vd+75)/4.));
            Iar=Pyramneuron_gar*hArinf*(vd-Pyramneuron_vK);
    
            %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  DIFFERENTIAL FIELD MODEL EQUATIONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            dx(1+index)=(-(Il+Ina+Ik+Ia+Iks+Ikna)-gsd*(vs-vd)/Pyramneuron_As)/Pyramneuron_Cm - (Isyn_GABA/(Pyramneuron_Cm*Pyramneuron_As));
            dx(2+index)=(-(Ica+Ikca+INap+Iar)-gsd*(vd-vs)/Pyramneuron_Ad)/Pyramneuron_Cm - ((Isyn_AMPA + Isyn_NMDA)/(Pyramneuron_Cm*Pyramneuron_Ad));
            dx(3+index)=Pyramneuron_phi*(ah*(1-h)-bh*h);
            dx(4+index)=Pyramneuron_phi*(an*(1-n)-bn*n);
            dx(5+index)=Pyramneuron_phiHa*(haInf-ha)/Pyramneuron_tauHa;
            dx(6+index)=Pyramneuron_phiKs*(mksinf-mks)/tks;
            dx(7+index)=-Pyramneuron_aNa*(Pyramneuron_As*Ina+Pyramneuron_Ad*INap)-Pyramneuron_Rpump*((Na^3)/((Na^3)+3375)-((Pyramneuron_NaEq^3)/((Pyramneuron_NaEq^3)+3375)));
            dx(8+index)=-Pyramneuron_aCa*(Pyramneuron_Ad*Ica)-(Ca/Pyramneuron_tauCa);
            
            dx(9+index) = aAMPA*f_presyn-sAMPA/tauAMPA;
            dx(10+index) = aNMDA*xNMDAs*(1-sNMDA)-sNMDA/tauNMDA;
            dx(11+index) = aX*f_presyn-xNMDAs/tauX; 
            dx(12+index) = aGABA*f_presyn-sGABA/tauGABA;
            
        else
            %interneuron
            %voltage
            v=x(13+index);
            %ionic channels kinetics variables
            h=x(14+index); 
            n=x(15+index);
            %synaptic variables
            synAMPA = x(16+index);
            synNMDA = x(17+index);
            synxNMDA = x(18+index);
            synGABA = x(19+index);
    
            %update random params to current neuron
            gl = randomgL(postsyn_neuron+1);
            vL = randomvL(postsyn_neuron+1);
    
            %synaptic current impinging in the neuron
            Isyn_AMPA = fact_AMPA*(v-VsynAMPA);
            Isyn_NMDA = fact_NMDA*(v-VsynNMDA);
            Isyn_GABA = fact_GABA*(v-VsynGABA);
    
            %----------------------  f(V_presyn)  -----------------------
            f_presyn = (1/(1+exp(-(v-20)/2)));
    
    
            %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  PREVIOUS CALCULUS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    
            %leak current
            Il=gl*(v-vL);
    
            %sodium (Na)
            am=0.5*(v+35)/(1-exp(-(v+35)/10));
            bm=20*exp(-(v+60)/18);
            minf=am/(am+bm);
            Ina=Interneuron_gna*(minf^3)*h*(v-Interneuron_vNa);
            ah=0.35*exp(-(v+58)/20);
            bh=5/(1+exp(-(v+28)/10));
    
            %delayed rectifier potassium (K)
            Ik=Interneuron_gk*(n^4)*(v-Interneuron_vK);
            an=0.05*(v+34)/(1-exp(-(v+34)/10));
            bn=0.625*exp(-(v+44)/80);
    
            %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  DIFFERENTIAL FIELD MODEL EQUATIONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            dx(13+index) = (-(Il+Ina+Ik))/Interneuron_Cm - (Isyn_AMPA + Isyn_NMDA + Isyn_GABA)/(Interneuron_Cm*Interneuron_A); 
            dx(14+index) = Interneuron_phi*(ah*(1-h)-bh*h);
            dx(15+index) = Interneuron_phi*(an*(1-n)-bn*n);
            dx(16+index) = aAMPA*f_presyn-synAMPA/tauAMPA;
            dx(17+index) = aNMDA*synxNMDA*(1-synNMDA)-synNMDA/tauNMDA;
            dx(18+index) = aX*f_presyn-synxNMDA/tauX;
            dx(19+index) = aGABA*f_presyn-synGABA/tauGABA;
        end
    
    end
end