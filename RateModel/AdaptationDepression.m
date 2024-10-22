%Define discretization needed for simulation
delta = 10^(-4); %time and activation_function discretization
%% -------------------  EXAMPLES ACTIVATION FUNCTIONS  --------------------
%Plot activation function shape for different values of slope (a_k) and
% OX-axis displacement (b_k)

%----------------------------  Different slopes  --------------------------
%activation function interval (only to plot activation functions)
activ_interval = 0:delta:1000; 
%Different slope values
a_ks = [0,0.5,1,1.5];
%Set fixed OX-displacement value
b_k = 100;
%Colors
color = ['r',"#EDB120",'b',"#7E2F8E"]

%Figure to show activation function for different slopes
h=figure()
for i=1:length(a_ks)
    a_k_actual=a_ks(i);
    activ_output_actual = @(x)(a_k_actual*subplus(x-b_k))
    plot(activ_interval,activ_output_actual(activ_interval),'LineWidth',4,'Color',color(i),'DisplayName',strcat('\alpha_k = ',num2str(a_k_actual)))
    hold on;
end
legend('show','Location','northwest','NumColumns',2)
xlabel('x','FontSize',30)
ylabel('F(x)','FontSize',30)
ylim([-100 1500])
ax = gca; 
ax.FontSize = 30;
hold off;

%-----------------------  Different OX-displacements  ---------------------

%Different OX-displacement values
b_ks = [0,200,400,600];
%Set fixed slope value
a_k = 1;

%Figure to show activation function for different OX-displacements
h=figure()
for i=1:length(b_ks)
    b_k_actual=b_ks(i);
    activ_output_actual = @(x)(a_k*subplus(x-b_k_actual))
    plot(activ_interval,activ_output_actual(activ_interval),'LineWidth',4,'Color',color(i),'DisplayName',strcat('\beta_k = ',num2str(b_k_actual)))
    hold on;
end
legend('show','Location','northwest','NumColumns',2)
xlabel('x','FontSize',30)
xlabel('x','FontSize',30)
ylabel('F(x)','FontSize',30)
ax = gca; 
ax.FontSize = 30;
hold off;
%% ---------------------------ACTIVATION FUNCTION--------------------------
%excitatory activation function definition
alpha_e = 0.21;
beta_e = -25;
activ_output_e = @(x)(alpha_e*subplus(x-beta_e))

%inhibitory activation function definition
alpha_i = 0.60;
beta_i = 75;
activ_output_i = @(x)(alpha_i*subplus(x-beta_i))
%% --------------------------COMMON PARAMETERS-----------------------------
%Time interval to integrate the system
tspan = [0,20000];
        
%Synaptic connection parameters
wee = 10;
wei = 1.5;
wie = 9;
wii = 0.5;

%Time constant values
tau_e = 150;
tau_i = 320;
tau_se = 250; 
tau_si = 250;
tau_ae = 1500;
tau_ai = 500;
%ke=1;
%ki=1;

%Mass matrix for ode integrator
Mass = [tau_e 0 0 0 0 0;0 tau_i 0 0 0 0;0 0 tau_se 0 0 0;0 0 0 tau_si 0 0; 0 0 0 0 tau_ae 0; 0 0 0 0 0 tau_ai];
opts = odeset('Mass',Mass);

%External currents intputs
Iext_e = 0;
Iext_i = 0;

%Parameters controling adaptation sensitivity
Je=10;
Ji=3.5;

%Define effective depression functions
Fefectiva_e = @(fd)(0.3+exp(-(4*(1+0.09)))+(1-(0.3+exp(-(4*(1+0.09)))))*exp(10.8*(fd-1)));
Fefectiva_i = @(fd)(exp(-0.36)+(1-exp(-0.36))*exp(10*(fd-1)));

%Plot effective depression functions: controling excitatory and inhibitory
%depression factors
%activation function interval (only to plot depression functions)
activ_interval = 0:delta:1; 
h=figure()
plot(activ_interval,Fefectiva_e(activ_interval),'LineWidth',4,'Color','r','DisplayName','f_{De}')
hold on;
plot(activ_interval,Fefectiva_i(activ_interval),'LineWidth',4,'Color','b','DisplayName','f_{Di}')
hold on;
xlabel('f_D','FontSize',36);
ylabel('f_{D_k}','FontSize',36);
legend('NumColumns',2,'Location','northwest');
ax = gca; 
ax.FontSize = 24;
hold off;

%============ THE USER CAN CHANGE fd FACTOR FOR SIMULATIONS ==============
%Set fD factor for simulation
fd = 0.84;
fde = Fefectiva_e(fd);
fdi = Fefectiva_i(fd);

%Depression activation_function definition
%Define thresholds (depend on fD)
theta_de = fde*150;
theta_de_function = @(fd)(150*Fefectiva_e(fd));
theta_di = fdi*65;
theta_di_function = @(fd)(65*Fefectiva_i(fd));
%Define activation function for the depression
Ge = @(x)(0.*(x<(theta_de-10))+(1/20)*(x-(theta_de-10)).*(x>=(theta_de-10) & x<(theta_de+10))+1.*(x>=(theta_de+10)));
Gi = @(x)(0.*(x<(theta_di-10))+(1/20)*(x-(theta_di-10)).*(x>=(theta_di-10) & x<(theta_di+10))+1.*(x>=(theta_di+10)));
%% --------------------------  ODE INTEGRATION  ---------------------------
%r = [re,ri,se,si,ae,ai]

%Set initial condition
r0 = [0.5,0.5,1,1,0,0];

%Integrate system
[t,r] = ode45(@(t,r)ModelFunctionBothScaledDA(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,Ge,Gi,Je,Ji,fde,fdi),tspan,r0,opts);
%% ---------------------------  PLOTS FIGURE  -----------------------------
dr_e = @(r_e,r_i)((1/tau_e)*(-r_e+activ_output_e(fde*wee*r_e-fdi*wei*r_i+Iext_e)));
dr_i = @(r_e,r_i)((1/tau_i)*(-r_i+activ_output_i(-fdi*wii*r_i+fde*wie*r_e+Iext_i)));

%Define vector field meshgrid
spacing = 10;
[R_E,R_I] = meshgrid(-200:spacing:375);

%Figure 1: Plot nullclines with se=fde/si=fdi, vector field and system
%trajectory
h=figure()
quiver(R_E,R_I,dr_e(R_E,R_I),dr_i(R_E,R_I),'Color',[.7 .7 .7],'LineWidth',2,'DisplayName','VectorField');
hold on;
fimplicit(dr_e,'LineWidth',4,'Color','r','DisplayName','re-nullcline');
hold on;
fimplicit(dr_i,'LineWidth',4,'Color','b','DisplayName','ri-nullcline');
hold on;
plot(r(:,1),r(:,2),"Linewidth",4,'Color','k');
axis([min(r(:,1))-10 max(r(:,1))+10 min(r(:,2))-10 max(r(:,2))+10])
legend('vector field','re-nullcline','ri-nullcline','NumColumns',3,'Orientation','horizontal','Location','northoutside','Fontsize',20);
xlabel('r_e','FontSize',48);
ylabel('r_i','FontSize',48);
%current axes
ax = gca; 
ax.FontSize = 40;
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd), 'PhaseSpaceNullclines.png'),'png')

%Figure 2: Plot phase space excitatory and inhibitory population (re,ri) plane
h=figure()
quiver(R_E,R_I,dr_e(R_E,R_I),dr_i(R_E,R_I),'Color',[.7 .7 .7],'LineWidth',2,'DisplayName','VectorField');
hold on;
hold on;
plot(r(:,1),r(:,2),"Linewidth",4,'Color','k');
axis([min(r(:,1))-10 max(r(:,1))+10 min(r(:,2))-10 max(r(:,2))+10])
xlabel('r_e','FontSize',48);
ylabel('r_i','FontSize',48);
% current axes
ax = gca; 
ax.FontSize = 36;
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd), 'PhaseSpace.png'),'png')

%Figure 3: Plot firing rate excitatory and inhibitory population 
h=figure()
plot(t/1000,r(:,3),'-',"LineWidth",4,'Color',"#EDB120");
hold on;
plot(t/1000,r(:,4),'-',"LineWidth",4,'Color',"#7E2F8E");
hold on;
xlabel('Time t(s)','FontSize',48);
ylabel('Solution s_D(t)','FontSize',48);
% current axes
ylim([min(min(r(:,3)),min(r(:,4)))-0.1 max(max(r(:,3)),max(r(:,4)))+0.1])
ax = gca; 
ax.FontSize = 36;
h.Position = [100 100 800 400];
legend('s_{De}','s_{Di}',"Location","southeast","Orientation",'horizontal','Fontsize',24)
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd), 'Depression.png'),'png')

%Figure 4: Plot adaptation in excitatory and inhibitory population 
h=figure()
plot(t/1000,r(:,5),'-',"LineWidth",4,'Color','r');
hold on;
plot(t/1000,r(:,6),'-',"LineWidth",4,'Color','b');
xlabel('Time t(s)','FontSize',48);
ylabel('Solution a(t)','FontSize',48);
% current axes
ylim([min(min(r(:,5),min(r(:,6))))-10 max(max(r(:,5),max(r(:,6))))+25])
ax = gca; 
legend('a_e','a_i',"Location","northeast","Orientation",'horizontal','Fontsize',24)
ax.FontSize = 36;
h.Position = [100 100 800 400];
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd), 'Adaptation.png'),'png')

%Figure 5: Plot firing rate excitatory and inhibitory population 
h=figure()
plot(t/1000,r(:,1),'-',"LineWidth",4,'Color','r');
hold on;
plot(t/1000,r(:,2),'-',"LineWidth",4,'Color','b');
hold on;
xlabel('Time t(s)','FontSize',48);
ylabel('Solution r(t)','FontSize',48);
% current axes
ylim([min(min(r(:,1),min(r(:,2))))-10 max(max(r(:,1)),max(r(:,2)))+25]);
ax = gca; 
ax.FontSize = 36;
h.Position = [100 100 800 400];
legend('r_e','r_i',"Location","northeast","Orientation",'horizontal','Fontsize',24)
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd), 'FiringRate.png'),'png')
%% -------------------------- COMPLETE VIDEO  -----------------------------
%Define vector field meshgrid
spacing = 10;
[R_E,R_I] = meshgrid(-200:spacing:375);

%Define general figure
h = figure()
set(h,'Position',[100 100 2000 1500])

%Define time steps to include frames in the video
video_speed = 2;
for idx=1:length(t)
    clf;
    if mod(idx,video_speed)==0
        dr_e_t = @(r_e,r_i)((1/tau_e)*(-r_e+activ_output_e(r(idx,3)*wee*r_e-r(idx,4)*wei*r_i+Iext_e-r(idx,5))));
        dr_i_t = @(r_e,r_i)((1/tau_i)*(-r_i+activ_output_i(-r(idx,4)*wii*r_i+r(idx,3)*wie*r_e+Iext_i-r(idx,6))));
        F_e = @(r_e,r_i)(r(idx,3)*wee*r_e-r(idx,4)*wei*r_i+Iext_e-r(idx,5)-beta_e);
        F_i = @(r_e,r_i)(-r(idx,4)*wii*r_i+r(idx,3)*wie*r_e+Iext_i-r(idx,6)-beta_i);

        %Subplot (1,1) --> VectorField/PhaseSpace/Nullclines
        subplot(2,2,1);
        quiver(R_E,R_I,dr_e_t(R_E,R_I),dr_i_t(R_E,R_I),'Color',[.7 .7 .7],'LineWidth',2,'DisplayName','VectorField');
        hold on;
        fimplicit(dr_e_t,'LineWidth',4,'Color','r','DisplayName','re-nullcline');
        hold on;
        fimplicit(dr_i_t,'LineWidth',4,'Color','b','DisplayName','ri-nullcline');
        hold on;
        fimplicit(F_e,'LineWidth',4,'Color','k','DisplayName','Fe');
        hold on;
        fimplicit(F_i,'LineWidth',4,'Color','k','DisplayName','Fi');
        hold on;
        plot(r(idx,1),r(idx,2),'.',"MarkerSize",40,'Color','k');
        axis([-50 150 -200 375])
        xlabel('r_e','FontSize',36);
        ylabel('r_i','FontSize',36);
        yticks([-200 -100 0 100 200 300])
        title('Vector field and nullclines','FontSize',18);
        % current axes
        ax = gca; 
        ax.FontSize = 24;
        hold on;
    
        %Subplot (1,2) --> FiringRate
        subplot(2,2,2);
        %excitatory
        re_idx = nan(1,length(t));
        re_idx(1:idx) = r(1:idx,1);
        re_idx(end) = r(end,1);
        %inhibitory
        ri_idx = nan(1,length(t));
        ri_idx(1:idx) = r(1:idx,2);
        ri_idx(end) = r(end,2);
        plot(t/1000,re_idx,'-',"LineWidth",4,'Color','r');
        hold on;
        plot(t/1000,ri_idx,'-',"LineWidth",4,'Color','b');
        hold on;
        title('Firing rate','FontSize',18);
        xlabel('Time t (s)','FontSize',24);
        ylabel('Solution r(t)','FontSize',24);
        % current axes
        ylim([min(min(r(:,1),min(r(:,2))))-10 max(max(r(:,1)),max(r(:,2)))+25]);
        ax = gca; 
        ax.FontSize = 24;

        %Subplot (2,1) --> Depression
        subplot(2,2,3);
        %excitatory
        se_idx = nan(1,length(t));
        se_idx(1:idx) = r(1:idx,3);
        se_idx(end) = r(end,3);
        %inhibitory
        si_idx = nan(1,length(t));
        si_idx(1:idx) = r(1:idx,4);
        si_idx(end) = r(end,4);
        plot(t/1000,se_idx,'-',"LineWidth",4,'Color',"#EDB120");
        hold on;
        plot(t/1000,si_idx,'-',"LineWidth",4,'Color',"#7E2F8E");
        hold on;
        title('Depression','FontSize',18);
        xlabel('Time t (s)','FontSize',24);
        ylabel('Solution s(t)','FontSize',24);
        % current axes
        ylim([min(min(r(:,3)),min(r(:,4)))-0.1 max(max(r(:,3)),max(r(:,4)))+0.1])
        ax = gca; 
        ax.FontSize = 24;
        %yticks([0.5, 0.6, 0.7, 0.8,0.9,1])
    
        %Subplot (2,2) --> Adaptation
        subplot(2,2,4);
        %excitatory
        ae_idx = nan(1,length(t));
        ae_idx(1:idx) = r(1:idx,5);
        ae_idx(end) = r(end,5);
        %inhibitory
        ai_idx = nan(1,length(t));
        ai_idx(1:idx) = r(1:idx,6);
        ai_idx(end) = r(end,6);
        plot(t/1000,ae_idx,'-',"LineWidth",4,'Color','r');
        hold on
        plot(t/1000,ai_idx,'-',"LineWidth",4,'Color','b');
        title('Adaptation','FontSize',18);
        xlabel('Time t(s)','FontSize',24);
        ylabel('Solution a(t)','FontSize',24);
        % current axes
        ylim([min(min(r(:,5),min(r(:,6))))-10 max(max(r(:,5),max(r(:,6))))+25])
        ax = gca; 
        ax.FontSize = 24;
        
        sgtitle(strcat('f_D = ',num2str(fd)),'FontSize',40)
        drawnow;
        movieVector(floor(idx/video_speed)) = getframe(h);
        
    end
end

%Comment the following lines for not saving the video
myWriter = VideoWriter(strcat('CompleteAnimationfD',num2str(fd)),'MPEG-4')
myWriter.FrameRate = 5
open(myWriter)
writeVideo(myWriter,movieVector)
close(myWriter)
clear movieVector
%% -----------------------  IMPROVED STABILITY VIDEO  ---------------------
%Define vector field meshgrid
spacing = 10;
[R_E,R_I] = meshgrid(-200:spacing:375);

%Define general figure
h = figure()
set(h,'Position',[100 100 2000 1500])

%Initialize solutions vectors
solspecificR1e = NaN(1,length(t));
solspecificR1i = NaN(1,length(t));
solspecificR2e = NaN(1,length(t));
solspecificR2i = NaN(1,length(t));
solspecificR4e = NaN(1,length(t));
solspecificR4i = NaN(1,length(t));

%Backwards integration
tspanunstableR2 = 0:0.01:1500;
epsilonunstableR2 = 5;

%Backwards integration
tspanbackwards = 1000:-0.01:0;
epsilonbackwards = 20;

%Forward integration
%Mass matrix for ode integrator
tspanforwards = 0:0.01:500;
MassForward = [tau_e 0;0 tau_i];
optsForward = odeset('Mass',MassForward);

%Time-frozen system
for idx=1:length(t)
    clf;
    if mod(idx,video_speed)==0
        %Function definition
        dr = @(rvar)[(1/tau_e)*(-rvar(1)+activ_output_e(r(idx,3)*wee*rvar(1)-r(idx,4)*wei*rvar(2)+Iext_e-r(idx,5)));(1/tau_i)*(-rvar(2)+activ_output_i(-r(idx,4)*wii*rvar(2)+r(idx,3)*wie*rvar(1)+Iext_i-r(idx,6)))];
        F_e = @(r_e,r_i)(r(idx,3)*wee*r_e-r(idx,4)*wei*r_i+Iext_e-r(idx,5)-beta_e);
        F_i = @(r_e,r_i)(-r(idx,4)*wii*r_i+r(idx,3)*wie*r_e+Iext_i-r(idx,6)-beta_i);

        %Forward integration - current solution

        %Set initial condition
        r0_forward = [r(idx,1), r(idx,2)];
        s_e = r(idx,3);
        s_i = r(idx,4);
        a_e = r(idx,5);
        a_i = r(idx,6);

        %Integrate system
        [tforward1,rforward1] = ode45(@(t,r)DefProject6Dinto2D(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,s_e,s_i,a_e,a_i),tspanforwards,r0_forward,optsForward);

        %Plot foward trajectory
        plot(rforward1(:,1),rforward1(:,2),'LineWidth',4,'Color','k','LineStyle','-.')
        hold on;


        %1. Compute the solution in each region
        solspecificR1e_iter = 0;
        solspecificR1i_iter = 0;
        solspecificR2e_iter= (alpha_e*(beta_e-(Iext_e-r(idx,5))))/(alpha_e*wee*r(idx,3)-1);
        solspecificR2i_iter = 0;
        solspecificR4e_iter = (alpha_e*alpha_i*wei*r(idx,4)*(Iext_i-r(idx,6)-beta_i)-alpha_e*(alpha_i*wii*r(idx,4)+1)*(Iext_e-r(idx,5)-beta_e))/((alpha_e*wee*r(idx,3)-1)*(alpha_i*wii*r(idx,4)+1)-alpha_e*alpha_i*wie*r(idx,3)*wei*r(idx,4));
        solspecificR4i_iter = ((alpha_i*wie*r(idx,3))/(alpha_i*wii*r(idx,4)+1))*solspecificR4e_iter+(alpha_i*(Iext_i-r(idx,6)-beta_i))/(alpha_i*wii*r(idx,4)+1);
        
        %2. Save fixed point in each region
        if F_e(solspecificR1e_iter,solspecificR1i_iter)<0 && F_i(solspecificR1e_iter,solspecificR1i_iter)<0
            solspecificR1e(1,idx) = solspecificR1e_iter;
            solspecificR1i(1,idx) = solspecificR1i_iter;
        end
        if F_e(solspecificR2e_iter,solspecificR2i_iter)>=0 && F_i(solspecificR2e_iter,solspecificR2i_iter)<0
            solspecificR2e(1,idx) = solspecificR2e_iter;
            solspecificR2i(1,idx) = solspecificR2i_iter;
            
            %2. Compute the Jacobian matrix and evaluate at the fixed point
            J = @(rvar_e,rvar_i)[(1/tau_e)*(-1+alpha_e*r(idx,3)*wee*hardlim(r(idx,3)*wee*rvar_e-r(idx,4)*wei*rvar_i+Iext_e-r(idx,5)-beta_e)) (1/tau_e)*(-alpha_e*r(idx,4)*wei*hardlim(r(idx,3)*wee*rvar_e-r(idx,4)*wei*rvar_i+Iext_e-r(idx,5)-beta_e)); (1/tau_i)*(alpha_i*r(idx,3)*wie*hardlim(-r(idx,4)*wii*rvar_i+r(idx,3)*wie*rvar_e+Iext_i-r(idx,6)-beta_i)) (1/tau_i)*(-1-alpha_i*r(idx,4)*wii*hardlim(-r(idx,4)*wii*rvar_i+r(idx,3)*wie*rvar_e+Iext_i-r(idx,6)-beta_i))]; 
            Jsubs = J(solspecificR2e_iter,solspecificR2i_iter);
            
            %3. Find vaps and veps of the Jacobian matrix
            [veps,vaps] = eig(Jsubs);

            %Backward integration - stable manifold
            %Set initial condition
            if vaps(2,2)<0
                %StableManifild
                r2_up = [solspecificR2e_iter+epsilonbackwards*veps(1,2),solspecificR2i_iter+epsilonbackwards*veps(2,2),r(idx,3),r(idx,4),r(idx,5),r(idx,6)]
                r2_down = [solspecificR2e_iter-epsilonbackwards*veps(1,2),solspecificR2i_iter-epsilonbackwards*veps(2,2),r(idx,3),r(idx,4),r(idx,5),r(idx,6)];
                %UnstableManifold
                r2_upUnstable = [solspecificR2e_iter+epsilonunstableR2*veps(1,1),solspecificR2i_iter+epsilonunstableR2*veps(2,1),r(idx,3),r(idx,4),r(idx,5),r(idx,6)]
                r2_downUnstable = [solspecificR2e_iter-epsilonunstableR2*veps(1,1),solspecificR2i_iter-epsilonunstableR2*veps(2,1),r(idx,3),r(idx,4),r(idx,5),r(idx,6)];
            
            elseif vaps(1,1)<0
                %StableManifold
                r2_up = [solspecificR2e_iter+epsilonbackwards*veps(1,1),solspecificR2i_iter+epsilonbackwards*veps(2,1),r(idx,3),r(idx,4),r(idx,5),r(idx,6)]
                r2_down = [solspecificR2e_iter-epsilonbackwards*veps(1,1),olspecificR2i_iter-epsilonbackwards*veps(2,1),r(idx,3),r(idx,4),r(idx,5),r(idx,6)];
                %UnstableManifold
                r2_upUnstable = [solspecificR2e_iter+epsilonunstableR2*veps(1,2),solspecificR2i_iter+epsilonunstableR2*veps(2,2),r(idx,3),r(idx,4),r(idx,5),r(idx,6)]
                r2_downUnstable = [solspecificR2e_iter-0.1*veps(1,2),solspecificR2i_iter-0.1*veps(2,2),r(idx,3),r(idx,4),r(idx,5),r(idx,6)];
            
            end
            %Integrate system StableManifold
            [tbackwards3,rbackwards2_up] = ode45(@(t,r)ModelFunctionBothScaledDA(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,Ge,Gi,Je,Ji,fde,fdi),tspanbackwards,r2_up,opts);
            [tbackwards4,rbackwards2_down] = ode45(@(t,r)ModelFunctionBothScaledDA(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,Ge,Gi,Je,Ji,fde,fdi),tspanbackwards,r2_down,opts);
            %Integrate system UnstableManifold
            [tbackwards3Unstable,rbackwards2_upUnstable] = ode45(@(t,r)ModelFunctionBothScaledDA(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,Ge,Gi,Je,Ji,fde,fdi),0:0.01:2000,r2_upUnstable,opts);
            [tbackwards4Unstable,rbackwards2_downUnstable] = ode45(@(t,r)ModelFunctionBothScaledDA(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,Ge,Gi,Je,Ji,fde,fdi),0:0.01:5000,r2_downUnstable,opts);

            %Plot stable manifold
            plot(rbackwards2_up(:,1),rbackwards2_up(:,2),'LineWidth',4,'Color',[.4 .4 .4],'LineStyle','-.')
            hold on;
            plot(rbackwards2_down(:,1),rbackwards2_down(:,2),'LineWidth',4,'Color',[.4 .4 .4],'LineStyle','-.')
            hold on
            plot(rbackwards2_upUnstable(:,1),rbackwards2_upUnstable(:,2),'LineWidth',4,'Color','r','LineStyle','-.')
            hold on;
            %plot(rbackwards2_downUnstable(:,1),rbackwards2_downUnstable(:,2),'LineWidth',4,'Color','r','LineStyle','-.')
            hold on
        end
        if F_e(solspecificR4e_iter,solspecificR4i_iter)>=0 && F_i(solspecificR4e_iter,solspecificR4i_iter)>=0
            solspecificR4e(1,idx) = solspecificR4e_iter;
            solspecificR4i(1,idx) = solspecificR4i_iter;
        end

        %Define subplots
    
        %VectorField/PhaseSpace/Nullclines/FixedPoints
        dr_e_t = @(r_e,r_i)((1/tau_e)*(-r_e+activ_output_e(r(idx,3)*wee*r_e-r(idx,4)*wei*r_i+Iext_e-r(idx,5))));
        dr_i_t = @(r_e,r_i)((1/tau_i)*(-r_i+activ_output_i(-r(idx,4)*wii*r_i+r(idx,3)*wie*r_e+Iext_i-r(idx,6))));
        
        quiver(R_E,R_I,dr_e_t(R_E,R_I),dr_i_t(R_E,R_I),'Color',[.7 .7 .7],'LineWidth',2,'DisplayName','VectorField');
        hold on;
        fimplicit(dr_e_t,'LineWidth',4,'Color','r','DisplayName','re-nullcline');
        hold on;
        fimplicit(dr_i_t,'LineWidth',4,'Color','b','DisplayName','ri-nullcline');
        hold on;
        fimplicit(F_e,'LineWidth',4,'Color','k','DisplayName','Fe');
        hold on;
        fimplicit(F_i,'LineWidth',4,'Color','k','DisplayName','Fi');
        hold on
        %NOTE: Change idx for : to plot fixed point trajectory
        plot(solspecificR1e(1,idx),solspecificR1i(1,idx),'*',"MarkerSize",20,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot(solspecificR2e(1,idx),solspecificR2i(1,idx),'*',"MarkerSize",20,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot(solspecificR4e(1,idx),solspecificR4i(1,idx),'*',"MarkerSize",20,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot(r(idx,1),r(idx,2),'.',"MarkerSize",40,'Color','k');
        axis([-50 150 -200 375])
        xlabel('r_e','FontSize',36);
        ylabel('r_i','FontSize',36);
        yticks([-200 -100 0 100 200 300])
        % current axes
        ax = gca; 
        ax.FontSize = 24;
        hold on;

        sgtitle(strcat('f_D = ',num2str(fd)),'FontSize',40)
        drawnow;
        movieVector(floor(idx/video_speed)) = getframe(h);
    

    end
end
%Comment the following lines for not saving the video
myWriter = VideoWriter(strcat('ZoomAnimationfD',num2str(fd)),'MPEG-4')
myWriter.FrameRate = 3
open(myWriter)
writeVideo(myWriter,movieVector)
close(myWriter)
clear movieVector
%% ------------- CHECKING EQUILIBRIUM POINTS EXPRESSION  ------------------

%Initialize vectors to save solution
solspecificR1e = NaN(1,length(t));
solspecificR1i = NaN(1,length(t));
solspecificR2e = NaN(1,length(t));
solspecificR2i = NaN(1,length(t));
solspecificR4e = NaN(1,length(t));
solspecificR4i = NaN(1,length(t));

%Initialize vectors to save eigenvalues
vaps_R1 = NaN(2,length(t));
vaps_R2 = NaN(2,length(t));
vaps_R4 = NaN(2,length(t));

for idx=1:length(t)
    %1. Initialize the function and input-output funftion
    dr = @(rvar)[(1/tau_e)*(-rvar(1)+activ_output_e(r(idx,3)*wee*rvar(1)-r(idx,4)*wei*rvar(2)+Iext_e-r(idx,5)));(1/tau_i)*(-rvar(2)+activ_output_i(-r(idx,4)*wii*rvar(2)+r(idx,3)*wie*rvar(1)+Iext_i-r(idx,6)))];
    F_e = @(r_e,r_i)(r(idx,3)*wee*r_e-r(idx,4)*wei*r_i+Iext_e-r(idx,5)-beta_e);
    F_i = @(r_e,r_i)(-r(idx,4)*wii*r_i+r(idx,3)*wie*r_e+Iext_i-r(idx,6)-beta_i);

    %2. Compute the Jacobian matrix and evaluate at the fixed point
    J = @(rvar_e,rvar_i)[(1/tau_e)*(-1+alpha_e*r(idx,3)*wee*hardlim(r(idx,3)*wee*rvar_e-r(idx,4)*wei*rvar_i+Iext_e-r(idx,5)-beta_e)) (1/tau_e)*(-alpha_e*r(idx,4)*wei*hardlim(r(idx,3)*wee*rvar_e-r(idx,4)*wei*rvar_i+Iext_e-r(idx,5)-beta_e)); (1/tau_i)*(alpha_i*r(idx,3)*wie*hardlim(-r(idx,4)*wii*rvar_i+r(idx,3)*wie*rvar_e+Iext_i-r(idx,6)-beta_i)) (1/tau_i)*(-1-alpha_i*r(idx,4)*wii*hardlim(-r(idx,4)*wii*rvar_i+r(idx,3)*wie*rvar_e+Iext_i-r(idx,6)-beta_i))]; 
    
    %3. Compute the solution in each region
    solspecificR1e_iter = 0;
    solspecificR1i_iter = 0;

    solspecificR2e_iter= (alpha_e*(beta_e-(Iext_e-r(idx,5))))/(alpha_e*wee*r(idx,3)-1);
    solspecificR2i_iter = 0;

    solspecificR4e_iter = (alpha_e*alpha_i*wei*r(idx,4)*(Iext_i-r(idx,6)-beta_i)-alpha_e*(alpha_i*wii*r(idx,4)+1)*(Iext_e-r(idx,5)-beta_e))/((alpha_e*wee*r(idx,3)-1)*(alpha_i*wii*r(idx,4)+1)-alpha_e*alpha_i*wie*r(idx,3)*wei*r(idx,4));
    solspecificR4i_iter = ((alpha_i*wie*r(idx,3))/(alpha_i*wii*r(idx,4)+1))*solspecificR4e_iter+(alpha_i*(Iext_i-r(idx,6)-beta_i))/(alpha_i*wii*r(idx,4)+1);

    %4. Save fixed point in each region
    if F_e(solspecificR1e_iter,solspecificR1i_iter)<=0 && F_i(solspecificR1e_iter,solspecificR1i_iter)<=0
        solspecificR1e(1,idx) = solspecificR1e_iter;
        solspecificR1i(1,idx) = solspecificR1i_iter;
        JsubsR1 = J(solspecificR1e_iter,solspecificR1i_iter);
        [veps,vaps] = eig(JsubsR1);
        vaps_R1(1,idx) = vaps(1,1);
        vaps_R1(2,idx) = vaps(2,2);
    end
    if F_e(solspecificR2e_iter,solspecificR2i_iter)>=0 && F_i(solspecificR2e_iter,solspecificR2i_iter)<=0
        solspecificR2e(1,idx) = solspecificR2e_iter;
        solspecificR2i(1,idx) = solspecificR2i_iter;
        JsubsR2 = J(solspecificR2e_iter,solspecificR2i_iter);
        [veps,vaps] = eig(JsubsR2);
        vaps_R2(1,idx) = vaps(1,1);
        vaps_R2(2,idx) = vaps(2,2);
    end
    if F_e(solspecificR4e_iter,solspecificR4i_iter)>=0 && F_i(solspecificR4e_iter,solspecificR4i_iter)>=0
        solspecificR4e(1,idx) = solspecificR4e_iter;
        solspecificR4i(1,idx) = solspecificR4i_iter;
        JsubsR4 = J(solspecificR4e_iter,solspecificR4i_iter);
        [veps,vaps] = eig(JsubsR4);
        vaps_R4(1,idx) = vaps(1,1);
        vaps_R4(2,idx) = vaps(2,2);
    
    end
end
%% ----------------  PLOTING EQUILIBRIUM POINT EXPRESSION  ----------------
%Equilibrium point R1
h1a=figure()
scatter(t/1000,solspecificR1e(1,:),60,'r','filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,solspecificR1i(1,:),60,'b','filled','o','MarkerFaceAlpha',0.3)
xlabel('Time (s)','FontSize',48);
ylabel('Fixed point R_1','FontSize',48);
% current axes
ax = gca; 
ax.FontSize = 36;
h1a.Position = [100 100 800 400];
saveas(h1a,strcat('.\fD', mat2str(fd),'SolR1.png'),'png')
hold off;

%Equilibrium point R2
h2a=figure()
scatter(t/1000,solspecificR2e(1,:),60,'r','filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,solspecificR2i(1,:),60,'b','filled','o','MarkerFaceAlpha',0.3)
xlabel('Time (s)','FontSize',48);
ylabel('Fixed point R_2','FontSize',48);
% current axes
ax = gca; 
ax.FontSize = 36;
h2a.Position = [100 100 800 400];
saveas(h2a,strcat('.\fD', mat2str(fd),'SolR2.png'),'png')
hold off;

%Equilibrium point R4
h3a=figure()
scatter(t/1000,solspecificR4e(1,:),60,'r','filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,solspecificR4i(1,:),60,'b','filled','o','MarkerFaceAlpha',0.3)
xlabel('Time (s)','FontSize',48);
ylabel('Fixed point R_4','FontSize',48);
ylim([0,250])
% current axes
ax = gca; 
ax.FontSize = 36;
h3a.Position = [100 100 800 400];
saveas(h3a,strcat('.\fD', mat2str(fd),'SolR4.png'),'png')
hold off;

%Eigenvalues equilibrium point each region
heig = figure()
R1color = [0.6350 0.0780 0.1840]; %magenta
R2color = [0.9290 0.6940 0.1250]; %groc
R4color = [0.4940 0.1840 0.5560]; %lila
scatter(t/1000,vaps_R1(1,:),60,R1color,'filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,vaps_R1(2,:),60,R1color,'filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,vaps_R2(1,:),60,R2color,'filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,vaps_R2(2,:),60,R2color,'filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,real(vaps_R4(1,:)),60,R4color,'filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,real(vaps_R4(2,:)),60,R4color,'filled','o','MarkerFaceAlpha',0.3)
hold on;
yline(0,'Linewidth',4,'Color',[.3 .3 .3],'LineStyle','-.','Alpha',0.8)
hold off;
xlabel('Time (s)','FontSize',48);
ylabel('R(eigenvalue)','FontSize',48);
% current axes
ax = gca; 
ax.FontSize = 36;
xlim([10,20])
heig.Position = [100 100 800 400];
saveas(heig,strcat('.\fD', mat2str(fd), 'Eigenvalues.png'),'png')
hold off;
%% -----------------  BIFURCATION DIAGRAM ADAPTATION MODIFIED -------------
s_e = 1;
s_i = 1;

%Define adaptation vectors
adapt_e_vector = 0:650;
adapt_i_vector = 0:650;

%Initialize matrix to save solution for each adaptation pair (ae,ai)
solR1e = NaN(length(adapt_e_vector),length(adapt_i_vector));
solR1i = NaN(length(adapt_e_vector),length(adapt_i_vector));
solR2e = NaN(length(adapt_e_vector),length(adapt_i_vector));
solR2i = NaN(length(adapt_e_vector),length(adapt_i_vector));
solR4e = NaN(length(adapt_e_vector),length(adapt_i_vector));
solR4i = NaN(length(adapt_e_vector),length(adapt_i_vector));

for adapt_e_iter=1:length(adapt_e_vector)
    disp(['Adaptation_e: ', mat2str(adapt_e_iter)])
    for adapt_i_iter=1:length(adapt_i_vector)
        adapt_e = adapt_e_vector(adapt_e_iter);
        adapt_i = adapt_i_vector(adapt_i_iter);
        %1. Initialize the function and input-output funftion
        dr = @(rvar)[(1/tau_e)*(-rvar(1)+activ_output_e(s_e*wee*rvar(1)-s_i*wei*rvar(2)+Iext_e-adapt_e));(1/tau_i)*(-rvar(2)+activ_output_i(-s_i*wii*rvar(2)+s_e*wie*rvar(1)+Iext_i-adapt_i))];
        F_e = @(r_e,r_i)(s_e*wee*r_e-s_i*wei*r_i+Iext_e-adapt_e-beta_e);
        F_i = @(r_e,r_i)(-s_i*wii*r_i+s_e*wie*r_e+Iext_i-adapt_i-beta_i);

        %2. Compute the Jacobian matrix and evaluate at the equilibrium point
        J = @(rvar_e,rvar_i)[(1/tau_e)*(-1+alpha_e*r(idx,3)*wee*hardlim(r(idx,3)*wee*rvar_e-r(idx,4)*wei*rvar_i+Iext_e-r(idx,5)-beta_e)) (1/tau_e)*(-alpha_e*r(idx,4)*wei*hardlim(r(idx,3)*wee*rvar_e-r(idx,4)*wei*rvar_i+Iext_e-r(idx,5)-beta_e)); (1/tau_i)*(alpha_i*r(idx,3)*wie*hardlim(-r(idx,4)*wii*rvar_i+r(idx,3)*wie*rvar_e+Iext_i-r(idx,6)-beta_i)) (1/tau_i)*(-1-alpha_i*r(idx,4)*wii*hardlim(-r(idx,4)*wii*rvar_i+r(idx,3)*wie*rvar_e+Iext_i-r(idx,6)-beta_i))]; 

        solR1e_iter = 0;
        solR1i_iter = 0;
        solR2e_iter= (alpha_e*(beta_e-(Iext_e-adapt_e)))/(alpha_e*wee*s_e-1);
        solR2i_iter = 0;
        solR4e_iter = (alpha_e*alpha_i*wei*s_e*(Iext_i-adapt_i-beta_i)-alpha_e*(alpha_i*wii*s_i+1)*(Iext_e-adapt_e-beta_e))/((alpha_e*wee*s_e-1)*(alpha_i*wii*s_i+1)-alpha_e*alpha_i*wie*s_e*wei*s_i);
        solR4i_iter = ((alpha_i*wie*s_e)/(alpha_i*wii*s_i+1))*solR4e_iter+(alpha_i*(Iext_i-adapt_i-beta_i))/(alpha_i*wii*s_i+1);

        %4. Save equilibrium point in each region
        if F_e(solR1e_iter,solR1i_iter)<=0 && F_i(solR1e_iter,solR1i_iter)<=0
            solR1e(adapt_e_iter,adapt_i_iter) = solR1e_iter;
            solR1i(adapt_e_iter,adapt_i_iter) = solR1i_iter;
        end
        if F_e(solR2e_iter,solR2i_iter)>=0 && F_i(solR2e_iter,solR2i_iter)<=0
            solR2e(adapt_e_iter,adapt_i_iter) = solR2e_iter;
            solR2i(adapt_e_iter,adapt_i_iter) = solR2i_iter;
        end
        if F_e(solR4e_iter,solR4i_iter)>=0 && F_i(solR4e_iter,solR4i_iter)>=0
            solR4e(adapt_e_iter,adapt_i_iter) = solR4e_iter;
            solR4i(adapt_e_iter,adapt_i_iter) = solR4i_iter;
        end

    end
end

%--------------  PLOTS  ----------------
%Rename the vectors
re_traj = r(:,1);
ri_traj = r(:,2);

%Define colors for each region
r1_color = "#A2142F";
r2_color = "#EDB120";
r4_color = "#7E2F8E";

color3Dshort = [1:length(transpose(r(:,6)))]/length(transpose(r(:,6)));

h1a=figure()
mesh(0:650,0:650,solR1e,'EdgeColor',r1_color,'FaceColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on
mesh(0:650,0:650,solR2e,'EdgeColor',r2_color,'FaceColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on
mesh(0:650,0:650,solR4e,'EdgeColor',r4_color,'FaceColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on;
set(gca,'ZScale','log')
ax = gca; 
ax.FontSize = 20;
xlabel('a_i','FontSize',24);
ylabel('a_e','FontSize',24);
zlabel('r_e^*','FontSize',24);
xticks([0 100 200 300 400 500 600])
yticks([0 100 200 300 400 500 600])
zticks([10^(-5) 10^(-2) 10^0 10^2])
xlim([0,length(adapt_e_vector)])
ylim([0,length(adapt_i_vector)])
zlim([10^(-5), 10^3])
patch([transpose(r(:,6)) nan],[transpose(r(:,5)) nan],[transpose(re_traj) nan],[color3Dshort nan],'Linewidth',6,'EdgeColor','interp','EdgeAlpha',0.8,'FaceColor','none')
cmap = colormap(bone);
cbar = colorbar();
cbar.Ticks = 0:0.1:1;
cbar.TickLabels = num2cell(0:1:10);
caxis([0, 1]);
hold off;

h1b=figure()
mesh(0:650,0:650,solR1e,'EdgeColor',r1_color,'FaceColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.5)
hold on
mesh(0:650,0:650,solR2e,'EdgeColor',r2_color,'FaceColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.5)
hold on
mesh(0:650,0:650,solR4e,'EdgeColor',r4_color,'FaceColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.5)
ax = gca; 
ax.FontSize = 20;
xlabel('a_i','FontSize',24);
ylabel('a_e','FontSize',24);
zlabel('r_e^*','FontSize',24);
xticks([0 100 200 300 400 500 600])
yticks([0 100 200 300 400 500 600])
xlim([0,length(adapt_e_vector)])
ylim([0,length(adapt_i_vector)])
zlim([0,200])
patch([transpose(r(:,6)) nan],[transpose(r(:,5)) nan],[transpose(re_traj) nan],[color3Dshort nan],'Linewidth',6,'EdgeColor','interp','EdgeAlpha',0.8,'FaceColor','none')
view(3);
cmap = colormap(bone);
cbar = colorbar();
cbar.Ticks = 0:0.1:1;
cbar.TickLabels = num2cell(0:1:10);
caxis([0, 1]);
hold off;


h2a=figure()
mesh(0:650,0:650,solR1i,'EdgeColor',r1_color,'FaceColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on
mesh(0:650,0:650,solR2i,'EdgeColor',r2_color,'FaceColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on
mesh(0:650,0:650,solR4i,'EdgeColor',r4_color,'FaceColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on;
set(gca,'ZScale','log')
ax = gca; 
ax.FontSize = 20;
xlabel('a_i','FontSize',24);
ylabel('a_e','FontSize',24);
zlabel('r_i^*','FontSize',24);
xticks([0 100 200 300 400 500 600])
yticks([0 100 200 300 400 500 600])
zticks([10^(-5) 10^(-2) 10^0 10^2])
xlim([0,length(adapt_e_vector)])
ylim([0,length(adapt_i_vector)])
zlim([10^(-5),10^3])
patch([transpose(r(:,6)) nan],[transpose(r(:,5)) nan],[transpose(ri_traj) nan],[color3Dshort nan],'Linewidth',6,'EdgeColor','interp','EdgeAlpha',0.8,'FaceColor','none')
view(3);
cmap = colormap(bone);
cbar = colorbar();
cbar.Ticks = 0:0.1:1;
cbar.TickLabels = num2cell(0:1:10);
caxis([0, 1]);
hold off;

h2b=figure()
ax = gca;
mesh(ax,0:650,0:650,solR1i,'EdgeColor',r1_color,'FaceColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.5)
hold on
mesh(ax,0:650,0:650,solR2i,'EdgeColor',r2_color,'FaceColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.5)
hold on
mesh(ax,0:650,0:650,solR4i,'EdgeColor',r4_color,'FaceColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.5)
hold on;
ax.FontSize = 20;
xlabel(ax,'a_i','FontSize',24);
ylabel(ax,'a_e','FontSize',24);
zlabel(ax,'r_i^*','FontSize',24);
xticks(ax,[0 100 200 300 400 500 600])
yticks(ax,[0 100 200 300 400 500 600])
xlim(ax,[0,length(adapt_e_vector)])
ylim(ax,[0,length(adapt_i_vector)])
zlim([0,500])
patch([transpose(r(:,6)) nan],[transpose(r(:,5)) nan],[transpose(ri_traj) nan],[color3Dshort nan],'Linewidth',6,'EdgeColor','interp','EdgeAlpha',0.8,'FaceColor','none')
hold on;
view(3);
cmap = colormap(bone);
cbar = colorbar();
cbar.Ticks = 0:0.1:1;
cbar.TickLabels = num2cell(0:1:10);
caxis([0, 1]);
hold off;
%% -----------------  3D-DEPRESSION FIRING RATE VIDEO  --------------------
%Define color in each region
r1_color = "#A2142F";
r2_color = "#EDB120";
r4_color = "#7E2F8E";

%Define values range for se,si
spacing = 10;
[R_E,R_I] = meshgrid(-200:spacing:375);
se_vector = 0.4:0.01:1;
si_vector = 0.4:0.01:1;
zero_plane = zeros(length(si_vector),length(se_vector));

h=figure()
set(h,'Position',[100 100 2000 1500])

for idx=1:length(t)
    clf
    if mod(idx,1)==0
        %Take current a_e,a_i values
        ae_current = r(idx,5);
        ai_current = r(idx,6);

        %Initialize solutions vectors
        solvideoR1e = NaN(length(si_vector),length(se_vector));
        solvideoR1i = NaN(length(si_vector),length(se_vector));
        solvideoR2e = NaN(length(si_vector),length(se_vector));
        solvideoR2i = NaN(length(si_vector),length(se_vector));
        solvideoR4e = NaN(length(si_vector),length(se_vector));
        solvideoR4i = NaN(length(si_vector),length(se_vector));
        %Initialize eigenvalues vectors
        vaps_R4_first = NaN(length(si_vector),length(se_vector));
        vaps_R4_second = NaN(length(si_vector),length(se_vector));

        for se_value=1:length(se_vector)
            se_current = se_vector(se_value);
            for si_value=1:length(si_vector)
                si_current = si_vector(si_value);

                %Define region functions
                F_e = @(r_e,r_i)(se_current*wee*r_e-si_current*wei*r_i+Iext_e-r(idx,5)-beta_e);
                F_i = @(r_e,r_i)(-si_current*wii*r_i+se_current*wie*r_e+Iext_i-r(idx,6)-beta_i);
                J = @(rvar_e,rvar_i)[(1/tau_e)*(-1+alpha_e*se_current*wee*hardlim(se_current*wee*rvar_e-si_current*wei*rvar_i+Iext_e-r(idx,5)-beta_e)) (1/tau_e)*(-alpha_e*si_current*wei*hardlim(se_current*wee*rvar_e-si_current*wei*rvar_i+Iext_e-r(idx,5)-beta_e)); (1/tau_i)*(alpha_i*se_current*wie*hardlim(-si_current*wii*rvar_i+se_current*wie*rvar_e+Iext_i-r(idx,6)-beta_i)) (1/tau_i)*(-1-alpha_i*si_current*wii*hardlim(-si_current*wii*rvar_i+se_current*wie*rvar_e+Iext_i-r(idx,6)-beta_i))]; 

                %1. Compute the solution in each region
                solspecificR1e_iter = 0;
                solspecificR1i_iter = 0;
        
                solspecificR2e_iter= (alpha_e*(beta_e-(Iext_e-r(idx,5))))/(alpha_e*wee*se_current-1);
                solspecificR2i_iter = 0;
            
                solspecificR4e_iter = (alpha_e*alpha_i*wei*si_current*(Iext_i-r(idx,6)-beta_i)-alpha_e*(alpha_i*wii*si_current+1)*(Iext_e-r(idx,5)-beta_e))/((alpha_e*wee*se_current-1)*(alpha_i*wii*si_current+1)-alpha_e*alpha_i*wie*se_current*wei*si_current);
                solspecificR4i_iter = ((alpha_i*wie*se_current)/(alpha_i*wii*si_current+1))*solspecificR4e_iter+(alpha_i*(Iext_i-r(idx,6)-beta_i))/(alpha_i*wii*si_current+1);
                %2. Save fixed point in each region
                if F_e(solspecificR1e_iter,solspecificR1i_iter)<=0 && F_i(solspecificR1e_iter,solspecificR1i_iter)<=0
                    solvideoR1e(si_value,se_value) = solspecificR1e_iter;
                    solvideoR1i(si_value,se_value) = solspecificR1i_iter;
                end
                if F_e(solspecificR2e_iter,solspecificR2i_iter)>=0 && F_i(solspecificR2e_iter,solspecificR2i_iter)<=0
                    solvideoR2e(si_value,se_value) = solspecificR2e_iter;
                    solvideoR2i(si_value,se_value) = solspecificR2i_iter;
                end
                if F_e(solspecificR4e_iter,solspecificR4i_iter)>=0 && F_i(solspecificR4e_iter,solspecificR4i_iter)>=0
                    solvideoR4e(si_value,se_value) = solspecificR4e_iter;
                    solvideoR4i(si_value,se_value) = solspecificR4i_iter;
                    JsubsR4 = J(solspecificR4e_iter,solspecificR4i_iter);
                    [veps,vaps] = eig(JsubsR4);
                    vaps_R4_first(si_value,se_value) = vaps(1,1);
                    vaps_R4_second(si_value,se_value) = vaps(2,2);
                end

            end
        end

        subplot(2,2,1);
        mesh(si_vector,se_vector,solvideoR1e,'FaceColor',r1_color,'EdgeColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,solvideoR2e,'FaceColor',r2_color,'EdgeColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,solvideoR4e,'FaceColor',r4_color,'EdgeColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on;
        ax = gca; 
        ax.FontSize = 20;
        xlabel('s_i','FontSize',24);
        ylabel('s_e','FontSize',24);
        zlabel('r_e^*','FontSize',24);
        title('Equilibrium point r_e^*','FontSize',18);
        xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        zticks([0 50 100 150 200 ])
        xlim([0.4,1])
        ylim([0.4,1])
        zlim([0, 150])
        hold on;
        plot3(r(1:idx,3),r(1:idx,4),r(1:idx,1),'Linewidth',8,'Color',[.4 .4 .4 .3])
        hold on
        scatter3(r(idx,3),r(idx,4),r(idx,1),100,'k','filled','o')
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR1e(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR2e(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR4e(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        set(gca,'view',[120 20])
        % current axes
        ax = gca; 
        ax.FontSize = 24;
        hold on;


        subplot(2,2,2);
        mesh(si_vector,se_vector,solvideoR1i,'EdgeColor',r1_color,'FaceColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,solvideoR2i,'EdgeColor',r2_color,'FaceColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,solvideoR4i,'EdgeColor',r4_color,'FaceColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on;
        ax = gca; 
        ax.FontSize = 20;
        xlabel('s_e','FontSize',24);
        ylabel('s_i','FontSize',24);
        zlabel('r_i^*','FontSize',24);
        title('Equilibrium point r_i^*','FontSize',18);
        xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        zticks([0 50 100 150 200 250 300])
        xlim([0.4,1])
        ylim([0.4,1])
        zlim([0, 300])
        hold on;
        plot3(r(1:idx,3),r(1:idx,4),r(1:idx,2),'Linewidth',8,'Color',[.4 .4 .4 .3])
        hold on;
        scatter3(r(idx,3),r(idx,4),r(idx,2),100,'k','filled','o')
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR1i(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR2i(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR4i(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        set(gca,'view',[150 20])
        % current axes
        ax = gca; 
        ax.FontSize = 24;
        hold on;


        subplot(2,2,3)
        mesh(si_vector,se_vector,real(vaps_R4_first),'EdgeColor',r4_color,'FaceColor',r4_color,'FaceAlpha',0.3,'EdgeAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,real(vaps_R4_second),'EdgeColor',r4_color,'FaceColor',r4_color,'FaceAlpha',0.3,'EdgeAlpha',0.3)
        hold on;
        mesh(si_vector,se_vector,zero_plane,'EdgeColor',[.4 .4 .4],'EdgeAlpha',0.3,'FaceColor',[.4 .4 .4],'FaceAlpha',0.3)
        hold on;
        ax = gca; 
        ax.FontSize = 20;
        xlabel('s_e','FontSize',24);
        ylabel('s_i','FontSize',24);
        zlabel('R(eigenvalue)','FontSize',24);
        title('Eigenvalues R_4','FontSize',18);
        xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        xlim([0.4,1])
        ylim([0.4,1])
        zlim([-2*10^(-3), 5*10^(-3)])
        hold on;
        scatter3(r(idx,3),r(idx,4),real(vaps_R4(1,idx)),100,'k','filled','o')
        hold on;
        scatter3(r(idx,3),r(idx,4),real(vaps_R4(2,idx)),100,'k','filled','o')
        hold on;
        set(gca,'view',[170 10])
        % current axes
        ax = gca; 
        ax.FontSize = 24;
        hold on;

        %Subplot (2,2) --> FiringRate
        subplot(2,2,4);
        %show time dependency r_e, r_i
        %excitatory
        re_idx = nan(1,length(t));
        re_idx(1:idx) = r(1:idx,1);
        re_idx(end) = r(end,1);
        %inhibitory
        ri_idx = nan(1,length(t));
        ri_idx(1:idx) = r(1:idx,2);
        ri_idx(end) = r(end,2);
        plot(t/1000,re_idx,'-',"LineWidth",4,'Color','r');
        hold on;
        plot(t/1000,ri_idx,'-',"LineWidth",4,'Color','b');
        hold on;
        title('Firing rate','FontSize',18);
        xlabel('Time t (s)','FontSize',24);
        ylabel('Solution r(t)','FontSize',24);
    
        % current axes
        ylim([-10 max(max(r(:,1)),max(r(:,2)))+10]);
        ax = gca; 
        ax.FontSize = 24;
        sgtitle(strcat('f_D = ',num2str(fd)),'FontSize',40)
        drawnow;
        movieVector(floor(idx/1)) = getframe(h);
    end
end
hold off;
myWriter = VideoWriter(strcat('DepressionBifurcationFiringRate',num2str(fd)),'MPEG-4')
myWriter.FrameRate = 2
open(myWriter)
writeVideo(myWriter,movieVector)
close(myWriter)
clear movieVector
%% -----------------  3D-DEPRESSION PHASE SPACE VIDEO  --------------------
r1_color = "#A2142F";
r2_color = "#EDB120";
r4_color = "#7E2F8E";

spacing = 10;
[R_E,R_I] = meshgrid(-200:spacing:375);
se_vector = 0.4:0.01:1;
si_vector = 0.4:0.01:1;
zero_plane = zeros(length(si_vector),length(se_vector));

h=figure()
set(h,'Position',[100 100 2000 1500])

for idx=1:length(t)
    clf
    if mod(idx,video_speed)==0
        %Initialize solutions vectors
        solvideoR1e = NaN(length(si_vector),length(se_vector));
        solvideoR1i = NaN(length(si_vector),length(se_vector));
        solvideoR2e = NaN(length(si_vector),length(se_vector));
        solvideoR2i = NaN(length(si_vector),length(se_vector));
        solvideoR4e = NaN(length(si_vector),length(se_vector));
        solvideoR4i = NaN(length(si_vector),length(se_vector));
        %Initialize eigenvalues vectors
        vaps_R4_first = NaN(length(si_vector),length(se_vector));
        vaps_R4_second = NaN(length(si_vector),length(se_vector));

        dr_e_t = @(r_e,r_i)((1/tau_e)*(-r_e+activ_output_e(r(idx,3)*wee*r_e-r(idx,4)*wei*r_i+Iext_e-r(idx,5))));
        dr_i_t = @(r_e,r_i)((1/tau_i)*(-r_i+activ_output_i(-r(idx,4)*wii*r_i+r(idx,3)*wie*r_e+Iext_i-r(idx,6))));
        F_e_nullcline = @(r_e,r_i)(r(idx,3)*wee*r_e-r(idx,4)*wei*r_i+Iext_e-r(idx,5)-beta_e);
        F_i_nullcline = @(r_e,r_i)(-r(idx,4)*wii*r_i+r(idx,3)*wie*r_e+Iext_i-r(idx,6)-beta_i);

        for se_value=1:length(se_vector)
            se_current = se_vector(se_value);
            for si_value=1:length(si_vector)
                si_current = si_vector(si_value);

                %Define region functions
                F_e = @(r_e,r_i)(se_current*wee*r_e-si_current*wei*r_i+Iext_e-r(idx,5)-beta_e);
                F_i = @(r_e,r_i)(-si_current*wii*r_i+se_current*wie*r_e+Iext_i-r(idx,6)-beta_i);
                J = @(rvar_e,rvar_i)[(1/tau_e)*(-1+alpha_e*se_current*wee*hardlim(se_current*wee*rvar_e-si_current*wei*rvar_i+Iext_e-r(idx,5)-beta_e)) (1/tau_e)*(-alpha_e*si_current*wei*hardlim(se_current*wee*rvar_e-si_current*wei*rvar_i+Iext_e-r(idx,5)-beta_e)); (1/tau_i)*(alpha_i*se_current*wie*hardlim(-si_current*wii*rvar_i+se_current*wie*rvar_e+Iext_i-r(idx,6)-beta_i)) (1/tau_i)*(-1-alpha_i*si_current*wii*hardlim(-si_current*wii*rvar_i+se_current*wie*rvar_e+Iext_i-r(idx,6)-beta_i))]; 

                %1. Compute the solution in each region
                solspecificR1e_iter = 0;
                solspecificR1i_iter = 0;
        
                solspecificR2e_iter= (alpha_e*(beta_e-(Iext_e-r(idx,5))))/(alpha_e*wee*se_current-1);
                solspecificR2i_iter = 0;
            
                solspecificR4e_iter = (alpha_e*alpha_i*wei*si_current*(Iext_i-r(idx,6)-beta_i)-alpha_e*(alpha_i*wii*si_current+1)*(Iext_e-r(idx,5)-beta_e))/((alpha_e*wee*se_current-1)*(alpha_i*wii*si_current+1)-alpha_e*alpha_i*wie*se_current*wei*si_current);
                solspecificR4i_iter = ((alpha_i*wie*se_current)/(alpha_i*wii*si_current+1))*solspecificR4e_iter+(alpha_i*(Iext_i-r(idx,6)-beta_i))/(alpha_i*wii*si_current+1);
                %2. Save fixed point in each region
                if F_e(solspecificR1e_iter,solspecificR1i_iter)<=0 && F_i(solspecificR1e_iter,solspecificR1i_iter)<=0
                    solvideoR1e(si_value,se_value) = solspecificR1e_iter;
                    solvideoR1i(si_value,se_value) = solspecificR1i_iter;
                end
                if F_e(solspecificR2e_iter,solspecificR2i_iter)>=0 && F_i(solspecificR2e_iter,solspecificR2i_iter)<=0
                    solvideoR2e(si_value,se_value) = solspecificR2e_iter;
                    solvideoR2i(si_value,se_value) = solspecificR2i_iter;
                end
                if F_e(solspecificR4e_iter,solspecificR4i_iter)>=0 && F_i(solspecificR4e_iter,solspecificR4i_iter)>=0
                    solvideoR4e(si_value,se_value) = solspecificR4e_iter;
                    solvideoR4i(si_value,se_value) = solspecificR4i_iter;
                    JsubsR4 = J(solspecificR4e_iter,solspecificR4i_iter);
                    [veps,vaps] = eig(JsubsR4);
                    vaps_R4_first(si_value,se_value) = vaps(1,1);
                    vaps_R4_second(si_value,se_value) = vaps(2,2);
                end

            end
        end

        subplot(2,2,1);
        mesh(si_vector,se_vector,solvideoR1e,'FaceColor',r1_color,'EdgeColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,solvideoR2e,'FaceColor',r2_color,'EdgeColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,solvideoR4e,'FaceColor',r4_color,'EdgeColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on;
        ax = gca; 
        ax.FontSize = 20;
        xlabel('s_e','FontSize',24);
        ylabel('s_i','FontSize',24);
        zlabel('r_e^*','FontSize',24);
        title('Equilibrium point r_e^*','FontSize',18);
        xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        zticks([0 50 100 150 200 ])
        xlim([0.4,1])
        ylim([0.4,1])
        zlim([0, 150])
        hold on;
        plot3(r(1:idx,3),r(1:idx,4),r(1:idx,1),'Linewidth',8,'Color',[.4 .4 .4 .3])
        hold on
        scatter3(r(idx,3),r(idx,4),r(idx,1),100,'k','filled','o')
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR1e(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR2e(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR4e(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        set(gca,'view',[120 20])
        % current axes
        ax = gca; 
        ax.FontSize = 24;
        hold on;


        subplot(2,2,2);
        mesh(si_vector,se_vector,solvideoR1i,'EdgeColor',r1_color,'FaceColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,solvideoR2i,'EdgeColor',r2_color,'FaceColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,solvideoR4i,'EdgeColor',r4_color,'FaceColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
        hold on;
        ax = gca; 
        ax.FontSize = 20;
        xlabel('s_e','FontSize',24);
        ylabel('s_i','FontSize',24);
        zlabel('r_i^*','FontSize',24);
        title('Equilibrium point r_i^*','FontSize',18);
        xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        zticks([0 50 100 150 200 250 300])
        xlim([0.4,1])
        ylim([0.4,1])
        zlim([0, 300])
        hold on;
        plot3(r(1:idx,3),r(1:idx,4),r(1:idx,2),'Linewidth',8,'Color',[.4 .4 .4 .3])
        hold on;
        scatter3(r(idx,3),r(idx,4),r(idx,2),100,'k','filled','o')
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR1i(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR2i(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot3(r(idx,3),r(idx,4),solspecificR4i(idx),'*',"MarkerSize",15,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        set(gca,'view',[150 20])
        % current axes
        ax = gca; 
        ax.FontSize = 24;
        hold on;


        subplot(2,2,3)
        mesh(si_vector,se_vector,real(vaps_R4_first),'EdgeColor',r4_color,'FaceColor',r4_color,'FaceAlpha',0.3,'EdgeAlpha',0.3)
        hold on
        mesh(si_vector,se_vector,real(vaps_R4_second),'EdgeColor',r4_color,'FaceColor',r4_color,'FaceAlpha',0.3,'EdgeAlpha',0.3)
        hold on;
        mesh(si_vector,se_vector,zero_plane,'EdgeColor',[.4 .4 .4],'EdgeAlpha',0.3,'FaceColor',[.4 .4 .4],'FaceAlpha',0.3)
        ax = gca; 
        ax.FontSize = 20;
        xlabel('s_e','FontSize',24);
        ylabel('s_i','FontSize',24);
        zlabel('R(eigenvalue)','FontSize',24);
        title('Eigenvalues R_4','FontSize',18);
        xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
        xlim([0.4,1])
        ylim([0.4,1])
        zlim([-2*10^(-3), 5*10^(-3)])
        hold on;
        scatter3(r(idx,3),r(idx,4),real(vaps_R4(1,idx)),100,'k','filled','o')
        hold on;
        scatter3(r(idx,3),r(idx,4),real(vaps_R4(2,idx)),100,'k','filled','o')
        hold on;
        set(gca,'view',[170 10])
        % current axes
        ax = gca; 
        ax.FontSize = 24;
        hold on;

        %Subplot (2,2) --> PhaseSpace
        subplot(2,2,4);
        quiver(R_E,R_I,dr_e_t(R_E,R_I),dr_i_t(R_E,R_I),'Color',[.7 .7 .7],'LineWidth',2,'DisplayName','VectorField');
        hold on;
        fimplicit(dr_e_t,'LineWidth',4,'Color','r','DisplayName','re-nullcline');
        hold on;
        fimplicit(dr_i_t,'LineWidth',4,'Color','b','DisplayName','ri-nullcline');
        hold on;
        fimplicit(F_e_nullcline,'LineWidth',4,'Color','k','DisplayName','Fe');
        hold on;
        fimplicit(F_i_nullcline,'LineWidth',4,'Color','k','DisplayName','Fi');
        hold on;
        %NOTE: Change idx for : to plot fixed point trajectory
        plot(solspecificR1e(idx),solspecificR1i(idx),'*',"MarkerSize",20,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot(solspecificR2e(idx),solspecificR2i(idx),'*',"MarkerSize",20,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot(solspecificR4e(idx),solspecificR4i(idx),'*',"MarkerSize",20,'Color',[.4 .4 .4],'LineWidth',4);
        hold on;
        plot(r(idx,1),r(idx,2),'.',"MarkerSize",40,'Color','k');
        axis([-50 150 -200 375])
        xlabel('r_e','FontSize',36);
        ylabel('r_i','FontSize',36);
        yticks([-200 -100 0 100 200 300])
        title('Vector field and nullclines','FontSize',18);
        % current axes
        ax = gca; 
        ax.FontSize = 24;
        hold on;
    
        sgtitle(strcat('f_D = ',num2str(fd)),'FontSize',40)
        drawnow;
        movieVector(floor(idx/video_speed)) = getframe(h);
    end
end
hold off;
myWriter = VideoWriter(strcat('DepressionBifurcationPhaseSpace',num2str(fd)),'MPEG-4')
myWriter.FrameRate = 2
open(myWriter)
writeVideo(myWriter,movieVector)
close(myWriter)
clear movieVector
%% -----------------  3D-DEPRESSION BIFURCATION IMAGE  --------------------
%Select colors in each region
r1_color = "#A2142F";
r2_color = "#EDB120";
r4_color = "#7E2F8E";

%Select se,si values range
spacing = 10;
[R_E,R_I] = meshgrid(-200:spacing:375);
se_vector = 0.4:0.01:1;
si_vector = 0.4:0.01:1;
zero_plane = zeros(length(si_vector),length(se_vector));
color3Dshort = [1:length(transpose(r(:,1)))]/length(transpose(r(:,1)));

%Take current a_e,a_i values
ae_current = r(50,5);
ai_current = r(50,6);

%Initialize solutions vectors
solvideoR1e = NaN(length(si_vector),length(se_vector));
solvideoR1i = NaN(length(si_vector),length(se_vector));
solvideoR2e = NaN(length(si_vector),length(se_vector));
solvideoR2i = NaN(length(si_vector),length(se_vector));
solvideoR4e = NaN(length(si_vector),length(se_vector));
solvideoR4i = NaN(length(si_vector),length(se_vector));
%Initialize eigenvalues vectors
vaps_R4_first = NaN(length(si_vector),length(se_vector));
vaps_R4_second = NaN(length(si_vector),length(se_vector));

for se_value=1:length(se_vector)
    se_current = se_vector(se_value);
    for si_value=1:length(si_vector)
        si_current = si_vector(si_value);

        %Define region functions
        F_e = @(r_e,r_i)(se_current*wee*r_e-si_current*wei*r_i+Iext_e-ae_current-beta_e);
        F_i = @(r_e,r_i)(-si_current*wii*r_i+se_current*wie*r_e+Iext_i-ai_current-beta_i);
        J = @(rvar_e,rvar_i)[(1/tau_e)*(-1+alpha_e*se_current*wee*hardlim(se_current*wee*rvar_e-si_current*wei*rvar_i+Iext_e-ae_current-beta_e)) (1/tau_e)*(-alpha_e*si_current*wei*hardlim(se_current*wee*rvar_e-si_current*wei*rvar_i+Iext_e-ae_current-beta_e)); (1/tau_i)*(alpha_i*se_current*wie*hardlim(-si_current*wii*rvar_i+se_current*wie*rvar_e+Iext_i-ai_current-beta_i)) (1/tau_i)*(-1-alpha_i*si_current*wii*hardlim(-si_current*wii*rvar_i+se_current*wie*rvar_e+Iext_i-ai_current-beta_i))]; 

        %1. Compute the solution in each region
        solspecificR1e_iter = 0;%0.1909;
        solspecificR1i_iter = 0;%0.0442804;

        solspecificR2e_iter= (alpha_e*(beta_e-(Iext_e-ae_current)))/(alpha_e*wee*se_current-1);
        solspecificR2i_iter = 0;%0.0442804;
    
        solspecificR4e_iter = (alpha_e*alpha_i*wei*si_current*(Iext_i-ai_current-beta_i)-alpha_e*(alpha_i*wii*si_current+1)*(Iext_e-ae_current-beta_e))/((alpha_e*wee*se_current-1)*(alpha_i*wii*si_current+1)-alpha_e*alpha_i*wie*se_current*wei*si_current);
        solspecificR4i_iter = ((alpha_i*wie*se_current)/(alpha_i*wii*si_current+1))*solspecificR4e_iter+(alpha_i*(Iext_i-ai_current-beta_i))/(alpha_i*wii*si_current+1);
        %2. Save fixed point in each region
        if F_e(solspecificR1e_iter,solspecificR1i_iter)<=0 && F_i(solspecificR1e_iter,solspecificR1i_iter)<=0
            solvideoR1e(si_value,se_value) = solspecificR1e_iter;
            solvideoR1i(si_value,se_value) = solspecificR1i_iter;
        end
        if F_e(solspecificR2e_iter,solspecificR2i_iter)>=0 && F_i(solspecificR2e_iter,solspecificR2i_iter)<=0
            solvideoR2e(si_value,se_value) = solspecificR2e_iter;
            solvideoR2i(si_value,se_value) = solspecificR2i_iter;
        end
        if F_e(solspecificR4e_iter,solspecificR4i_iter)>=0 && F_i(solspecificR4e_iter,solspecificR4i_iter)>=0
            solvideoR4e(si_value,se_value) = solspecificR4e_iter;
            solvideoR4i(si_value,se_value) = solspecificR4i_iter;
            JsubsR4 = J(solspecificR4e_iter,solspecificR4i_iter);
            [veps,vaps] = eig(JsubsR4);
            vaps_R4_first(si_value,se_value) = vaps(1,1);
            vaps_R4_second(si_value,se_value) = vaps(2,2);
        end

    end
end

h=figure()
%set(h,'Position',[100 100 2000 1500])
mesh(si_vector,se_vector,solvideoR1e,'FaceColor',r1_color,'EdgeColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on
mesh(si_vector,se_vector,solvideoR2e,'FaceColor',r2_color,'EdgeColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on
mesh(si_vector,se_vector,solvideoR4e,'FaceColor',r4_color,'EdgeColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on;
ax = gca; 
ax.FontSize = 20;
xlabel('s_e','FontSize',24);
ylabel('s_i','FontSize',24);
zlabel('r_e^*','FontSize',24);
title('Equilibrium point r_e^*','FontSize',18);
xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
zticks([0 50 100 150 200 ])
%xlim([0,1])
%ylim([0,1])
xlim([0.4,1])
ylim([0.4,1])
zlim([0, 150])
hold on;
patch([transpose(r(:,3)) nan],[transpose(r(:,4)) nan],[transpose(r(:,1)) nan],[color3Dshort nan],'Linewidth',6,'EdgeColor','interp','EdgeAlpha',0.8,'FaceColor','none')
hold on;
view(3);
cmap = colormap(bone);
cbar = colorbar();
cbar.Ticks = 0:0.1:1;
cbar.TickLabels = num2cell(0:1:10);
caxis([0, 1]);
set(gca,'view',[120 20])
% current axes
ax = gca; 
ax.FontSize = 24;
hold off;


h=figure()
%set(h,'Position',[100 100 2000 1500])
mesh(si_vector,se_vector,solvideoR1i,'EdgeColor',r1_color,'FaceColor',r1_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on
mesh(si_vector,se_vector,solvideoR2i,'EdgeColor',r2_color,'FaceColor',r2_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on
mesh(si_vector,se_vector,solvideoR4i,'EdgeColor',r4_color,'FaceColor',r4_color,'EdgeAlpha',0.3,'FaceAlpha',0.3)
hold on;
ax = gca; 
ax.FontSize = 20;
xlabel('s_e','FontSize',24);
ylabel('s_i','FontSize',24);
zlabel('r_i^*','FontSize',24);
title('Equilibrium point r_i^*','FontSize',18);
xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
zticks([0 50 100 150 200 250 300])
xlim([0.4,1])
ylim([0.4,1])
zlim([0, 150])
hold on;
patch([transpose(r(:,3)) nan],[transpose(r(:,4)) nan],[transpose(r(:,2)) nan],[color3Dshort nan],'Linewidth',6,'EdgeColor','interp','EdgeAlpha',0.8,'FaceColor','none')
hold on;
view(3);
cmap = colormap(bone);
cbar = colorbar();
cbar.Ticks = 0:0.1:1;
cbar.TickLabels = num2cell(0:1:10);
caxis([0, 1]);
set(gca,'view',[150 20])
%current axes
ax = gca; 
ax.FontSize = 24;
hold off;


h=figure()
mesh(si_vector,se_vector,real(vaps_R4_first),'EdgeColor',r4_color,'FaceColor',r4_color,'FaceAlpha',0.3,'EdgeAlpha',0.3)
hold on
mesh(si_vector,se_vector,real(vaps_R4_second),'EdgeColor',r4_color,'FaceColor',r4_color,'FaceAlpha',0.3,'EdgeAlpha',0.3)
hold on;
mesh(si_vector,se_vector,zero_plane,'EdgeColor',[.4 .4 .4],'EdgeAlpha',0.3,'FaceColor',[.4 .4 .4],'FaceAlpha',0.3)
ax = gca; 
ax.FontSize = 20;
xlabel('s_e','FontSize',24);
ylabel('s_i','FontSize',24);
zlabel('R(eigenvalue)','FontSize',24);
title('Eigenvalues R_4','FontSize',18);
xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])
zticks([0 50 100 150 200 ])
xlim([0.4,1])
ylim([0.4,1])
set(gca,'view',[170 10])
% current axes
ax = gca; 
ax.FontSize = 24;
hold off;
