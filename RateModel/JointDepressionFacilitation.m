%Define discretization needed for simulation
delta = 10^(-4); %time and activation_function discretization
activ_interval = 0:delta:1; 
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
%Time interval
tspan = [0,20000];
        
%System parameters
wee = 10;
wei = 1.5;
wie = 9;
wii = 0.5;

%Provisional values
tau_e = 150;
tau_i = 320;%320
tau_ae = 1500;
tau_ai = 500;
tau_sde = 250; 
tau_sdi = 250;
tau_sfe = 30; 
tau_sfi = 30;


%Mass matrix for ode integrator
Mass = [tau_e 0 0 0 0 0 0 0;0 tau_i 0 0 0 0 0 0;0 0 tau_ae 0 0 0 0 0;0 0 0 tau_ai 0 0 0 0; 0 0 0 0 tau_sde 0 0 0; 0 0 0 0 0 tau_sdi 0 0; 0 0 0 0 0 0 tau_sfe 0; 0 0 0 0 0 0 0 tau_sfi];
opts = odeset('Mass',Mass);


%External intputs
Iext_e = 0;
Iext_i = 0;
Je=10;
Ji=3.5;

%GENERAL fD+fF parameters
%========== THE USER CAN CHANGE THESE PARAMTERS FOR SIMULATIONS ===========
%Probability of release facilitation
P0 = 0.7;
%Depression and Facilitation factors
fd=0.9;
ff=0.5;

%--------------------------Depression parameters---------------------------
%Auxiliar functions for depression steady-states
%h = @(P0)(((0.25-0.8)/(1-0.1))*(P0-1.4));
h_e = @(P0)(0.3+exp(-(4*(P0+0.09))));
h_i = @(P0)(exp(-0.36*P0));
FDefectiva_e = @(fd,P0)(h_e(P0)+(1-h_e(P0))*exp(10.8*(fd-1))); 
FDefectiva_i = @(fd,P0)(h_i(P0)+(1-h_i(P0))*exp(10*(fd-1)));
%Depression steady-state for current combination of fD,P0
fde = FDefectiva_e(fd,P0)-0.2*(1-ff).*(fd<1);
fdi = FDefectiva_i(fd,P0)-0.2*(1-ff).*(fd<1);

%h=figure()
%plot(activ_interval,FDefectiva_e(activ_interval),'LineWidth',4,'Color','r','DisplayName','f_{De}');
%hold on;
%plot(activ_interval,FDefectiva_i(activ_interval),'LineWidth',4,'Color','b','DisplayName','f_{Di}');
%ax = gca; 
%ax.FontSize = 24;
%legend('Location','northwest','NumColumns',2);
%xlabel('Depression factor f_D')
%ylabel('Effective f_k')
%hold off;

%Depression activation_function definition (low-pass filter)

%Depression threshold
theta_de = (fde)*150;%+250*(1-P0); 
theta_di = (fdi)*65;%+250*(1-P0); 
%theta_de = 60;
%theta_di = 50;

Gde = @(x)(0.*(x<(theta_de-10))+(1/20)*(x-(theta_de-10)).*(x>=(theta_de-10) & x<(theta_de+10))+1.*(x>=(theta_de+10)));
Gdi = @(x)(0.*(x<(theta_di-10))+(1/20)*(x-(theta_di-10)).*(x>=(theta_di-10) & x<(theta_di+10))+1.*(x>=(theta_di+10)));
%Unset fsolve messages
%options = optimset('Display','off');
interval_G = 0:delta:400; 
h=figure()
plot(interval_G,Gde(interval_G),'LineWidth',4,'Color','r','DisplayName','G_{e}');
hold on;
plot(interval_G,Gdi(interval_G),'LineWidth',4,'Color','b','DisplayName','G_{i}');
ax = gca; 
ax.FontSize = 24;
legend('Location','southeast','NumColumns',2);
xlabel('r_k')
ylabel('G_k')
ylim([0,1])
hold off;

%--------------------------Facilitation parameters-------------------------

%Adjust bifurcation point fF^* depending on P0 and fD
%when this threshold is achieved s_Fe,s_Fi will have as steady state P0stf
thetaFD = @(P0,fD)(min(1,(((-0.4)/(0.9-0.1))*(P0-0.9)+exp(-5*(fD+((0.5)/(0.7-0.1))*P0-0.58)))));

%Theta_fk_factor: Ratio to reduce fF to achieve silent state at the bifurcation point fF^* for facilitation
%thetaFefectiva = @(fD,fF)(1.*(fF>thetaFD(P0,fd))+(0).*(fF<=thetaFD(P0,fd) | thetaFD(P0,fd)>=1)); %posar thetaFD=0.72 per recuperar independencia de fD
%theta_fe_factor = thetaFefectiva(fd,ff);
%theta_fi_factor = thetaFefectiva(fd,ff);
%FFefectiva_e= @(P0,ff)(0.25*((P0+ff)/2)+0.75);

%Auxiliary functions for facilitation steady-states
OY_e0 = @(P0)((1-0.72)*P0+0.72);
OY_e1 = @(P0)((1-0.82)*P0+0.82);
FFefectiva_e= @(P0,ff)((OY_e1(P0)-OY_e0(P0))*ff+OY_e0(P0));
OY_i = @(P0)(-0.15*P0+0.15);
FFefectiva_i = @(P0,ff)(P0+OY_i(P0)*ff);
%FFefectiva_i= @(P0,ff)(P0+0.15*ff).*(P0<0.85)+1.*(P0>=0.85);
%Facilitation steady-state for current combination of fF,P0
ffi = (FFefectiva_i(P0,ff)).*(ff>thetaFD(P0,fd))+P0.*(ff<=thetaFD(P0,fd));
ffe = (FFefectiva_e(P0,ff)).*(ff>thetaFD(P0,fd))+P0.*(ff<=thetaFD(P0,fd));


%Facilitation activation function definition(band-pass filter)

%Threshold definition
%Theta_fk_lower: Lower threshold for facilitation
theta_fe_lower = 2;%/ff;%3.25;%Facilitation only: 3.25;
theta_fi_lower = 5;
%Theta_fk_upper: Higher threshold for facilitation
theta_fe_upper = (max((1/(3))*ff*theta_de,theta_fe_lower)).*(P0>=0.25 & fd<1)+(200).*(P0<0.25 | fd==1);%theta_de-20;%Millor: 400
theta_fi_upper = (max((1/(3))*ff*theta_di,theta_fi_lower)).*(P0>=0.25 & fd<1)+(300).*(P0<0.25 | fd==1);%theta_di-20;%Millor: 400
%theta_fe_upper = (1/(2))*ff*theta_de;
%Facilitation activation function definition
%Gfi = @(x)(((1/theta_fi_lower)*theta_fi_factor*x).*(x<theta_fi_lower)+(1/theta_fi_lower)*theta_fi_factor*theta_fi_lower.*(x>=theta_fi_lower & x<theta_fi_upper)+(-(1/200)*(x-theta_fi_upper)+(1/theta_fi_lower)*theta_fe_factor*theta_fi_lower).*(x>=theta_fi_upper));
%Gfe = @(x)(((1/theta_fe_lower)*theta_fe_factor*x).*(x<theta_fe_lower)+(1/theta_fe_lower)*theta_fe_factor*theta_fe_lower.*(x>=theta_fe_lower & x<theta_fe_upper)+(-(1/500)*(x-theta_fe_upper)+(1/theta_fe_lower)*theta_fi_factor*theta_fe_lower).*(x>=theta_fe_upper));
Gfi = @(x)(((1/theta_fi_lower)*x).*(x<theta_fi_lower)+(1/theta_fi_lower)*theta_fi_lower.*(x>=theta_fi_lower & x<theta_fi_upper)+(-(1/150)*(x-theta_fi_upper)+(1/theta_fi_lower)*theta_fi_lower).*(x>=theta_fi_upper));
Gfe = @(x)(((1/theta_fe_lower)*x).*(x<theta_fe_lower)+(1/theta_fe_lower)*theta_fe_lower.*(x>=theta_fe_lower & x<theta_fe_upper)+(-(1/150)*(x-theta_fe_upper)+(1/theta_fe_lower)*theta_fe_lower).*(x>=theta_fe_upper));

interval_G = 0:delta:400; 
h=figure()
plot(interval_G,Gfe(interval_G),'LineWidth',4,'Color','r','DisplayName','G_{e}');
hold on;
plot(interval_G,Gfi(interval_G),'LineWidth',4,'Color','b','DisplayName','G_{i}');
ax = gca; 
ax.FontSize = 24;
legend('Location','southeast','NumColumns',2);
xlabel('r_k')
ylabel('G_{F_k}')
ylim([0,1])
hold off;
%% --------------------------  PARAMETERS PLOTS  --------------------------

%-----------Bifurcation point f_F^* ------------
P0_vector = [0,0.1:0.2:1,1];
h=figure()
for i=1:length(P0_vector)
    P0_current = P0_vector(i);
    thetaFDplot = @(fD)(min(1,(((-0.4)/(0.9-0.1))*(P0_current-0.9)+exp(-5*(fD+((0.5)/(0.7-0.1))*P0_current-0.58)))));
    plot(activ_interval,thetaFDplot(activ_interval),'LineWidth',4,'DisplayName',strcat('P0_{STF} = ',num2str(P0_current)));
    hold on;
end

ax = gca; 
ax.FontSize = 24;
h.Position = [100 150 800 400];
legend('Location','southwest','NumColumns',3,'FontSize',20);
xlabel('Depression factor f_D')
ylabel('Bif. Point f_F^*')
ylim([-0.8,1.1])
xticks([0,0.2,0.4,0.6,0.8,1])
hold off;

%-------------Equilibrium points ffe--------------
h=figure()
for i=1:length(P0_vector)
    fD_plot = 1; %Facilitation case
    P0_current = P0_vector(i);
    FFefectiva_e_plot= @(ff)((OY_e1(P0_current)-OY_e0(P0_current))*ff+OY_e0(P0_current));
    thetaFDplot = min(1,(((-0.4)/(0.9-0.1))*(P0_current-0.9)+exp(-5*(fD_plot+((0.5)/(0.7-0.1))*P0_current-0.58))));
    ffe = @(ff)((FFefectiva_e_plot(ff)).*(ff>thetaFDplot)+P0_current.*(ff<=thetaFDplot));
    plot(activ_interval,ffe(activ_interval),'LineWidth',4,'DisplayName',strcat('P0_{STF} = ',num2str(P0_current)));
    hold on
end

ax = gca; 
ax.FontSize = 24;
h.Position = [100 150 800 400];
legend('Location','southwest','NumColumns',3,'FontSize',20);
xlabel('Facilitation factor f_F')
ylabel('f_{Fe}')
ylim([-0.75,1.1])
yticks([0,0.2,0.4,0.6,0.8,1])
xticks([0,0.2,0.4,0.6,0.8,1])
hold off;


%-------------Equilibrium points ffi--------------
h=figure()
for i=1:length(P0_vector)
    fD_plot = 1; %Facilitation case
    P0_current = P0_vector(i);
    FFefectiva_i_plot = @(ff)(P0_current+OY_i(P0_current)*ff);
    thetaFDplot = min(1,(((-0.4)/(0.9-0.1))*(P0_current-0.9)+exp(-5*(fD_plot+((0.5)/(0.7-0.1))*P0_current-0.58))));
    ffi = @(ff)((FFefectiva_i_plot(ff)).*(ff>thetaFDplot)+P0_current.*(ff<=thetaFDplot));
    plot(activ_interval,ffi(activ_interval),'LineWidth',4,'DisplayName',strcat('P0_{STF} = ',num2str(P0_current)));
    hold on;
end

ax = gca; 
ax.FontSize = 24;
h.Position = [100 150 800 400];
legend('Location','southwest','NumColumns',3,'FontSize',20);
xlabel('Facilitation factor f_F')
ylabel('f_{Fi}')
ylim([-0.75,1.1])
yticks([0,0.25,0.5,0.75,1])
xticks([0,0.2,0.4,0.6,0.8,1])
hold off;


%-------------Equilibrium points fde--------------
%h_e = @(P0)(0.3+exp(-(4*(P0+0.09))));
%h_i = @(P0)(exp(-0.36*P0));
%FDefectiva_e = @(fd,P0)(h_e(P0)+(1-h_e(P0))*exp(10.8*(fd-1))); 
%FDefectiva_i = @(fd,P0)(h_i(P0)+(1-h_i(P0))*exp(10*(fd-1)));
%fde = FDefectiva_e(fd,P0)-0.2*(1-ff).*(fd<1);
%fdi = FDefectiva_i(fd,P0)-0.2*(1-ff).*(fd<1);

h=figure()
for i=1:length(P0_vector)
    P0_current = P0_vector(i);
    FDefectiva_e_plot= @(fd)(h_e(P0_current)+(1-h_e(P0_current))*exp(10.8*(fd-1)));
    fde_current = @(fd)FDefectiva_e_plot(fd);
    plot(activ_interval,fde_current(activ_interval),'LineWidth',4,'DisplayName',strcat('P0_{STD} = ',num2str(P0_current)));
    hold on
end

ax = gca; 
ax.FontSize = 24;
h.Position = [100 150 800 400];
legend('Location','southwest','NumColumns',3,'FontSize',20);
xlabel('Depression factor f_D')
ylabel('f_{De}')
ylim([-0.25,1.1])
yticks([0,0.2,0.4,0.6,0.8,1])
xticks([0,0.2,0.4,0.6,0.8,1])
hold off;


%-------------Equilibrium points ffi--------------
h=figure()
for i=1:length(P0_vector)
    P0_current = P0_vector(i);
    FDefectiva_i_plot = @(fd)(h_i(P0_current)+(1-h_i(P0_current))*exp(10*(fd-1)));
    fdi_current = @(fd)FDefectiva_i_plot(fd);
    plot(activ_interval,fdi_current(activ_interval),'LineWidth',4,'DisplayName',strcat('P0_{STD} = ',num2str(P0_current)));
    hold on;
end

ax = gca; 
ax.FontSize = 24;
h.Position = [100 150 800 400];
legend('Location','southwest','NumColumns',3,'FontSize',20);
xlabel('Depression factor f_D')
ylabel('f_{Di}')
ylim([0.45,1.01])
yticks([0.5,0.6,0.7,0.8,0.9,1])
xticks([0,0.2,0.4,0.6,0.8,1])
hold off;
%% --------------------------  ODE INTEGRATION  ---------------------------
%Set initial condition
r0 = [0.5,0.5,0,0,1,1,P0,P0];

%Integrate system
[t,r] = ode45(@(t,r)ModelFunctionBothScaledDFA(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,Gde,Gdi,Je,Ji,fde,fdi,Gfe,Gfi,ffe,ffi,P0),tspan,r0,opts);
%% -------------------------------  PLOTS  --------------------------------
dr_e = @(r_e,r_i)((1/tau_e)*(-r_e+activ_output_e(fde*ffe*wee*r_e-fdi*ffi*wei*r_i+Iext_e)));
dr_i = @(r_e,r_i)((1/tau_i)*(-r_i+activ_output_i(-fdi*ffi*wii*r_i+fde*ffe*wie*r_e+Iext_i)));
%Define vector field meshgrid
spacing = 5;
[R_E,R_I] = meshgrid(-200:spacing:405);

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
xlabel('r_e','FontSize',36);
ylabel('r_i','FontSize',36);
yticks([-100 0 100 200 300])
% current axes
ax = gca; 
ax.FontSize = 24;
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0),'PhaseSpaceNullclines.png'),'png')

%Figure 2: Plot phase space excitatory and inhibitory population (re,ri) plane
h=figure()
plot(r(:,1),r(:,2),'-',"LineWidth",4,'Color','k');
hold on;
quiver(R_E,R_I,dr_e(R_E,R_I),dr_i(R_E,R_I),'Color',[.7 .7 .7],'LineWidth',2,'DisplayName','VectorField');
hold on;
xlabel('r_e','FontSize',48);
ylabel('r_i','FontSize',48);
% current axes
axis([min(r(:,1))-10 max(r(:,1))+10 min(r(:,2))-10 max(r(:,2))+10])
ax = gca; 
ax.FontSize = 36;
hold off;
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0), 'PhaseSpace.png'),'png')

%Figure 3: Plot adaptation excitatory and inhibitory population 
h=figure()
plot(t/1000,r(:,3),'-',"LineWidth",4,'Color','r');
hold on;
plot(t/1000,r(:,4),'-',"LineWidth",4,'Color','b');
xlabel('Time t(s)','FontSize',48);
ylabel('Solution a(t)','FontSize',48);
%current axes
ylim([-10 max(max(r(:,3),max(r(:,4))))+10])
ax = gca; 
legend('a_e','a_i',"Location","northeast","Orientation",'horizontal','Fontsize',24)
ax.FontSize = 36;
h.Position = [100 150 800 400];
hold off;
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0), 'Adaptation.png'),'png')

%Figure 4: Plot depression in excitatory and inhibitory population 
h=figure()
plot(t/1000,r(:,5),'-',"LineWidth",4,'Color',"#EDB120");
hold on;
plot(t/1000,r(:,6),'-',"LineWidth",4,'Color',"#7E2F8E");
hold on;
xlabel('Time t(s)','FontSize',48);
ylabel('Solution s_D(t)','FontSize',48);
% current axes
ylim([-0.01+min(min(r(:,5)),min(r(:,6))) max(max(r(:,5)),max(r(:,6)))+0.01])
ax = gca; 
ax.FontSize = 36;
h.Position = [100 100 800 400];
legend('s_{De}','s_{Di}',"Location","southeast","Orientation",'horizontal','Fontsize',24)
hold off;
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0), 'Depression.png'),'png')

%Figure 5: Plot facilitation in excitatory and inhibitory population 
h=figure()
plot(t/1000,r(:,7),'-',"LineWidth",4,'Color',"#EDB120");
hold on;
plot(t/1000,r(:,8),'-',"LineWidth",4,'Color',"#7E2F8E");
hold on;
xlabel('Time t(s)','FontSize',48);
ylabel('Solution s_F(t)','FontSize',48);
% current axes
ylim([-0.01+min(min(r(:,7)),min(r(:,8))) max(max(r(:,7)),max(r(:,8)))+0.01])
ax = gca; 
ax.FontSize = 36;
h.Position = [100 100 800 400];
legend('s_{Fe}','s_{Fi}',"Location","northeast","Orientation",'horizontal','Fontsize',24)
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0), 'Facilitation.png'),'png')

%Figure 6: Plot firing rate excitatory and inhibitory population 
h=figure()
plot(t/1000,r(:,1),'-',"LineWidth",4,'Color','r');
hold on;
plot(t/1000,r(:,2),'-',"LineWidth",4,'Color','b');
hold on;
xlabel('Time t(s)','FontSize',48);
ylabel('Solution r(t)','FontSize',48);
% current axes
ylim([-10 max(max(r(:,1)),max(r(:,2)))+10]);
ax = gca; 
ax.FontSize = 36;
h.Position = [100 100 800 400];
legend('r_e','r_i',"Location","northeast","Orientation",'horizontal','Fontsize',24)
%Uncomment the following line to save Figure in current folder
%saveas(h,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0), 'FiringRate.png'),'png')
%% --------------------------  COMPLETE VIDEO  ----------------------------
%Define vector field meshgrid
spacing = 10;
[R_E,R_I] = meshgrid(-300:spacing:475);

%Define general figure
h = figure()
set(h,'Position',[100 100 2000 1500])

%Define speed for video
video_speed = 2;

for idx=1:length(t)
    clf;
    if mod(idx,video_speed)==0
        dr_e_t = @(r_e,r_i)((1/tau_e)*(-r_e+activ_output_e(r(idx,5)*r(idx,7)*wee*r_e-r(idx,6)*r(idx,8)*wei*r_i+Iext_e-r(idx,3))));
        dr_i_t = @(r_e,r_i)((1/tau_i)*(-r_i+activ_output_i(-r(idx,6)*r(idx,8)*wii*r_i+r(idx,5)*r(idx,7)*wie*r_e+Iext_i-r(idx,4))));
        F_e = @(r_e,r_i)(r(idx,5)*r(idx,7)*wee*r_e-r(idx,6)*r(idx,8)*wei*r_i+Iext_e-r(idx,3)-beta_e);
        F_i = @(r_e,r_i)(-r(idx,6)*r(idx,8)*wii*r_i+r(idx,5)*r(idx,7)*wie*r_e+Iext_i-r(idx,4)-beta_i);

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
        axis([-150 350 -300 475])
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
    
        %Subplot (2,2) --> Depression
        subplot(2,2,3);
        %excitatory
        seD_idx = nan(1,length(t));
        seD_idx(1:idx) = r(1:idx,5);
        seD_idx(end) = r(end,5);
        %inhibitory
        siD_idx = nan(1,length(t));
        siD_idx(1:idx) = r(1:idx,6);
        siD_idx(end) = r(end,6);
        plot(t/1000,seD_idx,'-',"LineWidth",4,'Color',"#EDB120");
        hold on;
        plot(t/1000,siD_idx,'-',"LineWidth",4,'Color',"#7E2F8E");
        hold on;
        title('Depression','FontSize',18);
        xlabel('Time t (s)','FontSize',24);
        ylabel('Solution s_D(t)','FontSize',24);
        % current axes
        ylim([0.3 max(max(r(:,5)),max(r(:,6)))+0.1])
        ax = gca; 
        ax.FontSize = 24;
    
        %Subplot (2,1) --> Facilitation
        subplot(2,2,4);
        %excitatory
        se_idx = nan(1,length(t));
        se_idx(1:idx) = r(1:idx,7);
        se_idx(end) = r(end,7);
        %inhibitory
        si_idx = nan(1,length(t));
        si_idx(1:idx) = r(1:idx,8);
        si_idx(end) = r(end,8);
        plot(t/1000,se_idx,'-',"LineWidth",4,'Color',"#EDB120");
        hold on;
        plot(t/1000,si_idx,'-',"LineWidth",4,'Color',"#7E2F8E");
        hold on;
        title('Facilitation','FontSize',18);
        xlabel('Time t (s)','FontSize',24);
        ylabel('Solution s_F(t)','FontSize',24);
        % current axes
        ylim([P0 max(max(r(:,5)),max(r(:,6)))+0.1])
        ax = gca; 
        ax.FontSize = 24;
        
        sgtitle(strcat('f_D = ',num2str(fd),'; f_F = ',num2str(ff),'; P0_{STF} = ',num2str(P0)),'FontSize',40)
        drawnow;
        movieVector(floor(idx/video_speed)) = getframe(h);
        
    end
end

myWriter = VideoWriter(strcat('CompleteAnimationDepressionfD',num2str(fd),'FacilitationfF'),'MPEG-4')
myWriter.FrameRate = 4
open(myWriter)
writeVideo(myWriter,movieVector)
close(myWriter)
clear movieVector;
%% -----------------------  IMPROVED STABILITY VIDEO  ---------------------
%Define vector field meshgrid
spacing = 10;
[R_E,R_I] = meshgrid(-300:spacing:475);

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
        dr_e_t = @(r_e,r_i)((1/tau_e)*(-r_e+activ_output_e(r(idx,5)*r(idx,7)*wee*r_e-r(idx,6)*r(idx,8)*wei*r_i+Iext_e-r(idx,3))));
        dr_i_t = @(r_e,r_i)((1/tau_i)*(-r_i+activ_output_i(-r(idx,6)*r(idx,8)*wii*r_i+r(idx,5)*r(idx,7)*wie*r_e+Iext_i-r(idx,4))));
        F_e = @(r_e,r_i)(r(idx,5)*r(idx,7)*wee*r_e-r(idx,6)*r(idx,8)*wei*r_i+Iext_e-r(idx,3)-beta_e);
        F_i = @(r_e,r_i)(-r(idx,6)*r(idx,8)*wii*r_i+r(idx,5)*r(idx,7)*wie*r_e+Iext_i-r(idx,4)-beta_i);
        dr = @(rvar)[(1/tau_e)*(-rvar(1)+activ_output_e(r(idx,5)*r(idx,7)*wee*rvar(1)-r(idx,6)*r(idx,8)*wei*rvar(2)+Iext_e-r(idx,3)));(1/tau_i)*(-rvar(2)+activ_output_i(-r(idx,6)*r(idx,8)*wii*rvar(2)+r(idx,5)*r(idx,7)*wie*rvar(1)+Iext_i-r(idx,4)))];

        %Forward integration - current solution

        %Set initial condition
        r0_forward = [r(idx,1), r(idx,2)];
        a_e = r(idx,3);
        a_i = r(idx,4);
        s_De = r(idx,5);
        s_Di = r(idx,6);
        s_Fe = r(idx,7);
        s_Fi = r(idx,8);

        %Integrate system
        [tforward1,rforward1] = ode45(@(t,r)DefProject8Dinto2D(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,s_De,s_Di,s_Fe,s_Fi,a_e,a_i),tspanforwards,r0_forward,optsForward);

        %Plot foward trajectory
        plot(rforward1(:,1),rforward1(:,2),'LineWidth',4,'Color',[.4 .4 .4],'LineStyle','-.')
        hold on;


        %1. Compute the solution in each region
        solspecificR1e_iter = 0;
        solspecificR1i_iter = 0;
        solspecificR2e_iter= (alpha_e*(beta_e-(Iext_e-r(idx,3))))/(alpha_e*wee*r(idx,5)*r(idx,7)-1);
        solspecificR2i_iter = 0;
        solspecificR4e_iter = (alpha_e*alpha_i*wei*r(idx,6)*r(idx,8)*(Iext_i-r(idx,4)-beta_i)-alpha_e*(alpha_i*wii*r(idx,6)*r(idx,8)+1)*(Iext_e-r(idx,3)-beta_e))/((alpha_e*wee*r(idx,5)*r(idx,7)-1)*(alpha_i*wii*r(idx,6)*r(idx,8)+1)-alpha_e*alpha_i*wie*r(idx,5)*r(idx,7)*wei*r(idx,6)*r(idx,8));
        solspecificR4i_iter = ((alpha_i*wie*r(idx,5)*r(idx,7))/(alpha_i*wii*r(idx,6)*r(idx,8)+1))*solspecificR4e_iter+(alpha_i*(Iext_i-r(idx,4)-beta_i))/(alpha_i*wii*r(idx,6)*r(idx,8)+1);
        
        %2. Save fixed point in each region
        if F_e(solspecificR1e_iter,solspecificR1i_iter)<0 && F_i(solspecificR1e_iter,solspecificR1i_iter)<0
            solspecificR1e(1,idx) = solspecificR1e_iter;
            solspecificR1i(1,idx) = solspecificR1i_iter;
        end
        if F_e(solspecificR2e_iter,solspecificR2i_iter)>=0 && F_i(solspecificR2e_iter,solspecificR2i_iter)<0
            solspecificR2e(1,idx) = solspecificR2e_iter;
            solspecificR2i(1,idx) = solspecificR2i_iter;
        end
        if F_e(solspecificR4e_iter,solspecificR4i_iter)>=0 && F_i(solspecificR4e_iter,solspecificR4i_iter)>=0
            solspecificR4e(1,idx) = solspecificR4e_iter;
            solspecificR4i(1,idx) = solspecificR4i_iter;
        end

        %Define subplots
    
        %VectorField/PhaseSpace/Nullclines/FixedPoints
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
        axis([-150 350 -300 475])
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

myWriter = VideoWriter(strcat('ZoomAnimationDepressionfD',num2str(fd),'FacilitationfF'),'MPEG-4')
myWriter.FrameRate = 2
open(myWriter)
writeVideo(myWriter,movieVector)
close(myWriter)
%% ------------------- CHECKING FIXED POINTS EXPRESSION  ------------------

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
    F_e = @(r_e,r_i)(r(idx,5)*r(idx,7)*wee*r_e-r(idx,6)*r(idx,8)*wei*r_i+Iext_e-r(idx,3)-beta_e);
    F_i = @(r_e,r_i)(-r(idx,6)*r(idx,8)*wii*r_i+r(idx,5)*r(idx,7)*wie*r_e+Iext_i-r(idx,4)-beta_i);
    dr = @(rvar)[(1/tau_e)*(-rvar(1)+activ_output_e(r(idx,5)*r(idx,7)*wee*rvar(1)-r(idx,6)*r(idx,8)*wei*rvar(2)+Iext_e-r(idx,3)));(1/tau_i)*(-rvar(2)+activ_output_i(-r(idx,6)*r(idx,8)*wii*rvar(2)+r(idx,5)*r(idx,7)*wie*rvar(1)+Iext_i-r(idx,4)))];

    %2. Compute the Jacobian matrix and evaluate at the fixed point
    J = @(rvar_e,rvar_i)[(1/tau_e)*(-1+alpha_e*r(idx,5)*r(idx,7)*wee*hardlim(r(idx,5)*r(idx,7)*wee*rvar_e-r(idx,6)*r(idx,8)*wei*rvar_i+Iext_e-r(idx,3)-beta_e)) (1/tau_e)*(-alpha_e*r(idx,6)*r(idx,8)*wei*hardlim(r(idx,5)*r(idx,7)*wee*rvar_e-r(idx,6)*r(idx,8)*wei*rvar_i+Iext_e-r(idx,3)-beta_e)); (1/tau_i)*(alpha_i*r(idx,5)*r(idx,7)*wie*hardlim(-r(idx,6)*r(idx,8)*wii*rvar_i+r(idx,5)*r(idx,7)*wie*rvar_e+Iext_i-r(idx,4)-beta_i)) (1/tau_i)*(-1-alpha_i*r(idx,6)*r(idx,8)*wii*hardlim(-r(idx,6)*r(idx,8)*wii*rvar_i+r(idx,5)*r(idx,7)*wie*rvar_e+Iext_i-r(idx,4)-beta_i))]; 
    
    %1. Compute the solution in each region
    solspecificR1e_iter = 0;
    solspecificR1i_iter = 0;
    solspecificR2e_iter= (alpha_e*(beta_e-(Iext_e-r(idx,3))))/(alpha_e*wee*r(idx,5)*r(idx,7)-1);
    solspecificR2i_iter = 0;
    solspecificR4e_iter = (alpha_e*alpha_i*wei*r(idx,6)*r(idx,8)*(Iext_i-r(idx,4)-beta_i)-alpha_e*(alpha_i*wii*r(idx,6)*r(idx,8)+1)*(Iext_e-r(idx,3)-beta_e))/((alpha_e*wee*r(idx,5)*r(idx,7)-1)*(alpha_i*wii*r(idx,6)*r(idx,8)+1)-alpha_e*alpha_i*wie*r(idx,5)*r(idx,7)*wei*r(idx,6)*r(idx,8));
    solspecificR4i_iter = ((alpha_i*wie*r(idx,5)*r(idx,7))/(alpha_i*wii*r(idx,6)*r(idx,8)+1))*solspecificR4e_iter+(alpha_i*(Iext_i-r(idx,4)-beta_i))/(alpha_i*wii*r(idx,6)*r(idx,8)+1);
    
    %2. Save fixed point in each region
    if F_e(solspecificR1e_iter,solspecificR1i_iter)<0 && F_i(solspecificR1e_iter,solspecificR1i_iter)<0
        solspecificR1e(1,idx) = solspecificR1e_iter;
        solspecificR1i(1,idx) = solspecificR1i_iter;
        JsubsR1 = J(solspecificR1e_iter,solspecificR1i_iter);
        [veps,vaps] = eig(JsubsR1);
        vaps_R1(1,idx) = vaps(1,1);
        vaps_R1(2,idx) = vaps(2,2);
    end
    if F_e(solspecificR2e_iter,solspecificR2i_iter)>=0 && F_i(solspecificR2e_iter,solspecificR2i_iter)<0
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
%% ----------------- PLOTING EQUILIBRIUM POINT EXPRESSION -----------------

%Equilibrium point R1
h1a=figure()
scatter(t/1000,solspecificR1e(1,:),60,'r','filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,solspecificR1i(1,:),60,'b','filled','o','MarkerFaceAlpha',0.3)
xlabel('Time (s)','FontSize',36);
ylabel('Fixed point R_1','FontSize',36);
% current axes
ax = gca; 
ax.FontSize = 24;
h1a.Position = [100 100 800 400];
hold off;
%Uncomment the following line to save Figure in current folder
%saveas(h1a,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0),'SolR1.png'),'png')

%Equilibrium point R2
h2a=figure()
scatter(t/1000,solspecificR2e(1,:),60,'r','filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,solspecificR2i(1,:),60,'b','filled','o','MarkerFaceAlpha',0.3)
xlabel('Time (s)','FontSize',36);
ylabel('Fixed point R_2','FontSize',36);
% current axes
ax = gca; 
ax.FontSize = 24;
h2a.Position = [100 100 800 400];
hold off;
%Uncomment the following line to save Figure in current folder
%saveas(h2a,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0),'SolR2.png'),'png')

%Equilibrium point R4
h3a=figure()
scatter(t/1000,solspecificR4e(1,:),60,'r','filled','o','MarkerFaceAlpha',0.3)
hold on;
scatter(t/1000,solspecificR4i(1,:),60,'b','filled','o','MarkerFaceAlpha',0.3)
xlabel('Time (s)','FontSize',36);
ylabel('Fixed point R_4','FontSize',36);
% current axes
ax = gca; 
ax.FontSize = 24;
h3a.Position = [100 100 800 400];
hold off;
%Uncomment the following line to save Figure in current folder
%saveas(h3a,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0),'SolR4.png'),'png')

heig = figure()
%eigenvalues
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
xlabel('Time (s)','FontSize',36);
ylabel('R(eigenvalue)','FontSize',36);
% current axes
ax = gca; 
ax.FontSize = 24;
heig.Position = [100 100 800 400];
hold off;
%Uncomment the following line to save Figure in current folder
%saveas(heig,strcat('.\fD', mat2str(fd),'ff',mat2str(ff),'P0_stf',mat2str(P0),'Eigenvalues.png'),'png')