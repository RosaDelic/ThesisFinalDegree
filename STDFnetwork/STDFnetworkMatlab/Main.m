%Define simulation parameters
Dnumber = 1;
Fnumber = 1;
t0 = 0;
tf = 10000;
h = 0.05;
p0_stf = 1;

%Run simulation
[t_final, ti, wi, pRelAMPA, pRelNMDA, pRelGABA, pRel_stfAMPA, pRel_stfNMDA, pRel_stfGABA] = NetworkSTDall(Dnumber,Fnumber,t0,tf,h,p0_stf)