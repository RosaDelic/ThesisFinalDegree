include("NetworkSTDall.jl")

#Define simulation parameters
t0 = 0;
tf = 10000;
h = 0.05;
Dnumber = 1;
Fnumber = 1;
p0_stf = 1;

#Run simulation
NetworkSTDall(Dnumber,Fnumber,t0,tf,h,p0_stf)
