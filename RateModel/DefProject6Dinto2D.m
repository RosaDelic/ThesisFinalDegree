function drdt = DefProject6Dinto2D(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,s_e,s_i,a_e,a_i)
%System definition:
%drdt(1)=drdt_e | drdt(2)=drdt_i 
%r(1)=r_e | r(2)=r_i 

drdt = [-r(1)+activ_output_e(wee*r(1)*s_e-wei*r(2)*s_i+Iext_e-a_e);-r(2)+activ_output_i(-wii*r(2)*s_i+wie*r(1)*s_e+Iext_i-a_i)];

end