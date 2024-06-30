function drdt = ModelFunctionBothScaledDA(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,Ge,Gi,Je,Ji,fde,fdi)
%System definition:
%drdt(1)=drdt_e | drdt(2)=drdt_i | drdt(1)=dsdt_e | drdt(2)=dsdt_i
%r(1)=r_e | r(2)=r_i | r(3)=s_e | r(4)=s_i | r(5) = a_e | r(6) = a_i

drdt = [-r(1)+activ_output_e(wee*r(1)*r(3)-wei*r(2)*r(4)+Iext_e-r(5));-r(2)+activ_output_i(-wii*r(2)*r(4)+wie*r(1)*r(3)+Iext_i-r(6));1-r(3)*(1+Ge(r(1))*((1/fde)-1));1-r(4)*(1+Gi(r(2))*((1/fdi)-1));-r(5)+Je*(fde)*r(1);-r(6)+Ji*(fdi)*r(2)];
end