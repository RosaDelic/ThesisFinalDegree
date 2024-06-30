function drdt = ModelFunctionBothScaledDFA(t,r,wee,wii,wie,wei,Iext_e,Iext_i,activ_output_e,activ_output_i,Gde,Gdi,Je,Ji,fde,fdi,Gfe,Gfi,ffe,ffi,P0stf)
%System definition:
%drdt(1)=drdt_e | drdt(2)=drdt_i | drdt(1)=dsdt_e | drdt(2)=dsdt_i
%r(1)=r_e | r(2)=r_i | r(3)=a_e | r(4)=a_i | r(5) = sd_e | r(6) = sd_i | r(7) = sf_e | r(8) = sf_i

drdt = [-r(1)+activ_output_e(wee*r(1)*r(5)*r(7)-wei*r(2)*r(6)*r(8)+Iext_e-r(3));-r(2)+activ_output_i(-wii*r(2)*r(6)*r(8)+wie*r(1)*r(5)*r(7)+Iext_i-r(4));-r(3)+Je*(fde)*r(1);-r(4)+Ji*(fdi)*r(2); 1-r(5)*(1+Gde(r(1))*((1/fde)-1));1-r(6)*(1+Gdi(r(2))*((1/fdi)-1));P0stf-r(7)*(1+Gfe(r(1))*((P0stf/ffe)-1)); P0stf-r(8)*(1+Gfi(r(2))*((P0stf/ffi)-1))];

end