function SD = SDNumber(len)
    SD = normrnd(0,1,[1,len]);
    SD = (SD-floor(SD)).*(SD>1)+(SD-ceil(SD)).*(SD<-1)+SD.*(abs(SD)<=1);
end