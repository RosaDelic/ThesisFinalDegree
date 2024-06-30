import Random as rnd
function SDNumber(len)
    SD = rnd.randn(len);
    for i in eachindex(SD)
        if SD[i]>1
            SD[i] = SD[i]-floor(SD[i]);
        elseif SD[i]<-1
            SD[i] = SD[i]-ceil(SD[i]);
        end
    end
    return SD
end