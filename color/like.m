function [ f,g ] = like( W,wine )
    r = wine(:,12);
    sigm = 1./(1+ exp( -W(2:end)'*(wine(:,1:11)') - W(1)));
    y = sigm';
    
    f = -sum(r.*log(y)+(1-r).*log(1-y));
    comb = r'-y';
    g = -[ sum(comb), comb*wine(:,1:11) ]';
end

