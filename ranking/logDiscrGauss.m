function [ g ] = logDiscrGauss( d,invS,ldetS,lp )
g = -0.5*d'*invS*d;
g = g+lp;%+0.5*ldetS;
end

