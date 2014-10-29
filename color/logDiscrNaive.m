function [ logfrac ] = logDiscrNaive( x,mu1,mu2,v1,v2,p1,p2 )
logfrac = -(( x - mu1 ).^2)/v1 + ((x - mu2).^2)/v2 + 0.5*log(v2/v1)+log(p1/p2); 
end

