function [ logfrac ] = logDiscrGauss( x,mu1,mu2,v1,v2,p1,p2 )
frac = -(x-mu1)*(inv(v1))*((x-mu1)')+(x-mu2)*(inv(v2))*((x-mu2)');
logterms = 0.5*log(det(v2)/det(v1))+log(p1/p2); 
logfrac = frac+logterms;
end

