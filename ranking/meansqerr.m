function [ tot_err ] = meansqerr( pred_class, true_class )
    tot_err = (1/size(true_class,1))*sqrt(sum(( pred_class - true_class ).^2));
end

