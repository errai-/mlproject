function [ x1m, x2m, errm ] = optimumCoeff( P1, P2, P3, compare, lattice, x1m, x1M, x2m, x2M )
    errm = 100000000;
    for x1 = x1m:lattice:x1M
        upper = min((1-x1),x2M);
        for x2 = x2m:lattice:upper
            tmpErr = sum(abs( round(x1*P1+x2*P2+(1-x1-x2)*P3) ~= compare));
            if (tmpErr < errm)
                x1m = x1;
                x2m = x2;
                errm = tmpErr;
            end
        end
    end
end

