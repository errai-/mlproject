function [ dist ] = varDist( x, invS )
    dist = x*invS*(x');
end

