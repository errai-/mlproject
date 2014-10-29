function [ f,g ] = like( W,wine )
    exP = zeros( size(wine,1),7 ); y = zeros( size(wine,1),7 );
    for klass=1:7
        exP(:,klass) = (exp( -W(1,klass)-W(2:end,klass)'*wine(:,1:11)' ))';
    end
    for ind=1:size(exP,1)
        y(ind,:) = exP(ind,:)/sum( exP(ind,:) );
    end
    f = 0;
    g = zeros( 12, 7 );
    for klass=1:7
        r = wine(:,13)==klass;
        comb = r'-y(:,klass)';
        f = f-sum( r.*log(y(:,klass)) );
        g(:,klass) = -[sum(comb), comb*wine(:,1:11)]';
    end
end

