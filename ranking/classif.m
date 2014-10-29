function [ klass ] = classif( votes )
    classFreq = [7,4,3,1,2,5,6]';
    amounts = (1:7)';
    for i=1:7
        amounts(i) = sum( votes==i );
    end
    maxim = max(amounts);
    amounts = (amounts==maxim);
    amounts = amounts./classFreq;
    maxim = 0;
    for i=1:7
        if (amounts(i)>maxim)
            klass = i;
            maxim=amounts(i);
        end
    end
end

