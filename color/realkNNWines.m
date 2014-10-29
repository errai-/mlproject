% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs=1:10;
situ=1:1;
for i=1:1
    for h=1:10
        h
        lower = 1+500*(h-1); upper = 500*h;
        indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];
        training = winefacts(indices(1:tra),:);
        validation = winefacts(indices(tra+1:tra+val),:);

        reds = training(strcmp(training.type, 'Red'),:);
        whites = training(strcmp(training.type, 'White'),:);

        trainArray = table2array(training(:,1:11));
        validArray = table2array(validation(:,1:11));
        S = cov(trainArray);
        invS = inv(S);
        diagS = diag(invS);

        rednessT = strcmp(training.type,'Red')';
        redness0 = strcmp(validation.type,'Red')';
        dists = zeros(val,1);
        errorLog = 0;
        initN = 3;
        rednessRes = 1:val;
        %hatDist = 29+i;
        for j = 1:val
            nearVote = zeros(initN,1);
            nearDist = Inf*ones(initN,1);
            for k = 1:tra
                vekt = trainArray(k,1:11)-validArray(j,1:11);
                %simple = vekt*(diagS.*vekt');
                %if (simple > hatDist)
                %    continue;
                %end
                dists(j) = dists(j)+1;
                currDist = varDist(vekt,invS);
                if (currDist >= nearDist(initN))
                    continue;
                end
                ind = initN;
                while ( ind > 1 && currDist < nearDist(ind-1) )
                    ind = ind-1;
                end
                nearDist(ind+1:end) = nearDist(ind:end-1);
                nearDist(ind) = currDist;
                nearVote(ind+1:end) = nearVote(ind:end-1);
                nearVote(ind) = rednessT(k);
            end
            samus = sum(nearVote);
            if (samus == Inf)
                rednessRes(j)=0;
                errorLog = errorLog+1;
            elseif (sum(nearVote) > initN/2)
                rednessRes(j)=1;
            else
                rednessRes(j)=0;
            end
        end

        errs(h)=sum( abs(redness0-rednessRes) )/val;
    end
    %situ(i)=mean(errs);
end