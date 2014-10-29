% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs=1:10;
situ=1:1;
rednessRes = (1:val)';
classFreq = [7,4,3,1,2,5,6];
lim=12;

for i=1:1
    for h=1:10
        h
        lower = 1+500*(h-1); upper = 500*h;
        indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];
        training = winefacts(indices(1:tra),:);
        validation = winefacts(indices(tra+1:tra+val),:);

        traing = [table2array(training(:,1:11)),strcmp('Red',training.type ), training.quality ];
        testng = [table2array(validation(:,1:11)),strcmp('Red',validation.type ), validation.quality ];
        S = cov(traing(:,1:lim));
        invS = inv(S);
        diagS = diag(invS);

        errorLog = 0;
        initN = 5;
        for j = 1:val
            nearVote = zeros(initN,1);
            nearDist = Inf*ones(initN,1);
            for k = 1:tra
                vekt = traing(k,1:lim)-testng(j,1:lim);
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
                nearVote(ind) = traing(k,13);
            end
            samus = sum(nearVote);
            if (samus == Inf)
                rednessRes(j)=0;
                errorLog = errorLog+1;
            else
                rednessRes(j)=classif(nearVote);
            end
        end

        errs(h)=sum( abs(testng(:,13)-rednessRes) )/val;
    end
    %situ(i)=mean(errs);
end
