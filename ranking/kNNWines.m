% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs=1:10;
situ=1:1;
rednessRes = (1:val)';
classFreq = [7,4,3,1,2,5,6];
lim=11;

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
            sumDist = zeros(7,1);
            for k = 1:tra
                vekt = traing(k,1:lim)-testng(j,1:lim);
                currDist = varDist(vekt,invS);
                if (currDist >= 10)
                    continue;
                end
                class = traing(k,13);
                sumDist(class) = sumDist(class) + exp( -0.1*currDist );
            end
            maxim = 0; ind = 0;
            for loop=1:7
                if (sumDist(loop)>maxim)
                    maxim = sumDist(loop);
                    ind = loop;
                end
            end
            rednessRes(j)=ind;
        end
        errs(h)=sum( abs(testng(:,13)-rednessRes) )/val;
    end
    %situ(i)=mean(errs);
end
