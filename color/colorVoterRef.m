% This script is used to classify a wine to either red or white
winefacts = readtable('../training_dataset.csv');

tra=4500; val=500;

disc = 9883-5000; NN = 9904-5000; forest = 9917.5-5000;
all = disc+NN+forest; disc=disc/all; NN = NN/all; forest = forest/all;
disc = exp(disc)-1; NN = exp(NN)-1; forest = exp(forest)-1; 

priorDiscr = disc/(disc+NN+forest);
priorNN = NN/(disc+NN+forest);
priorForest = forest/(disc+NN+forest);

errs = 1:10;
options = optimoptions('fminunc','GradObj','on'); % indicate gradient is provided
options.MaxIter = 1000;

for h=1:10
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    W = rand(12,1)/50-0.01;

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];
    testng = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red')];
    W = fminunc(@(W) like(W,traing),W,options);

    rpredict = (1./(1 + exp( -W(1)-W(2:end)'*testng(:,1:11)' ) ))';

    S = cov(traing(:,1:11));
    invS = inv(S);
    diagS = diag(invS);

    errorLog = 0;
    initN = 3;
    rednessRes = (1:val)';
    for j = 1:val
        nearVote = zeros(initN,1);
        nearDist = Inf*ones(initN,1);
        for k = 1:tra
            vekt = traing(k,1:11)-testng(j,1:11);
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
            nearVote(ind) = traing(k,12);
        end
        samus = sum(nearVote);
        if (samus == Inf)
            rednessRes(j)=0;
            errorLog = errorLog+1;
        else
            rednessRes(j) = sum(nearVote)/initN;
        end
    end
    
    BaggedTreeEns = TreeBagger(30,traing(:,1:11),traing(:,12),'NVarToSample',3);
    [gresults,gprobs]=predict(BaggedTreeEns,testng(:,1:11));
    
    errs(h) = sum(abs( ((priorDiscr*rpredict + priorNN*rednessRes + priorForest*gprobs(:,2)) > 0.5) - testng(:,12)))/val;
end
mean(errs)
