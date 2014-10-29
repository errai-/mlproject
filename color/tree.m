% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs = 1:10;
rerrs = 1:10;
options = optimoptions('fminunc','GradObj','on'); % indicate gradient is provided
options.MaxIter=1000;

for h=1:10
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];
    testng = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red')];

    BaggedTreeEns = TreeBagger(30,traing(:,1:11),traing(:,12),'NVarToSample',3);%,'oobpred','on');
    [fresults,fprobs]=predict(BaggedTreeEns,traing(:,1:11));
    fresults=cell2mat(fresults);
    [gresults,gprobs]=predict(BaggedTreeEns,testng(:,1:11));
    gresults=cell2mat(gresults);

    errs(h) = sum(abs(traing(:,12)-fresults+48))/tra;
    rerrs(h) = sum(abs(testng(:,12)-gresults+48))/val;
end
mean(errs)
mean(rerrs)

%ctree = fitctree(traing(:,1:11),traing(:,12),'SplitCriterion','deviance');
%view(ctree,'mode','graph');

%[tresults,tprobs]=predict(ctree,traing(:,1:11));
%[results,probs]=predict(ctree,testng(:,1:11));


%traintime = sum(abs(traing(:,12)-tresults))/tra;
%testtime = sum(abs(testng(:,12)-results))/val;
