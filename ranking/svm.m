winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs = 1:10;
rerrs = 1:10;
options = optimoptions('fminunc','GradObj','on'); % indicate gradient is provided
options.MaxIter=1000;
lim=11;

for h=1:1
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)),strcmp(training.type,'Red'), training.quality];
    testng = [table2array(validation(:,1:11)), strcmp(validation.type,'Red'),validation.quality];

    Mdl = fitcecoc(traing(:,1:lim),traing(:,13));
    %BaggedTreeEns = TreeBagger(30,traing(:,1:lim),traing(:,13),'NVarToSample',3);%,'oobpred','on');
    %[fresults,fprobs]=predict(BaggedTreeEns,traing(:,1:lim));
    %fresults=cell2mat(fresults); fresults=fresults-48;
    %[gresults,gprobs]=predict(BaggedTreeEns,testng(:,1:lim));
    %gresults=cell2mat(gresults); gresults=gresults-48;

    %errs(h) = sum( fresults~=traing(:,13) )/tra;
    %rerrs(h) = sum( gresults~=testng(:,13) )/val;
end
%mean(errs)
%mean(rerrs)