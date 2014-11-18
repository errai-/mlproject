% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

tra=4500; val=500; tes=1000;

%% 10-fold cross-validation %%

nerrs = 1:10; npriorerrs = 1:10; gerrs = 1:10; gpriorerrs = 1:10;
derrs = 1:10; kerrs = 1:10; terrs = 1:10;

for h=1:10
    %% Boundaries for cross-validation
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    %% Set convenient training and validation sets
    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red'), training.quality];
    valid = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red'), validation.quality];

    %% Prior values, if needed
    rPrior = sum(traing(:,12))/tra; wPrior = 1-rPrior;

    %% Naive (only 1 variable) Gaussian discrimination
    %m1 = mean(traing(traing(:,12)==1,8)); % Reds
    %v1 = var(traing(traing(:,12)==1,8));
    %m2 = mean(traing(traing(:,12)==0,8)); % Whites
    %v2 = var(traing(traing(:,12)==0,8));

    % With no prior and with prior
    %nclass1 = (logDiscrNaive( valid(:,8),m1,m2,v1,v2,1,1)>0);
    %nclass2 = (logDiscrNaive( valid(:,8),m1,m2,v1,v2,rPrior,wPrior)>0);

    %% Gaussian discrimination
    %m1 = mean(traing(traing(:,12)==1,1:11)); % Reds
    %v1 = cov(traing(traing(:,12)==1,1:11));
    %m2 = mean(traing(traing(:,12)==0,1:11)); % Whites
    %v2 = cov(traing(traing(:,12)==0,1:11));

    %gclass1=(1:val)'; gclass2=(1:val)'; % No prior and prior
    %for i = 1:val
    %    gclass1(i) = (logDiscrGauss(valid(i,1:11),m1,m2,v1,v2,1,1)>0);
    %    gclass2(i) = (logDiscrGauss(valid(i,1:11),m1,m2,v1,v2,rPrior,wPrior)>0);
    %end

    %% Linear discriminant
    diskr = fitcdiscr(traing(:,1:11),traing(:,13),'DiscrimType','Linear');
    dclass=predict(diskr,valid(:,1:11));

    %% Quadratic discriminant
    qdiskr = fitcdiscr(traing(:,1:11),traing(:,13),'DiscrimType','pseudoQuadratic');
    qdclass = predict(qdiskr,valid(:,1:11));
    
    %% SVM
    svm = fitcecoc(traing(:,1:11),traing(:,13));
    sclass = predict(svm,valid(:,1:11));
    
    %% kNN, k=3
    kNN = fitcknn(traing(:,1:11),traing(:,13),'Distance','mahalanobis','NumNeighbors',3);
    kclass=predict(kNN,valid(:,1:11));

    %% Random forest
    BaggedTreeEns = TreeBagger(1000,traing(:,1:11),traing(:,13)','NVarToSample',2);
    [tclass,treeProbs]=predict(BaggedTreeEns,valid(:,1:11));

    %% Random forest, regression
    BaggedTreeEns = TreeBagger(1000,traing(:,1:11),traing(:,13)','NVarToSample',2,'Method','regression');
    [trclass,rtreeProbs]=predict(BaggedTreeEns,valid(:,1:11));    
    
    %% Standard errors
    nerrs(h) = sum( nclass1 ~= valid(:,12) )/val;
    npriorerrs(h) = sum( nclass2 ~= valid(:,12) )/val;
    gerrs(h) = sum( gclass1 ~= valid(:,12) )/val;
    gpriorerrs(h) = sum( gclass2 ~= valid(:,12) )/val;
    derrs(h) = sum( dclass ~= valid(:,12) )/val;
    kerrs(h) = sum( kclass ~= valid(:,12) )/val;
    terrs(h) = sum( round(treeProbs(:,2)) ~= valid(:,12) )/val;
end

traing = [table2array(winefacts(:,1:11)),strcmp(winefacts.type,'Red'), winefacts.quality]';
testng = [table2array(winechall(:,1:11)),strcmp(winechall.type,'Red')]';
trainTarg = zeros(7,tra);

BaggedTreeEns = TreeBagger(1000,traing(1:lim,:)',traing(13,:)','NVarToSample',2);
[fresults,fprobs]=predict(BaggedTreeEns,traing(1:lim,:)');
fresults=cell2mat(fresults); fresults=fresults-48;
[gresults,gprobs]=predict(BaggedTreeEns,testng(1:lim,:)');
gresults=cell2mat(gresults); gresults=gresults-48;

fterrs = sum( fresults~=traing(13,:)' )/tra;

