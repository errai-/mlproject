% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

tra=4500; val=500; tes=1000;

%% 10-fold cross-validation %%

derrs = zeros(10,3); qderrs = zeros(10,3); serrs = zeros(10,3); 
kerrs = zeros(10,3); lerrs = zeros(10,3); trerrs = zeros(10,3); 
terrs = zeros(10,3);

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
    tclass = round(treeProbs(:,2));

    %% Random forest, regression
    BaggedTreeEns = TreeBagger(1000,traing(:,1:11),traing(:,13)','NVarToSample',2,'Method','regression');
    trclass=round(predict(BaggedTreeEns,valid(:,1:11)));    
    
    %% Least squares, regression
    w = lscov(traing(:, 1:11), traing(:, 13));
    lclass = round(valid(:, 1:11) * w);
    
    %% Standard errors
    derrs(h,1) = sum( dclass ~= valid(:,13) )/val;
    qderrs(h,1) = sum( qdclass ~= valid(:,13) )/val;
    serrs(h,1) = sum( sclass ~= valid(:,13) )/val;
    kerrs(h,1) = sum( kclass ~= valid(:,13) )/val;
    terrs(h,1) = sum( tclass ~= valid(:,13) )/val;
    trerrs(h,1) = sum( trclass ~= valid(:,13) )/val;
    lerrs(h,1) = sum( lclass ~= valid(:,13) )/val
    
    derrs(h,2) = tot_fscore(dclass,valid(:,13));
    qderrs(h,2) = tot_fscore(qdclass,valid(:,13));
    serrs(h,2) = tot_fscore(sclass,valid(:,13));
    kerrs(h,2) = tot_fscore(kclass,valid(:,13));
    terrs(h,2) = tot_fscore(tclass,valid(:,13));
    trerrs(h,2) = tot_fscore(trclass,valid(:,13));
    lerrs(h,2) = tot_fscore(lclass,valid(:,13));
end
[mean(derrs),mean(qderrs),mean(serrs),mean(kerrs)]
[mean(terrs),mean(trerrs),mean(lerrs)]

