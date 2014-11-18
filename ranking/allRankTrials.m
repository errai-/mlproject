% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

tra=4500; val=500; tes=1000;

%% 10-fold cross-validation %%

derrs = zeros(10,3); qderrs = zeros(10,3); serrs = zeros(10,3); 
kerrs = zeros(10,3); lerrs = zeros(10,3); trerrs = zeros(10,3); 
terrs = zeros(10,3); nerrs = zeros(10,3);

for h=1:10
    %% Boundaries for cross-validation
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    %% Set convenient training and validation sets
    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red'), training.quality];
    valid = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red'), validation.quality];

    %% Ultimately naive prediction
    nclass = ones(val,1)*4;
    
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
    nerrs(h,1) = sum( nclass ~= valid(:,13) )/val;
    derrs(h,1) = sum( dclass ~= valid(:,13) )/val;
    qderrs(h,1) = sum( qdclass ~= valid(:,13) )/val;
    serrs(h,1) = sum( sclass ~= valid(:,13) )/val;
    kerrs(h,1) = sum( kclass ~= valid(:,13) )/val;
    terrs(h,1) = sum( tclass ~= valid(:,13) )/val;
    trerrs(h,1) = sum( trclass ~= valid(:,13) )/val;
    lerrs(h,1) = sum( lclass ~= valid(:,13) )/val
    
    nerrs(h,2) = tot_fscore(nclass,valid(:,13));
    derrs(h,2) = tot_fscore(dclass,valid(:,13));
    qderrs(h,2) = tot_fscore(qdclass,valid(:,13));
    serrs(h,2) = tot_fscore(sclass,valid(:,13));
    kerrs(h,2) = tot_fscore(kclass,valid(:,13));
    terrs(h,2) = tot_fscore(tclass,valid(:,13));
    trerrs(h,2) = tot_fscore(trclass,valid(:,13));
    lerrs(h,2) = tot_fscore(lclass,valid(:,13));
    
    nerrs(h,3) = meansqerr(nclass,valid(:,13));
    derrs(h,3) = meansqerr(dclass,valid(:,13));
    qderrs(h,3) = meansqerr(qdclass,valid(:,13));
    serrs(h,3) = meansqerr(sclass,valid(:,13));
    kerrs(h,3) = meansqerr(kclass,valid(:,13));
    terrs(h,3) = meansqerr(tclass,valid(:,13));
    trerrs(h,3) = meansqerr(trclass,valid(:,13));
    lerrs(h,3) = meansqerr(lclass,valid(:,13));
end
% Order:
% discr, quadr. discr, svm, knn, tree, regr. tree, least squares

% Absolute fraction of errors
[mean(nerrs(:,1)),mean(derrs(:,1)),mean(qderrs(:,1)),mean(serrs(:,1)),mean(kerrs(:,1)),mean(terrs(:,1)),mean(trerrs(:,1)),mean(lerrs(:,1))]
% Fscores
[mean(nerrs(:,2)),mean(derrs(:,2)),mean(qderrs(:,2)),mean(serrs(:,2)),mean(kerrs(:,2)),mean(terrs(:,2)),mean(trerrs(:,2)),mean(lerrs(:,2))]
% Square mean error
[mean(nerrs(:,3)),mean(derrs(:,3)),mean(qderrs(:,3)),mean(serrs(:,3)),mean(kerrs(:,3)),mean(terrs(:,3)),mean(trerrs(:,3)),mean(lerrs(:,3))]
