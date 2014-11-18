% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

tra=4500; val=500; tes=1000;

derrs = zeros(1,3); qderrs = zeros(1,3); serrs = zeros(1,3); 
kerrs = zeros(1,3); lerrs = zeros(1,3); trerrs = zeros(1,3); 
terrs = zeros(1,3); nerrs = zeros(1,3);

%% Set training and testing sets

training = [table2array(winefacts(:,1:11)), strcmp(winefacts.type, 'Red'), winefacts.quality];
testing = [table2array(winetests(:,1:11)), strcmp(winetests.type, 'Red'), winetests.quality];

%% Ultimately naive prediction
nclass = ones(tes,1)*4;

%% Linear discriminant
diskr = fitcdiscr(training(:,1:11),training(:,13),'DiscrimType','Linear');
dclass=predict(diskr,testing(:,1:11));

%% Quadratic discriminant
qdiskr = fitcdiscr(training(:,1:11),training(:,13),'DiscrimType','pseudoQuadratic');
qdclass = predict(qdiskr,testing(:,1:11));

%% SVM
svm = fitcecoc(training(:,1:11),training(:,13));
sclass = predict(svm,testing(:,1:11));

%% kNN, k=3
kNN = fitcknn(training(:,1:11),training(:,13),'Distance','mahalanobis','NumNeighbors',3);
kclass=predict(kNN,testing(:,1:11));

%% Random forest
BaggedTreeEns = TreeBagger(1000,training(:,1:11),training(:,13),'NVarToSample',2);
tclass=predict(BaggedTreeEns,testing(:,1:11));
tclass=cell2mat(tclass); tclass=tclass-48;

%% Random forest, regression
BaggedTreeEns = TreeBagger(1000,training(:,1:11),training(:,13),'NVarToSample',2,'Method','regression');
trclass=round(predict(BaggedTreeEns,testing(:,1:11)));    

%% Least squares, regression
w = lscov(training(:, 1:11), training(:, 13));
lclass = round(testing(:, 1:11) * w);

%% Standard errors
nerrs(1) = sum( nclass ~= testing(:,13) )/tes;
derrs(1) = sum( dclass ~= testing(:,13) )/tes;
qderrs(1) = sum( qdclass ~= testing(:,13) )/tes;
serrs(1) = sum( sclass ~= testing(:,13) )/tes;
kerrs(1) = sum( kclass ~= testing(:,13) )/tes;
terrs(1) = sum( tclass ~= testing(:,13) )/tes;
trerrs(1) = sum( trclass ~= testing(:,13) )/tes;
lerrs(1) = sum( lclass ~= testing(:,13) )/tes;

nerrs(2) = tot_fscore(nclass,testing(:,13));
derrs(2) = tot_fscore(dclass,testing(:,13));
qderrs(2) = tot_fscore(qdclass,testing(:,13));
serrs(2) = tot_fscore(sclass,testing(:,13));
kerrs(2) = tot_fscore(kclass,testing(:,13));
terrs(2) = tot_fscore(tclass,testing(:,13));
trerrs(2) = tot_fscore(trclass,testing(:,13));
lerrs(2) = tot_fscore(lclass,testing(:,13));

nerrs(3) = meansqerr(nclass,testing(:,13));
derrs(3) = meansqerr(dclass,testing(:,13));
qderrs(3) = meansqerr(qdclass,testing(:,13));
serrs(3) = meansqerr(sclass,testing(:,13));
kerrs(3) = meansqerr(kclass,testing(:,13));
terrs(3) = meansqerr(tclass,testing(:,13));
trerrs(3) = meansqerr(trclass,testing(:,13));
lerrs(3) = meansqerr(lclass,testing(:,13));

% Order:
% discr, quadr. discr, svm, knn, tree, regr. tree, least squares

% Absolute fraction of errors
[nerrs(1),derrs(:,1),qderrs(:,1),serrs(:,1),kerrs(:,1),terrs(:,1),trerrs(:,1),lerrs(:,1)]
% Fscores
[nerrs(:,2),derrs(:,2),qderrs(:,2),serrs(:,2),kerrs(:,2),terrs(:,2),trerrs(:,2),lerrs(:,2)]
% Square mean error
[nerrs(:,3),derrs(:,3),qderrs(:,3),serrs(:,3),kerrs(:,3),terrs(:,3),trerrs(:,3),lerrs(:,3)]