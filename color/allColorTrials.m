% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

tra=4500; val=500; tes=1000;


%% 10-fold cross-validation %%

nerrs = 1:10; npriorerrs = 1:10; gerrs = 1:10; gpriorerrs = 1:10;
derrs = 1:10; kerrs = 1:10; terrs = 1:10; unerrs = 1:10;

for h=1:10
    %% Boundaries for cross-validation
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    %% Set convenient training and validation sets
    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];
    valid = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red')];

    %% Prior values, if needed
    rPrior = sum(traing(:,12))/tra; wPrior = 1-rPrior;

    %% Ultimate naivity
    unclass = zeros(val,1);
    
    %% Naive (only 1 variable) Gaussian discrimination
    m1 = mean(traing(traing(:,12)==1,8)); % Reds
    v1 = var(traing(traing(:,12)==1,8));
    m2 = mean(traing(traing(:,12)==0,8)); % Whites
    v2 = var(traing(traing(:,12)==0,8));

    % With no prior and with prior
    nclass1 = (logDiscrNaive( valid(:,8),m1,m2,v1,v2,1,1)>0);
    nclass2 = (logDiscrNaive( valid(:,8),m1,m2,v1,v2,rPrior,wPrior)>0);

    %% Gaussian discrimination
    m1 = mean(traing(traing(:,12)==1,1:11)); % Reds
    v1 = cov(traing(traing(:,12)==1,1:11));
    m2 = mean(traing(traing(:,12)==0,1:11)); % Whites
    v2 = cov(traing(traing(:,12)==0,1:11));

    gclass1=(1:val)'; gclass2=(1:val)'; % No prior and prior
    for i = 1:val
        gclass1(i) = (logDiscrGauss(valid(i,1:11),m1,m2,v1,v2,1,1)>0);
        gclass2(i) = (logDiscrGauss(valid(i,1:11),m1,m2,v1,v2,rPrior,wPrior)>0);
    end

    %% Linear discriminant
    diskr = fitcdiscr(traing(:,1:11),traing(:,12),'DiscrimType','Linear');
    dclass=predict(diskr,valid(:,1:11));

    %% kNN, k=1 or k=3
    kNN = fitcknn(traing(:,1:11),traing(:,12),'Distance','mahalanobis','NumNeighbors',3);
    kclass=predict(kNN,valid(:,1:11));

    %% Random forest 30
    BaggedTreeEns = TreeBagger(200,traing(:,1:11),traing(:,12),'NVarToSample',2);
    [tclass,treeProbs]=predict(BaggedTreeEns,valid(:,1:11));

    %% Standard errors
    unerrs(h) = sum( unclass ~= valid(:,12) )/val;
    nerrs(h) = sum( nclass1 ~= valid(:,12) )/val;
    npriorerrs(h) = sum( nclass2 ~= valid(:,12) )/val;
    gerrs(h) = sum( gclass1 ~= valid(:,12) )/val;
    gpriorerrs(h) = sum( gclass2 ~= valid(:,12) )/val;
    derrs(h) = sum( dclass ~= valid(:,12) )/val;
    kerrs(h) = sum( kclass ~= valid(:,12) )/val;
    terrs(h) = sum( round(treeProbs(:,2)) ~= valid(:,12) )/val;
end

%% Output errors
% logarithmic discrimination with and without prior
%[mean(unerrs),mean(nerrs),mean(npriorerrs),mean(gerrs),mean(gpriorerrs)]
% equal, mean from loop, and afterwards, minimal possible error for each h
%[mean(derrs),mean(kerrs),mean(terrs),mean(serrs)] % Single errors


%% Testing: %%

training = [table2array(winefacts(:,1:11)), strcmp(winefacts.type, 'Red')];
testing = [table2array(winetests(:,1:11)), strcmp(winetests.type, 'Red')];

% Use whole training data for final training:

% Naive (only 1 variable) Gaussian discrimination
m1 = mean(training(training(:,12)==1,8)); % Reds
v1 = var(training(training(:,12)==1,8));
m2 = mean(training(training(:,12)==0,8)); % Whites
v2 = var(training(training(:,12)==0,8));
% Gaussian discrimination
gm1 = mean(training(training(:,12)==1,1:11)); % Reds
gv1 = cov(training(training(:,12)==1,1:11));
gm2 = mean(training(training(:,12)==0,1:11)); % Whites
gv2 = cov(training(training(:,12)==0,1:11));
% Linear discriminant
diskr = fitcdiscr(training(:,1:11),training(:,12),'DiscrimType','Linear');
% kNN, k=1 or k=3
kNN = fitcknn(training(:,1:11),training(:,12),'Distance','mahalanobis','NumNeighbors',1);
% Random forest
BaggedTreeEns = TreeBagger(200,training(:,1:11),training(:,12),'NVarToSample',2);

%% Predict:

% Ultimate naivity
unclass = zeros(tes,1);
% Naive discrimination
nclass1 = (logDiscrNaive( testing(:,8),m1,m2,v1,v2,1,1)>0);
nclass2 = (logDiscrNaive( testing(:,8),m1,m2,v1,v2,rPrior,wPrior)>0);
% Gaussian discrimination
gclass1=(1:tes)'; gclass2=(1:tes)'; % No prior and prior
for i = 1:tes
    gclass1(i) = (logDiscrGauss(testing(i,1:11),gm1,gm2,gv1,gv2,1,1)>0);
    gclass2(i) = (logDiscrGauss(testing(i,1:11),gm1,gm2,gv1,gv2,rPrior,wPrior)>0);
end
% Linear discriminant
dclass=predict(diskr,testing(:,1:11));
% kNN, k=1 or k=3[kclass,knnProbs]=predict(kNN,valid(:,1:11));
kclass=predict(kNN,testing(:,1:11));
% Random forest
[tclass,treeProbs]=predict(BaggedTreeEns,testing(:,1:11));

% errors:
[sum(nclass1~=testing(:,12)),sum(nclass2~=testing(:,12)),sum(gclass1~=testing(:,12)),sum(gclass2~=testing(:,12))]/tes
[sum(dclass~=testing(:,12)),sum(kclass~=testing(:,12)),sum(round(treeProbs(:,2))~=testing(:,12))]/tes


figure(1); errplot(nclass1, 'naive_noprior');
figure(2); errplot(nclass2, 'naive_prior');
figure(3); errplot(gclass1, 'gauss_noprior');
figure(4); errplot(gclass2, 'gauss_prior');
figure(5); errplot(dclass, 'linear_discr');
figure(6); errplot(kclass, 'knn1');
figure(7); errplot(cellfun(@str2num,tclass), 'random_forest');
figure(8); errplot(unclass, 'ultimately_naive');
