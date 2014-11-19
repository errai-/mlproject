% This script is used to classify a wine to either red or white
% Usage: line 15 should be uncommented only during the first run
% line 70 should be uncommented only if new values for l's and f's are
% given

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

tra=4500; val=500; tes=1000;

% Naive priors
priorDiscr = 1/3; priorNN = 1/3; priorForest = 1/3;
% Initial priors for more complicated cases, comment out if these are being
% modified
l1=1/3;l2=1/3;l3=1/3;f1=1/3;f2=1/3;f3=1/3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 10-fold cross-validation %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

errs = 1:10; opterrs = 1:10; lerrs = 1:10; ferrs = 1:10;
X1 = 1:10; X2 = 1:10;

alld=zeros(val,10);allk=zeros(val,10);allt=zeros(val,10);allc=zeros(val,10);

err_rates = repmat([0], 10, 3)
f_scores = repmat([0], 10, 3)

for h=1:10
    % Boundaries for cross-validation
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    % Set convenient training and validation sets
    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];
    valid = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red')];

    % Linear discriminant
    diskr = fitcdiscr(traing(:,1:11),traing(:,12),'DiscrimType','Linear');
    [dclass,diskrProbs]=predict(diskr,valid(:,1:11));

    % kNN, k=1 or k=3
    kNN = fitcknn(traing(:,1:11),traing(:,12),'Distance','mahalanobis','NumNeighbors',3);
    [kclass,knnProbs]=predict(kNN,valid(:,1:11));

    % Random forest 30
    BaggedTreeEns = TreeBagger(200,traing(:,1:11),traing(:,12),'NVarToSample',2);
    [tclass,treeProbs]=predict(BaggedTreeEns,valid(:,1:11));

    % Save probabilities
    alld(:,h)=diskrProbs(:,2);
    allk(:,h)=knnProbs(:,2);
    allt(:,h)=treeProbs(:,2);
    allc(:,h)=valid(:,12);

%    [err_rates(h, 1), f_scores(h, 1)] = errplotg(valid, round(priorDiscr*diskrProbs(:,2) + priorNN*knnProbs(:,2) + priorForest*treeProbs(:,2)), 'cvv2n');
%    [err_rates(h, 2), f_scores(h, 2)] = errplotg(valid, round(l1*diskrProbs(:,2) + l2*knnProbs(:,2) + l3*treeProbs(:,2)), 'cvv2l');
%    [err_rates(h, 3), f_scores(h, 3)] = errplotg(valid, round(f1*diskrProbs(:,2) + f2*knnProbs(:,2) + f3*treeProbs(:,2)), 'cvv2f');

    % Standard errors
    errs(h) = sum( round(priorDiscr*diskrProbs(:,2) + priorNN*knnProbs(:,2) + priorForest*treeProbs(:,2)) ~= valid(:,12) )/val;
    lerrs(h) = sum( round(l1*diskrProbs(:,2) + l2*knnProbs(:,2) + l3*treeProbs(:,2)) ~= valid(:,12) )/val;
    ferrs(h) = sum( round(f1*diskrProbs(:,2) + f2*knnProbs(:,2) + f3*treeProbs(:,2)) ~= valid(:,12) )/val;
    % Optimize coefficients
    [x1m,x2m,tmpErr]=optimumCoeff(diskrProbs(:,2),knnProbs(:,2),treeProbs(:,2),valid(:,12),0.01,0.1,0.5,0.1,0.5);
    opterrs(h) = tmpErr/val;
    X1(h)=x1m;
    X2(h)=x2m;
end
[z1m,z2m,tmpErr]=optimumCoeffMulti(alld,allk,allt,allc,0.01,0.1,0.5,0.1,0.5);
l1 = mean(X1); l2 = mean(X2); l3 = 1-l1-l2; f1 = z1m; f2 = z2m; f3 = 1-f1-f2;

[l1, l2, l3, f1, f2, f3]

% Output errors
[mean(errs),mean(lerrs),mean(ferrs),mean(opterrs)] % coeffs:
% equal, mean from loop, and afterwards, minimal possible error for each h

fprintf('ERRR\n');
err_rates
fprintf('FSCR\n');
f_scores

%%%%%%%%%%%%
% Testing: %
%%%%%%%%%%%%

training = [table2array(winefacts(:,1:11)), strcmp(winefacts.type, 'Red')];
testing = [table2array(winetests(:,1:11)), strcmp(winetests.type, 'Red')];

% Use whole training data for final training:

% Linear discriminant
diskr = fitcdiscr(training(:,1:11),training(:,12),'DiscrimType','Linear');
% kNN, k=1 or k=3
kNN = fitcknn(training(:,1:11),training(:,12),'Distance','mahalanobis','NumNeighbors',3);
% Random forest
BaggedTreeEns = TreeBagger(200,training(:,1:11),training(:,12),'NVarToSample',2);

% Predict:

% Linear discriminant
[dclass,diskrProbs]=predict(diskr,testing(:,1:11));
% kNN, k=1 or k=3[kclass,knnProbs]=predict(kNN,valid(:,1:11));
[kclass,knnProbs]=predict(kNN,testing(:,1:11));
% Random forest
[tclass,treeProbs]=predict(BaggedTreeEns,testing(:,1:11));

predikt = round(priorDiscr*diskrProbs(:,2) + priorNN*knnProbs(:,2) + priorForest*treeProbs(:,2));
lpredikt = round(l1*diskrProbs(:,2) + l2*knnProbs(:,2) + l3*treeProbs(:,2));
fpredikt = round(f1*diskrProbs(:,2) + f2*knnProbs(:,2) + f3*treeProbs(:,2));

valid = testing
errplot(valid, round(priorDiscr*diskrProbs(:,2) + priorNN*knnProbs(:,2) + priorForest*treeProbs(:,2)), 'cvv2f_p');
errplot(valid, round(l1*diskrProbs(:,2) + l2*knnProbs(:,2) + l3*treeProbs(:,2)), 'cvv2f_l');
errplot(valid, round(f1*diskrProbs(:,2) + f2*knnProbs(:,2) + f3*treeProbs(:,2)), 'cvv2f_f');

% errors: (lpredikt is usually the best, fpredikt almost equal)
[sum(predikt~=testing(:,12)),sum(lpredikt~=testing(:,12)),sum(fpredikt~=testing(:,12))]/tes
[l1,l2,f1,f2]
