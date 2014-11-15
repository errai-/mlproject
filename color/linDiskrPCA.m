% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

tra=4500; val=500; tes=1000;


%% 10-fold cross-validation %%

derrs = zeros(10,11);

for h=1:10
    %% Boundaries for cross-validation
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    %% Set convenient training and validation sets
    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];
    valid = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red')];
    
    %% Linear discriminant with the given data
    diskr = fitcdiscr(traing(:,1:11),traing(:,12),'DiscrimType','Linear');
    dclass=predict(diskr,valid(:,1:11));
    derrs(h,11) = sum( dclass ~= valid(:,12) )/val;
    
    for k = 10:-1:1
        %% PCA
        [vecs,coeffs] = pca(traing(:,1:11),'NumComponents',k);
        means = mean(traing(:,1:11));
        %% Linear discriminant with the given data
        diskr = fitcdiscr(coeffs,traing(:,12),'DiscrimType','Linear');
        dclass=predict(diskr,(vecs'*(valid(:,1:11)'-repmat(means,val,1)'))');
           
        %% Standard errors
        derrs(h,k) = sum( dclass ~= valid(:,12) )/val;
    end
end

%% Output errors
% equal, mean from loop, and afterwards, minimal possible error for each h
mean(derrs) % Single errors


%% Testing: %%

training = [table2array(winefacts(:,1:11)), strcmp(winefacts.type, 'Red')];
testing = [table2array(winetests(:,1:11)), strcmp(winetests.type, 'Red')];

% Use whole training data for final training:

Derrs=zeros(11,1);
%% Linear discriminant with the given data
diskr = fitcdiscr(training(:,1:11),training(:,12),'DiscrimType','Linear');
dclass=predict(diskr,testing(:,1:11));
Derrs(11) = sum( dclass ~= testing(:,12) )/tes;
    
for k = 10:-1:1
    %% PCA
    [vecs,coeffs] = pca(training(:,1:11),'NumComponents',k);
    means = mean(training(:,1:11));
    %% Linear discriminant with the given data
    diskr = fitcdiscr(coeffs,training(:,12),'DiscrimType','Linear');
    dclass=predict(diskr,(vecs'*(testing(:,1:11)'-repmat(means,tes,1)'))');
           
    %% Standard errors
    Derrs(k) = sum( dclass ~= testing(:,12) )/tes;
end

% errors:
Derrs'