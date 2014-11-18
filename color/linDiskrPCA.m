% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

tra=4500; val=500; tes=1000;


%% 10-fold cross-validation %%

derrs = zeros(10,11);kerrs = zeros(10,11);terrs = zeros(10,11);

for h=1:10
    %% Boundaries for cross-validation
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    %% Set convenient training and validation sets
    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];
    valid = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red')];
    valid(:,1:11) = valid(:,1:11)./repmat(sqrt(var(traing(:,1:11))),size(valid,1),1);
    traing(:,1:11) = traing(:,1:11)./repmat(sqrt(var(traing(:,1:11))),size(traing,1),1);
    
    %% PCA
    [vecs,coeffs] = pca(traing(:,1:11));
    means = mean(traing(:,1:11));
    for k = 11:-1:1
        %% Linear discriminant with the given data
        diskr = fitcdiscr(coeffs(:,1:k),traing(:,12),'DiscrimType','Linear');
        dclass=predict(diskr,(valid(:,1:11)-repmat(means,val,1))*vecs(:,1:k));
        
        %% KNN with the given data
        kNN = fitcknn(coeffs(:,1:k),traing(:,12),'Distance','mahalanobis','NumNeighbors',3);
        kclass=predict(kNN,(valid(:,1:11)-repmat(means,val,1))*vecs(:,1:k));
        
        %% Random forest
        BaggedTreeEns = TreeBagger(200,coeffs(:,1:k),traing(:,12),'NVarToSample',2);
        [tclass,treeProbs]=predict(BaggedTreeEns,(valid(:,1:11)-repmat(means,val,1))*vecs(:,1:k));
        
        %% Standard errors
        derrs(h,k) = sum( dclass ~= valid(:,12) )/val;
        kerrs(h,k) = sum( kclass ~= valid(:,12) )/val;
        terrs(h,k) = sum( round(treeProbs(:,2)) ~= valid(:,12) )/val;
    end
end

%% Output errors
% equal, mean from loop, and afterwards, minimal possible error for each h
mean(derrs) % Single errors
mean(kerrs)
mean(terrs)
%% Testing: %%

training = [table2array(winefacts(:,1:11)), strcmp(winefacts.type, 'Red')];
testing = [table2array(winetests(:,1:11)), strcmp(winetests.type, 'Red')];
testing(:,1:11) = testing(:,1:11)./repmat(sqrt(var(training(:,1:11))),size(testing,1),1);
training(:,1:11) = training(:,1:11)./repmat(sqrt(var(training(:,1:11))),size(training,1),1);

% Use whole training data for final training:

Derrs=zeros(11,1);Kerrs=zeros(11,1);Terrs=zeros(11,1);

%% PCA
[vecs,coeffs] = pca(training(:,1:11));
means = mean(training(:,1:11));

for k = 11:-1:1
    %% Linear discriminant with the given data
    diskr = fitcdiscr(coeffs(:,1:k),training(:,12),'DiscrimType','Linear');
    dclass=predict(diskr,(testing(:,1:11)-repmat(means,tes,1))*vecs(:,1:k));
    
    %% KNN with the given data
    kNN = fitcknn(coeffs(:,1:k),training(:,12),'Distance','mahalanobis','NumNeighbors',3);
    kclass=predict(kNN,(testing(:,1:11)-repmat(means,tes,1))*vecs(:,1:k));
            
    %% Random forest
    BaggedTreeEns = TreeBagger(200,coeffs(:,1:k),training(:,12),'NVarToSample',2);
    [tclass,treeProbs]=predict(BaggedTreeEns,(testing(:,1:11)-repmat(means,tes,1))*vecs(:,1:k));
    
    %% Standard errors
    Derrs(k) = sum( dclass ~= testing(:,12) )/tes;
    Kerrs(k) = sum( kclass ~= testing(:,12) )/tes;
    Terrs(k) = sum( round(treeProbs(:,2)) ~= testing(:,12) )/tes;
end


% errors:
Derrs'
Kerrs'
Terrs'