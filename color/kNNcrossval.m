% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

tra=4500; val=500; tes=1000;


%% 10-fold cross-validation %%
errs = zeros(10,26);
fscores = zeros(10,26);

for h=1:10
    %% Boundaries for cross-validation
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    %% Set convenient training and validation sets
    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];
    valid = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red')];

    %% kNN
    for i = 1:2:51
        kNN = fitcknn(traing(:,1:11),traing(:,12),'Distance','mahalanobis','NumNeighbors',i);
        kclass=predict(kNN,valid(:,1:11));

        %% Standard errors
        errs(h,(i+1)/2) = sum( kclass ~= valid(:,12) )/val;
        [w,scores] = evaluate_quality( kclass, valid(:,12) );
        fscores(h,(i+1)/2) = w'*scores/(sum(w));
    end
end

mean(errs)
mean(fscores)


%% Testing: %%

training = [table2array(winefacts(:,1:11)), strcmp(winefacts.type, 'Red')];
testing = [table2array(winetests(:,1:11)), strcmp(winetests.type, 'Red')];

Errs = zeros(1,26);
Fscores = zeros(1,26);

% Use whole training data for final training:

%% kNN
for i = 1:2:51
    kNN = fitcknn(training(:,1:11),training(:,12),'Distance','mahalanobis','NumNeighbors',i);
    kclass=predict(kNN,testing(:,1:11));

    %% Standard errors
    Errs((i+1)/2) = sum( kclass ~= testing(:,12) )/tes;
    [w,scores] = evaluate_quality( kclass, testing(:,12) );
    Fscores((i+1)/2) =  w'*scores/sum(w);
end

% errors:
Errs
Fscores
hold off; hold on;
plot(1:2:51,1-mean(fscores),'r');
plot(1:2:51,mean(errs),'b');
xlabel('k');
ylabel('1-fscore/error rate');
legend('1-fscore','error rate');

