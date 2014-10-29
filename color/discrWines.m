% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs = 1:10;
rerrs = 1:10;
options = optimoptions('fminunc','GradObj','on'); % indicate gradient is provided
options.MaxIter=1000;

for h=1:10
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    W = rand(12,1)/50-0.01;

    traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];
    testng = [table2array(validation(:,1:11)), strcmp(validation.type, 'Red')];
    W = fminunc(@(W) like(W,traing),W,options);

    predict = (1./(1 + exp( -W(1)-W(2:end)'*traing(:,1:11)' ) ))';
    correcto = ( ((predict>0.5) == traing(:,12))==0 );
    errors = sum(correcto);

    rpredict = (1./(1 + exp( -W(1)-W(2:end)'*testng(:,1:11)' ) ))';
    rcorrecto = ( ((rpredict>0.5) == testng(:,12))==0 );
    rerrors = sum(rcorrecto);

    errs(h) = errors/tra;
    rerrs(h) = rerrors/val;
end
mean(errs)
mean(rerrs)