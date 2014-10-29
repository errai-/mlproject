% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs = 1:10;
rerrs = 1:10;
options = optimoptions('fminunc','GradObj','on'); % indicate gradient is provided
options.MaxIter=3;
options.MaxFunEvals=3;

for h=1:10
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)),strcmp('Red',training.type ), training.quality ];
    testng = [table2array(validation(:,1:11)),strcmp('Red',validation.type ), validation.quality ];
        
    nero = fitcdiscr(traing(:,1:11),traing(:,13),'DiscrimType','pseudoQuadratic');
    
    res=predict(nero,traing(:,1:11));
    vres=predict(nero,testng(:,1:11));

    correcto = (res ~= traing(:,13));
    errors = sum(correcto);

    rcorrecto = (vres ~= testng(:,13));
    rerrors = sum(rcorrecto);

    errs(h) = errors/tra;
    rerrs(h) = rerrors/val;
end
mean(errs)
mean(rerrs)