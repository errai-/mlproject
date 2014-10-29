% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');
winepreds = readtable('challenge_dataset.csv');

reds = winefacts(strcmp(winefacts.type, 'Red'),:);
redArray = table2array(reds(:,1:11));
whites = winefacts(strcmp(winefacts.type, 'White'),:);
whiteArray = table2array(whites(:,1:11));
rPrior = size(reds,1)/tra; wPrior = size(whites,1)/tra;

m1 = mean(redArray);
m2 = mean(whiteArray);
v1 = cov(redArray);
v2 = cov(whiteArray);

validArray = table2array(winepreds(:,1:11));

redness1=1:1000; redness2=1:1000;
for i = 1:size(validArray,1)
    redness1(i) = (logDiscrGauss(validArray(i,:),m1,m2,v1,v2,1,1)>0);
    redness2(i) = (logDiscrGauss(validArray(i,:),m1,m2,v1,v2,rPrior,wPrior)>0);
end
r12=sum( abs(redness1-redness2) );
