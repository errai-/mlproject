% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');
winepreds = readtable('challenge_dataset.csv');

tra=size(winefacts,1); val=size(winepreds,1);

reds = winefacts(strcmp(winefacts.type, 'Red'),:);
whites = winefacts(strcmp(winefacts.type, 'White'),:);
rPrior = size(reds,1)/tra; wPrior = size(whites,1)/tra;

m1 = sum(reds.density)/size(reds,1); m2 = sum(whites.density)/size(whites,1);
v1 = sum( (reds.density-m1).^2 )/size(reds,1);
v2 = sum( (whites.density-m2).^2 )/size(whites,1);

redness1 = (logDiscrNaive( winepreds.density,m1,m2,v1,v2,1,1)>0);
redness2 = (logDiscrNaive( winepreds.density,m1,m2,v1,v2,rPrior,wPrior)>0);
r12=sum( abs(redness1-redness2) );

