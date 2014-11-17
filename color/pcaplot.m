% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');

%% Set convenient training and validation sets
training = winefacts(1:5000,:);

traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];

%% PCA
[vecs,coeffs] = pca(traing(:,1:11),'NumComponents',2);

pcapts = [traing(:, 1:11) * vecs, traing(:, 12)]

redc = ( pcapts(:, 3) == 1 )
whitec = ( pcapts(:, 3) == 0 )

red = pcapts(redc,:)
white = pcapts(whitec,:)

plot(red(:,1),red(:,2),'r+')
hold on
plot(white(:,1),white(:,2),'y+')

print('-depsc','-r300','pca')
