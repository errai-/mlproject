% This script is used to classify a wine to either red or white

winefacts = readtable('../training_dataset.csv');
winetests = readtable('../test_dataset.csv');

%% Set convenient training and validation sets
training = winefacts(1:5000,:);

traing = [table2array(training(:,1:11)),training.quality,strcmp(training.type, 'Red')];

%% PCA
[vecs,coeffs] = pca(traing(:,1:11)./repmat(sqrt(var(traing(:,1:11))),5000,1),'NumComponents',2);

%pcapts = [(traing(:, 1:11) * vecs, traing(:, 12)]
pcapts = [coeffs, traing(:, end)];

red = pcapts(pcapts(:, end) == 1,:);
white = pcapts(pcapts(:, end) == 0,:);

plot(red(:,1),red(:,2),'r+');
hold on
plot(white(:,1),white(:,2),'y+');

%print('-depsc','-r300','pca')

hold off