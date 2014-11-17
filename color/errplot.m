function errplot(preds, name)

winefacts = readtable('../test_dataset.csv');

%% Set convenient training and validation sets
training = winefacts(:,:);

traing = [table2array(training(:,1:11)), strcmp(training.type, 'Red')];

%% PCA
centered = detrend(traing(:, 1:11), 'constant');

[vecs,coeffs, NONE, NONE, explained] = pca(centered,'NumComponents',2);

pcapts = [coeffs, traing(:, 12), preds];

pcapts

redOK = pcapts(( pcapts(:, 3) == 1 & pcapts(:, 4) == 1 ),:);
redNO = pcapts(( pcapts(:, 3) == 1 & pcapts(:, 4) == 0 ),:);
whiOK = pcapts(( pcapts(:, 3) == 0 & pcapts(:, 4) == 0 ),:);
whiNO = pcapts(( pcapts(:, 3) == 0 & pcapts(:, 4) == 1 ),:);

clf;
hold all;

plot(redOK(:,1),redOK(:,2),'r.', 'DisplayName', 'Red, Correct');
plot(whiOK(:,1),whiOK(:,2),'b.', 'DisplayName', 'White, Correct');
plot(redNO(:,1),redNO(:,2),'r^', 'DisplayName', 'Red, Incorrect');
plot(whiNO(:,1),whiNO(:,2),'b^', 'DisplayName', 'White, Incorrect');

legend('-DynamicLegend');

print('-depsc','-r300',name);

fprintf('      OK   NO\n');
fprintf('Red %4d %4d\n', size(redOK, 1), size(redNO, 1));
fprintf('Whi %4d %4d\n', size(whiOK, 1), size(whiNO, 1));

nerrs = size(redNO, 1) + size(whiNO, 1);
fprintf('Error rate: %.2f%%\n', nerrs / (size(traing, 1) * 1.0) * 100.0);

end
