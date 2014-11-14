winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs = 1:10;
rerrs = 1:10;
cerrs = 1:10;
lim=11;

for h=1:10
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)),strcmp(training.type,'Red'), training.quality];
    testng = [table2array(validation(:,1:11)), strcmp(validation.type,'Red'),validation.quality];

    x = lscov(traing(:, 1:lim), traing(:, 13));
    fresults = traing(:, 1:lim) * x;
    gresults = testng(:, 1:lim) * x;

    errs(h) = sum( abs(traing(:, 13) - fresults) )/tra;
    rerrs(h) = sum( abs(testng(:, 13) - gresults) )/val;
    cerrs(h) = sum( round(gresults)~=testng(:,13) )/val;
end
mean(errs)
mean(rerrs)
mean(cerrs)
