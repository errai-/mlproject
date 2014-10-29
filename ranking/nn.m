winefacts = readtable('training_dataset.csv');

tra = 4500; val = 500;

indices = 1:5000;
training = winefacts(indices(1:tra),:);
validation = winefacts(indices(tra+1:tra+val),:);

traing = [table2array(training(:,1:11)),strcmp(training.type,'Red'), training.quality]';
trainTarg = zeros(7,tra);
testng = [table2array(validation(:,1:11)), strcmp(validation.type,'Red'),validation.quality]';

for i=1:tra
    trainTarg(traing(13,i),i)=1;
end

longshot = 0;

%net = feedforwardnet(15);%,'trainbr');
net = patternnet([30,15],'trainscg');
net.divideParam.trainRatio = 100/100;%80
net.divideParam.valRatio   = 0/100;%20
net.divideParam.testRatio  = 0/100;
net.trainParam.max_fail = 20;
net.trainParam.epochs = 1000;

numNN = 40;
NN = cell(1,numNN);

yTot = zeros(7,tra); zTot = zeros(7,val);
for i=1:numNN
    i
    ind = datasample((1:tra)',tra);
    variables = randperm(12);
    vars = variables(1:10);
    NN{i} = train(net,traing(vars,ind),trainTarg(:,ind));
    y = NN{i}(traing(vars,:));
    yTot = yTot+y;
    z = NN{i}(testng(vars,:));
    zTot = zTot+z;
end
yTot = yTot/numNN;
zTot = zTot/numNN;

classes = vec2ind(yTot);
tclasses = vec2ind(zTot);

errs = sum( classes~=traing(13,:) )/tra
rerrs = sum( tclasses~=testng(13,:) )/val

if (longshot)

    nat = patternnet([30,15],'trainscg');
    nat.divideParam.trainRatio = 100/100;%80
    nat.divideParam.valRatio   = 0/100;%20
    nat.divideParam.testRatio  = 0/100;
    nat.trainParam.max_fail = 20;
    nat.trainParam.epochs = 1000;

    numNN = 100;
    NN = cell(1,numNN);

    yTot = zeros(7,tra); zTot = zeros(7,val);
    for i=1:numNN
        i
        ind = datasample((1:tra)',tra);
        variables = randperm(12);
        vars = variables(1:6);
        NN{i} = train(nat,traing(vars,ind),trainTarg(:,ind));
        y = NN{i}(traing(vars,:));
        yTot = yTot+y;
        z = NN{i}(testng(vars,:));
        zTot = zTot+z;
    end
    yTot = yTot/numNN;
    zTot = zTot/numNN;

    classes = vec2ind(yTot);
    tclasses = vec2ind(zTot);

    errs = sum( classes~=traing(13,:) )/tra
    rerrs = sum( tclasses~=testng(13,:) )/val
end