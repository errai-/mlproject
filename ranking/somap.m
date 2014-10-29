winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs = 1:10;
rerrs = 1:10;
options = optimoptions('fminunc','GradObj','on'); % indicate gradient is provided
options.MaxIter=1000;
lim=11;

for h=1:1
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)),strcmp(training.type,'Red'), training.quality]';
    testng = [table2array(validation(:,1:11)), strcmp(validation.type,'Red'),validation.quality]';

    trainTarg = zeros(7,tra);
    testTarg = zeros(7,val);
    
    for i=1:tra
        trainTarg(traing(13,i),i)=1;
    end
    net = selforgmap( [1 7] );
    %net = feedforwardnet(25);
    %net = patternnet(25);
    %net = lvqnet(25);
    net = train(net,traing(1:12,:),trainTarg);
    %net = trainlm(net,traing(1:12,:),trainTarg);
    y = net( traing(1:12,:) );
    classes = vec2ind(y);
    ry = net( testng(1:12,:) );
    rclasses = vec2ind(ry);

    errs(h) = sum( classes~=traing(13,:) )/tra;
    rerrs(h) = sum( rclasses~=testng(13,:) )/val;
end
