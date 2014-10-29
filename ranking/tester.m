winefacts = readtable('training_dataset.csv');

tra = 4500; val = 500;
lim = 12;
%terrs = 1:10; trerrs = 1:10; errs = 1:10; rerrs = 1:10;

%tstore = zeros(7,45000);
%nstore = zeros(7,45000);

%ttstore = zeros(7,5000);
%tnstore = zeros(7,5000);

for h=1:1
    h
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)),strcmp(training.type,'Red'), training.quality]';
    testng = [table2array(validation(:,1:11)), strcmp(validation.type,'Red'),validation.quality]';
    trainTarg = zeros(7,tra);
    
    BaggedTreeEns = TreeBagger(1000,traing(1:lim,:)',traing(13,:)','NVarToSample',2);%,'oobpred','on');
    [fresults,fprobs]=predict(BaggedTreeEns,traing(1:lim,:)');
    fresults=cell2mat(fresults); fresults=fresults-48;
    [gresults,gprobs]=predict(BaggedTreeEns,testng(1:lim,:)');
    gresults=cell2mat(gresults); gresults=gresults-48;

    tstore(:,1+(h-1)*4500:h*4500)=fprobs';
    ttstore(:,1+(h-1)*500:h*500)=gprobs';
    
    terrs(h) = sum( fresults~=traing(13,:)' )/tra;
    trerrs(h) = sum( gresults~=testng(13,:)' )/val;

    for i=1:tra
        trainTarg(traing(13,i),i)=1;
    end

    net = patternnet([30,15],'trainscg');
    net.divideParam.trainRatio = 100/100;%80
    net.divideParam.valRatio   = 0/100;%20
    net.divideParam.testRatio  = 0/100;
    net.trainParam.max_fail = 20;
    net.trainParam.epochs = 1000;

    numNN = 40;
    NN = cell(1,numNN);

    h
    
    yTot = zeros(7,tra); zTot = zeros(7,val);
    for i=1:numNN
        i
        ind = datasample((1:tra)',tra);
        variables = randperm(12);
        vars = variables(1:6);
        NN{i} = train(net,traing(vars,ind),trainTarg(:,ind));
        y = NN{i}(traing(vars,:));
        yTot = yTot+y;
        z = NN{i}(testng(vars,:));
        zTot = zTot+z;
    end
    yTot = yTot/numNN;
    zTot = zTot/numNN;

    nstore(:,1+(h-1)*4500:h*4500)=yTot;
    tnstore(:,1+(h-1)*500:h*500)=zTot;
    
    classes = vec2ind(yTot);
    tclasses = vec2ind(zTot);

    errs(h) = sum( classes~=traing(13,:) )/tra;
    rerrs(h) = sum( tclasses~=testng(13,:) )/val;
end