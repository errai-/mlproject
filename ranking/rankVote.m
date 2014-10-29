winefacts = readtable('training_dataset.csv');
winechall = readtable('test_dataset.csv');

tra=size(winefacts,1); val=size(winechall,1);

lim = 12;

traing = [table2array(winefacts(:,1:11)),strcmp(winefacts.type,'Red'), winefacts.quality]';
testng = [table2array(winechall(:,1:11)),strcmp(winechall.type,'Red')]';
trainTarg = zeros(7,tra);

BaggedTreeEns = TreeBagger(1000,traing(1:lim,:)',traing(13,:)','NVarToSample',2);
[fresults,fprobs]=predict(BaggedTreeEns,traing(1:lim,:)');
fresults=cell2mat(fresults); fresults=fresults-48;
[gresults,gprobs]=predict(BaggedTreeEns,testng(1:lim,:)');
gresults=cell2mat(gresults); gresults=gresults-48;

fterrs = sum( fresults~=traing(13,:)' )/tra;

for i=1:tra
    trainTarg(traing(13,i),i)=1;
end

net = patternnet([30,15],'trainscg');
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio   = 0/100;
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
    vars = variables(1:6);
    NN{i} = train(net,traing(vars,ind),trainTarg(:,ind));
    y = NN{i}(traing(vars,:));
    yTot = yTot+y;
    z = NN{i}(testng(vars,:));
    zTot = zTot+z;
end
yTot = yTot/numNN;
zTot = zTot/numNN;

classes = vec2ind(yTot);
fclasses = vec2ind(zTot);

ferrs = sum( classes~=traing(13,:) )/tra;

a = 1.1066; b=1-a;

trpred = vec2ind(a*fprobs'+b*yTot);
tepred = vec2ind(a*gprobs'+b*zTot);

faerrs = sum( trpred~=traing(13,:) )/tra;

finalerr1 = sum( tepred~=fclasses );
finalerr2 = sum( tepred~=gresults' );
finalerr3 = sum( gresults'~=fclasses );
