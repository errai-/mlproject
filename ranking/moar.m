winefacts = readtable('training_dataset.csv');

tra = 4500; val = 500;
lim = 12;
serrs = 1:10; srerrs = 1:10;

%a = (0.416-mean(trerrs))/(0.82-mean(trerrs)-mean(rerrs)); b = 1-a;
a = 1.1066; b=1-a;
for h=1:10
    h
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)),strcmp(training.type,'Red'), training.quality]';
    testng = [table2array(validation(:,1:11)), strcmp(validation.type,'Red'),validation.quality]';

    trpred = vec2ind(a*tstore(:,1+(h-1)*4500:h*4500)+b*nstore(:,1+(h-1)*4500:h*4500));
    tepred = vec2ind(a*ttstore(:,1+(h-1)*500:h*500)+b*tnstore(:,1+(h-1)*500:h*500));
    
    serrs(h) = sum( trpred~=traing(13,:) )/tra;
    srerrs(h) = sum( tepred~=testng(13,:) )/val;
end