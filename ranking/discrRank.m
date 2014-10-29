% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');

tra=4500; val=500;

errs = 1:10;
rerrs = 1:10;
options = optimoptions('fminunc','GradObj','on'); % indicate gradient is provided
options.MaxIter=3;
options.MaxFunEvals=3;

for h=1:1
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    %W = rand(12,7)/50-0.01;

    traing = [table2array(training(:,1:11)),strcmp('Red',training.type ), training.quality ];
    testng = [table2array(validation(:,1:11)),strcmp('Red',validation.type ), validation.quality ];
        
    %W = fminunc(@(W) like(W,traing),W,options);
    for ses=1:1000
        [fo,go] = like(W,traing);
        %go
        sus = 1;
        step = 1/max(abs(max(go./W)));
        [fg,gg] = like(W+sus*step*go,traing);
        while (fg >= fo)
            sus = sus/2;
            if (sus == 0)
                sus = 1/100000;
                break;
            end
            [fg,gg] = like(W+sus*step*go,traing);
        end
        fo
        sus
        W = W+sus*step*go;
    end
    
    exPones = zeros(tra,7); predict = zeros(tra,7);
    for ind=1:7
        exPones(:,ind)=(exp( -W(1,ind)-W(2:end,ind)'*traing(:,1:11)' ))';
    end
    for ind=1:size(exPones,1)
        predict(ind,:) = exPones(ind,:)/sum( exPones(ind,:) );
    end
    [ma,I1] = max(predict,[],2);
    correcto = ( (I1 == traing(:,13))==0 );
    errors = sum(correcto);

    rexPones = zeros(val,7); rpredict = zeros(val,7);
    for ind=1:7
        rexPones(:,ind)=(exp( -W(1,ind)-W(2:end,ind)'*testng(:,1:11)' ))';
    end
    for ind=1:size(rexPones,1)
        rpredict(ind,:) = rexPones(ind,:)/sum( rexPones(ind,:) );
    end
    [ma,I2] = max(rpredict,[],2);
    rcorrecto = ( (I2 == testng(:,13))==0 );
    rerrors = sum(rcorrecto);

    errs(h) = errors/tra;
    rerrs(h) = rerrors/val;
end
mean(errs)
mean(rerrs)
