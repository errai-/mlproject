% This script is used to classify a wine to either red or white
winefacts = readtable('training_dataset.csv');


tra=4500; val=500;

errs = 1:10;
rerrs = 1:10;
options = optimoptions('fminunc','GradObj','on'); % indicate gradient is provided

m=(1:11)';
S=zeros(11,77); invS=zeros(11,77); prior=(1:7)'; ldetS = (1:7)';

for h=1:1
    lower = 1+500*(h-1); upper = 500*h;
    indices = [ 1:(lower-1), (upper+1):5000, lower:upper ];

    training = winefacts(indices(1:tra),:);
    validation = winefacts(indices(tra+1:tra+val),:);

    traing = [table2array(training(:,1:11)),strcmp('Red',training.type ), training.quality ];
    testng = [table2array(validation(:,1:11)),strcmp('Red',validation.type ), validation.quality ];

    m = (mean( traing(:,1:11) ))';
    
    for i=1:7
        prior(i) = log(sum( traing(:,13)==i )/size(traing,1));
        S(:,(1+11*(i-1)):11*i)=cov( traing(traing(:,13)==i,1:11) );
        if (log(rcond(S(:,(1+11*(i-1)):11*i)))>-20)
            invS(:,(1+11*(i-1)):11*i)=inv( S(:,(1+11*(i-1)):11*i) );
        else
            invS(:,(1+11*(i-1)):11*i)=pinv( S(:,(1+11*(i-1)):11*i) );
        end
        ldetS(i) = log( det( S(:,(1+11*(i-1)):11*i) ) );
        ldetS(1) = ldetS(2)+5;
    end

    res = (1:tra)';
    for i=1:tra
        g = logDiscrGauss( traing(i,1:11)'-m, invS(:,1:11),ldetS(1),prior(1) );
        ind = 1;
        for j=2:7
            tmpG = logDiscrGauss( traing(i,1:11)'-m, invS(:,1+11*(j-1):11*j),ldetS(j),prior(j) );
            if (tmpG>g)
                g = tmpG;
                ind = j;
            end
        end
        res(i) = ind;
    end
    vres = (1:val)';
    for i=1:val
        g = logDiscrGauss( testng(i,1:11)'-m, invS(:,1:11),ldetS(1),prior(1) );
        ind = 1;
        g
        for j=2:7
            tmpG = logDiscrGauss( testng(i,1:11)'-m, invS(:,1+11*(j-1):11*j),ldetS(j),prior(j) );
            tmpG
            if (tmpG>g)
                g = tmpG;
                ind = j;
            end
        end
        vres(i) = ind;
    end
    
    errs(h)=sum( res~=traing(:,13) )/tra;
    rerrs(h)=sum( vres~=testng(:,13) )/val;
end
