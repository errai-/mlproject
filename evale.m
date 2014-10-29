function [ fscores, weights ] = evale( pred_class, true_class )
    
    t1 = []; t2 = [];
    for i=1:7
       pos1 = (pred_class == i);
       pos2 = (true_class == i);
       neg1 = pos1==0;
       neg2 = pos2==0;
       tp = sum(pos1(pos2,:));
       fp = sum(pos2(pos1,:)==0);
       tn = sum(neg1(neg2,:));
       fn = sum(neg2(neg1,:)==0);
       t1 = [t1; f_score(tp,tn,fp,fn)];
       t2 = [t2; sum(pos2)]
    end
    fscores = t1;
    weights = t2;
end

