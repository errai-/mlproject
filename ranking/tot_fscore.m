function [ score ] = tot_fscore( pred_class, true_class )
    [weights,fscores] = evaluate_quality( pred_class, true_class );
    score = (weights'*fscores)/sum(weights);
end

