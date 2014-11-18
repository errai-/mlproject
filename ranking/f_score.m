function [ fscore ] = f_score( n_tp, n_tn, n_fp, n_fn )
    if (n_tp == 0)
        fscore = 0;
    else
        precision = n_tp/(n_tp + n_fp);
        recall = n_tp/(n_tp + n_fn);
        fscore = 2*precision*recall/(precision+recall);
    end 
end
