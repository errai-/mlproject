function [ weights, fscores ] = evaluate_quality( pred_class, true_class )
    fscores = zeros(2,1);
    weights = zeros(2,1);
    for i=0:1 % loop over classes
        ntp = sum( true_class( (pred_class == i), : ) == i );
        nfp = sum( true_class( (pred_class == i), : ) ~= i );
        nfn = sum( true_class( (pred_class ~= i), : ) == i );
        ntn = sum( true_class( (pred_class ~= i), : ) ~= i );
        fscores(i+1) = f_score(ntp, ntn, nfp, nfn);
        weights(i+1) = size( true_class( (true_class == i), : ), 1 );
    end
end
