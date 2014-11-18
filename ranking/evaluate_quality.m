function [ weights, fscores ] = evaluate_quality( pred_class, true_class )
    fscores = zeros(7,1);
    weights = zeros(7,1);
    for i=1:7 % loop over classes
        ntp = sum( true_class( (pred_class == i), : ) == i );
        nfp = sum( true_class( (pred_class == i), : ) ~= i );
        nfn = sum( true_class( (pred_class ~= i), : ) == i );
        ntn = sum( true_class( (pred_class ~= i), : ) ~= i );
        fscores(i) = f_score(ntp, ntn, nfp, nfn);
        weights(i) = sum( true_class( (true_class == i), : ) );
    end
end

