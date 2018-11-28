function score = compare_two(test_feat,refer_feat)
% function description: compare two images "test" and "refer"; 
% test, refer: mask_channel*N where mask_channel is the number of regions
% per image (defined as 200) and N is the dimensionality. each row is
% l2-normalized

% the number of row equal to the number of regions per image.
mask_channel = size(test_feat,1);

% a 512*512 matrix where each row represents the distances between each region and all other regions. Using dot product.
% Because test_feat and refer_feat are both l2_normalized, this actually
% implements the formula (6) in the paper. 
test_temp = test_feat*refer_feat';

% column max is each reference region to all regions in the test image
% Ranging from 1 (cos0, most similar) to -1 (cos180, least similar)
[col_value,col_idx] = max(test_temp,[],1); 

% row max is each region in the test to all regions in the reference image.
[row_value,row_idx] = max(test_temp,[],2); 
        
mutual = 0;

word_size = 10000; % A vocabulary with 10000 words
load(['build_vocabulary/reverse_' num2str(word_size) '.mat']); % load the reverse
load(['build_vocabulary/word_' num2str(word_size) '.mat']); % load the clustering center


for inner = 1:mask_channel % for each region in the test image
    %if(col_idx(inner) ~= row_idx(col_idx(inner))) % if the match is not mutual
    if(col_idx(row_idx(inner)) ~= inner)  %if the match is not mutual
        row_value(inner) = 0; % we don't take that match into account.
                %row_value(inner) = row_value(inner);
    else
        if(row_value(inner) ~=0) % in case there are some mutual matches where both match to the other with a 0 score, this can happen when that region is pooled by a completely black mask
            mutual = mutual + 1; % not identify a mutual match as test region:inner, reference region:row_idx(inner)
                    %mutual_matrix = [mutual_matrix;inner row_idx(inner)];
        else
            row_value(inner) = 0;
        end
    end
    
end
    %the index where there is mutual match between the 'nonzero' and 'row_idx(nonzero)'
    nonzero = find(row_value ~= 0); % 
    test_nonzero = test_feat(nonzero,:); % the features of test regions which have mutual match with the reference image
    refer_nonzero = refer_feat(row_idx(nonzero),:); % the features of the reference which match to the 'test_nonzero',   'row_value(nonzero) = col_value(row_idx(nonzero))'
                
    testdist = pdist2(C,test_nonzero); % each column is a testing region to all clustering centers
    referdist = pdist2(C,refer_nonzero);% each column is a reference region to all clustering centers
    [t_value,t_index] = min(testdist); % t_index is the index of the assignment words
    [r_value,r_index] = min(referdist); % r_index is the index of the assignment words
        
    row_value(nonzero) = (row_value(nonzero).*(log10(totalimg./wordcnt(t_index)))').*(log10(totalimg./wordcnt(r_index)))';
        
    score = sum(row_value)/mask_channel; % calculate the average matching score, may replace 'mutual' by 'mask_channel' to encourage more mutual (but maybe less similar) matches.
        


end