function [output] = encode_feat(feat,mask)
%Function Description: extract clustered features from 'feat' and 'mask';
% feat: an input conv feat like 512*14*14, to be pooled using mask
% mask: an input conv feat like 512*14*14, used for pooling
% output: an output matrix to represent the image

    woff = 0;

    num_pool = 200; % the number of regions we choose to pool

    allmean = [];
    allbox = [];
    whichmask = [];
    
    for mask_idx = 1:size(mask,1)
        temp_mask = squeeze(mask(mask_idx,:,:)); % temp_mask: a 14*14 matrix
        
        % a binary mask, where all non-zero elements in temp_mask are set to 1
        binary_mask = im2bw(temp_mask,min(min(temp_mask))); 

        % find all the 8-connected components in the binary_mask;
        allprops = regionprops(binary_mask,temp_mask,'MeanIntensity','BoundingBox'); 
        
        % for each cluster, represented as S_j in formula (2) in the paper.
        for center_idx = 1:length(allprops)  
            % calculate the mean intensity of this cluster, E_j in formulat (2) in the paper
            allmean = [allmean;allprops(center_idx).MeanIntensity]; 
            
            % the position of these clusters
            allbox = [allbox;allprops(center_idx).BoundingBox]; 
            
            % the index of the channels
            whichmask = [whichmask;mask_idx]; 
        end
    end
    
    % sort the mean intensity from large to small
    sort_mean = sort(allmean,'descend');  
    
    % find the top 200 mean values
    threshold = sort_mean(num_pool);
    upper = find(allmean>=threshold); 
    
    % the bounding box to pool
    final_box = allbox(upper,:); 
    
    % which (out of 256) mask to pool
    final_mask = whichmask(upper); 
    
    output = [];
    for upper_idx = 1:length(final_mask)
        
        % the location of the mask to pool. 
        region = (final_box(upper_idx,:)); 
        
        % pick the right channel
        temp_mask = squeeze(mask(final_mask(upper_idx),:,:));
        
        % the feat to be pooled
        pool_feat = feat(:,(region(2):region(2)+region(4)-1),region(1):(region(1)+region(3)-1)); 
        
        % the mask used for pooling
        pool_mask = temp_mask((region(2):region(2)+region(4)-1),region(1):(region(1)+region(3)-1));
        
%         if(woff == 0)
%             ww = warning('query','last');
%             idd = ww.identifier;
%             warning('off',idd);
%             woff = 1;
%         end
        
        % flatten the feature
        pool_feat = reshape(pool_feat,size(pool_feat,1),[]);
        
        % flatten and normalize the mask
        flatten_pool = pool_mask(:);
        
        flatten_pool = flatten_pool/norm(flatten_pool);
        
        % perform feature pooling, this is a 512*1 feature
        pool_multi = pool_feat*flatten_pool; 
        
        % perform l2_normalization again
        temp_norm = norm(pool_multi);
        if(temp_norm ~=0)
            pool_multi = pool_multi./temp_norm;
        end
        
        % a matrix with 512*num_pool, where each column is a l2_normalize vector representing a region and there are num_pool regions.
        output = [output pool_multi];
    end


end