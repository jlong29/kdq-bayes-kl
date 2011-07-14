function [KLs, null_stats] = kdq_bayes_KL(data,win,splitmin,stepsize,alpha,null,berr)
% This code calculates the KL-divergence according to a range of available null
% hypotheses, which may be specified by the user (see below). It employs the
% kdq-tree to adaptively quantize the domain of ensemble states based upon the
% observed data. The kdq-tree is derived from the work of Dasu, Krishnan,
% Venkatasubramanian and Yi 2006 in: "An Information-Theoretic Approach to
% Detecting Changes in Multi-Dimensional Data Streams" (AT&T Labs - Research).
% The KL-Divergence is a bayes's estimator derived according to the
% convolution/Laplace transform method for Dirichlet distributions described 
% by Wolpert and Wolf 1995 in: "Estimating functions of probability 
% distributions from a finite set of samples" (Physical Review E vol. 52 # 6).
%
% In addition, the code for calculating the second moment of the KL-divergence
% has been linearized using the procedure of Wolpert 1995 in: "Determining 
% whether two data sets are from the same distribution." (Maximum Entropy and
% Bayesian Methods, Eds. K. Hanson, R. Silver)
%
% This is code from the paper "A Statistical Description of Neural 
% Ensemble Dynamics," Long and Carmena 2011, submitted to  
% Frontiers in Computational Neuroscience.
% by: John D. Long II
% contact: jlong29@berkeley.edu or jlong29@gmail.com.
% Download at: http://code.google.com/p/kdq-bayes-kl
%
% DETAILS:
% In all cases the kdq-tree is constructed based upon the complete data.
% We use the convension that D(p1||p2) is the KL-divergence from p2 to p1. Below,
% any data labeled 'n_2' are counts from the distribution p_2 and data labeled 
% 'n_1' are counts from the distribution p_1.
%
% INPUT:
% data - an MxN matrix of M time points and N units
% win  - an integer specifying the number of samples to include in the sliding
%        window.
% splitmin - an integer specifying the number of samples to include in a bin 
%       before splitting the bin via the kdq-tree (see 'Bern_tree.m').
% stepsize - an integer specifying the number of samples to slide between each
%       window evaluations.
% alpha - a real number setting the Dirichlet prior parameters to the same
%         value. Default: alpha = 1 (uniform prior). alpha < 1 biases output
%         toward non-uniform posterior and alpha > 1 biases output toward
%         uniform posterior.
% null - a string indicating which null hypothesis to evaluate (options below)
% berr - if this input is not empty, the standard deviation of the Bayes's
%        estimator will be calculated at each point. This is computationally 
%        expensive.
% 
% null hypotheses: 
% fixed - deviation of subsequent samples from an initial window's sample
%         distribution, syntax: 'fixed'
% derivative - change in KL-Divergence between adjacent windows slid by
%              stepsize, syntax: 'derivative','dkl' (default)
% independent - compares each sample within a window against a surrogate 
%               independent sample--generated by permuating each channel's 
%               time indices to break correlations--to test for correlations in
%               the data, syntax: 'independent','ind'
% different correlations - tests adjacent sliding windows for differences in the
%                          correlation structures. This test is 2 nested
%                          hypotheses:
%                          1) Are the samples the 2 adjacent windows 
%                             independent or correlated?
%                          2) Are the samples in adjacent windows from the same
%                             distribution?
%                          syntax, 'dcorr'
%
% OUTPUT:
% KLs - a time-series of KL-Divergence values relative to the null hypothesis
%       evaluated. In the case of the null hypothesis of 'different 
%       correlations' this will be a structure with fields for the tests of
%       independence and difference between adjacent samples. If stepsize is >1, 
%       then there will only be non-zero values at regular intervals of stepsize.
%       Also, different null hypotheses do not evaluate the KL-divergence at the
%       begining, or end of the time-series due to windowing (see code for more
%       info).
% null_stats - a structure providing the confidence interval for each estimate
%              of the KL-divergence estimate (optional), the mode, mean, and 
%              standard deviation of the detrended KL values, and a label 
%              indicating the null hypothesis evaluated.

% Argument check
[dp,N] = size(data);
if dp < N
    error(['More variables than datapoints. Check tranpose: data should be M '...
    'datapoints by N variables'])
end

if nargin < 2 || isempty(win)
    win      = 200;
    disp(['Win is set at ' num2str(win) '.'])
end
if nargin < 3 || isempty(splitmin)
    splitmin = 5;
    disp(['Splitmin is set at ' num2str(splitmin) '.'])
end
if nargin < 4 || isempty(stepsize)
    stepsize = 1;
    disp(['Stepsize is set at ' num2str(stepsize) '.'])
end
if nargin < 5 || isempty(alpha)
    alpha = 0.5;
    disp(['Dirichlet alpha set at ' num2str(alpha) ': Jeffrey''s prior'])
end
if nargin < 6 || isempty(null)
    null = 'dkl';
    disp('No null hypothesis selected: Evaluating KL-Divergence between sliding windows.')
end
if nargin < 7 || isempty(berr)
    berr = [];
end

% Generate shuffled data for estimating standard deviation of KL-divergence
% time-series under the assumption of independence between subsequent samples,
% i.e. detrend the data
shuff   = randsample(dp,dp);
data_sh = data(shuff,:);

% Null hypothesis options
switch lower(null)
   % evaluate null hypothesis that all samples come from the same distribution
   % that generated the data within the initial segment of the data.
   case {'fixed'}
        % KL-divergence vector to be populated
        KLs    = zeros(size(data,1),1);
        KLs_sh = zeros(size(data,1),1);
        if ~isempty(berr)
            ci  = zeros(size(data,1),1);
        end
        
        % Grab data for the initial, fixed segment
        W2       = single(data(1:win,:));
        W2_sh    = single(data_sh(1:win,:));
        
        %%%%%%%%%%%%%%%%%%%
        %%% Compression %%%
        %%%%%%%%%%%%%%%%%%%
        % Construct Bern_tree from complete data
        [comp_t] = Bern_tree(data,splitmin);
        
        % Determine number of leafs
        leafs    = find(comp_t.children(:,1)==0);
        
        % End of data for window size
        max_edge = length(KLs) - win+1;
        
        % Evaluate W1 relative to comp_t
        [W2_t]   = Bern_tree_update(comp_t, W2);
        [W2_tsh] = Bern_tree_update(comp_t, W2_sh);
        
        % Generate counts for initial segment
        n_2      = W2_t.nodesize(leafs);
        n_2sh    = W2_tsh.nodesize(leafs);
        
        % Calculate KL-Divergence
        for n = 2:stepsize:max_edge
            
            % Grab data for this window
            W1     = single(data(n:win+n-1,:));
            W1_sh  = single(data_sh(n:win+n-1,:));
            
            % Evaluate W2 relative to comp_t
            [W1_t]   = Bern_tree_update(comp_t, W1);
            [W1_tsh] = Bern_tree_update(comp_t, W1_sh);
            
            % Generate counts for this new window
            n_1    = W1_t.nodesize(leafs);
            n_1sh  = W1_tsh.nodesize(leafs);
            
            % Calculate KL-Divergence
            if isempty(berr)
                KLs(n+win-1)    = bayes_est_KL_priors(n_1,n_2,alpha);
                KLs_sh(n+win-1) = bayes_est_KL_priors(n_1sh,n_2sh,alpha);
            else
                [KLs(n+win-1),Q2,ci(n+win-1)] = bayes_est_KL_priors(n_1,n_2,alpha);
                KLs_sh(n+win-1)               = bayes_est_KL_priors(n_1sh,n_2sh,alpha);
            end
        end
        
        % Null hypothesis Information
        % estimate the mode of the values to minimize the influence of outliers.
        [F,X]     = ksdensity(KLs(win+1:stepsize:end));
        [val,ind] = max(F);
        
        null_stats.mode  = X(ind);
        null_stats.mean  = mean(KLs_sh(win+1:stepsize:end));
        null_stats.std   = std(KLs_sh(win+1:stepsize:end));
        if ~isempty(berr)
            null_stats.ci    = ci;
        end
        null_stats.model = 'fixed initial window';
        
    % evaluate null hypothesis that the samples in adjacent windows come from
    % the same distribution. This is a measure of the difference between samples.
    % This test includes differences in any statistics of the distributions.
    case {'derivative','dkl'}
        % KL-divergence vectors to be populated
        KLs    = zeros(size(data,1),1);
        KLs_sh = zeros(size(data,1),1);
        if ~isempty(berr)
            ci     = zeros(size(data,1),1);
        end
        
        %%%%%%%%%%%%%%%%%%%
        %%% Compression %%%
        %%%%%%%%%%%%%%%%%%%
        % construct the Bern tree from the complete data
        comp_t = Bern_tree(data,splitmin);
        % Determine number of leafs
        leafs  = find(comp_t.children(:,1)==0);
        
        % End of data for window size (two windows here)
        max_edge       = length(KLs) - 2*win+1;
        
        % Calculate KL-Divergence
        for n = 1:stepsize:max_edge
            
            % Grab data for this window
            W2    = single(data(n:win+n-1,:));
            W2_sh = single(data_sh(n:win+n-1,:));
            
            W1    = single(data(win+n:2*win+n-1,:));
            W1_sh = single(data_sh(win+n:2*win+n-1,:));
            
            % compress W1 and W2 data using comp_t
            [W2_t]   = Bern_tree_update(comp_t, W2);
            [W2_tsh] = Bern_tree_update(comp_t, W2_sh);
            [W1_t]   = Bern_tree_update(comp_t, W1);
            [W1_tsh] = Bern_tree_update(comp_t, W1_sh);
            
            % Generate counts for each window
            n_2   = W2_t.nodesize(leafs);
            n_2sh = W2_tsh.nodesize(leafs);
            n_1   = W1_t.nodesize(leafs);
            n_1sh = W1_tsh.nodesize(leafs);
            
            % Calculate KL-Divergence
            if isempty(berr)
                KLs(n+win-1)    = bayes_est_KL_priors(n_1,n_2,alpha);
                KLs_sh(n+win-1) = bayes_est_KL_priors(n_1sh,n_2sh,alpha);
            else
                [KLs(n+win-1),Q2,ci(n+win-1)] = bayes_est_KL_priors(n_1,n_2,alpha);
                KLs_sh(n+win-1)                  = bayes_est_KL_priors(n_1sh,n_2sh,alpha);
            end
        end
        
        % Null hypothesis Information
        % estimate the mode of the values to avoid the influence of outliers.
        [F,X]     = ksdensity(KLs(win:stepsize:end-win));
        [val,ind] = max(F);
        
        null_stats.mode  = X(ind);
        null_stats.mean  = mean(KLs_sh(win:stepsize:end-win));
        null_stats.std   = std(KLs_sh(win:stepsize:end-win));
        if ~isempty(berr)
            null_stats.ci    = ci;
        end
        null_stats.model = 'basic KL change detection';

    % evaluate null hypothesis of independence within each window. This detects
    % changes in ensemble correlations distinct from changes in ensemble firing
    % rate. The kdq-tree is still constructed from the whole dataset.
    case {'ind', 'independent'}
        % KL-divergence vectors to be populated
        KLs    = zeros(size(data,1),1);
        KLs_sh = zeros(size(data,1),1);
        if ~isempty(berr)
            ci     = zeros(size(data,1),1);
        end
        
        %%%%%%%%%%%%%%%%%%%
        %%% Compression %%%
        %%%%%%%%%%%%%%%%%%%
        % Construct the Bern tree using complete data
        [comp_t] = Bern_tree(data,splitmin);
        % Determine number of leafs
        leafs  = find(comp_t.children(:,1)==0);
        
        % End of data for window size
        max_edge    = length(KLs) - win+1;
        
        % allocate memory for independent samples
        indsamp     = zeros(win,N,'single');
        indsamp_sh  = zeros(win,N,'single');
        
        % Calculate KL-Divergence
        for n = 1:stepsize:max_edge
            
            % data for this window
            W1      = single(data(n:win+n-1,:));
            
            if n == 1
                % Generate independent samples by scrambling the time indices on
                % a per unit basis
                for q = 1:N
                    ind1            = randsample(1:win,win);
                    ind2            = randsample(1:win,win);
                    indsamp(:,q)    = W1(ind1,q);
                    indsamp_sh(:,q) = W1(ind2,q);
                end
                % INITIAL overhead
                nosp    = cell(N,1);
                nosp_sh = cell(N,1);
                sp      = cell(N,1);
                sp_sh   = cell(N,1);
                for q = 1:N
                    nosp{q}   = find(indsamp(:,q)==0);
                    nosp_sh{q}= find(indsamp_sh(:,q)==0);
                    sp{q}     = find(indsamp(:,q)==1);
                    sp_sh{q}  = find(indsamp_sh(:,q)==1);
                end
            else
                % just scramble incoming data while matching the firing rates of the
                % original data
                old_dp     = data(n-1,:);
                new_dp     = data(win+n-1,:);
                
                for q = 1:N
                    % If new element and old element are the same for this channel, do
                    % nothing
                    if old_dp(q)~=new_dp(q)
                        % If they are different
                        if old_dp(q) == 0
                            % sample an index where a zero is found
                            ind = randsample(nosp{q},1);
                            % delete this index from the list of zeros
                            nosp{q} = nosp{q}(nosp{q}~=ind);
                            % switch the value of this index in ind_sample
                            indsamp(ind,q)=1;
                            % add this index for the list of ones for this channel
                            sp{q} = [sp{q};ind];
                        else
                            % sample an index where a one is found
                            ind = randsample(sp{q},1);
                            % delete this index from the list of ones
                            sp{q} = sp{q}(sp{q}~=ind);
                            % switch the value of this index in indsamp
                            indsamp(ind,q)=0;
                            % add this index for the list of zeros for this channel
                            nosp{q} = [nosp{q};ind];
                        end
                    end
                    if old_dp(q)~=new_dp(q)
                        % If they are different
                        if old_dp(q) == 0
                            % sample an index where a zero is found
                            ind = randsample(nosp_sh{q},1);
                            % delete this index from the list of zeros
                            nosp_sh{q} = nosp_sh{q}(nosp_sh{q}~=ind);
                            % switch the value of this index in ind_sample
                            indsamp_sh(ind,q)=1;
                            % add this index for the list of ones for this channel
                            sp_sh{q} = [sp_sh{q};ind];
                        else
                            % sample an index where a one is found
                            ind = randsample(sp_sh{q},1);
                            % delete this index from the list of ones
                            sp_sh{q} = sp_sh{q}(sp_sh{q}~=ind);
                            % switch the value of this index in indsamp
                            indsamp_sh(ind,q)=0;
                            % add this index for the list of zeros for this channel
                            nosp_sh{q} = [nosp_sh{q};ind];
                        end
                    end
                end
            end
            
            % Transform data relative to comp_t
            [W1_t]   = Bern_tree_update(comp_t, W1);
            [W2_i]   = Bern_tree_update(comp_t, indsamp);
            [W1_ish] = Bern_tree_update(comp_t, indsamp_sh);
            
            % Generate counts for each data
            n_1    = W1_t.nodesize(leafs);
            n_2i   = W2_i.nodesize(leafs);
            n_1ish = W1_ish.nodesize(leafs);
            
            % Calculate KL-Divergence
            if isempty(berr)
                KLs(n+win-1)    = bayes_est_KL_priors(n_1,n_2i,alpha);
                KLs_sh(n+win-1) = bayes_est_KL_priors(n_1ish,n_2i,alpha);
            else
                [KLs(n+win-1),Q2,ci(n+win-1)] = bayes_est_KL_priors(n_1,n_2i,alpha);
                KLs_sh(n+win-1)               = bayes_est_KL_priors(n_1ish,n_2i,alpha);
            end
        end

        % Null hypothesis Information
        % estimate the mode of the values to avoid the influence of outliers.
        [F,X]     = ksdensity(KLs(win:stepsize:end));
        [val,ind] = max(F);
        
        null_stats.mode  = X(ind);
        null_stats.mean  = mean(KLs_sh(win:stepsize:end));
        null_stats.std   = std(KLs_sh(win:stepsize:end));
        if ~isempty(berr)
            null_stats.ci    = ci;
        end
        null_stats.model = 'independent';
        
        
    % evaluate null hypothesis that correlation structure among the units 
    % within adjacent sliding windows are the same. Rejecting this null
    % hypothesis requires rejecting two hypotheses: 1) Is each sample in the 2 
    % adjacent windows independent or correlated? 2) Are the samples in adjacent
    % windows from the same distribution?
    case 'dcorr'
        % KL-divergence vectors to be populated for each null hypothesis
        KLs_ind    = zeros(size(data,1),1);
        KLs_ind_sh = zeros(size(data,1),1);
        
        KLs_dkl    = zeros(size(data,1),1);
        KLs_dkl_sh = zeros(size(data,1),1);
        
        if ~isempty(berr)
            ci_ind = zeros(size(data,1),1);
            ci_dkl = zeros(size(data,1),1);
        end
        
        %%%%%%%%%%%%%%%%%%%
        %%% Compression %%%
        %%%%%%%%%%%%%%%%%%%
        % Construct Bern_tree from complete data
        [comp_t] = Bern_tree(data,splitmin);
        
        % Determine number of leafs
        leafs  = find(comp_t.children(:,1)==0);
        
        % End of data for window size (two windows here)
        max_edge       = length(KLs_ind) - 2*win+1;
        
        % allocate memory for independent sample
        indsamp    = zeros(win,N,'single');
        indsamp_sh = zeros(win,N,'single');
        
        % Calculate KL-Divergence
        for n = 1:stepsize:max_edge
            
            % Grab data for adjacent windows
            W2    = single(data(n:win+n-1,:));
            W2_sh = single(data_sh(n:win+n-1,:));
            W1    = single(data(win+n:2*win+n-1,:));
            W1_sh = single(data_sh(win+n:2*win+n-1,:));
            
            if n == 1
                % Generate independent samples by scrambling the time indices on
                % a per unit basis
                for q = 1:N
                    ind1            = randsample(1:win,win);
                    ind2            = randsample(1:win,win);
                    indsamp(:,q)    = W2(ind1,q);
                    indsamp_sh(:,q) = W2(ind2,q);
                end
                % INITIAL overhead
                nosp    = cell(N,1);
                nosp_sh = cell(N,1);
                sp      = cell(N,1);
                sp_sh   = cell(N,1);
                for q = 1:N
                    nosp{q}    = find(indsamp(:,q)==0);
                    nosp_sh{q} = find(indsamp_sh(:,q)==0);
                    sp{q}      = find(indsamp(:,q)==1);
                    sp_sh{q}   = find(indsamp_sh(:,q)==1);
                end
            else
                % just scramble incoming data while matching the firing rates of the
                % original data
                old_dp     = data(n-1,:);
                new_dp     = data(win+n-1,:);
                
                for q = 1:N
                    % If new element and old element are the same for this channel, do
                    % nothing
                    if old_dp(q)~=new_dp(q)
                        % If they are different
                        if old_dp(q) == 0
                            % sample an index where a zero is found
                            ind = randsample(nosp{q},1);
                            % delete this index from the list of zeros
                            nosp{q} = nosp{q}(nosp{q}~=ind);
                            % switch the value of this index in ind_sample
                            indsamp(ind,q)=1;
                            % add this index for the list of ones for this channel
                            sp{q} = [sp{q};ind];
                        else
                            % sample an index where a one is found
                            ind = randsample(sp{q},1);
                            % delete this index from the list of ones
                            sp{q} = sp{q}(sp{q}~=ind);
                            % switch the value of this index in indsamp
                            indsamp(ind,q)=0;
                            % add this index for the list of zeros for this channel
                            nosp{q} = [nosp{q};ind];
                        end
                    end
                    if old_dp(q)~=new_dp(q)
                        % If they are different
                        if old_dp(q) == 0
                            % sample an index where a zero is found
                            ind = randsample(nosp_sh{q},1);
                            % delete this index from the list of zeros
                            nosp_sh{q} = nosp_sh{q}(nosp_sh{q}~=ind);
                            % switch the value of this index in ind_sample
                            indsamp_sh(ind,q)=1;
                            % add this index for the list of ones for this channel
                            sp_sh{q} = [sp_sh{q};ind];
                        else
                            % sample an index where a one is found
                            ind = randsample(sp_sh{q},1);
                            % delete this index from the list of ones
                            sp_sh{q} = sp_sh{q}(sp_sh{q}~=ind);
                            % switch the value of this index in indsamp
                            indsamp_sh(ind,q)=0;
                            % add this index for the list of zeros for this channel
                            nosp_sh{q} = [nosp_sh{q};ind];
                        end
                    end
                end
            end
            
            % compress W1 and W2 data using comp_t
            [W2_t]   = Bern_tree_update(comp_t, W2);
            [W2_tsh] = Bern_tree_update(comp_t, W2_sh);
            [W1_t]   = Bern_tree_update(comp_t, W1);
            [W1_tsh]   = Bern_tree_update(comp_t, W1_sh);
            
            % Compress independent samples using comp_t
            [W2_i]   = Bern_tree_update(comp_t, indsamp);
            [W2_ish] = Bern_tree_update(comp_t, indsamp_sh);
            
            % Generate counts for each window
            n_2    = W2_t.nodesize(leafs);
            n_2sh  = W2_tsh.nodesize(leafs);
            n_2i   = W2_i.nodesize(leafs);
            n_2ish = W2_ish.nodesize(leafs);
            n_1    = W1_t.nodesize(leafs);
            n_1sh  = W1_tsh.nodesize(leafs);
            
            % Calculate KL-Divergence from surrogate independent sample
            if isempty(berr)
                KLs_ind(n+win-1)    = bayes_est_KL_priors(n_2,n_2i,alpha);
                KLs_ind_sh(n+win-1) = bayes_est_KL_priors(n_2i,n_2ish,alpha);
                % Calculate KL-Divergence between adjacent samples
                KLs_dkl(n+win-1) = bayes_est_KL_priors(n_1,n_2,alpha);
                KLs_dkl_sh(n+win-1) = bayes_est_KL_priors(n_1sh,n_2sh,alpha);
            else
                [KLs_ind(n+win-1),Q2,ci_ind(n+win-1)] = bayes_est_KL_priors(n_2,n_2i,alpha);
                KLs_ind_sh(n+win-1) = bayes_est_KL_priors(n_2i,n_2ish,alpha);
                % Calculate KL-Divergence between adjacent samples
                [KLs_dkl(n+win-1),Q2,ci_dkl(n+win-1)] = bayes_est_KL_priors(n_1,n_2,alpha);
                KLs_dkl_sh(n+win-1) = bayes_est_KL_priors(n_1sh,n_2sh,alpha);
            end
        end
        
        % Generate structure of KLs for each null hypothesis
        KLs.dkl = KLs_dkl;
        KLs.ind = KLs_ind;
        
        % Null hypothesis Information
        % estimate the mode of the values to avoid the influence of outliers.
        [F,X]     = ksdensity(KLs_dkl(win:stepsize:end-win));
        [val,ind] = max(F);
        null_stats.mode_dkl = X(ind);
        null_stats.mean_dkl = mean(KLs_dkl_sh(win:stepsize:end-win));
        null_stats.std_dkl  = std(KLs_dkl_sh(win:stepsize:end-win));
        
        [F,X]     = ksdensity(KLs_ind(win:stepsize:end-win));
        [val,ind] = max(F);
        null_stats.mode_ind = X(ind);
        null_stats.mean_ind = mean(KLs_ind_sh(win:stepsize:end-win));
        null_stats.std_ind  = std(KLs_ind_sh(win:stepsize:end-win));
        if ~isempty(berr)
            null_stats.ci_ind = ci_ind;
            null_stats.ci_dkl = ci_dkl;
        end
        
        null_stats.model = 'different correlations';
    otherwise
        fprintf('\nUnrecognized null hypothesis declaration.\n')
        fprintf('Available options:\n')
        fprintf('''fixed''\n')
        fprintf('''derivative'' or ''dkl''\n')
        fprintf('''independent'' or ''ind''\n')
        fprintf('''dcorr''\n')
        fprintf('See the comments for this m-file for details, or\n')
        fprintf('type ''help kdq_bayes_KL''\n')
        KLs = NaN;
        null_stats = NaN;
        return
end
