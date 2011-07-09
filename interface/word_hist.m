function [expectation_values,sorted_hist,joint_prob,ind_prob] = word_hist(spikes, sort_words)
% Purpose:
% To sort the binary ensemble data according to the 2^N available states (N is 
% the number of binary variables). The main sorting engine is 'Bern_tree' and 
% 'Bern_tree_update': two very efficient sorting algorithms
%
% Inputs:
% spikes: Written with neural data in mind. 
%         In general this code will accept any MXN binary matrix (M is data
%         points and N is the number of variables).
% 
% sort_words: The variable sort_words takes the values of '1' indicating sort the
%             states according to frequency and '0' indicating leaving the states in
%             the original lexographic order.
%
% Outputs:
% expectation_values: expectation_values for individual variables
% sorted_hist:        ensemble states sorted by frequency with last row being
%                     these counts.
% joint_prob:         empirical probabilities for each ensemble state.
% ind_prob:           expected probabilities of each ensemble state assuming the
%                     independent model.
%
% Note: Keep in mind that memory will limit the possible ensemble size.
%
% This is code from the paper "A Statistical Description of Neural 
% Ensemble Dynamics," Long and Carmena 2011, submitted to  
% Frontiers in Computational Neuroscience.
% by: John D. Long II
% contact: jlong29@berkeley.edu or jlong29@gmail.com.
% Download at: http://code.google.com/p/kdq-bayes-kl
%
%% Generates Matrix of Binary Words and Counts each instance

%Generate library of Possible States with bottom row for Counts

bits=size(spikes,2);

if nargin < 2 || isempty(sort_words)
    sort_words=0;
end

% Matrix for discrete state space description, plus one row for counts
word_library=zeros(bits + 1,2^(bits),'single');

% Fun little algorithm for doing this that is compatible with the
% the Bernoulli tree sorting algorithm referenced below

for n=1:bits
    alt=2^(bits-n);
    zero=zeros(alt,1);
    one=ones(alt,1);
    zeros_ones=[one; zero];
    repeats=2^bits/length(zeros_ones);
    sequence=repmat(zeros_ones,[1 repeats]);
    word_library(n,:)=sequence(:);
end

%% Now to Populate Counts

% Construct Bernoulli tree for sorting data
[bern_tree]   = Bern_tree(word_library(1:end-1,:)',1);

% Now use sorting tree to sort data
[sorted_data] = Bern_tree_update(bern_tree,spikes);

% see 'Bern_tree_update.m' for details of these steps
leafs         = logical(sorted_data.children(:,1) == 0);

counts        = sorted_data.nodesize(leafs)';
word_library(end,:) = counts(end:-1:1); 


%% Now to arrange the words in descending order of occurrence

if sort_words==1
[sorted,ind]=sort(word_library(end,:),2,'descend');
sorted_hist=word_library(:,ind);
else
    sorted_hist=word_library;
end

%% Construct Independent Model

%total sampling size
total=size(spikes,1);

%i.e. meaning firing rate for each neuron
expectation_values=sum(spikes,1)/total;

ind_model=zeros(size(sorted_hist,1)-1,size(sorted_hist,2),'single');

for n=1:size(spikes,2)
    %find the ones
    a=logical(sorted_hist(n,:));
    
    %find the zeros
    b=logical(sorted_hist(n,:)<.5);
    
    %convert to binomial probabilities
    ind_model(n,a)=expectation_values(n);
    ind_model(n,b)=1-expectation_values(n);
end

% Calculate Empirical and Independent model probabilities
% Empirical probabilities
% Convert counts to probabilities while accounting for states with zeros 
% by utilizing the posterior probability of the data given a dirichlet 
% distribution with 0.5 for all parameters on the prior (from Krichevsky and Trofimov 1981)
if any(sorted_hist(end,:) == 0)
    joint_prob = (sorted_hist(end,:) + 0.5)./(total + size(sorted_hist,2)/2);
else
    joint_prob = sorted_hist(end,:)./total;
end

% Independent Model Probabilities
ind_prob=prod(ind_model,1);
