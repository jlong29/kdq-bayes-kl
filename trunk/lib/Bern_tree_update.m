function [Bern_Tree_update] = Bern_tree_update(Bern_Tree,data)
% Companion to 'Bern_tree.m'
% This code takes in a previously constructed Bern-tree and data.
% This usually means data with the number of variables(necessary) and number
% of data points (not necessary for program but usually required for
% analysis). The output is the same Bern-tree, but now the leaves will have
% counts reflecting the data entered as input
%
% This is code from the paper "A Statistical Description of Neural 
% Ensemble Dynamics," Long and Carmena 2011, submitted to  
% Frontiers in Computational Neuroscience.
% by: John D. Long II
% contact: jlong29@berkeley.edu or jlong29@gmail.com.
% Download at: http://code.google.com/p/kdq-bayes-kl
%
% OUTPUT:
% The Bern-tree struct should have the fields:
%  .node       node number
%  .parent     parent node number
%  .children   matrix of child nodes (2 cols, 1st is left child)
%  .nodesize   number of points at this node
%  .axis       dimension along which this node was split
%  .split      cutoff value for split (X<= split goes to left child node)              
%  .bounds     bounds for cell relative to axis of split

% Check for appropriate inputs
if nargin < 2
    error('The Second input should be the data you wish to use to update the Bern-tree')
end

% Check to see if the user put in a structure, indicating a kdq-tree
if isstruct(Bern_Tree) == 0
    error('The first input must be a Bern-tree. See m-file for details.')
end

% Check orientation of data matrix
if size(data,1) < size(data,2)
    error('Fewer data points than dimensions. Data Matrix should be (Npoints x Mdimensions)')
end

% Determine dimensions and number of data points
Dim           = size(data,2);
N             = size(data,1);

%%% Outer loop runs over the data searching through the dimensions for the
%%% node within the input Bern-tree that best matches its value. The search
%%% continues until a leaf is confronted. At every node a data point passes
%%% through, I will increment a counter, to match the output of Bern_tree.

% Counter of per node datapoints to populated
nodesize      = zeros(length(Bern_Tree.nodesize),1);

% Need to know about the children of a given node
children      = Bern_Tree.children;

max_depth     = Dim+1;

for n = 1:N
    
    % Get the data
    Xnode  = data(n,:);
    
    % Keep processing nodes until done starting with first dimension
    axis   = 1;
    node   = 1;
    isleaf = 0;
    
    % Increment counter for root node
    nodesize(node) = nodesize(node) + 1;
    
    while isleaf == 0
        
        % If you're at the max depth, then you've hit a leaf
        if axis == max_depth || children(node,1) == 0
            isleaf                   = 1;
            
        else
        
            % Get relevant data point for comparison
            point    = Xnode(axis);

            if point <= 0.5

                % Increment counter for left child of node
                nodesize(children(node,1)) = nodesize(children(node,1))+1;

                % Make next node left child
                node                       = children(node,1);

                % Update partition axis
                axis                       = axis+1;
            else
                %Increment counter for right child of node
                nodesize(children(node,2)) = nodesize(children(node,2))+1;

                %Make next node right child
                node                       = children(node,2);

                %Update partition axis
                axis                       = axis+1;
            end
        end
    end
end

% Construct output
Bern_Tree_update          = Bern_Tree;
Bern_Tree_update.nodesize = nodesize;
