function [Bern_Tree] = Bern_tree(data,splitmin)
% This code implements a variant of the kdq-tree from the work of Dasu, Krishnan,
% Venkatasubramanian and Yi 2006 in: "An Information-Theoretic Approach to
% Detecting Changes in Multi-Dimensional Data Streams" (AT&T Labs - Research).
%
% This is code from the paper "A Statistical Description of Neural 
% Ensemble Dynamics," Long and Carmena 2011, submitted to  
% Frontiers in Computational Neuroscience.
% by: John D. Long II
% contact: jlong29@berkeley.edu or jlong29@gmail.com.
% Download at: http://code.google.com/p/kdq-bayes-kl
%
% INPUT:
% data - an MxN matrix of M time points and N units
% splitmin - an integer specifying the number of samples to include in a bin 
%       before splitting the bin via the kdq-tree (see 'Bern_tree.m').
%
% OUTPUT:
% Bern_Tree is a structure with the following fields:
%  .node       node ID number
%  .parent     parent node number
%  .children   matrix of child nodes (2 cols, 1st is left child)
%  .nodesize   number of points at this node
%  .axis       dimension along which this node was split

if nargin < 2
    error('User must supply an integer indicating the maximum bin count')
end
% Check orientation of data matrix
if size(data,1) < size(data,2)
    error(['Fewer data points than dimensions. Data Matrix should be '...
        'Npoints x Mdimensions and N > M'])
end
% Check whether data is Bernoulli variables or not
if ~isempty(find(data > 1,1))
    error(['This code only accepts Bernoulli data i.e. only 1 and 0 are' ... 
        'admissable values'])
end

% Generate for cycling through dimensions at each cut
Dim           = size(data,2); 

% Number of maximum nodes (conversative)
P             = min([2*size(data,1)-1 2^(Dim+1)-1]);

nodenumber    = zeros(P,1,'uint32');
parent        = zeros(P,1,'uint32');
children      = zeros(P,2,'uint32');
nodesize      = zeros(P,1,'uint32');
axis          = zeros(P,1,'uint32');
    
nodenumber(1) = 1;
axis(1)       = 1;

N              = size(data,1);
assignednode   = ones(N,1);
nextunusednode = 2;

% Keep processing nodes until done starting with first dimension
tnode = 1;

while(tnode < nextunusednode)
    
    % For the Bern_tree there should be at most one cut per dimension
    if axis(tnode) > Dim
        % Record information about this node
        noderows = find(assignednode==tnode);
        Nnode    = length(noderows);
        nodesize(tnode)   = Nnode;
        
        tnode = tnode + 1;
        continue
    end
        
    % Record information about this node
    noderows        = find(assignednode==tnode);
	Nnode           = length(noderows);
	nodesize(tnode) = Nnode;
    
    % Consider splitting this node
    if (Nnode> splitmin)      % split only nodes with counts above splitmin
		% Grab data at this node for this dimension
        Xnode = data(noderows,axis(tnode));
		
        % Set leftside bounds and allocate data
        leftside                   = Xnode <= 0.5;
        
        % Set rightside bounds and allocate data
        rightside                  = Xnode > 0.5;
        
        % abort a split if either resulting node has zero elements
        if isempty(find(leftside,1,'first')) || isempty(find(rightside,1,'first'))
            tnode         = tnode +1;
            continue
        end
        
        % Label appropriate children for this node
        children(tnode,:) = nextunusednode + (0:1);
        
        % Designate appropriate axis for each child
        next_axis                   = axis(tnode)+1;
        axis(nextunusednode+(0:1))  = [next_axis next_axis];
        
        % Update list of nodes to include these children
        nodenumber(nextunusednode+(0:1))  = nextunusednode + uint32((0:1)');
        
        % Assign data to appropriate node
        assignednode(noderows(leftside))  = nextunusednode;
        assignednode(noderows(rightside)) = nextunusednode+1;

        % Update parent node list
        parent(nextunusednode+(0:1)) = tnode;

        % Designate next unused node
        nextunusednode = nextunusednode+2;    
    end

    % Move on to next node
    tnode = tnode + 1;

end

topnode            = nextunusednode - 1;

Bern_Tree.node      = double(nodenumber(1:topnode));
Bern_Tree.parent    = double(parent(1:topnode));
Bern_Tree.children  = double(children(1:topnode,:));
Bern_Tree.nodesize  = double(nodesize(1:topnode));
Bern_Tree.axis      = double(axis(1:topnode));
