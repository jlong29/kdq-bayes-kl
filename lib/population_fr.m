function pop_fr = population_fr(spikes,win,bin)
% This code calculates the population, or ensemble, firing rate for a neural
% ensemble. The time-scale of this calculation is determined by 'win' and the
% conversion to spikes/second requires the 'bin' input for accurate conversion.
%
% INPUT
% spikes - datapoints x variables binary matrix of spikes
% win  - an integer specifying the number of samples to include in the sliding
%        window.
% bin - an integer indicating the number of samples that were binned.
%
% OUTPUT
% pop_fr - a vector indicating the population firing rate
%
% This is code from the paper "A Statistical Description of Neural 
% Ensemble Dynamics," Long and Carmena 2011, submitted to  
% Frontiers in Computational Neuroscience.
% by: John D. Long II
% contact: jlong29@berkeley.edu or jlong29@gmail.com.
% Download at: http://code.google.com/p/kdq-bayes-kl

% Calculate population firing rate
[dp,units] = size(spikes);
if dp < units
    spikes = spikes';
    disp('Input transposed')
end
    
sum_fr  = sum(spikes,2);

% End of data for window size
max_edge       = dp - win+1;

pop_fr = zeros(size(sum_fr));
for n = 1:max_edge
    % Calculate running average population firing rate
    pop_fr(n+win-1) = (1000/bin)*mean(sum_fr(n:win+n-1));
end
