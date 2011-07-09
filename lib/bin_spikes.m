function [binned_spikes]=bin_spikes(spikes, bin, sum_or_binary)
% You specify how many samples you want included in a bin
% and whether or not you want the bins to reflect the sum of spikes within
% that window, or just a binary string of whether or not there was a spike
% at any time within the bin.
%
% This is code from the paper "A Statistical Description of Neural 
% Ensemble Dynamics," Long and Carmena 2011, submitted to  
% Frontiers in Computational Neuroscience.
% by: John D. Long II
% contact: jlong29@berkeley.edu or jlong29@gmail.com.
% Download at: http://code.google.com/p/kdq-bayes-kl
%
% INPUT
% spikes - datapoints x variables binary matrix of spikes
% bin - an integer indicating the number of samples to be binned
% sum_or_binary - if 1, then binary string
%                   else 0, then sum of spikes in bin
%
% OUTPUT
% binned_spikes - binary matrix of binned spike data in uint8 format

% Dimension Check
if size(spikes,2) > size(spikes,1)
    spikes = spikes';
end

%Correct for size of data
resid=mod(size(spikes,1),bin);

if resid==0
    binned=size(spikes,1)/bin;
else
    binned=(size(spikes,1)+bin-resid)/bin;
    spikes=[spikes; zeros(bin-resid,size(spikes,2),'uint8')];
end

binned_spikes=zeros(binned, size(spikes,2),'uint8');

for n=1:size(spikes,2)
    bin_chan=reshape(uint16(spikes(:,n))',[bin binned]);
    bin_chan=sum(bin_chan,1);
    if sum_or_binary==1
        a=logical(bin_chan>1);
        bin_chan(a)=1;
    end
    binned_spikes(:,n)=bin_chan';
end
