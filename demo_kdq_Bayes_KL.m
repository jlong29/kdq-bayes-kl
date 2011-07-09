% This is the demo script for the code from the paper 
% "A Statistcal Description of Neural Ensemble Dynamics," 
% Long and Carmena 2011, submitted to  
% Frontiers in Computational Neuroscience 2011.
% by: John D. Long II
% contact: jlong29@berkeley.edu or jlong29@gmail.com.
% Download at: http://code.google.com/p/kdq-bayes-kl
%
% Instructions:
%  - Change to the code directory and run demo_kdq_Bayes_KL.m
%  - The script will take you through the functions .
%  - After each step, the script will pause. 
%  - To continue, hit any button.
%  - Read along in the demo.m file to follow what's happening.
%

fprintf('\nDemo program for TOWARD A STATISTICAL DESCRIPTION OF NEURAL ENSEMBLE DYNAMICS, Long and Carmena.\n')

%% The Binary kdq-tree (Figure 3)
fprintf('\nIntroducing the kdq-tree...\n')
fprintf('\nRecreating Figure 3...\n')
scrnsz = get(0,'screensize');
fig1 = figure('Position',scrnsz);

% Surrogate ensemble specs
num_var       = 5;      % number of variables in surrogate ensemble
N             = 1000;   % number of sample data points
splitmin      = 5;      % density parameter for Bern_tree

% The complete kdq-tree
data_complete = binornd(1,.5,N,num_var);    % Generate Bernoulli variables
[dc_tree]     = Bern_tree(data_complete,splitmin);  % Generate binary kdq-tree

% Plot complete data and set panel specs
subplot(2,2,1,'fontname','Times New Roman','fontsize',10),
    treeplot(dc_tree.parent')
    title([{'Complete Tree'};{''}],'fontname','Times New Roman','fontsize',10)
    ylabel('Tree Depth','fontname','Times New Roman','fontsize',10)
    xlabel('Terminal Nodes')
    kids = get(gca,'children');
    if length(kids)>1
        delete(kids(2))
    end
    set(kids(1),'linewidth',1)
    set(gca,'xticklabel','','yticklabel','','ticklength',[0 0])
    axis tight

% The partial kdq-tree
beta          = .05; % set probability of a one

data_incomp   = binornd(1,beta,N,num_var);  % Generate Bernoulli data
[dic_tree]    = Bern_tree(data_incomp,splitmin);    % Generate binary kdq-tree

% Count and sort ensemble patterns
leafs         = find(dic_tree.children(:,1)==0);
pruned_data   = sort(dic_tree.nodesize(leafs),'descend')./N;
[expectation_values,sorted_hist,joint_prob,ind_prob] = word_hist(data_incomp, 1);

% Plot pruned data and set panel specs
subplot(2,2,2,'fontname','Times New Roman','fontsize',10),
    treeplot(dic_tree.parent')
    title([{'Pruned Tree via'}; {'Kdq-tree'}],'fontname','Times New Roman','fontsize',10)
    xlabel('Terminal Nodes')
    kids = get(gca,'children');
    if length(kids)>1
        delete(kids(2))
    end
    set(kids(1),'linewidth',1)
    set(gca,'xticklabel','','yticklabel','','ticklength',[0 0])
    axis tight

% Elimination of missing data
missing_data1 = 100*length(find(sorted_hist(end,:)==0))/size(sorted_hist,2);
missing_data2 = 100*length(find(pruned_data==0))/length(pruned_data);
% Empirical Entropy
bitse         = -joint_prob*log2(joint_prob)';
bitsk         = -pruned_data'*log2(pruned_data);
% Check the amount of compression via the kdq-tree
compression   = 100*(1 - length(pruned_data)/size(sorted_hist,2));

% Plot log10(probability) of uncompressed data and set panel specs
subplot(2,2,3,'fontname','Times New Roman','fontsize',10)
    semilogy(joint_prob,'k'),hold on
    text(2^num_var-1-.05*(2^num_var-1),.65,[{sprintf('Empirical Bits = %5.3f',bitse)};...
    {sprintf('Missing Patterns = %3.0f%%',missing_data1)};...
    {sprintf('Units = %6.0f',num_var)}],...
    'VerticalAlignment','top','HorizontalAlignment','right')

    xlim([1 2^num_var])
    ylim([min(joint_prob) 1])
    title([{'Empirical Probability of'}; {'Simulated Data'}])
    ylabel('log_{10}(Probability)','fontname','Times New Roman','fontsize',10)
    xlabel('Ranked Ensemble Patterns','fontname','Times New Roman','fontsize',10)
    
% Plot log10(probability) of compressed data and set panel specs
subplot(2,2,4,'fontname','Times New Roman','fontsize',10)
    semilogy(pruned_data,'k')
    text(length(pruned_data)-.05*length(pruned_data),.65,[{sprintf('Empirical Bits = %5.3f',bitsk)};...
    {sprintf('Missing Patterns = %3.1f%%',missing_data2)};...
    {sprintf('Compression = %3.0f%%',compression)};...
    {sprintf('Splitmin = %6.0f',splitmin)}],...
    'VerticalAlignment','top','HorizontalAlignment','right')

    ylim([min(joint_prob) 1])
    set(gca,'yticklabel','')
    xlim([1 length(pruned_data)])
    title([{'Empirical Probability of'}; {'Compressed Data'}])
    xlabel('Ranked Ensemble Patterns','fontname','Times New Roman','fontsize',10)

fprintf('\nTo proceed to see basic demo of method, hit any key...\n')
pause
close(fig1)
clear

%% Demonstration of the basic method (Figure 4)
fprintf('\nDEMONSTRATOIN OF BASIC METHOD...\n')
load('data\basic_demo_data.mat')
scrnsz = get(0,'screensize');
fig1 = figure('Position',scrnsz);

fprintf('\nProcessing data...\n')

% Analysis specs
win      = 500;
splitmin = 5;
stepsize = 1;
alpha    = 0.5;
null     = 'fixed';

% Process some data and check speed
fprintf('\nDetecting Changes in ensemble firing rate...\n')
t   = clock;
KL1 = kdq_bayes_KL(data_fr,win,splitmin,stepsize,alpha,null);
t   = etime(clock,t);
fprintf('\nProcessed 4000 ensemble patterns from a 10 variable ensemble in %3.1f seconds.\n',t)
fprintf('\nDetecting Changes in strength of ensemble correlations...\n')
t   = clock;
KL2 = kdq_bayes_KL(data_sc,win,splitmin,stepsize,alpha,null);
t   = etime(clock,t);
fprintf('\nProcessed 4000 ensemble patterns from a 10 variable ensemble in %3.1f seconds.\n',t)

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Firing Rate Rasters %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,1,'fontname','Times New Roman','fontsize',10)
    line([2000 2000],[0.5 10.5],'linestyle',':','linewidth',2,'color',[0.5569  0.5843  0.5765]),hold on
    plot(firings1(:,1),firings1(:,2),'ok','markerfacecolor','k','markersize',2);

    title([{'Change in Ensemble'}; {'Firing Rates'}],'fontname','Times New Roman','fontsize',10)
    ylabel('Neuron #','fontname','Times New Roman','fontsize',10)
    set(gca,'ylim',[0.5 10.5],'ytick',1:10,'yticklabel','1|||||||||10',...
        'xtick',[0 1000 2000 3000 4000],'xticklabel','0||2000||4000')

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Correlated Rasters %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,2,'fontname','Times New Roman','fontsize',10)
    line([2000 2000],[0.5 10.5],'linestyle',':','linewidth',2,'color',[0.5569  0.5843  0.5765]),hold on
    plot(firings2(:,1),firings2(:,2),'ok','markerfacecolor','k','markersize',2);
    
    title([{'Change in Ensemble'}; {'Spatial Correlations'}],'fontname','Times New Roman','fontsize',10)
    set(gca,'ylim',[0.5 10.5],'ytick',1:10,'yticklabel','1|||||||||10',...
        'xtick',[0 1000 2000 3000 4000],'xticklabel','0||2000||4000')

%%%%%%%%%%%%%%%%%%%%%%
%%% Firing Rate KL %%%
%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,3,'fontname','Times New Roman','fontsize',10)
    line([2000 2000],[0 0.09],'linestyle',':','linewidth',2,'color',[0.5569  0.5843  0.5765]),hold on
    plot(KLs1(1:end-1),'k','linewidth',2),
    ylim([min(KLs1(KLs1>0)) max(KLs1)]),xlim([1 size(data_fr,1)])

    set(gca,'xtick',[0 1000 2000 3000 4000],'xticklabel','0||2000||4000')
    ylabel('KL-Divergence (a.u.)','fontname','Times New Roman','fontsize',10)
    xlabel('Time (ms)','fontname','Times New Roman','fontsize',10)

%%%%%%%%%%%%%%%%%%%%%
%%% Correlated KL %%%
%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,4,'fontname','Times New Roman','fontsize',10)
    line([2000 2000],[0 0.1],'linestyle',':','linewidth',2,'color',[0.5569  0.5843  0.5765]),hold on
    plot(KLs2(1:end-1),'k','linewidth',2)
    ylim([min(KLs2(KLs2>0)) max(KLs2)]),xlim([1 size(data_fr,1)])
    
    set(gca,'xtick',[0 1000 2000 3000 4000],'xticklabel','0||2000||4000')
    xlabel('Time (ms)','fontname','Times New Roman','fontsize',10)

disp('To proceed to correlation analysis, hit any key...')
pause
close(fig1)
clear

%% Detection of changes in correlation structure (Figure 6)
fprintf('\nDETECTION OF ENSEMBLE CORRELATIONS...\n')
load('data\correlation_increase_data.mat')
scrnsz = get(0,'screensize');
fig1 = figure('Position',scrnsz);

% Rasters: Correlations go up while ensemble firing rate stays the same
subplot(3,2,1,'fontname','Times New Roman','fontsize',10);
    ind1 = find(firings1(:,1)<21);
    ind2 = find(firings1(:,1)>=21 & firings1(:,1)<=40);
    ind3 = find(firings1(:,1)>=41);
    ind  = [ind1;ind2;ind3];
    line([21 21],[0.5 10.5],'color','k','linestyle','--'),hold on
    line([41 41],[0.5 10.5],'color','k','linestyle','--')
    plot(firings1(ind,1),firings1(ind,2),'ok','markerfacecolor','k','markersize',2),
        ylim([0.5 10.5]),
        xlim([0 61]),
        ylabel('Neuron #')
        set(gca,'ytick',1:10,'xtick',[],'fontname','Times New Roman','fontsize',10)    
        title([{'Transient ensemble correlations'}; ...
            {'Constant ensemble firing rate'}])
        xlabel('Subsets of Samples')
        set(gca,'fontname','Times New Roman','fontsize',10)

% PROCESS INCREASE IN CORRELATION DATA
% Analysis specs
win      = 100;
splitmin = 5;
stepsize = 1;
alpha    = 0.5;
null     = 'fixed';

fprintf('\nDetecting in strength of ensemble correlations...\n')
t   = clock;
[KLs1, null_stats] = kdq_bayes_KL(spikes1,win,splitmin,stepsize,alpha,null);
t   = etime(clock,t);
fprintf('\nProcessed 2600 ensemble patterns from a 10 variable ensemble in %3.1f seconds.\n',t)

% Output of method under null hypothesis of homogeneity with intial windows
% samples
subplot(3,2,3,'fontname','Times New Roman','fontsize',10);hold on
    m_KLs1  = null_stats.mode;
    se_KLs1 = null_stats.std;
    patch([1 2600 2600 1],[m_KLs1-se_KLs1 m_KLs1-se_KLs1 m_KLs1+se_KLs1 m_KLs1+se_KLs1],[0.8824 0.8824 0.8824],'line','none')
    line([size(c1,1) size(c1,1)],[min(KLs1(KLs1>0)) max(KLs1)],'linestyle','--','color','k'),hold on
    line([size(c1,1)+size(g,1) size(c1,1)+size(g,1)],[min(KLs1(KLs1>0)) max(KLs1)],'linestyle','--','color','k')
    line([1 size(spikes1,1)],[m_KLs1 m_KLs1],'color',[0.3912 0.3990 0.350]),
    plot(KLs1,'k','linewidth',1.5),
        set(gca,'ylim',[min(KLs1(KLs1>0)) max(KLs1)],'xticklabel','')
        xlim([1 length(KLs1)])
        ylabel('KL-Divergence')
        set(gca,'fontname','Times New Roman','fontsize',10)

% Ensemble firing rate
subplot(3,2,5,'fontname','Times New Roman','fontsize',10);
    line([size(c1,1) size(c1,1)],[3.9 6.1],'linestyle','--','color','k'),hold on
    line([size(c1,1)+size(g,1) size(c1,1)+size(g,1)],[3.9 6.1],'linestyle','--','color','k')
    patch([1 2600 2600 1],m_pop_fr1-se_pop_fr1-m_pop_fr1+se_pop_fr1,[0.8824    0.8824    0.8824],'line','none')
    plot(pop_fr1,'k','linewidth',1.5),
    line([1 size(spikes1,1)],[m_pop_fr1 m_pop_fr1],'color',[0.3912 0.3990 0.350]),
        set(gca,'xlim',[1 size(spikes1,1)],'ylim',[4.4 5.5],'ytick',[4.5 5 5.5])
        set(gca,'xtick',[500 1500 2500])
        xlabel('Samples')
        ylabel([{'Ensemble'};{'Firing Rate'}])
        set(gca,'fontname','Times New Roman','fontsize',10)

% Change in correlation structure
load('data\correlation_structure_change_data')

% Rasters: change in the structure of ensemble correlations
subplot(3,2,2,'fontname','Times New Roman','fontsize',10);
    ind1  = find(firings2(:,1)<=200);
    ind2 = find(firings2(:,1)>=1239 & firings2(:,1)<1439);
    ind3 = find(firings2(:,1)>=2519 & firings2(:,1)<2719);
    ind4 = find(firings2(:,1)>=3201 & firings2(:,1)<3401);
    ind  = [ind1;ind2;ind3;ind4];
    line([200 200],[0.5 10.5],'color','k','linestyle','--'),hold on
    line([400 400],[0.5 10.5],'color','k','linestyle','--')
    line([600 600],[0.5 10.5],'color','k','linestyle','--')
    plot(firings2(ind1,1),firings2(ind1,2),'ok','markerfacecolor','k','markersize',2),hold on
    plot(firings2(ind2,1)-1239+200,firings2(ind2,2),'ok','markerfacecolor','k','markersize',2),
    plot(firings2(ind3,1)-2519+400,firings2(ind3,2),'ok','markerfacecolor','k','markersize',2),
    plot(firings2(ind4,1)-3201+600,firings2(ind4,2),'ok','markerfacecolor','k','markersize',2),
        ylim([0.5 10.5]),
        xlim([1 800]),
        set(gca,'ytick',1:10,'xtick',[])
        ylabel('Neuron #','fontsize',10)
        xlabel('Subsets of Samples','horizontalalignment','center')
        title([{'Transient ensemble correlations'};...
            {'Change in correlation structure'}],'horizontalalignment','left')
        set(gca,'ytick',1:10,'xtick',[],'fontname','Times New Roman','fontsize',10)
    
% Change analysis specs for new data
null = 'dcorr';

fprintf('\nDetecting Changes in structure of ensemble correlations...\n')
t    = clock;
[KLs2,null_stats] = kdq_bayes_KL(data_change_corr,win,splitmin,stepsize,alpha,null);
t    = etime(clock,t);
fprintf('\nProcessed 4000 ensemble patterns from a 10 variable ensemble in %3.1f seconds.\n',t)

% Out of method under assumption of no change in correlation structure
subplot(3,2,4,'fontname','Times New Roman','fontsize',10);hold on
    temp1 = null_stats.mode_ind-null_stats.std_ind;
    temp2 = null_stats.mode_ind+null_stats.std_ind;
    patch([1 4000 4000 1],[temp1 temp1 temp2 temp2],[0.8824    0.8824    0.8824],'line','none'),
    line([1000 1000],[0.02 0.34],'linestyle','--','color','k'),
    line([2000 2000],[0.02 0.34],'linestyle','--','color','k'),
    line([3000 3000],[0.02 0.34],'linestyle','--','color','k'),
    line([1 4000],[null_stats.mode_ind null_stats.mode_ind],'color',[0.3912 0.3990 0.350]),
    plot(KLs2.ind,'k','linewidth',1.5),
        set(gca,'ylim',[0.02 0.34],'ytick',[.1 .3],'xtick',[1000 2000 3000],'xticklabel','')
        xlim([1 length(KLs2.ind)])
        ylabel('KL-Divergence')
        title('Null hypothesis: Independence')
        text(70,.13,'Null S.D.')
        set(gca,'fontname','Times New Roman','fontsize',10)
    
subplot(3,2,6,'fontname','Times New Roman','fontsize',10);hold on
    temp1 = null_stats.mode_dkl-null_stats.std_dkl;
    temp2 = null_stats.mode_dkl+null_stats.std_dkl;
    patch([1 4000 4000 1],[temp1 temp1 temp2 temp2],[0.8824    0.8824    0.8824],'line','none'),
    line([1000 1000],[0.01 0.5],'linestyle','--','color','k'),
    line([2000 2000],[0.01 0.5],'linestyle','--','color','k'),
    line([3000 3000],[0.01 0.5],'linestyle','--','color','k'),
    line([1 4000],[null_stats.mode_dkl null_stats.mode_dkl],'color',[0.3912 0.3990 0.350]),
    plot(KLs2.dkl,'k','linewidth',1.5),
        set(gca,'ylim',[0.01 0.5],'ytick',[0.2 0.4],'xtick',[1000 2000 3000])
        xlim([1 length(KLs2.dkl)])
        title('Null hypothesis: Homogeneity')
		xlabel('Samples')
        ylabel('KL-Divergence')
        set(gca,'fontname','Times New Roman','fontsize',10)

disp('To proceed to example visualization, hit any key...')
pause
close(fig1)
clear

%% Example visualization
fprintf('\nAPPLICATION TO REAL DATA\n')
fprintf('\nExample visualization of neural data and behavioral events.\n')
load('data\rat_s1bf_behavior.mat')

% Reconstruct time_line
time_line = T1:sr:N*sr-sr;

% % UNCOMMENT IF YOU WISH TO SEE HOW LONG IT TAKES TO RUN THE ANALYSIS
% bin      = sed_output.bin;
% win      = sed_output.win;
% splitmin = 5;
% alpha    = 0.5;
% 
% data   = sed_output.data;
% spikes = zeros(N,length(sp_list),'uint8');
% for i = 1:size(data,1)
%     spikes(data(i,1),data(i,2))=1;
% end
% fprintf('\nRunning null hypothesis of independence over 72 minutes of data from a 13 unit ensemble...\n')
% t    = clock;
% [sed_output] = stat_ensemble_dynamics(spikes,win,bin,'ind',splitmin,1,alpha,[],'sort');
% t    = etime(clock,t);
% fprintf('\nProcessed in %3.1f seconds.\n',t)

% Run visualization
fprintf('\nHorizontal zoom and pan are linked. Scroll through the data...\n\n')
stat_ensemble_dynamics_visualization(sed_output,time_line,behavior_events,'mode')
