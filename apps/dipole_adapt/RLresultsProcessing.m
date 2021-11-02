%%
clear;
close all;
%%
set(groot,'defaultLineLineWidth',1.5);
set(groot,'defaultFigureColor','w');
set(groot,'defaultTextFontsize',12);
set(groot,'defaultAxesFontsize',12);
set(groot,'defaultPolarAxesFontsize',12);
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultPolarAxesTickLabelInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultAxesLineWidth',1);
%% direction policy training process
name = 'ego2sensorLRCFDGrad1';
dir = [name,'/agent_00_rank_000_cumulative_rewards.dat'];
data = data_extract(dir);
figure, plot(data.reward,'.','markersize',0.1);
hold on, plot(movmean(data.reward, 200));
xlabel('episode');
ylabel('reward');
title(name)
exportgraphics(gcf,['/home/yusheng/Dropbox/transfer/dipole_log/',name,'.eps'])
%     savefig(['/home/yusheng/Dropbox/transfer/dipole_log/',setting, env, num2str(i),'.fig'])
name = 'ego2sensorLRCFDGrad2';
dir = [name,'/agent_00_rank_000_cumulative_rewards.dat'];
data = data_extract(dir);
figure, plot(data.reward,'.','markersize',0.1);
hold on, plot(movmean(data.reward, 200));
xlabel('episode');
ylabel('reward');
title(name)
exportgraphics(gcf,['/home/yusheng/Dropbox/transfer/dipole_log/',name,'.eps'])

name = 'ego2sensorFBCFDGrad1';
dir = [name,'/agent_00_rank_000_cumulative_rewards.dat'];
data = data_extract(dir);
figure, plot(data.reward,'.','markersize',0.1);
hold on, plot(movmean(data.reward, 200));
xlabel('episode');
ylabel('reward');
title(name)
exportgraphics(gcf,['/home/yusheng/Dropbox/transfer/dipole_log/',name,'.eps'])

name = 'ego2sensorLRCFDclosersensor1';
dir = [name,'/agent_00_rank_000_cumulative_rewards.dat'];
data = data_extract(dir);
figure, plot(data.reward,'.','markersize',0.1);
hold on, plot(movmean(data.reward, 200));
xlabel('episode');
ylabel('reward');
title(name)
exportgraphics(gcf,['/home/yusheng/Dropbox/transfer/dipole_log/',name,'.eps'])

% setting = 'ego2sensorLR';
% env = 'CFD';
% for i = 1:2
%     dir = [setting, env, num2str(i),'/agent_00_rank_000_cumulative_rewards.dat'];
%     data = data_extract(dir);
%     figure, plot(data.reward,'.','markersize',0.1);
%     hold on, plot(movmean(data.reward, 200));
%     xlabel('episode');
%     ylabel('reward');
%     title([setting, env, num2str(i)])
%     exportgraphics(gcf,['/home/yusheng/Dropbox/transfer/dipole_log/',setting, env, num2str(i),'.eps'])
% end
% setting = 'ego2sensorFB';
% env = 'CFD';
% for i = 1:2
%     dir = [setting, env, num2str(i),'/agent_00_rank_000_cumulative_rewards.dat'];
%     data = data_extract(dir);
%     figure, plot(data.reward,'.','markersize',0.1);
%     hold on, plot(movmean(data.reward, 200));
%     xlabel('episode');
%     ylabel('reward');
%     title([setting, env, num2str(i)])
%     exportgraphics(gcf,['/home/yusheng/Dropbox/transfer/dipole_log/',setting, env, num2str(i),'.eps'])
% %     savefig(['/home/yusheng/Dropbox/transfer/dipole_log/',setting, env, num2str(i),'.fig'])
% end
% only when each training has same length
% c_trun = struct2cell(data);
% reward_trun = cell2mat(c_trun(3,:));
% n_episode_trun = cell2mat(c_trun(1,:));

% c = get(gca,'colororder');
% 
% Reward_median_trun = median(reward_trun,2);
% RewardLowerBound_trun = quantile(reward_trun,0.25,2);
% RewardUpperBound_trun = quantile(reward_trun,0.75,2);
% x_axis = n_episode_trun(:,1)';lColor = [0,0,0];aColor = [0,0,0]+0.5;lWidth = 1.5;
% [patch_trun,l_trun] = plot_shadederrorbar(Reward_median_trun,RewardUpperBound_trun,RewardLowerBound_trun,x_axis,lColor,aColor,lWidth);

% Reward_median_dirBS = median(reward_BSwoN1,2);
% RewardLowerBound_dirBS = quantile(reward_BSwoN1,0.25,2);
% RewardUpperBound_dirBS = quantile(reward_BSwoN1,0.75,2);
% 
% x_axis = n_episode_BSwoN1(:,1)';lColor = c(1,:);aColor = c(1,:)+0.5*(1-c(1,:));lWidth = 1.5;
% legend([l_trun,l_dirBS],{'truncation',...
%     'bootstrap'});
% xlabel('episode');ylabel('reward');
% set(gcf,'position',[440 625 233 173]);

%%
function [patch,h] = plot_shadederrorbar(mean,upBound,lowBound,x_axis,lColor,aColor,lWidth)
alpha = 0.4;
x_vector = [x_axis, fliplr(x_axis)];
patch = fill(x_vector, [upBound',fliplr(lowBound')], aColor);
patch.EdgeColor = 'none';
patch.FaceAlpha = alpha;
hold on;
h = plot(x_axis, mean', 'color', lColor, 'LineWidth', lWidth);

end
function [patch,h] = plot_shadederror2d(xmean,xupBound,xlowBound,ymean,yupBound,ylowBound,lColor,aColor,lWidth)
alpha = 0.4;
x_vector = [xupBound', fliplr(xlowBound')];
patch = fill(x_vector, [yupBound',fliplr(ylowBound')], aColor);
patch.EdgeColor = 'none';
patch.FaceAlpha = alpha;
hold on;
h = plot(xmean, ymean, 'color', lColor, 'LineWidth', lWidth);

end