clear; close all;
set(0,'defaultaxesfontsize',16);
set(0,'defaultlinelinewidth',2);

savedir = './';

load macro_output_low.mat

idx = [10:20:100]; % traj index
t_idx = [1:70]; % time index


figure(1);
set(gcf,'Position',[100,100,800,400])
for k = 1:length(idx)
    plot(x_traj(t_idx ,idx(k)), y_traj( t_idx, idx(k)),'o-','MarkerSize',3) ; hold on
end
legend({'traj 1','traj 2', 'traj 3', 'traj 4'},'location','se')

figure(2);
subplot(121)
for k = 1:length(idx)
    plot(time_traj(t_idx), G_traj(t_idx ,idx(k)),'-') ; hold on
end
title('G(t)')

subplot(122)
for k = 1:length(idx)
    plot(time_traj(t_idx), R_traj(t_idx ,idx(k)),'-') ; hold on
end
title('R(t)')
