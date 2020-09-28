% clear; close all;
set(0,'defaultaxesfontsize',16);
set(0,'defaultlinelinewidth',2);

savedir = './';

load macro_output_low_exp1.mat

sz = size(G_traj);
ntraj = sz(2); 
ntime = sz(1);

idx = [5:10: ntraj/2-5]; % traj index
t_idx = [1: ntime-10]; % time index


figure(1);
set(gcf,'Position',[100,100,800,400])
for k = 1:length(idx)
    plot(x_traj(t_idx ,idx(k)), y_traj( t_idx, idx(k)),'o-','MarkerSize',3) ; hold on
end
legend({'traj 1','traj 2', 'traj 3', 'traj 4', 'traj 5'},'location','se')


% plot contours
contour_lvl = [ 1:10: ntime-10];
for j = 1:length(contour_lvl)
    h1=plot(x_traj(contour_lvl(j),: ), y_traj( contour_lvl(j), :),'k--'); hold on;
    h1.Annotation.LegendInformation.IconDisplayStyle = 'off'; % to turn of legend off
    if j == 1 || j==length(contour_lvl)
        text(x_traj(contour_lvl(j), end), y_traj(contour_lvl(j), end),sprintf('t = %.2f', time_traj(contour_lvl(j)) ))
    end
end
axis equal;
axis([7e-3 11e-3 -2e-3 0]);
xlabel('x')
ylabel('y')

% print('-depsc', sprintf('%s/macro_traj.eps',savedir), '-r300' )



% plot G, R, Tdot
t_idx = [1:70]; % time index
figure(2);
set(gcf,'Position',[100,100,1200,400])
subplot(131)
for k = 1:length(idx)
    plot(time_traj(t_idx), G_traj(t_idx ,idx(k)),'-') ; hold on
end
title('G(t)')
xlabel('t')

subplot(132)
for k = 1:length(idx)
    plot(time_traj(t_idx), R_traj(t_idx ,idx(k)),'-') ; hold on
end
title('R(t)')
xlabel('t')

% cooling rate
subplot(133)
Tdot = - G_traj .* R_traj;
for k = 1:length(idx)
    plot(time_traj(t_idx), Tdot(t_idx ,idx(k)) ,'-') ; hold on
end
title('cooling rate dT/dt')
xlabel('t')

% print('-depsc', sprintf('%s/macro_GR.eps',savedir), '-r300' )


% save data
savedir = './';
t_list = 1:60;
for k = 1:length(idx)
    
    t_macro = time_traj(t_list);
    G_t = G_traj(t_list,idx(k))' / 1e6; % convert from K/m to K/um
    R_t = R_traj(t_list,idx(k))' * 1e6; % convert from m/s to um/s
    R_t(1) = 0;
   
    
    % save(sprintf('%s/macroGR_traj%d.mat',savedir, k), 'G_t', 'R_t', 't_macro');
end




