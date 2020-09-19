clear; close all;
set(0,'defaultaxesfontsize',16);
set(0,'defaultlinelinewidth',2);

savedir = '/Users/yuanxun/workspace/ammfrwrd20/tex/figures';

% M1 =csvread('macro_output_track_t03.20e+00_dt1.00e-02_dx2.24e-04_Tt6.50e+00.csv');
M1 =csvread('macro_output_Ts_t03.20e+00_dt1.00e-02_dx2.24e-04_Tt6.50e+00.csv');
time1 = M1(:,1);
xj_arr1 = M1(:,2);
yj_arr1 = M1(:,3);
Tj_arr1 = M1(:,4);
Gj_arr1 = M1(:,5);
Rj_arr1 = M1(:,6);
% j = 1; nstep =80;
% timeflag = M1(j,1);
% time_sp = zeros(nstep,1);len=length(M1(:,1));
% G_sp = zeros(nstep,len);R_sp = zeros(nstep,len);Rad_sp=zeros(nstep,len);
% xj_sp = zeros(nstep,len);yj_sp = zeros(nstep,len);
% for i = 1:len
%     
%    if  M1(i,1)==timeflag
%        
%        G_sp(j,i) = Gj_arr1(i);
%        R_sp(j,i) = Rj_arr1(i);
% 
%        time_sp(j) = timeflag; 
%        xj_sp(j,i) = xj_arr1(i);yj_sp(j,i) = yj_arr1(i);
%    
%    else 
%        j=j+1;
%        timeflag=M1(i,1);
%        G_sp(j,i) = Gj_arr1(i);
%        R_sp(j,i) = Rj_arr1(i);
% 
%        time_sp(j) = timeflag; 
%        xj_sp(j,i) = xj_arr1(i);yj_sp(j,i) = yj_arr1(i);
%    end
%   
%     
% end 
% 
% 
% figure(2)
% 
% for j =1:4
%   for i=1:4
%     
%     time_num = (4*j+i-4)+8;
%     subplot(4,4,4*j+i-4)
%     Yp = G_sp(time_num,:);
%     Xp = R_sp(time_num,:);
%     plot( Xp(Xp~=0),Yp(Xp~=0),'o');
%     %axis([0 4 0 12e5 ]);
%     xlabel('R');ylabel('G');title(strcat('t=',num2str(time_sp(time_num))))
%     
%     
%   end
% end
% 
% figure(4)
% title('R distribution')    
% for j =1:4
%   for i=1:4
%     
%     
%     subplot(4,4,4*j+i-4)
%     time_num = (4*j+i-4)+1;
%     Yp = yj_sp(time_num,:);
%     Xp = xj_sp(time_num,:);
%     Zp = R_sp(time_num,:);
%     %plot( Xp(Xp~=0),Yp(Xp~=0),'o');
%     scatter(Xp(Xp~=0),Yp(Xp~=0),25,Zp(Xp~=0))
%     axis([0 8e-4 -4e-4 0]);
%     xlabel('x');ylabel('y');title(strcat('t=',num2str(time_sp(time_num))))
%     colorbar
%     
%   end
% end   


i=1;
while time1(i)==time1(1)
    i = i +1;
    
end
num_sam = i-1; %89
num_time = round(length(time1)/num_sam);

% G_ = zeros(num_sam,num_time);R_ = zeros(num_sam,num_time);
% Y_ = zeros(num_sam,num_time);X_ = zeros(num_sam,num_time);
% beta_ = zeros(num_sam,num_time);
time_ = reshape(time1,[num_sam,num_time]);
G_ = reshape(Gj_arr1,[num_sam,num_time]);
R_ = reshape(Rj_arr1,[num_sam,num_time]);
T_ = reshape(Tj_arr1,[num_sam,num_time]);
X_ = reshape(xj_arr1,[num_sam,num_time]);
Y_ = reshape(yj_arr1,[num_sam,num_time]);
Beta_ = reshape(M1(:,7),[num_sam,num_time]);


%% set start and end time to make sure R is in a reasonable range
t_st = 170; t_end = num_time-10;

%% trajector plot
traj = [10,15,30, 48];
figure(1);
set(gcf,'Position',[100,100,800,400])
for k = 1:length(traj)
    plot(X_(traj(k),t_st:t_end),Y_(traj(k),t_st:t_end),'o-','MarkerSize',3) ; hold on
end
legend({'traj 1','traj 2', 'traj 3', 'traj 4'},'location','se')

% plot contours
cline = [0:21:t_end-t_st];
for j = 1:length(cline)
    h1=plot(X_(:,t_st+cline(j)),Y_(:,t_st+cline(j)),'k--'); hold on;
    h1.Annotation.LegendInformation.IconDisplayStyle = 'off'; % to turn of legend off
end
axis equal;
axis([0 24e-3 -12e-3 0]);
xlabel('x');ylabel('y');title('sampled trajectory')

print('-depsc', sprintf('%s/macro_traj.eps',savedir), '-r300' )

%% G(t), R(t) plot
figure(2)
set(gcf,'Position',[100,100,1200,400])
subplot(121)
for k = 1:length(traj)
    plot(time_(traj(k),t_st:t_end),G_(traj(k),t_st:t_end)); hold on;
end     
xlabel('time');ylabel('G');title('Thermal gradient')
legend({'traj 1','traj 2', 'traj 3', 'traj 4'},'location','ne')

subplot(122)
for k = 1:length(traj)
    plot(time_(traj(k),t_st:t_end),R_(traj(k),t_st:t_end)); hold on;
end     
xlabel('time');ylabel('R');title('Thermal velocity')
legend({'traj 1','traj 2', 'traj 3', 'traj 4'},'location','se')

print('-depsc', sprintf('%s/macro_GR.eps',savedir), '-r300' )

figure(3)
set(gcf,'Position',[100,100,800,400])
for k = 1:length(traj)
    plot(time_(traj(k),t_st:t_end),Beta_(traj(k),t_st:t_end)); hold on;
end
xlabel('time');ylabel('thermal orientation');title('\theta over time')
legend('point 1','point 2','point 3','point 4','point 5')



X_ = X_(:,t_st:t_end);Y_ = Y_(:,t_st:t_end);
G_ = G_(:,t_st:t_end);R_ = R_(:,t_st:t_end);T_ = T_(:,t_st:t_end);
Beta_ = Beta_(:,t_st:t_end);
time_ = time_(:,t_st:t_end);
time_ = time_-time_(:,1);

save traj_low_try.mat X_ Y_ G_ R_ Beta_ time_ T_ t_st



% save data
savedir = '/Users/yuanxun/workspace/direct-solid/codes';
for k = 1:length(traj)
    
    t_macro = time_(traj(k),:);
    G_t = G_(traj(k),:) / 1e4; % convert from K/m to K/um
    R_t = R_(traj(k),:) * 1e6; % convert from m/s to um/s
    R_t(1) = 0;
    
    G_t = [G_t(1), G_t];
    t_trans = 0.1;
    t_macro = [ 0, t_macro + t_trans];
    R_t = [0, R_t];
    
    save(sprintf('%s/macroGR_traj%d.mat',savedir, k), 'G_t', 'R_t', 't_macro');
end







