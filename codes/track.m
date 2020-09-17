
set(0,'defaultaxesfontsize',16);
set(0,'defaultlinelinewidth',2);

M1 =csvread('macro_output_track_t03.20e+00_dt1.00e-02_dx2.24e-04_Tt6.50e+00.csv');
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

i1 =5; i2 = 25; i3 = 15; i4 = 10; i5 = 20; 
%% set start and end time to make sure R is in a reasonable range
t_st = 159; t_end = num_time-10;

figure(1)
subplot(221)
plot(X_(i1,t_st:t_end),Y_(i1,t_st:t_end),'o-','MarkerSize',3)
hold on;
plot(X_(i2,t_st:t_end),Y_(i2,t_st:t_end),'o-','MarkerSize',3)
hold on;
plot(X_(i3,t_st:t_end),Y_(i3,t_st:t_end),'o-','MarkerSize',3)
hold on;
plot(X_(i4,t_st:t_end),Y_(i4,t_st:t_end),'o-','MarkerSize',3)
hold on;
plot(X_(i5,t_st:t_end),Y_(i5,t_st:t_end),'o-','MarkerSize',3)
hold on;
plot(X_(:,t_st),Y_(:,t_st),'--')
hold on;
plot(X_(:,t_st+60),Y_(:,t_st+60),'--')
hold on;
plot(X_(:,t_st+100),Y_(:,t_st+100),'--')
hold on;
plot(X_(:,t_end),Y_(:,t_end),'--')
hold off;
axis equal;
axis([0 24e-3 -12e-3 0]);
xlabel('x');ylabel('y');title('trajectory')
legend('point 1','point 2','point 3','point 4','point 5')
subplot(222)
plot(time_(i1,t_st:t_end),G_(i1,t_st:t_end))
hold on;
plot(time_(i2,t_st:t_end),G_(i2,t_st:t_end))
hold on;
plot(time_(i3,t_st:t_end),G_(i3,t_st:t_end))
hold on;
plot(time_(i4,t_st:t_end),G_(i4,t_st:t_end))
hold on;
plot(time_(i5,t_st:t_end),G_(i5,t_st:t_end))
hold off;
xlabel('time');ylabel('G');title('G over time')
legend('point 1','point 2','point 3','point 4','point 5')
subplot(223)
plot(time_(i1,t_st:t_end),R_(i1,t_st:t_end))
hold on;
plot(time_(i2,t_st:t_end),R_(i2,t_st:t_end))
hold on;
plot(time_(i3,t_st:t_end),R_(i3,t_st:t_end))
hold on;
plot(time_(i4,t_st:t_end),R_(i4,t_st:t_end))
hold on;
plot(time_(i5,t_st:t_end),R_(i5,t_st:t_end))
hold off;
xlabel('time');ylabel('R');title('R over time')
legend('point 1','point 2','point 3','point 4','point 5')
subplot(224)
plot(time_(i1,t_st:t_end),Beta_(i1,t_st:t_end))
hold on;
plot(time_(i2,t_st:t_end),Beta_(i2,t_st:t_end))
hold on;
plot(time_(i3,t_st:t_end),Beta_(i3,t_st:t_end))
hold on;
plot(time_(i4,t_st:t_end),Beta_(i4,t_st:t_end))
hold on;
plot(time_(i5,t_st:t_end),Beta_(i5,t_st:t_end))
hold off;
xlabel('time');ylabel('thermal orientation');title('\theta over time')
legend('point 1','point 2','point 3','point 4','point 5')



X_ = X_(:,t_st:t_end);Y_ = Y_(:,t_st:t_end);
G_ = G_(:,t_st:t_end);R_ = R_(:,t_st:t_end);T_ = T_(:,t_st:t_end);
Beta_ = Beta_(:,t_st:t_end);
time_ = time_(:,t_st:t_end);
time_ = time_-time_(:,1);

save traj_low_try.mat X_ Y_ G_ R_ Beta_ time_ T_ t_st






