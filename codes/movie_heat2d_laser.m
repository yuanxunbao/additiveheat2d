clear; close all;
set(0,'defaultaxesfontsize',16);
set(0,'defaultlinelinewidth',2);

odir = './movie/';

L=200;
T_s =40;
T_l = 110;

load para.mat;

load Glatdt0.1xgrids513.mat
% load latdt0.00625xgrids2049.mat;
% load nolatnoraddt0.00625xgrids2049.mat;
% load nolatdt0.00625xgrids2049.mat;

lx= cast(lx,'double'); nxs= cast(nxs,'double'); nys= cast(nys,'double');
x = linspace(0,lx,nxs);
y = linspace(0,-ly,nys);
[xx,yy] = meshgrid(x,y);

temp = temp513;
cmax = max(temp513(:));
cmin = min(temp513(:));

sz = size(temp); nxny=sz(1); Nt = sz(2);
nframes = Nt/2;
nskip = floor(Nt/nframes);
frames = [1:nskip:Nt-1,Nt];

%% compute phase
temp_c = mat2cell(temp, nxny, ones(1,Nt));
Tphase_c = cellfun( @(y) segment_T(y,T_s,T_l), temp_c,'UniformOutput',0);
Tphase = cell2mat(Tphase_c); 

%% compute G/R ratio
GoR = G.^2 ./ abs(GR); 
GoR(~isfinite(GoR)) = 0; % get rid of NaN's and Inf's

figure(1)
set(gcf,'Position',[100,100,900,900])

for k = 1:length(frames)
    
    
    subplot(3,1,1)
    surf(xx,yy, reshape(temp(:,frames(k)),[nys,nxs])); shading interp; colormap(gca,(hot)); 
    title('Temperature')
    view(2); axis equal; axis tight; axis off; 
    caxis([cmin,cmax])
    c1 = colorbar; c1.Limits = [cmin,cmax];
   
    
    subplot(3,1,2)
    surf(xx,yy, reshape(Tphase(:,frames(k)),[nys,nxs])); shading interp; colormap(gca, gray); 
    view(2); axis equal; axis tight; axis off;
    caxis([0,1])
    title('Phase')
    c2 = colorbar; c2.Limits = [0,1];
    
    
    subplot(3,1,3)
    surf(xx,yy, reshape(GoR(:,frames(k)),[nys,nxs])); shading interp; colormap(gca, jet); 
    view(2); axis equal; axis tight; axis off;
    title('G\\R')
    caxis([0,1e5])
    c3 = colorbar; c3.Limits = [0,1e5];

    print('-dpng',sprintf('%s/heat_movie%03d.png',odir, k-1),'-r200')
    
end

% movie(laser_mv)

% segmentation of temperature field 
function seg = segment_T(temp, T_s, T_l)

    Tf = (temp - T_s)/(T_l-T_s);
    
    seg = (Tf < 0).*0 + (Tf >=0 & Tf <= 1).*Tf + (Tf>1).*1.0;
    
end




