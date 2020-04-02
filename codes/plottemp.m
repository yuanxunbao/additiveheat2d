set(0,'defaultaxesfontsize',16);
set(0,'defaultlinelinewidth',2);

odir = '../tex/figures/';

load para.mat;

load latdt0.00625xgrids2049.mat; fname = 'temp_wlat_wrad'; tname = 'w. latent heat/rad/convec';
% load nolatdt0.00625xgrids2049.mat; fname = 'temp_nlat_wrad'; tname = 'wo. latent heat, w. rad/convec';
% load nolatnoraddt0.00625xgrids2049.mat; fname = 'temp_nlat_nrad'; tname = 'wo. latent heat/rad/convec';

temp = temp2049;

lx= cast(lx,'double');nxs= cast(nxs,'double');nys= cast(nys,'double');
x = linspace(0,lx,nxs);
y = linspace(0,-ly,nys);
[xx,yy] = meshgrid(x,y);

frames = [11,21, 31, 41, 51];

cmin = 0;
cmax = 400;

for k = 1:length(frames)
    
    figure(1)
    set(gcf,'position',[100,100,600,900])
    subplot(length(frames),1,k);
    surf(xx,yy, reshape(temp(:,frames(k)),[nys,nxs])); shading interp; colormap(hot); 
    view(2); axis equal; axis tight; axis off;
    caxis([cmin,cmax]);
    if k==1, title(sprintf('Temperature (%s)',tname)); end
    c1 = colorbar; c1.Limits = [cmin,cmax];
    
    
end

% figure(1);
% print('-dpng',sprintf('%s/%s.png',odir,fname));




