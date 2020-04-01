set(0,'defaultaxesfontsize',16);
set(0,'defaultlinelinewidth',2);


load para.mat;

load latdt0.00625xgrids2049.mat;
%load nolatnoraddt0.00625xgrids2049.mat;
%load nolatdt0.00625xgrids2049.mat;

lx= cast(lx,'double');nxs= cast(nxs,'double');nys= cast(nys,'double');
x = linspace(0,lx,nxs);
y = linspace(0,-ly,nys);
[xx,yy] = meshgrid(x,y);


figure(1)
subplot(5,1,1);
surf(xx,yy, reshape(temp2049(:,11),[nys,nxs])); shading interp; colormap gray; 
view(2); axis equal; axis tight; set(gca,'visible','off')
c1 = colorbar; c1.Limits = [0 400];
subplot(5,1,2);
surf(xx,yy, reshape(temp2049(:,21),[nys,nxs])); shading interp; colormap gray; 
view(2); axis equal; axis tight; set(gca,'visible','off')
c2 = colorbar; c2.Limits = [0 400];
subplot(5,1,3);
surf(xx,yy, reshape(temp2049(:,31),[nys,nxs])); shading interp; colormap gray; 
view(2); axis equal; axis tight; set(gca,'visible','off')
c3 = colorbar; c3.Limits = [0 400];
subplot(5,1,4);
surf(xx,yy, reshape(temp2049(:,41),[nys,nxs])); shading interp; colormap gray; 
view(2); axis equal; axis tight; set(gca,'visible','off')
c4 = colorbar; c4.Limits = [0 400];
subplot(5,1,5);
surf(xx,yy, reshape(temp2049(:,51),[nys,nxs])); shading interp; colormap gray; 
view(2); axis equal; axis tight; set(gca,'visible','off')
c5 = colorbar; c5.Limits = [0 400];





