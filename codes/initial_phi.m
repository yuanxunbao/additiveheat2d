Lx = 1;
Lz = 2;

nx = 100;
nz = 200;

dx = Lx/nx;
dz = Lz/nz;

x = 0:dx:Lx;
z = 0:dy:Lz;

[zz,xx] = meshgrid(z,x);

w = 2e-2;
z0 = 1e-1;
n = 10;
phi0 = -tanh((zz-z0-0.01*sin(2*pi/Lx*xx*n))/w);

k = 0.3;
U0 = 1/(1-k) * ( k./((1-phi0)/2+k*(1+phi0)/2) - 1);

figure(1);
subplot(121)
surf(xx,zz,phi0);
shading interp
view(2);
colorbar
title('phi')

subplot(122) 
surf(xx,zz,U0);
shading interp
view(2);
colorbar
title('U')



print('-dpng','../tex/figures/phi_initial.png')

