Lx = 1;
Lz = 2;

nx = 100;
nz = 200;

dx = Lx/nx;
dz = Lz/nz;

x = 0:dx:Lx;
z = 0:dz:Lz;

[zz,xx] = meshgrid(z,x);

w = 2e-2;
z0 = 2e-1;
n = 1;
phi0 = -tanh((zz-z0-0.01*cos(2*pi/Lx*xx*n) )/w);

U0 = 0*phi0-1;

figure(1);
subplot(121);
surf(xx,zz,phi0);
shading interp
view(2);
colorbar
title('phi0')
subplot(122)
surf(xx,zz,U0);
shading interp
view(2);
colorbar
title('U0')


% print('-dpng','../tex/figures/phi_initial.png')

