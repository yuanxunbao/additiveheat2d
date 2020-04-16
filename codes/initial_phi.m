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
phi = -tanh((zz-z0-0.01*sin(2*pi/Lx*xx*n))/w);

figure(1);
surf(xx,zz,phi);
shading interp
view(2); 

print('-dpng','../tex/figures/phi_initial.png')

