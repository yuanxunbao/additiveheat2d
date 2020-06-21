Lx = 1;
Lz = 4;

nx = 100;
nz = 800;

dx = Lx/nx;
dz = Lz/nz;

x = 0:dx:Lx;
z = 0:dz:Lz;

k_max = floor(Lx/ (5 * dx)); % max wavenumber, 12 grid points to resolve the highest wavemode

eta = 1e-2;
A = (2*rand(1,k_max)-1) * eta; % amplitude, draw from [-1,1] * eta
x_c = rand(1,k_max)*Lx; % shift, draw from [0,Lx]

[zz,xx] = meshgrid(z,x);

W0 = 2e-2; % interfacial width
z0 = 1e-1; % level of z-axis

% sinusoidal perturbation
sp = 0; 
for k = 1:k_max
   
    sp = sp + A(k)*sin(2*pi*k/Lx* (xx-x_c(k)) );
    
end

phi0 = -tanh((zz-z0-sp)/W0);

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

figure(2);
% plot(x,sp); hold on
plot(x, sin(2*pi*k_max/Lx*x),'.-');



% print('-dpng','../tex/figures/phi_initial.png')

