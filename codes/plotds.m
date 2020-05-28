load a0W01.4131lx22.5.mat

xx3=xx;zz3=zz;




figure(3)
subplot(1,6,1);
surf(xx3,zz3,reshape(y(1:nz*nx,1),[nz,nx]));shading interp;
view(2); axis tight;axis equal; 
tit1=strcat('t=',num2str(0));
title(tit1)
subplot(1,6,2);
surf(xx3,zz3,reshape(y(1:nz*nx,8),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit2=strcat('t=',num2str(t/5));
title(tit2)
subplot(1,6,3);
surf(xx3,zz3,reshape(y(1:nz*nx,10),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit3=strcat('t=',num2str(2*t/5));
title(tit3)
subplot(1,6,4);
surf(xx3,zz3,reshape(y(1:nz*nx,11),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit4=strcat('t=',num2str(3*t/5));
title(tit4)
subplot(1,6,5);
surf(xx3,zz3,reshape(y(1:nz*nx,41),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit5=strcat('t=',num2str(4*t/5));
title(tit5)
subplot(1,6,6);
surf(xx3,zz3,reshape(y(1:nz*nx,51),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit6=strcat('t=',num2str(t));
title(tit6)

figure(2)
subplot(1,6,1);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,1),[nz,nx]));shading interp;
view(2); axis tight;axis equal; 
tit1=strcat('t=',num2str(0));
title(tit1)
subplot(1,6,2);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,11),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit2=strcat('t=',num2str(t/5));
title(tit2)
subplot(1,6,3);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,21),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit3=strcat('t=',num2str(2*t/5));
title(tit3)
subplot(1,6,4);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,31),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit4=strcat('t=',num2str(3*t/5));
title(tit4)
subplot(1,6,5);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,41),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit5=strcat('t=',num2str(4*t/5));
title(tit5)
subplot(1,6,6);
surf(xx3,zz3,reshape(y(nz*nx+1:2*nz*nx,51),[nz,nx])); shading interp; 
view(2); axis equal; axis tight; 
tit6=strcat('t=',num2str(t));
title(tit6)
