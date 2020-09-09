[rr, tt] = meshgrid(r,t_ma);
figure(1)
subplot(121)
surf(rr, tt,T_actual);shading interp; view(2); axis tight;colorbar;caxis([900 1000])
xlabel('r(m)');ylabel('t(s)');title('Actual temperature distribution')
subplot(122)
surf(rr, tt,T_FR);shading interp; view(2);  axis tight;colorbar;caxis([900 1000])
xlabel('r(m)');ylabel('t(s)');title('Frozen temperature assumption')