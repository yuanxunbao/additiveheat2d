load FR_check_low.mat


[rr, tt] = meshgrid(r,t_ma);
figure(1)
subplot(131)
surf(rr, tt,T_actual);shading interp; view(2); axis tight;colorbar;caxis([900 1000])
xlabel('r (meter)'); ylabel('time (seconds)'); title('macro solver T(t,r)')


subplot(132)
surf(rr, tt,T_FR);shading interp; view(2);  axis tight;colorbar;caxis([900 1000])
xlabel('r (meter)'); ylabel('time (seconds)'); title('frozen temp. approx. ')


err = abs(T_actual-T_FR)./abs(T_actual);
subplot(133)
surf(rr, tt, err);shading interp; view(2);  axis tight; colorbar
xlabel('r (meter)'); ylabel('time (seconds)'); title('pointwise error ')