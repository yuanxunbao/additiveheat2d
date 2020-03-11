% 2D discrete Laplacian on a regular grid with Dir BC
nx = 10;
ex = ones(nx,1);
Dxx = spdiags([-ex 2*ex -ex],[-1 0 1], nx, nx);
Ix = speye(nx); % sparse identity

ny = 20;
ey = ones(ny,1);
Dyy = spdiags([-ey 2*ey -ey], [-1 0 1], ny, ny);
Iy = speye(ny);

% row ordering
Lrow = kron(Dxx, Iy) + kron(Ix,Dyy);

% column ordering
Lcol= kron(Dyy, Ix) + kron(Iy,Dxx);

% several method for reordering.
p1 = dissect(Lcol);
p2 = amd(Lcol);
p3 = symrcm(Lcol);


figure(1)
subplot(121)
spy(Lrow)
title('row ordering')
subplot(122)
spy(chol(Lrow))
title('chol. factorization')

figure(2)
subplot(121)
spy(chol(Lcol))
title('col. ordering')
subplot(122)
spy(L2)
title('chol. factorization')

figure(3)
subplot(121)
spy(Lcol(p,p))
title('tested dissection')
subplot(122)
spy(chol(Lcol(p,p)))
title('chol. factorization')

figure(4)
subplot(121)
spy(Lcol(p1,p1))
title('nested dissection')
subplot(122)
spy(chol(Lcol(p1,p1)))
title('chol. factorization')


figure(5)
subplot(121)
spy(Lcol(p2,p2))
title('Approximate Minimum Degree')
subplot(122)
spy(chol(Lcol(p2,p2)))
title('chol. factorization')

figure(6)
subplot(121)
spy(Lcol(p3,p3))
title('Reverse Cuthill-McKee')
subplot(122)
spy(chol(Lcol(p3,p3)))
title('chol. factorization')



