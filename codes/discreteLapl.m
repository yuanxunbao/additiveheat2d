% 2D discrete Laplacian on a regular grid with Dir BC
% ref: https://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
nx = 50;
ex = ones(nx,1);
Dxx = spdiags([-ex 2*ex -ex],[-1 0 1], nx, nx);
Ix = speye(nx); % sparse identity

ny = 10;
ey = ones(ny,1);
Dyy = spdiags([-ey 2*ey -ey], [-1 0 1], ny, ny);
Iy = speye(ny);

% col ordering
Lcol = kron(Dxx, Iy) + kron(Ix,Dyy);

% row ordering
Lrow= kron(Dyy, Ix) + kron(Iy,Dxx);

% several method for reordering.
p1 = dissect(Lrow);
p2 = amd(Lrow);
p3 = symrcm(Lrow);


figure(1)
subplot(121)
spy(Lrow)
title('row ordering')
subplot(122)
spy(chol(Lrow))
title('chol. factorization')

figure(2)
subplot(121)
spy(Lcol)
title('col. ordering')
subplot(122)
spy(chol(Lcol))
title('chol. factorization')

figure(3)
subplot(121)
spy(Lrow(p1,p1))
title('nested dissection')
subplot(122)
spy(chol(Lrow(p1,p1)))
title('chol. factorization')

figure(4)
subplot(121)
spy(Lrow(p2,p2))
title('Approximate Minimum Degree')
subplot(122)
spy(chol(Lrow(p2,p2)))
title('chol. factorization')

figure(5)
subplot(121)
spy(Lrow(p3,p3))
title('Reverse Cuthill-McKee')
subplot(122)
spy(chol(Lrow(p3,p3)))
title('chol. factorization')



