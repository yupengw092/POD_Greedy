% Convergence history of the empirical interpolation approximation 
% and the EIM-POD-Greedy method for a target function f(x,t) in [0,1]^2 for iter iterations


clc
clear all;
format long;
warning off;
f = @(z,b) 1./sqrt((z(:,1)-b(1)).^2+(z(:,2)-b(2)).^2+1);
xmu = linspace(0.01, 0.1, 10)';
[Xmu, Tmu] = meshgrid(xmu, xmu);
muset = [Xmu(:) Tmu(:)];
musize = size(muset, 1);
tau = 2^(-7);

Nt = 2^7;
t = linspace(2^-7, 1, Nt)';
Nx = 100;
x = linspace(0.01, 1, Nx)';
[T, X] = meshgrid(t, x);
dofset = [T(:) X(:)];

iter = 20;
error1 = zeros(iter,1);error2 = error1; U = zeros(size(dofset,1),musize);

%%  EIM
B = zeros(iter,iter);q = zeros(iter,1);Q = zeros(size(dofset,1),iter);
for p = 1:musize
    U(:,p) = f(dofset,muset(p,:));
end
for n = 1:iter
    if n == 1
        r = U;
    else
        r = U-Q(:,1:n-1)*(B(1:n-1,1:n-1)\U(dofid(1:n-1),:));
    end
    L = max(abs(r));
    [~,muid] = max(L);
    u = r(:,muid);
    [~,dofid(n)] = max(abs(u));
    q(n) = u(dofid(n));
    Q(:,n) = u/q(n);
    B(n,1:n) = Q(dofid(n),1:n);
    error1(n) = abs(q(n));
end

clear B;
clear Q;
clear dofid;
clear r;


%%  EIM-POD-Greedy
B = zeros(iter,iter);q = zeros(iter,1);Q = zeros(Nx,iter);L = zeros(musize,1);
eta = error2;eta_avg = error2;kappa = error2;Lambda = error2;
for n = 1:iter
    if n == 1
        for p = 1:musize
            Up = reshape(U(:,p),Nx,Nt);
            r(:,:,p) = Up;
            l = max(abs(r(:,:,p)));
            L(p) = sqrt(tau*sum(l.^2));
        end
    else
        for p = 1:musize
            Up = reshape(U(:,p),Nx,Nt);
            r(:,:,p) = Up-Q(:,1:n-1)*(B(1:n-1,1:n-1)\Up(dofid(1:n-1),:));
            l = max(abs(r(:,:,p)));
            L(p) = sqrt(tau*sum(l.^2));
        end
    end
    [~,muid] = max(L);
    error2(n) = L(muid);
    rn = r(:,:,muid);
    [~,lam,v] = svds(rn,1);
    u = rn*v/lam(1,1);
    if n == 1
        w = u;
    else
        w = u-Q(:,1:n-1)*((B(1:n-1,1:n-1)\u(dofid(1:n-1))));
    end
    [~,dofid(n)] = max(abs(w));
    q(n) = w(dofid(n));
    Q(:,n) = w/q(n);
    B(n,1:n) = Q(dofid(n),1:n);
    %% eta_avg_n
    for p = 1:musize
        delta = sqrt(tau*sum(r(dofid(n),:,p).^2,2));
        eta(n) = eta(n)+delta/L(p);
    end
    eta_avg(n) = eta(n)/musize;
    %% kappa_n
    kappa(n) = cond(B(1:n,1:n));
    %% Lambda_n
    rho = sum(abs(Q(:,1:n)*(B(1:n,1:n)^(-1))),2);
    Lambda(n) = max(rho);
end
clear B;
clear Q;
clear dofid;
label = {'2', '4', '6', '8', '10', '12', '14', '16', '18', '20'};
figure;
hold on;
semilogy(1:iter, log10(error1), 'Marker', '+', 'LineWidth', 0.5);
semilogy(1:iter, log10(error2), 'Marker', '*', 'LineWidth', 0.5);
hold off;
xlabel('$N$','Interpreter','latex','Fontsize',24);
ylabel('$\log_{10} \widehat{E}_N$','Interpreter','latex','Fontsize',26);
xticks(2:2:20);
xticklabels({'2','4','6','8','10','12','14','16','18','20'});
set(gca, 'FontSize', 14);
grid on;
legend('EIM','EIM-POD-Greedy Method',  'FontSize', 14, 'Interpreter', 'latex');

saveas(gcf,'EIM_EIMPODGreedy','epsc');

