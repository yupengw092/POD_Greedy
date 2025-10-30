clc
clear
format long;

f = @(z,b) 1./sqrt((z(:,1)-b(1)).^2+(z(:,2)-b(2)).^2+1);
xmu = linspace(0.01, 0.1, 10)';
[Xmu, Tmu] = meshgrid(xmu, xmu);
muset = [Xmu(:) Tmu(:)];
musize = size(muset, 1);
tau = 2^(-7);
Nt = 2^7;
t = linspace(tau, 1, Nt)';
Nx = 100;
x = linspace(0.01, 1, Nx)';
[T, X] = meshgrid(t, x);
dofset = [T(:) X(:)];
U = zeros(Nx*Nt,musize);
for p = 1:musize
    U(:,p) = f(dofset,muset(p,:));
end
m = 4;iter = 20;
for k = 1:m
    [error(k,:),theta(k,:)] = EIMPODGreedy(U,k,iter,Nx,Nt,tau);
end

label = {'8','16','24','32','40','48','56','64','72','80'};
xtick1 = 1:1:20;
xtick2 = 2:2:40;
xtick3 = 3:3:60;
xtick4 = 4:4:80;
Nlabel = 32;
idx1 = xtick1<=Nlabel;
idx2 = xtick2<=Nlabel;
idx3 = xtick3<=Nlabel;
idx4 = xtick4<=Nlabel;
figure('Color',[1 1 1]);
hold on;
semilogy(xtick1(idx1), log10(error(1,idx1)), 'Marker', '*', 'LineWidth', 0.5);
semilogy(xtick2(idx2), log10(error(2,idx2)), 'Marker', '+', 'LineWidth', 0.5);
semilogy(xtick3(idx3), log10(error(3,idx3)), 'Marker', 's', 'LineWidth', 0.5);
semilogy(xtick4(idx4), log10(error(4,idx4)), 'Marker', 'o', 'LineWidth', 0.5);
hold on;
legend('$m=1$', '$m=2$', '$m=3$', '$m=4$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\log_{10}\widehat{E}_N$','Interpreter','latex','Fontsize',26); 
xlabel('$N$', 'Interpreter', 'latex','Fontsize',24);
set(gca, 'FontSize', 14); 
xticks(8:8:32);
%xticklabels({'8','16','24','32','40','48','56','64','72','80'});
xticklabels({'8','16','24','32'}); 
grid on;
saveas(gcf,'EIMPODGreedymbasis','fig');
%saveas(gcf,'EIMPODGreedymbasis','epsc');
%%  EIM-POD-Greedy (m POD basis)
function [error,theta] = EIMPODGreedy(U,m,iter,Nx,Nt,tau)
musize = size(U,2);
B = zeros(m*iter,m*iter);q = zeros(m*iter,1);Q = zeros(Nx,m*iter);dofid = zeros(m*iter,1);
error = zeros(iter,1);theta = error;
for n = 1:iter
    idx = (n-1)*m;
    if n == 1
        r = reshape(U(:,1),Nx,Nt);
        l = max(abs(r));
        error(n) = sqrt(tau*sum(l.^2));
        [~,lam,v] = svds(r,m);
        u = r*v/lam;
        theta(n) = lam(m,m)/lam(1,1);
        for k = 1:m
            idx = idx+1;
            if k == 1
                w = u(:,k);
                [~,dofid(idx)] = max(abs(w));
                q(idx) = w(dofid(idx));
                Q(:,idx) = w/q(idx);
                B(idx,1:idx) = Q(dofid(idx),1:idx);
            else
                w = u(:,k)-Q(:,1:idx-1)*((B(1:idx-1,1:idx-1)\u(dofid(1:idx-1),k)));
                [~,dofid(idx)] = max(abs(w));
                q(idx) = w(dofid(idx));
                Q(:,idx) = w/q(idx);
                B(idx,1:idx) = Q(dofid(idx),1:idx);
            end
        end
    else
        L = zeros(musize,1);
        for p = 1:musize
            Up = reshape(U(:,p),Nx,Nt);
            r(:,:,p) = Up-Q(:,1:idx)*(B(1:idx,1:idx)\Up(dofid(1:idx),:));
            l = max(abs(r(:,:,p)));
            L(p) = sqrt(tau*sum(l.^2));
        end
        [~,muid] = max(L);
        error(n) = L(muid);
        Un = reshape(U(:,muid),[Nx,Nt]);
        r = Un-Q(:,1:idx)*(B(1:idx,1:idx)\Un(dofid(1:idx),:));
        [~,lam,v] = svds(r,m);
        u = r*v/lam;
        theta(n) = lam(m,m)/lam(1,1);
        for k = 1:m
            idx = idx+1;
            w = u(:,k)-Q(:,1:idx-1)*((B(1:idx-1,1:idx-1)\u(dofid(1:idx-1),k)));
            [~,dofid(idx)] = max(abs(w));
            q(idx) = w(dofid(idx));
            Q(:,idx) = w/q(idx);
            B(idx,1:idx) = Q(dofid(idx),1:idx);
        end
    end
end
end

