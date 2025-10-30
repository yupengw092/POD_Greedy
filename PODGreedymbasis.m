% Compute the error of the weak POD-Greedy method with m = 1,2,3,4

clc
clear
format long;
f = @(t,z) exp(-t).*sin(pi*z(:,1)).*sin(pi*z(:,2));
g = @(z) sin(pi*z(:,1)).*sin(pi*z(:,2));
uD = @(t,z) zeros(size(z,1),1);
pde.f = f;pde.u0 = g;pde.uD = uD;
musize = 100;
muset = linspace(1, 2, musize)';
iter = 20;
finalT = 1;
s = 6;
tau = 2^(-s-1);
timeSteps = finalT / tau+1;

[node, elem] = squaremesh([-1,1,-1,1],2^-s);
NT = size(elem,1);NV = size(node,1);
M = sparse(NV,NV);A1 = M;A2 = M;A = M;
n1 = [];n2 = n1;
for k = 1:NT
    x = node(elem(k,:),1);
    if all(x<=0)
        n1 = [n1,k];
    end
end
n2 = setdiff(1:NT,n1);
[Dlambda, area] = gradbasis(node, elem);
for i = 1:3
    for j = 1:3
        A1ij = area(n1).*dot(Dlambda(n1, :, i), Dlambda(n1, :, j), 2);
        A1 = A1 + sparse(elem(n1,i), elem(n1,j), A1ij, NV, NV);
        A2ij = area(n2).*dot(Dlambda(n2, :, i), Dlambda(n2, :, j), 2);
        A2 = A2 + sparse(elem(n2,i), elem(n2,j), A2ij, NV, NV);
        Aij = area.*dot(Dlambda(:, :, i), Dlambda(:, :, j), 2);
        A = A + sparse(elem(:,i), elem(:,j), Aij, NV, NV);
        if i==j
            Mij = area/6.*ones(NT,1);
        else
            Mij = area/12.*ones(NT,1);
        end
        M = M + sparse(elem(:,i), elem(:,j), Mij, NV, NV);
    end
end

mid= (node(elem(:,1),:)+node(elem(:,2),:)+node(elem(:,3),:))/3;
gmid = g(mid);
btg = repmat(gmid/3.*area,1,3);
F0 = accumarray(elem(:),btg(:),[NV 1]);
uh0 = M\F0;

F = sparse(NV, timeSteps);
for j = 1:timeSteps
    fmid = f((j-1)*tau, mid);
    btf = repmat(fmid/3.*area,1,3);
    F(:,j) = accumarray(elem(:),btf(:),[NV 1]);
end
T = myauxstructure(elem);
bdNode = T.bdNode;
freeNode = setdiff(1:NV, bdNode);
B = chol(A(freeNode,freeNode));
load("PODGreedysmall.mat");
error = zeros(iter,4);theta = error;
for m = 1:4
    st = 1;
    V = zeros(NV,m*iter);
    for n = 1:iter
        if n == 1
            mu = muset(idx(n));
            uh = Solve(mu, uh0,pde,node, elem, A1, A2, M, tau, timeSteps, F);
            [~,lam,v] = svds(B*uh(freeNode,2:end),m);
            for i = 1:m
                V(freeNode,st) = uh(freeNode,2:end)*v(:,i)/lam(i,i);
                st = st+1;
            end
            theta(n,m) = lam(i,i)/lam(1,1);
            error(n,m) = sqrt(tau*trace(uh(:,2:end)'*A*uh(:,2:end)));
        else
            mu = muset(idx(n));
            uh = Solve(mu, uh0,pde,node, elem, A1, A2, M, tau, timeSteps, F);
            for j = 2:timeSteps
                for k = 1:st-1
                    uh(:,j) = uh(:,j)-V(:,k)'*A*uh(:,j)*V(:,k);
                end
            end
            [~,lam,v] = svds(B*uh(freeNode,2:timeSteps),m);
            for i = 1:m
                V(freeNode,st) = uh(freeNode,2:end)*v(:,i)/lam(i,i);
                st = st+1;
            end
            theta(n,m) = lam(i,i)/lam(1,1);
            L = zeros(musize,1);
            for p = 1:musize
                mu =muset(p);
                uh = Solve(mu,uh0,pde,node, elem, A1, A2, M, tau, timeSteps, F);
                for j = 2:timeSteps
                    for k = 1:st-1
                        uh(:,j) = uh(:,j)-V(:,k)'*A*uh(:,j)*V(:,k);
                    end
                end
                L(p) = sqrt(tau*trace(uh(:,2:end)'*A*uh(:,2:end)));
            end
            error(n,m) = max(L);
        end
        fprintf('Iteraton at %d-th step\n',n);
    end
end
save('PODGreedymbasis.mat','error');
figure('Color', [1 1 1]);
hold on;
semilogy(1:iter, log10(error(:,1)), 'Marker', 'o', 'linewidth', 0.5);
hold on;
semilogy(1:iter, log10(error(:,2)), 'Marker', 's', 'linewidth', 0.5);
hold on;
semilogy(1:iter, log10(error(:,3)), 'Marker', '*', 'linewidth', 0.5);
hold on;
semilogy(1:iter, log10(error(:,4)), 'Marker', '+', 'linewidth', 0.5);

legend('$m=1$','$m=2$','$m=3$','$m=4$','Interpreter', 'latex','Fontsize', 14);
xlabel('$N$','Interpreter','latex');
ylabel('${\rm log}_{10} E_N$','Interpreter','latex');
xticks(2:2:20);
xticklabels({'2', '4', '6', '8', '10', '12', '14', '16', '18', '20'});
grid on;
saveas(gcf,'PODGreedymbasis','epsc');

save('PODGreedymbasis.mat','error','theta');
