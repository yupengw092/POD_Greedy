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
idx =  [10,100,100,100,100,100,55,100,51,100,49,53,100,48,52,49,55,100,100,1];
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


function T = myauxstructure(elem)
%% AUXSTRUCTURE auxiliary structure for a 2-D triangulation.
totalEdge = sort([elem(:,[2,3]); elem(:,[3,1]); elem(:,[1,2])],2);
[edge,i2,j] = unique(totalEdge,'rows','legacy');
NT = size(elem,1);
elem2edge = reshape(j,NT,3);
i1(j(3*NT:-1:1)) = 3*NT:-1:1; 
i1 = i1';
k1 = ceil(i1/NT); 
k2 = ceil(i2/NT); 
t1 = i1 - NT*(k1-1);
t2 = i2 - NT*(k2-1);
ix = (i1 ~= i2); 
edge2elem = [t1,t2,k1,k2];
neighbor = accumarray([[t1(ix),k1(ix)];[t2,k2]],[t2(ix);t1],[NT 3]);
bdElem = t1(t1 == t2);


iy = ( t1 == t2 ); list = (1:size(edge,1))'; 
bdEI = list(iy); 
bdEdge = edge(bdEI,:);
bdk1 = k1(t1 == t2);
bdEdge2elem = [bdElem(bdk1==1);bdElem(bdk1==2);bdElem(bdk1==3)];


signedge = ones(NT,3);
signedge(:,1) = signedge(:,1) - 2* (elem(:,2)>elem(:,3));
signedge(:,2) = signedge(:,2) - 2* (elem(:,3)>elem(:,1));
signedge(:,3) = signedge(:,3) - 2* (elem(:,1)>elem(:,2));


bdLI = edge2elem(bdEI,3);


T = struct('neighbor',neighbor,'elem2edge',elem2edge,'edge',edge,'edge2elem',edge2elem,...
    'bdEdge',bdEdge,'bdNode',unique(bdEdge),'bdEI',bdEI,...
    'signedge',signedge,'bdElem',bdElem,'bdEdge2elem',bdEdge2elem);
end

