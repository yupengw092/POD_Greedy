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
tau = 2^-9;
timeSteps = finalT / tau+1;

[node, elem] = squaremesh([-1,1,-1,1],2^-8);
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
T = myauxstructure(elem);
bdNode = T.bdNode;
freeNode = setdiff(1:NV, bdNode);
F = sparse(NV, timeSteps);AinvF = zeros(NV,timeSteps-1);FF = 0;
for j = 1:timeSteps
    fmid = f((j-1)*tau, mid);
    btf = repmat(fmid/3.*area,1,3);
    F(:,j) = accumarray(elem(:),btf(:),[NV 1]);
    if j>=2
        AinvF(freeNode,j-1) = A(freeNode,freeNode)\F(freeNode,j);
        FF = FF+AinvF(:,j-1)'*A*AinvF(:,j-1);
    end
end

m = 1;
res = ones(iter,1);idx = res;theta = res;error = res;
B = chol(A(freeNode,freeNode));
V = zeros(NV,m*iter);
AinvM = zeros(NV,m*iter);AinvA1 = AinvM;AinvA2 = AinvM;
st = 1;
tic;
for n = 1:iter
    if n == 1
        idx(n) = 20;
        mu = muset(idx(n));
        AA = M/tau+mu*A1+A2;
        for j = 1:timeSteps
            if j == 1
                uh(:,j) = uh0;
            else
                uh(bdNode, j) = pde.uD((j - 1)*tau, node(bdNode, :));
                rhs = F(:,j)+M*uh(:,j-1)/tau-AA*uh(:,j);
                uh(freeNode,j) = AA(freeNode, freeNode) \ rhs(freeNode);
            end
        end
        [~,lam,v] = svds(B*uh(freeNode,2:end),m);
        for i = 1:m
            V(freeNode,st) = uh(freeNode,2:end)*v(:,i)/lam(i,i);
            st = st+1;
        end
        c = uh0'*A*V(:,1);
        uh0 = c*V(:,1);
        theta(n) = lam(m,m)/lam(1,1);
    else
        L = zeros(musize,1);L1 = L;
        AinvM(freeNode,st-1) = A(freeNode,freeNode)\(M(freeNode,:)*V(:,st-1));
        AinvA1(freeNode,st-1) = A(freeNode,freeNode)\(A1(freeNode,:)*V(:,st-1));
        AinvA2(freeNode,st-1) = A(freeNode,freeNode)\(A2(freeNode,:)*V(:,st-1));
        MM = AinvM(:,1:st-1)'*A*AinvM(:,1:st-1);
        MA1 = AinvM(:,1:st-1)'*A*AinvA1(:,1:st-1);
        MA2 = AinvM(:,1:st-1)'*A*AinvA2(:,1:st-1);
        A1A1 = AinvA1(:,1:st-1)'*A*AinvA1(:,1:st-1);
        A1A2 = AinvA1(:,1:st-1)'*A*AinvA2(:,1:st-1);
        A2A2 = AinvA2(:,1:st-1)'*A*AinvA2(:,1:st-1);
        FM = AinvF'*A*AinvM(:,1:st-1);FA1 = AinvF'*A*AinvA1(:,1:st-1);FA2 = AinvF'*A*AinvA2(:,1:st-1);
        Mt = V(:,1:st-1)'*M*V(:,1:st-1);
        A1t = V(:,1:st-1)'*A1*V(:,1:st-1);
        A2t = V(:,1:st-1)'*A2*V(:,1:st-1);
        Ft = V(:,1:st-1)'*F;
        for p = 1:musize
            mu = muset(p);
            AA = Mt/tau+mu*A1t+A2t;
            un = zeros(st-1,timeSteps);
            un(:,1) = Mt\(V(:,1:st-1)'*M*uh0);
            r = FF;
            for j = 2:timeSteps
                un(:,j) = AA\(Ft(:,j)+Mt*un(:,j-1)/tau);
                r = r+(un(:,j)-un(:,j-1))'*MM*(un(:,j)-un(:,j-1))/tau^2+...
                    un(:,j)'*(mu^2*A1A1+2*mu*A1A2+A2A2)*un(:,j)-...
                    2*FM(j-1,1:st-1)*(un(:,j)-un(:,j-1))/tau-...
                    2*(mu*FA1(j-1,1:st-1)+FA2(j-1,1:st-1))*un(:,j)+...
                    2*(un(:,j)-un(:,j-1))'*(mu*MA1+MA2)*un(:,j)/tau;
            end
            L(p) = sqrt(tau*r);
        end
        [~,idx(n)] = max(L);
        res(n) = L(idx(n));
        mu = muset(idx(n));
        AA = M/tau+mu*A1+A2;
        for j = 1:timeSteps
            if j == 1
                uh(:,j) = uh0;
            else
                uh(bdNode, j) = pde.uD((j - 1)*tau, node(bdNode, :));
                rhs = F(:,j)+M*uh(:,j-1)/tau-AA*uh(:,j);
                uh(freeNode,j) = AA(freeNode, freeNode) \ rhs(freeNode);
            end
        end
        aa = Mt/tau+mu*A1t+A2t;
        for j = 2:timeSteps
            un(:,j) = aa\(Ft(:,j)+Mt*un(:,j-1)/tau);
        end
        err = uh-V(:,1:st-1)*un;
        error(n) = sqrt(tau*trace(err(:,2:end)'*A*err(:,2:end)));
        for j = 2:timeSteps
            for k = 1:st-1
                uh(:,j) = uh(:,j)-V(:,k)'*A*uh(:,j)*V(:,k);
            end
        end
        %% POD basis
        [~, lam,v] = svds(B*uh(freeNode,2:end),m);
        for i = 1:m
            V(freeNode,st) = uh(freeNode,2:end)*v(:,i)/lam(i,i);
            st = st+1;
        end
        theta(n) = lam(m,m)/lam(1,1);
    end
end
toc

figure('Color', [1 1 1]);
hold on;
semilogy(1:iter, log10(res), 'Marker', 'o', 'linewidth', 0.5);
hold on;
semilogy(1:iter, log10(error), 'Marker', 'o', 'linewidth', 0.5);
legend('${\rm log}_{10} \Delta_n$', '${\rm log}_{10} e_n$','Interpreter', 'latex','Fontsize', 14);
xlabel('$n$','Interpreter','latex');
ylabel('${\rm log}_{10}\Delta_n$/ ${\rm log}_{10} e_n$','Interpreter','latex');
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






