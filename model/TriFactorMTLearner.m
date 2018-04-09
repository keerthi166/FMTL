function [W,C,F,S,G, Sigma, Omega] = TriFactorMTLearner( X,Y,rho_fr1,rho_fr2,opts)
%% Multi-task Dictionary learning
% Solve the following objective function
%
% $$\min_{\mathbf{F},\mathbf{G},C} \mathcal{L}(Y,\mathcal{F}(X,[\mathbf{F}\mathbf{G}^\top;C])) + \mu ||F||_F^2 + \rho_{l1} ||G||_1^1$$
%
% $X$ and $Y$ are the cell array of size K,
% $\mathcal{L}$ is the loss function (given by opts.loss),
% $C$ is the bias term (1xK) vector,
% $\mathbf{F}$ is the task dictionary (PxK) matrix,
% $\mathbf{G}$ is the task sparse code (PxK) matrix,
% $\rho_{l1},\mu$ are the regularization parameters,
%
%
% See also <MTSolver.m MTSolver>


K=length(Y);
N=cellfun(@(x) size(x,1),X);
[~,P]=size(X{1});

loss=opts.loss;
debugMode=opts.debugMode;

% Regularization Parameters
rho_l1=0.1; %reg. param for l1 penalty
if isfield(opts,'rho_l1')
    rho_l1=opts.rho_l1;
end
kappa1=3; %Number of feature clusters
if isfield(opts,'kappa1')
    kappa1=opts.kappa1;
end
kappa2=3; %Number of task clusters
if isfield(opts,'kappa2')
    kappa2=opts.kappa2;
end
learnOmega=true;
if isfield(opts,'learnOmega')
    learnOmega=opts.learnOmega;
end

learnSigma=true;
if isfield(opts,'learnSigma')
    learnSigma=opts.learnSigma;
end

sparseOmega=false;
if isfield(opts,'sparseOmega')
    iOmega=zeros(K);
    sparseOmega=opts.sparseOmega;
end

sparseSigma=false;
if isfield(opts,'sparseSigma')
    iSigma=zeros(P);
    sparseSigma=opts.sparseSigma;
end




% Initialize F, G
mu=0;
[W,~]= STLearner(X,Y,mu,opts);

[U,E,V] = svd(W);
[~,ind] = sort(diag(E),'descend');
ind=[ind;(min(P,K)+1:max(P,K))'];
if (kappa1>P)
    F = [U(:,ind(1:P)) randn(P,kappa1-P)];
else
    F = U(:,ind(1:kappa1)); % Pxkappa1 matrix
end

if (kappa2>K)
    G = [V(ind(1:K),:);randn(kappa2-K,K)]';
else
    G = V(ind(1:kappa2),:)'; % Kxkappa2 matrix
end
S=F'*W*G;

opts.Finit=F;
opts.Ginit=G;
opts.rho=rho_l1;
opts.rho_sr=rho_fr2;
maxIter=opts.maxOutIter;
rho_F=0;
rho_G=0;
rho_S=0.1;

Sigma=eye(P)/P;
if isfield(opts,'LF')
    if ~isempty(opts.LF)
        Sigma=opts.LF;
        learnSigma=false;
    end
end
Omega=eye(K)/K;

epsilon=1e-8;
obj=0;



vecF=zeros(P*kappa1,1);
vecG=zeros(K*kappa2,1);
vecS=zeros(kappa1*kappa2,1);
Gcell=mat2cell(G,ones(1,K),kappa2)';
for it=1:maxIter
    % Solve for S given (G,F)
    
    temp=cellfun(@(x,y,g) 1*F'*x'*y*g,X,Y,Gcell,'UniformOutput',false);
    B=sum(cat(3,temp{:}),3);
    [vecS,~,~,~,~]=pcg(@getAS,B(:),[],[],[],[],vecS);
    if sum(vecS)==0
        vecS=S(:);
    end
    S = reshape(vecS,kappa1,kappa2);
    
    
    
    % Solve for G given (F,S)
    XFS=cellfun(@(x) (x*F)*S,X,'UniformOutput',false);
    %opts.Omega=Omega;
    %G=StructMTLearner(XFS,Y,rho_fr2, opts)';
  
    
    temp=cellfun(@(xfs,y) xfs'*y,XFS,Y,'UniformOutput',false);
    RHS=cat(1,temp{:});
    temp=cellfun(@(xfs) (xfs'*xfs),XFS,'UniformOutput',false);
    LHS=blkdiag(temp{:})+rho_fr2*kron(eye(kappa2),Omega)+rho_G;
    clear temp
    [vecG,~,~,~,~]=pcg(LHS,RHS,[],[],[],[],vecG);
    G = reshape(vecG,kappa2,K)';
    
    clear LHS RHS
    %}
    
    Gcell=mat2cell(G,ones(1,K),kappa2)';
    
    % Solve for F, given (G,S)
    temp=cellfun(@(x,y,g) 1*x'*y*g*S',X,Y,Gcell,'UniformOutput',false);
    B=sum(cat(3,temp{:}),3);
    
    [vecF,~,~,~,~]=pcg(@getAF,B(:),[],[],[],[],vecF);
    F = reshape(vecF,P,kappa1);
    
    %{
    % Solve for Sigma, given F
    [U,S] = eig(F*F'+epsilon*eye(P));
    Smin=sqrt(abs(real(diag(S))));
    Smin=Smin/sum(Smin);
    Sigma = U * diag(1./(Smin)) * U';
    
    % Solve for Omega, given G
    [V,Q] = eig(G*G'+epsilon*eye(K));
    
    Qmin=sqrt(abs(real(diag(Q))));
    Qmin=Qmin/sum(Qmin);
    Omega = V * diag(1./(Qmin)) * V';
    %}
    
    
    if learnSigma
        if sparseSigma
            %[iSigma,Sigma]=GraphicalLasso((W*Omega*W')/K,rho_l1,[],0,1,1,1,1e-4,100,iSigma,Sigma);
            [Sigma,~] = graphicalLasso((W*Omega*W')/K,rho_l1, 5, 1e-4);
            Sigma=real(Sigma);
        else
            Sigma = fastOmega(F, epsilon);
        end
    end
    if learnOmega
        if sparseOmega
            %[iOmega,Omega] = GraphicalLasso((W'*Sigma*W)/P,rho_l1,[],0,1,1,1,1e-4,100,iOmega,Omega);
            [Omega,~] = graphicalLasso((W'*Sigma*W)/P,rho_l1, 5, 1e-4);
            Omega=real(Omega);
        else
            Omega = fastOmega(G, epsilon);
        end
    end
    
    
    obj=[obj;func(F,S,G)];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if mod(it,1)==0 && debugMode
        fprintf('Iteration %d, Objective:%f, Relative Obj:%f \n',it,obj(end),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= opts.tol)
        break;
    end
    
end
W=F*S*G';
C=zeros(1,K);

% Fast Computation of AX for solving AX=B via CG
    function vecAx=getAF(vecF)
        Fmat=reshape(vecF,P,kappa1);
        temp1=cellfun(@(x,g) (x'*x)*Fmat*((g*S')'*(g*S'))+(rho_F+rho_fr1*Sigma)*Fmat,X,Gcell,'UniformOutput',false);
        matAx=sum(cat(3,temp1{:}),3);
        vecAx=matAx(:);
    end

    function vecAx=getAS(vecS)
        matS=reshape(vecS,kappa1,kappa2);
        temp1=cellfun(@(x,g) ((x*F)'*(x*F))*matS*(g'*g)+rho_S,X,Gcell,'UniformOutput',false);
        matAx=sum(cat(3,temp1{:}),3);
        vecAx=matAx(:);
    end
% Compute Objective function
    function Fobj=func(F,S,G)
        Wcell=mat2cell(F*S*G',P,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                temp0=cellfun(@(x,w,y,n) mean(max(1-y.*(x*w),0)),X,Wcell,Y,Ncell,'UniformOutput',false);
            case 'logit'
                temp0=cellfun(@(x,w,y,n) mean(log(1+exp(-(x*w).*y))),X,Wcell,Y,Ncell,'UniformOutput',false);
            otherwise % Default Least Square Loss
                % Func of Squared Error Loss
                temp0=cellfun(@(x,w,y,n) 0.5*norm((y - x*w))^2,X,Wcell,Y,Ncell,'UniformOutput',false);
        end
        Fobj=sum(cell2mat(temp0))+0.5*rho_fr1*trace((F*F')*Sigma) + 0.5*rho_fr2*trace((G*G')*Omega);
    end


    function omega = fastOmega(W, epsilon)
        % fast way to compute Omega using thin SVD
        %
        %
        [U, S1] = svd(W, 0);
        S1 = diag(S1);
        S1 = S1 .^ 2 ; %W'*W: eigenvalues
        S1 = S1 + epsilon; %W'*W + epsilon: eigenvalues
        S1 = sqrt(S1);    % square root of W*W'+epsilon: sqrt
        totalTrace = sum(S1) + sqrt(epsilon)*(size(W,2)-length(S1));  % trace
        S1 = S1 /totalTrace; % normalize such that D will have trace of 1
        omega = U*diag(1./S1)*U';
    end
end

