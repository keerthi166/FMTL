function [W,C,Sigma, Omega] = MatrixNormalMTLearner( X,Y,rho_fr,opts)
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


learnOmega=true;
if isfield(opts,'learnOmega')
    learnOmega=opts.learnOmega;
end

learnSigma=true;
if isfield(opts,'learnSigma')
    learnSigma=opts.learnSigma;
end

opts.rho=rho_l1;
maxIter=opts.maxOutIter;

Sigma=eye(P)/P;
Omega=eye(K)/K;
iSigma=zeros(P);
iOmega=zeros(K);


obj=0;

vecW=zeros(P*K,1);
for it=1:maxIter
    
    
    temp=cellfun(@(x,y) x'*y,X,Y,'UniformOutput',false);
    RHS=cat(1,temp{:});
    %temp=cellfun(@(x) (x'*x),X,'UniformOutput',false);
    %LHS=blkdiag(temp{:})+rho_fr*kron(Sigma,Omega);
    %clear temp
    [vecW,~,~,~,~]=pcg(@getAX,RHS,[],[],[],[],vecW);
    clear LHS RHS
    W = reshape(vecW,P,K);
    
    
    if learnSigma
        [Sigma,~] = graphicalLasso((W*Omega*W')/K,rho_l1, 5, 1e-4);
        Sigma=real(Sigma);
    %[iSigma,Sigma]=GraphicalLasso((W*Omega*W')/K,rho_l1,[],0,1,1,1,1e-6,1000,iSigma,Sigma);
    end
    if learnOmega
        [Omega,~] = graphicalLasso((W'*Sigma*W)/P,rho_l1, 5, 1e-4);
        Omega=real(Omega);
    %[iOmega,Omega] = GraphicalLasso((W'*Sigma*W)/P,rho_l1,[],0,1,1,1,1e-6,1000,iOmega,Omega);
    end
    
    obj=[obj;func(W)];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if mod(it,1)==0 && debugMode
        fprintf('Iteration %d, Objective:%f, Relative Obj:%f \n',it,obj(end),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= opts.tol)
        break;
    end
    
end
C=zeros(1,K);

function vecAx=getAX(vecW)
         Wmat=reshape(vecW,P,K);
         temp=cellfun(@(x) (x'*x),X,'UniformOutput',false);
         vecLHS=blkdiag(temp{:})*vecW;
         clear temp
         LHS=reshape(vecLHS,P,K)+rho_fr*Sigma*Wmat*Omega;
         vecAx=LHS(:);
end

% Compute Objective function
    function Fobj=func(W)
        Wcell=mat2cell(W,P,ones(1,K));
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
        Fobj=sum(cell2mat(temp0))+0.5*rho_fr*trace((W*Omega*W')*Sigma);
    end


    function omega = fastOmega(W, epsilon)
        % fast way to compute inverse of Omega using thin SVD
        %
        %
        [U, S] = svd(W, 0);
        S = diag(S);
        S = S .^ 2 ; %W'*W: eigenvalues
        S = S + epsilon; %W'*W + epsilon: eigenvalues
        S = sqrt(S);    % square root of W*W'+epsilon: sqrt
        totalTrace = sum(S) + sqrt(epsilon)*(size(W,2)-length(S));  % trace
        S = S /totalTrace; % normalize such that D will have trace of 1
        omega = U*diag(1./S)*U'; % Compute inverse
    end
end

