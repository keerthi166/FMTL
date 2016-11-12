function [W,C,F,G, Sigma, Omega] = BiFactorMTLearner( X,Y,rho_fr1,rho_fr2,opts)
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
kappa=3; %Number of feature/task clusters
if isfield(opts,'kappa')
    kappa=opts.kappa;
end

% Initialize F, G
mu=0;
[W,~]= STLearner(X,Y,mu,opts);
[U,E,~] = svd(W,'econ');
[~,ind] = sort(diag(E),'descend');

if (kappa>min(P,K))
    F = [U(:,ind(1:min(P,K))) randn(P,kappa-min(P,K))];
    G = (pinv(F)*W)';
else
    F = U(:,ind(1:kappa)); % Pxkappa matrix
    G = (F'*W)'; % K*kappa matrix
end
opts.Finit=F;
opts.Ginit=G;
opts.maxIter=50;
opts.rho=rho_l1;
opts.rho_sr=rho_fr2;

Sigma=ones(P);
Omega=ones(K);

epsilon=1e-6;
obj=0;

vecF=zeros(P*kappa,1);
for it=1:100
    % Solve for G given F
    XF=cellfun(@(x) x*F,X,'UniformOutput',false);
    opts.Omega=Omega;
    G=StructMTLearner(XF,Y,rho_fr2, opts)';
    Gcell=mat2cell(G,ones(1,K),kappa)';
    
    % Solve for F, given G
    temp=cellfun(@(x,y,g) 1*x'*y*g,X,Y,Gcell,'UniformOutput',false);
    B=sum(cat(3,temp{:}),3);
    
    [vecF,~,~,~,~]=pcg(@getAX,B(:),1e-6,5,[],[],vecF);
    F = reshape(vecF,P,kappa);
    
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
    %Sigma = fastOmega(F, epsilon);
    %Omega = fastOmega(G, epsilon);
    
    
    obj=[obj;func(F,G)];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if mod(it,5)==0 && debugMode
        fprintf('Iteration %d, Objective:%f, Relative Obj:%f \n',it,obj(end),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= opts.tol)
        break;
    end
    
end
W=F*G';
C=zeros(1,K);

% Fast Computation of AX for solving AX=B via CG
    function vecAx=getAX(vecF)
        Fmat=reshape(vecF,P,kappa);
        temp1=cellfun(@(x,g) (x'*x)*Fmat*(g'*g)+rho_fr1*Sigma*Fmat,X,Gcell,'UniformOutput',false);
        matAx=sum(cat(3,temp1{:}),3);
        vecAx=matAx(:);
    end
% Compute Objective function
    function Fobj=func(F,G)
        Wcell=mat2cell(F*G',P,ones(1,K));
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
        Fobj=sum(cell2mat(temp0))+rho_fr1*trace((F*F')*Sigma) + rho_fr2*trace((G*G')*Omega);
    end


    function omega = fastOmega(W, epsilon)
        % fast way to compute Omega using thin SVD
        %
        %
        [U, S] = svd(W, 0);
        S = diag(S);
        S = S .^ 2 ; %W'*W: eigenvalues
        S = S + epsilon; %W'*W + epsilon: eigenvalues
        S = sqrt(S);    % square root of W*W'+epsilon: sqrt
        totalTrace = sum(S) + sqrt(epsilon)*(size(W,2)-length(S));  % trace
        S = S /totalTrace; % normalize such that D will have trace of 1
        omega = U*diag(1./S)*U';
    end
end

