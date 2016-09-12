function [W,C] = SPMTMLearner( X,Y,rho_fr,lambda,opts)
%% Self-based Manifold-based Multi-task learning
% Solve the following objective function
%
% $$\min_{\mathbf{W},C, \mathbf{D} \in \mathcal{C}} \mathcal{L}(Y,\mathcal{F}(X,[\mathbf{W};C])) + \rho_{fr}\mathbf{D}^{-1}(\mathbf{W}\mathbf{W}^\top + \epsilon I_P) +
% \rho_{l1}||\mathbf{W}||_1^1$$
%
% where $\mathcal{C}=\{\mathbf{D}|\mathbf{D} \in \mathbf{S}_{++}^P, trace{D}\leq 1\}$
% $X$ and $Y$ are the cell array of size K,
% $\mathcal{L}$ is the loss function (given by opts.loss),
% $C$ is the bias term (1xK) vector,
% $\mathbf{W}$ is the task parameter (PxK) matrix,
% $D$ is the task feature subspace (PxP) matrix.
% $\rho_{l1},\rho_{fr}$ are the regularization parameters,
%
%
% See also <MTSolver.m MTSolver>


K=length(Y);
N=cellfun(@(y) size(y,1),Y);
[~,P]=size(getX(1));

% Manifold settings
nIter=10; % Number of iterations used for learning the manifold
knn=10; % K-nearest neighbor for manifolds
if knn>=K
    knn=K-1;
end

loss=opts.loss;
debugMode=opts.debugMode;
maxIter=5;


% Regularization Parameters
%rho_fr: reg. param for feature regularization penalty
rho_l1=0;%reg. param for l1 regularization penalty
if isfield(opts,'rho_l1')
    rho_l1=opts.rho_l1;
end

rId=sprintf('%s_%s_%d',opts.dataset,opts.method,opts.rId);
baseLoc='/usr0/home/kmuruges/Research/repos/Code/workspace/FMTL/';
runScriptPath=sprintf('%s/lib/cems/R',baseLoc);
paramFilePath=sprintf('%s/results/cems',baseLoc);

obj=0;
Wm=zeros(P,K);
selftype='sparse';
stepSize=lambda;
for it=1:maxIter
    
    % Solve for W given D
    [W,C] = MTSolver(X, Y,@grad,@func,@proj,opts);
    opts.Winit=W;
    opts.W0init=C;
    
    
    % Solve for manifold, given W
    [~,taskF]=func(W,C);
    if strcmp(loss,'least')
        taskF=taskF./N;
    end
    
    if strcmp(selftype,'sparse')
        [~,sortObsIdx]=sort(taskF);
        selTasks=sortObsIdx(1:min(K,lambda));
        tau=ones(1,K)*0;
        tau(selTasks)=1;
        lambda=lambda+stepSize;
    elseif strcmp(selftype,'weight')
        tau=max((lambda*ones(1,K)-taskF),0.01);
        lambda=lambda*1.1;
    elseif strcmp(selftype,'prob')
        tau=exp(-taskF/lambda);
        lambda=lambda*1.1;
    end
    if (sum(tau)==0)
        tau=ones(1,K)*0.1;
    end
    %tau=tau./sum(tau);
    
    
    Wsub=W(:,logical(tau));
    [Wm,result]=computeManifoldProjections(Wsub,W,knn,nIter,runScriptPath,paramFilePath,rId);
    if isempty(Wm)
        result
    end
    
    
    obj=[obj;func(W,C)];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if debugMode
        fprintf('OutIteration %d, Objective:%f, Relative Obj:%f \n',it,func(W,C),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= opts.tol)
        break;
    end
end


% Gradient Function
    function [gW,gW0]=grad(W,W0)
        Wcell=mat2cell([W;W0],P+1,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                % Gradient of Hinge Loss
                Pr=cellfun(@(t,w,n) [getX(t),ones(n,1)]*w,num2cell(1:K),Wcell,Ncell,'UniformOutput',false);
                a=cellfun(@(p,y) p.*y<1,Pr,Y,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(t,y,a,n) (-[getX(t),ones(n,1)]'*diag(a)*y),num2cell(1:K),Y,a,Ncell,'UniformOutput',false))*diag(1./N); % PxK matrix
                
            case 'logit'
                % Gradient of Logistic Loss
                Pr=cellfun(@(t,y,w,n) 1./(1+exp(-([getX(t),ones(n,1)]*w).*y)),num2cell(1:K),Y,Wcell,Ncell,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(t,y,a,n) (-[getX(t),ones(n,1)]'*diag(1-a)*y),num2cell(1:K),Y,Pr,Ncell,'UniformOutput',false))*diag(1./N); % PxK matrix
                
            otherwise
                % Gradient of Squared Error Loss
                temp=cell2mat(cellfun(@(t,y,w,n) (([getX(t),ones(n,1)]'*[getX(t),ones(n,1)])*w)-[getX(t),ones(n,1)]'*y,num2cell(1:K),Y,Wcell,Ncell,'UniformOutput',false)); % PxK matrix
        end
        gW=temp(1:end-1,:)+2*rho_fr*(W-Wm);
        gW0=temp(end,:);
    end

% Objective Function
    function [F,taskF]=func(W,W0)
        Wcell=mat2cell([W;W0],P+1,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                temp=cellfun(@(t,w,y,n) mean(max(1-y.*([getX(t),ones(n,1)]*w),0)),num2cell(1:K),Wcell,Y,Ncell,'UniformOutput',false);
            case 'logit'
                temp=cellfun(@(t,w,y,n) mean(log(1+exp(-([getX(t),ones(n,1)]*w).*y))),num2cell(1:K),Wcell,Y,Ncell,'UniformOutput',false);
            otherwise % Default Least Square Loss
                % Func of Squared Error Loss
                temp=cellfun(@(t,w,y,n) 0.5*norm((y - [getX(t),ones(n,1)]*w))^2,num2cell(1:K),Wcell,Y,Ncell,'UniformOutput',false);
        end
        taskF=cell2mat(cellfun(@(lt,t) lt+rho_fr*((W(:,t)-Wm(:,t))'*(W(:,t)-Wm(:,t))),temp,num2cell(1:K),'UniformOutput',false));
        F=sum(cell2mat(temp))+rho_fr*trace((W-Wm)'*(W-Wm));
    end


% Projection Function
    function [W,ns]=proj(W,a)
        % L1 penalty
        W = sign(W).*max(0,abs(W)- a*rho_l1/2);
        ns = sum(sum(abs(W)));
    end

    function Xt=getX(taskId)
        if iscell(X)
            Xt=X{taskId};
        else
            Xt=X;
        end
        
    end
end

