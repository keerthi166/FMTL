function [U,C, theta] = MTASOLearner( X,Y,rho_fr,opts)
%% Multi-task learning with Alternating Structure Optimization
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
H=opts.h;
[~,P]=size(getX(1));


loss=opts.loss;
debugMode=opts.debugMode;
maxIter=opts.maxOutIter;



% Regularization Parameters
%rho_fr: reg. param for feature regularization penalty
rho_l1=0;%reg. param for l1 regularization penalty
if isfield(opts,'rho_l1')
    rho_l1=opts.rho_l1;
end


obj=0;
U=zeros(P,K);
theta=randn(H,P);

for it=1:maxIter
    
    % Solve for V, given Theta and U
    V=theta*U;
    
    % Solve for U given theta and V
    projU=theta'*V;
    [U,C] = MTSolver(X, Y,@grad,@func,@proj,opts);
    opts.Winit=U;
    opts.W0init=C;
    
    % Solve for Theta, given U
    U=sqrt(rho_fr)*U; 
    
    % Compute SVD of U
    [V1,D,V2] = svd(U,'econ');
    [vals,ind] = sort(diag(D),'descend');
    if (H>min(P,K))
        theta = [V1(:,ind(1:min(P,K))) randn(P,H-min(P,K))]';
    else
        theta = V1(:,ind(1:H))';
    end
    
    
    obj=[obj;func(U,C)];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if debugMode
        fprintf('OutIteration %d, Objective:%f, Relative Obj:%f \n',it,func(U,C),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= opts.tol)
        break;
    end
end


% Gradient Function
    function [gW,gW0]=grad(U,U0)
        Ucell=mat2cell([U;U0],P+1,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                % Gradient of Hinge Loss
                Pr=cellfun(@(t,w,n) [getX(t),ones(n,1)]*w,num2cell(1:K),Ucell,Ncell,'UniformOutput',false);
                a=cellfun(@(p,y) p.*y<1,Pr,Y,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(t,y,a,n) (-[getX(t),ones(n,1)]'*diag(a)*y),num2cell(1:K),Y,a,Ncell,'UniformOutput',false))*diag(1./N); % PxK matrix
                
            case 'logit'
                % Gradient of Logistic Loss
                Pr=cellfun(@(t,y,w,n) 1./(1+exp(-([getX(t),ones(n,1)]*w).*y)),num2cell(1:K),Y,Ucell,Ncell,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(t,y,a,n) (-[getX(t),ones(n,1)]'*diag(1-a)*y),num2cell(1:K),Y,Pr,Ncell,'UniformOutput',false))*diag(1./N); % PxK matrix
                
            otherwise
                % Gradient of Squared Error Loss
                temp=cell2mat(cellfun(@(t,y,w,n) (([getX(t),ones(n,1)]'*[getX(t),ones(n,1)])*w)-[getX(t),ones(n,1)]'*y,num2cell(1:K),Y,Ucell,Ncell,'UniformOutput',false)); % PxK matrix
        end
        gW=temp(1:end-1,:)+2*rho_fr*(U-projU);
        gW0=temp(end,:);
    end

% Objective Function
    function [F,taskF]=func(U,U0)
        Ucell=mat2cell([U;U0],P+1,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                temp=cellfun(@(t,w,y,n) mean(max(1-y.*([getX(t),ones(n,1)]*w),0)),num2cell(1:K),Ucell,Y,Ncell,'UniformOutput',false);
            case 'logit'
                temp=cellfun(@(t,w,y,n) mean(log(1+exp(-([getX(t),ones(n,1)]*w).*y))),num2cell(1:K),Ucell,Y,Ncell,'UniformOutput',false);
            otherwise % Default Least Square Loss
                % Func of Squared Error Loss
                temp=cellfun(@(t,w,y,n) 0.5*norm((y - [getX(t),ones(n,1)]*w))^2,num2cell(1:K),Ucell,Y,Ncell,'UniformOutput',false);
        end
        taskF=cellfun(@(lt,t) lt+rho_fr*(U(:,t)-projU(:,t)),temp,num2cell(1:K),'UniformOutput',false);
        F=sum(cell2mat(temp))+rho_fr*trace((U-projU)'*(U-projU));
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

