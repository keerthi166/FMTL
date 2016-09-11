function [W,C] = MMTLearner(X,Y,rho_sr,opts)
%% Mean Multi-task learner
% Solve the following objective function
%
% $$\min_{\mathbf{W},C} \mathcal{L}(Y,\mathcal{F}(X,[\mathbf{W};C])) + \rho_{sr}\mathbf{W}\Omega\mathbf{W}^\top+ \mu||\mathbf{W}||_2^2 +
% \rho_{l1}||\mathbf{W}||_1^1$$
%
% where $X$ and $Y$ are the cell array of size K,
% $\mathcal{L}$ is the loss function (given by opts.loss),
% $C$ is the bias term (1xK) vector,
% $\mathbf{W}$ is the task parameter (PxK) matrix,
% $\Omega$ is the task structure parameter (KxK) matrix.
% $\mu,\rho_{l1},\rho_{sr}$ are the regularization parameters,
%
%
% See also <MTSolver.m MTSolver>

debugMode=opts.debugMode;



% Regularization Parameters
%rho_sr :reg. param for structure regularization penalty
mu=0; % reg. param for l2-norm penalty
if isfield(opts,'mu')
    mu=opts.mu;
end
rho_l1=0; % reg. param for l1-norm penalty
if isfield(opts,'rho')
    rho_l1=opts.rho_l1;
end


loss=opts.loss;
debugMode=opts.debugMode;
maxIter=25;


K=length(Y);
N=cellfun(@(y) size(y,1),Y);
[~,P]=size(getX(1));

Wm=zeros(P,1);

for it=1:maxIter
    % Solve for W
    [W,C] = MTSolver(X, Y,@grad,@func,@proj,opts);
    Wm=mean(W,2);
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
        gW=temp(1:end-1,:)+2*mu*W + 2*rho_sr*(W-repmat(Wm,1,K));
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
        taskF=cell2mat(cellfun(@(lt,w) lt+rho_sr*((w(1:end-1)-Wm)'*(w(1:end-1)-Wm)),temp,Wcell,'UniformOutput',false));
        F=sum(cell2mat(temp))+mu*norm(W,'fro')^2 + rho_sr*trace((W-repmat(Wm,1,K))'*(W-repmat(Wm,1,K)));
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

