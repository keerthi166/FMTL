function [W,C] = StructMTLearner(X,Y,Omega, opts)
%% Structured Multi-task learner
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
mu=opts.mu; % reg. param for squared l2-norm penalty
rho_l1=0; % reg. param for l1-norm penalty
if isfield(opts,'rho')
    rho_l1=opts.rho_l1;
end
rho_sr=0.1; % reg. param for structure regularization penalty
if isfield(opts,'rho_sr')
    rho_sr=opts.rho_sr;
end

loss=opts.loss;

K=length(Y);
N=cellfun(@(x) size(x,1),X);
[~,P]=size(X{1});

% Solve for W
[W,C] = MTSolver(X, Y,@grad,@func,@proj,opts);


% Gradient Function
    function [gW,gW0]=grad(W,W0)
        Wcell=mat2cell([W;W0],P+1,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                % Gradient of Hinge Loss
                Pr=cellfun(@(x,w,n) [x,ones(n,1)]*w,X,Wcell,Ncell,'UniformOutput',false);
                a=cellfun(@(p,y) p.*y<1,Pr,Y,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(x,y,a,n) (-[x,ones(n,1)]'*diag(a)*y),X,Y,a,Ncell,'UniformOutput',false))*diag(1./N); % PxK matrix
                
            case 'logit'
                % Gradient of Logistic Loss
                Pr=cellfun(@(x,y,w,n) 1./(1+exp(-([x,ones(n,1)]*w).*y)),X,Y,Wcell,Ncell,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(x,y,a,n) (-[x,ones(n,1)]'*diag(1-a)*y),X,Y,Pr,Ncell,'UniformOutput',false))*diag(1./N); % PxK matrix
                
            otherwise
                % Gradient of Squared Error Loss
                temp=cell2mat(cellfun(@(x,y,w,n) (([x,ones(n,1)]'*[x,ones(n,1)])*w)-[x,ones(n,1)]'*y,X,Y,Wcell,Ncell,'UniformOutput',false)); % PxK matrix
        end
        gW=temp(1:end-1,:)+2*mu*W + 2*rho_sr*W*Omega;
        gW0=temp(end,:);
    end

% Objective Function
    function F=func(W,W0)
        Wcell=mat2cell([W;W0],P+1,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                temp=cellfun(@(x,w,y,n) mean(max(1-y.*([x,ones(n,1)]*w),0)),X,Wcell,Y,Ncell,'UniformOutput',false);
            case 'logit'
                temp=cellfun(@(x,w,y,n) mean(log(1+exp(-([x,ones(n,1)]*w).*y))),X,Wcell,Y,Ncell,'UniformOutput',false);
            otherwise % Default Least Square Loss
                % Func of Squared Error Loss
                temp=cellfun(@(x,w,y,n) 0.5*norm((y - [x,ones(n,1)]*w))^2,X,Wcell,Y,Ncell,'UniformOutput',false);
        end
        F=sum(cell2mat(temp))+mu*norm(W,'fro')^2 + rho_sr*trace(W*Omega*W');
    end


% Projection Function
    function [W,ns]=proj(W,a)
        % L1 penalty
        W = sign(W).*max(0,abs(W)- a*rho_l1/2);
        ns = sum(sum(abs(W)));
    end
end

