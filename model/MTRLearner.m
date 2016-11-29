function [W,C, Omega] = MTRLearner( X,Y,rho_sr,opts)
%% Multi-task Relationship learning
% Solve the following objective function
%
% $$\min_{\mathbf{W},C, \mathbf{D} \in \mathcal{C}} \mathcal{L}(Y,\mathcal{F}(X,[\mathbf{W};C])) + \rho_{rl}(\mathbf{W}\Omega\mathbf{W}^\top + \epsilon I_P) +
% \rho_{l1}||\mathbf{W}||_1^1$$
%
% where $\mathcal{C}=\{\mathbf{D}|\mathbf{D} \in \mathbf{S}_{++}^P, trace{D}\leq 1\}$
% $X$ and $Y$ are the cell array of size K,
% $\mathcal{L}$ is the loss function (given by opts.loss),
% $C$ is the bias term (1xK) vector,
% $\mathbf{W}$ is the task parameter (PxK) matrix,
% $\Omega$ is the task relationship (KxK) matrix.
% $\rho_{l1},\rho_{fr}$ are the regularization parameters,
%
%
% See also <MTSolver.m MTSolver>


K=length(Y);
N=cellfun(@(x) size(x,1),X);
[~,P]=size(X{1});


loss=opts.loss;
debugMode=opts.debugMode;
maxIter=opts.maxOutIter;

Omega=eye(K)/(K);
epsilon=1e-8;

% Regularization Parameters
rho_l1=0;
%rho_sr=0.1; %reg. param for feature regularization penalty
%if isfield(opts,'rho_sr')
%    rho_sr=opts.rho_sr;
%end
obj=0;
for it=1:maxIter
    
    % Solve for W given D
    %[W,C] = MTSolver(X, Y,@grad,@func,@proj,opts);
    opts.Omega=Omega;
    [W,C] = StructMTLearner(X, Y, rho_sr, opts);
    opts.Winit=W;
    opts.W0init=C;
    % Solve for D, given W
    [U,S] = eig(W'*W+epsilon*eye(K));
    Smin=sqrt(abs(diag(S)));
    Smin=Smin/sum(Smin);
    Omega = U * diag(1./(Smin)) * U';
    
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
        gW=temp(1:end-1,:)+2*rho_sr*W*Omega;
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
        F=sum(cell2mat(temp))+rho_sr*trace(W*Omega*W');
    end


% Projection Function
    function [W,ns]=proj(W,a)
        % L1 penalty
        W = sign(W).*max(0,abs(W)- a*rho_l1/2);
        ns = sum(sum(abs(W)));
    end

end

