function [W,C] = LifelongLearner( X,Y,rho_fr,opts)
%% Active Lifelong Learner for multi-task learning
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
for t = 1 : K
    X{t}(:,end+1) = 1;
end



N=cellfun(@(y) size(y,1),Y);
[~,P]=size(getX(1));
C=zeros(K,1);
h=opts.h;

loss=opts.loss;
debugMode=opts.debugMode;
activeSelType=opts.activeSelType;

% Regularization Parameters
%rho_fr: reg. param for feature regularization penalty
rho_l1=0.1;%reg. param for l1 regularization penalty
if isfield(opts,'rho_l1')
    rho_l1=opts.rho_l1;
end

useLogistic=true;
if strcmp(loss,'least')
    useLogistic=false;
end
% Initialize ELLA
model = initModelELLA(struct('k',h,...
    'd',P,...
    'mu',rho_l1,...
    'lambda',rho_fr,...
    'ridgeTerm',1e-1,...
    'initializeWithFirstKTasks',true,...
    'useLogistic',useLogistic,...
    'lastFeatureIsABiasTerm',true));
%{
model = initModelELLA(struct('k',2,...
    'd',P,...
    'mu',exp(-12),...
    'lambda',exp(-10),...
    'ridgeTerm',exp(-5),...
    'initializeWithFirstKTasks',true,...
    'useLogistic',useLogistic,...
    'lastFeatureIsABiasTerm',true));
%}
learned = logical(zeros(length(Y),1));
unlearned = find(~learned);
for it=1:K
    % change the last input to 1 for random, 2 for InfoMax, 3 for Diversity, 4 for Diversity++
    idx = selectTaskELLA(model,X(unlearned),Y(unlearned),activeSelType);
    model = addTaskELLA(model,X{unlearned(idx)},Y{unlearned(idx)},unlearned(idx));
    learned(unlearned(idx)) = true;
    unlearned = find(~learned);
    for tprime = 1 : length(unlearned)
        model = addTaskELLA(model,X{unlearned(tprime)},Y{unlearned(tprime)},unlearned(tprime),true);
    end
    %{
    % Solve for W given D
    [W,C] = MTSolver(X, Y,@grad,@func,@proj,opts);
    opts.Winit=W;
    opts.W0init=C;
    % Solve for D, given W
    [U,S] = eig(W*W'+epsilon*eye(P));
    Smin=sqrt(abs(diag(S)));
    Smin=Smin/sum(Smin);
    invD = U * diag(1./(Smin)) * U';
    
    
    obj=[obj;func(W,C)];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if debugMode
        fprintf('OutIteration %d, Objective:%f, Relative Obj:%f \n',it,func(W,C),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= opts.tol)
        break;
    end
    %}
end
W=zeros(d-1,T);
C=zeros(1,T);
for tt=1:T
temp = model.L*model.S(:,tt);
W(:,tt)=temp(1:d-1,1);
C(tt)=temp(d,1);
end

    function Xt=getX(taskId)
        if iscell(X)
            Xt=X{taskId};
        else
            Xt=X;
        end
        
    end
end

