function [W,C] = MTSolverNoBiasTerm(X,Y,grad,func,proj,opts)
% Multi-task Learner with accelerated gradient descent
% X and Y are the cell array of size K

% @param: proj(Z,lambda) -projection operator for non-smooth function
% @param: grad(W) -gives gradient of the objective function at W
% @param: func(W) -gives function value of the objective function at W (without nonsmooth term)
%


debugMode=opts.debugMode;
K=length(Y);
N=cellfun('length',X);
[~,P]=size(X{1});

% Initialization
%stepsize=0.5;

maxIter=100;
if isfield(opts,'maxIter')
    maxIter=opts.maxIter;
end
% Initialize W and C here (warm start)
if isfield(opts,'W0')
    W=opts.W0;
else
    W=zeros(P,K);
end
if isfield(opts,'C0')
    C=opts.C0;
else
    C=zeros(1,K);
end
Z=W; % Init Z


alpha=1;
gamma=1;
inc=2;

nGFlag=0;
obj=0;



for it=1:maxIter

    W_old=W;
    % Compute gradient and function value at Z
    gF=grad(Z);
    Fz=func(Z);
    
    % Step 1: Determine Step Size gammma
    while true
        % Step 2: Projection Step
        % solve 2/gamma (min_W gamma/2 ||W-Z|| + lambda* nsReg(W))
        [W,ns]= proj(Z-gF/gamma,2/gamma);
        F=func(W);
        
        delta=W-Z;
        if norm(delta,'fro')^2<1e-20
            nGFlag=1;
            break;
        end
        % Update step size according to the function values
        if F <= (Fz + sum(sum(delta.*gF)) + gamma/2*sum(sum(delta.*delta)))
            break;
        else
            gamma=gamma*inc;
        end
        
    end
    
    if nGFlag
        break;
    end
    % Compute alpha
    alpha_old=alpha;
    alpha= 0.5 * (1 + (1+ 4 * alpha^2)^0.5);
    % Compute Z
    Z=W+((alpha_old-1)/alpha)*(W-W_old);
 
    %{
        % Simple Gradient Descent
        gF=grad(W);
        W = W-(stepsize)*gF;
        F=func(W);
        ns=0;
        if relObj > 0 && (method==0)
        stepsize = stepsize*0.5;
        end
    %}
    
    obj=[obj;F+ns];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if mod(it,50)==0 && debugMode
        fprintf('InIteration %d, Objective:%f, Relative Obj:%f \n',it,obj(end),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= opts.tol)
        break;
    end

end

end

