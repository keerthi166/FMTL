function [F,G] = BiFactorMTSolver(X,Y,kappa,grad,func,projU,projV,opts)
%% Bifactorized Multi-task Solver with accelerated gradient descent
% for W=FG' factorization
% X and Y are the cell array of size K

% @param: projU(U,lambda1) -projection operator for U non-smooth function
% @param: projV(V,lambda2) -projection operator for V non-smooth function
% @param: grad(F,G) -gives gradient of the objective function at F,G
% @param: func(F,G) -gives function value of the objective function at F,G (without nonsmooth term)
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
% Initialize F,G and F0 here (warm start)
if isfield(opts,'Finit')
    F=opts.Finit;
else
    F=zeros(P,kappa);
end
if isfield(opts,'Ginit')
    G=opts.Ginit;
else
    G=zeros(K,kappa);
end

U=F; % Init U
V=G; % Init V

alpha=1;
gamma=1;
inc=2;

nGFlag=0;
obj=0;

for it=1:maxIter
    
    F_old=F;
    G_old=G;
    
    % Compute gradient and function value at Z
    [gF,gG]=grad(U,V);
    Fz=func(U,V);
    
    % Step 1: Determine Step Size gammma
    while true
        % Step 2: Projection Step
        % solve 2/gamma (min_F gamma/2 ||F-U|| + lambda1* nsReg(F))
        % solve 2/gamma (min_G gamma/2 ||G-V|| + lambda2* nsReg(G))
        
        [F,ns1]= projU(U-gF/gamma,2/gamma);
        [G,ns2]= projV(V-gG/gamma,2/gamma);
        
        Fobj=func(F,G);
        

        deltaF=F-U;
        deltaG=G-V;
        if norm(deltaF,'fro')+norm(deltaG,'fro')^2<1e-20
            nGFlag=1;
            break;
        end
        % Update step size according to the function values
        if Fobj <= (Fz + sum(sum(deltaF.*gF)) + sum(sum(deltaG.*gG)) ...
                 + gamma/2*sum(sum(deltaF.*deltaF)) + gamma/2*sum(sum(deltaG.*deltaG)))
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
    
    % Compute U,V,U0
    U=F+((alpha_old-1)/alpha)*(F-F_old);
    V=G+((alpha_old-1)/alpha)*(G-G_old);
    
    %{
        % Simple Gradient Descent
        [gF,gG]=grad(F,G);
        
        F = F-(stepsize)*gF;
        G = G-(stepsize)*gG;
    
        Fobj=func(F,G);
        ns=0;
        if relObj > 0 
        stepsize = stepsize*0.5;
        end
    %}
    
    obj=[obj;Fobj+ns1+ns2];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if mod(it,500)==0 && debugMode
        fprintf('Iteration %d, Objective:%f, Relative Obj:%f \n',it,obj(end),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= opts.tol)
        break;
    end
    
end

end

