function [W,C,Omega] = MTByClusteringLearner( X,Y,rho_fr,opts)
%% Multi-task learning byu Clustering based on Barzilai's Implementation
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
maxIter=opts.maxOutIter;

Omega=eye(K)/K;


% Construct linear kernel
data=[];
label=[];
task_index=[];
for tt=1:K
    data=[data;X{tt}]; %NxP matrix
    label=[label;Y{tt}]; % 1XN vector
    task_index=[task_index,tt*ones(1,size(X{tt},1))]; % 1XN vector
end
n=size(data,1);
insIndex=cell(1,K);
ins_indicator=zeros(K,n);
for tt=1:K
    insIndex{tt}=sort(find(task_index==tt));
    ins_indicator(tt,insIndex{tt})=1;
end
Km=data*data';


epsilon=1e-8;
stepSize=10;
kappa=0.1;
obj=0;

for it=1:maxIter

    % Solve for alpha given Omega
    MTKm=Km.*Omega(task_index,task_index);
    %alpha = (MTKm+rho_fr*eye(n))\label;
    alpha=pcg((MTKm+rho_fr*eye(n)),label);
    %model=MTRL_RR(MTKm,label,task_index,insIndex,N); % Compute alpha and b
    %alpha=model.alpha';
    fobj = alpha'*label-0.5*rho_fr*(alpha'*alpha)-0.5*alpha'*MTKm*alpha ;
    clear MTKm
    
    
    % Compute Gradient
    B=ins_indicator*diag(alpha);
    gradOmega=kappa*ones(K)-0.5*B*Km*B';
    tOmega=Omega-stepSize*gradOmega/sqrt(it);
    % Compute Omega by Gradient Projection
    Omega = gradProject(tOmega, epsilon);
    
    
    %{
    %Calculate Omega
        temp=Omega(:,task_index)*diag(alpha);
        temp=temp*Km*temp';
        [eigVector,eigValue]=eig(temp+epsilon*eye(K));
        clear temp;
        eigValue=sqrt(abs(diag(eigValue)));
        eigValue=eigValue/sum(eigValue);
        Omega=eigVector*diag(eigValue)*eigVector';
        %m_Cor=eigVector*diag(eigValue./(lambda1*eigValue+lambda2))*eigVector';
        clear eigVector eigValue;
    %}
    obj=[obj;fobj];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if mod(it,1)==0 && debugMode
        fprintf('Iteration %d, Objective:%f, Relative Obj:%f \n',it,obj(end),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= opts.tol)
        break;
    end
    
end
W=zeros(P,K);
C=zeros(1,K);
%C=model.b;

for i=1:n
   W=W+alpha(i)*data(i,:)'*Omega(task_index(i),:); 
end

%{
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
%}

    function omega = gradProject(P, epsilon)
        % Compute Omega using Gradient Projection
        % P computed from the gradient descent
        % Gradient Projection
        for in_it=1:opts.maxIter
            oldP=P;
            [eigVector,eigValue]=eig(P);
            R=eigVector*diag(max(0,diag(eigValue)/sum(diag(eigValue))))*eigVector';
            Q=R;
            Q=min(1,max(0,Q));
            Q(logical(eye(size(Q)))) = 1;
            P=P-(R-Q);
            if(norm(P-oldP,'fro'))<=epsilon*K*K
                break;
            end
        end
        omega = P;
        omega(omega<epsilon)=0;
    end
end

