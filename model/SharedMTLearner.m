function [W,C,gk] = SharedMTLearner( X,Y,mu,opts)
%% Shared Multi-task learning (SHAMO)
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
[~,P]=size(X{1});
kappa=opts.kappa;

loss=opts.loss;
debugMode=opts.debugMode;
maxIter=opts.maxOutIter;
epsilon=1e-8;

% Regularization Parameters
%rho_fr: reg. param for feature regularization penalty
rho_l1=0;%reg. param for l1 regularization penalty
if isfield(opts,'rho_l1')
    rho_l1=opts.rho_l1;
end


% Randomly assign tasks to kappa models
gk=randi(kappa,1,K);


%{
% Split task data into two sets to avoid bias
X1=cell(1,K);
Y1=cell(1,K);
X2=cell(1,K);
Y2=cell(1,K);

for tt=1:K
    if strcmp(loss,'hinge') || strcmp(loss,'logit')
        sel=cvpartition(Y{tt},'HoldOut',floor(N(tt)/2));
    else
        sel=cvpartition(N(tt),'HoldOut',floor(N(tt)/2));
    end
    X1{tt}=X{tt}(sel.test,:);
    Y1{tt}=Y{tt}(sel.test);
    
    X2{tt}=X{tt}(sel.training,:);
    Y2{tt}=Y{tt}(sel.training);
end
%}

obj=0;
if kappa~=1
    for it=1:maxIter
        % Step 1: Merge task data based on gk
        mX=cell(1,kappa);
        mY=cell(1,kappa);
        mN=zeros(1,kappa);
        for ii=1:kappa
            if(sum(gk==ii)<=0)
                gk(randi(K,1))=ii;
            end
            tempX=X(gk==ii);
            tempY=Y(gk==ii);
            mX{ii}=cat(1,tempX{:});
            mY{ii}=cat(1,tempY{:});
            mN(ii)=size(mY{ii},1);
            clear tempX tempY
        end
        [mW,mC] = MTSolver(mX, mY,@grad,@func,@proj,opts);
        clear mX mY
        
        
        
        % Step 2: Use [X2,Y2] to recompute gk via loss
        lossMat=computeLossMat(mW,mC); % K x kappa matrix
        [~,gk]=min(lossMat');
        emptyClusters= setdiff(1:kappa,unique(gk));
        while ~isempty(emptyClusters)
            for ii=1:length(emptyClusters)
                gk(randi(K,1))= emptyClusters(ii);
            end
            emptyClusters= setdiff(1:kappa,unique(gk));
        end
        
        
        %{
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
end
% Compute final weight matrix
% Merge task data based on gk
mX=cell(1,kappa);
mY=cell(1,kappa);
mN=zeros(1,kappa);
for ii=1:kappa
    tempX=X(gk==ii);
    tempY=Y(gk==ii);
    mX{ii}=cat(1,tempX{:});
    mY{ii}=cat(1,tempY{:});
    mN(ii)=size(mY{ii},1);
    clear tempX tempY
end
[mW,mC] = MTSolver(mX, mY,@grad,@func,@proj,opts);
clear mX mY

W=mW(:,gk);
C=mC(gk);

% Gradient Function
    function [gW,gW0]=grad(W,W0)
        Wcell=mat2cell([W;W0],P+1,ones(1,kappa));
        Ncell=num2cell(mN);
        switch (loss)
            case 'hinge'
                % Gradient of Hinge Loss
                Pr=cellfun(@(t,w,n) [getX(t),ones(n,1)]*w,num2cell(1:kappa),Wcell,Ncell,'UniformOutput',false);
                a=cellfun(@(p,y) p.*y<1,Pr,mY,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(t,y,a,n) (-[getX(t),ones(n,1)]'*diag(a)*y),num2cell(1:kappa),mY,a,Ncell,'UniformOutput',false))*diag(1./mN); % PxK matrix
                
            case 'logit'
                % Gradient of Logistic Loss
                Pr=cellfun(@(t,y,w,n) 1./(1+exp(-([getX(t),ones(n,1)]*w).*y)),num2cell(1:kappa),mY,Wcell,Ncell,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(t,y,a,n) (-[getX(t),ones(n,1)]'*diag(1-a)*y),num2cell(1:kappa),mY,Pr,Ncell,'UniformOutput',false))*diag(1./mN); % PxK matrix
                
            otherwise
                % Gradient of Squared Error Loss
                if(sum(cellfun('isempty',mX))>0)
                    disp('empty X');
                end
                temp=cell2mat(cellfun(@(t,y,w,n) (([getX(t),ones(n,1)]'*[getX(t),ones(n,1)])*w)-[getX(t),ones(n,1)]'*y,num2cell(1:kappa),mY,Wcell,Ncell,'UniformOutput',false)); % PxK matrix
            
        end
        gW=temp(1:end-1,:)+2*mu*W;
        gW0=temp(end,:);
    end

% Objective Function
    function F=func(W,W0)
        Wcell=mat2cell([W;W0],P+1,ones(1,kappa));
        Ncell=num2cell(mN);
        switch (loss)
            case 'hinge'
                temp=cellfun(@(t,w,y,n) mean(max(1-y.*([getX(t),ones(n,1)]*w),0)),num2cell(1:kappa),Wcell,mY,Ncell,'UniformOutput',false);
            case 'logit'
                temp=cellfun(@(t,w,y,n) mean(log(1+exp(-([getX(t),ones(n,1)]*w).*y))),num2cell(1:kappa),Wcell,mY,Ncell,'UniformOutput',false);
            otherwise % Default Least Square Loss
                % Func of Squared Error Loss
                temp=cellfun(@(t,w,y,n) 0.5*norm((y - [getX(t),ones(n,1)]*w))^2,num2cell(1:kappa),Wcell,mY,Ncell,'UniformOutput',false);
        end
        F=sum(cell2mat(temp))+mu*norm(W,'fro')^2;
    end

    function lossMat=computeLossMat(W,W0)
        Ncell=num2cell(cellfun(@(y) size(y,1),Y));
        switch (loss)
            case 'hinge'
                lossMat=cell2mat(cellfun(@(x,y,n) mean(max(1-repmat(y,1,kappa).*([x,ones(n,1)]*[W;W0]),0))',X,Y,Ncell,'UniformOutput',false));
            case 'logit'
                lossMat=cell2mat(cellfun(@(x,y,n) mean(log(1+exp(-([x,ones(n,1)]*[W;W0]).*repmat(y,1,kappa)))),X,Y,Ncell,'UniformOutput',false));
            otherwise % Default Least Square Loss
                % Func of Squared Error Loss

                temp=cellfun(@(x,y,n) 0.5*sum((repmat(y,1,kappa) - [x,ones(n,1)]*[W;W0]).^2),X,Y,Ncell,'UniformOutput',false)';
                lossMat=cell2mat(temp);
        end
        
    end

% Projection Function
    function [W,ns]=proj(W,a)
        % L1 penalty
        W = sign(W).*max(0,abs(W)- a*rho_l1/2);
        ns = sum(sum(abs(W)));
    end

    function Xt=getX(taskId)
        if iscell(mX)
            Xt=mX{taskId};
        else
            Xt=mX;
        end
        
    end
end

