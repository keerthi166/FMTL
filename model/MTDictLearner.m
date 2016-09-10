function [W,C,F,G] = MTDictLearner( X,Y,kappa,opts)
%% Multi-task Dictionary learning
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



K=length(Y);
N=cellfun(@(x) size(x,1),X);
[~,P]=size(X{1});

loss=opts.loss;
debugMode=opts.debugMode;

% Regularization Parameters
rho_l1=0.1; %reg. param for Sparse code regularization on G
if isfield(opts,'rho_l1')
    rho_l1=opts.rho_l1;
end
rho_fr=0.1; %reg. param for Feature/Dictionary regularization F penalty
if isfield(opts,'rho_fr')
    rho_fr=opts.rho_fr;
end


% Initialize F, G
opts.mu=0;
[W,~]= STLearner(X,Y,opts);
[U,E,~] = svd(W,'econ');
[~,ind] = sort(diag(E),'descend');

if (kappa>min(P,K))
    F = [U(:,ind(1:min(P,K))) randn(P,kappa-min(P,K))];
    G = (pinv(F)*W)';
else
    F = U(:,ind(1:kappa)); % Pxkappa matrix
    G = (F'*W)'; % K*kappa matrix
end
opts.Finit=F;
opts.Ginit=G;

opts.maxIter=20;
opts.rho=rho_l1;
vecF=zeros(P*kappa,1);
for it=1:100
    XF=cellfun(@(x) x*F,X,'UniformOutput',false);
    
    G= STLearner(XF,Y,opts)';
    Gcell=mat2cell(G,ones(1,K),kappa)';
    temp=cellfun(@(x,y,g) 1*x'*y*g,X,Y,Gcell,'UniformOutput',false);
    B=sum(cat(3,temp{:}),3);
    
    [vecF,~,~,~,~]=pcg(@getAX,B(:),1e-6,5,[],[],vecF);
    F = reshape(vecF,P,kappa);
end
W=F*G';
C=zeros(1,K);

% Fast Computation of AX for solving AX=B via CG
    function vecAx=getAX(vecF)
        Fmat=reshape(vecF,P,kappa);
        temp1=cellfun(@(x,g) (x'*x)*Fmat*(g'*g)+rho_fr*Fmat,X,Gcell,'UniformOutput',false);
        matAx=sum(cat(3,temp1{:}),3);
        vecAx=matAx(:);
    end


%{
% Gradient Function
function [gF,gG]=grad(F,G)
        
        Gcell=mat2cell(G,ones(1,K),kappa)';
        Ncell=num2cell(N);
        
        % Gradient of Squared Error Loss
        temp=cellfun(@(x,y,g,n) ((x'*x)*F*(g'*g)-x'*y*g)/n,X,Y,Gcell,Ncell,'UniformOutput',false); % Kxkappa matrix cell array
        tempF=sum(cat(3,temp{:}),3);
        tempG=cell2mat(cellfun(@(x,y,g,n) (((x*F)'*x*F)*g'-(x*F)'*y)/n,X,Y,Gcell,Ncell,'UniformOutput',false)); % Kxkappa matrix
        
        gF=tempF+2*rho_fr*F;
        gG=tempG';
        
    end


% Objective Function
function F=func(F,G)
        W=F*G';
        Wcell=mat2cell(W,P,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                temp=cellfun(@(x,w,y,n) mean(max(1-y.*(x*w),0)),X,Wcell,Y,Ncell,'UniformOutput',false);
            case 'logit'
                temp=cellfun(@(x,w,y,n) mean(log(1+exp(-(x*w).*y))),X,Wcell,Y,Ncell,'UniformOutput',false);
            otherwise % Default Least Square Loss
                % Func of Squared Error Loss
                temp=cellfun(@(x,w,y,n) 0.5*norm((y - x*w))^2,X,Wcell,Y,Ncell,'UniformOutput',false);
        end
        F=sum(cell2mat(temp))+rho_fr*norm(F,'fro')^2;
    end

% Projection Function for F
function [F,ns]=projF(F,a)
        % No non-smooth term for F
        ns=0;
    end

% Projection Function for G
function [G,ns]=projG(G,a)
        
        
        % L1 penalty
        for kk=1:K
            G(kk,:) = L1GeneralProjection(funObjmtl,G(kk,:)',a*(rho_l1/2)*ones(kappa,1),[], X{kk},Y{kk},F)'; %sign(G).*max(0,abs(G)- a*rho_l1/2);
        end
        ns = sum(sum(abs(G)));
    end
%}
end

