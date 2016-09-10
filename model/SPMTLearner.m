function [W,C] = SPMTLearner( X,Y,opts)
% Self-paced Multi-task learning
% X and Y are the cell array of size K
% Uses Different steps as tau

method=opts.method;
debugMode=opts.debugMode;
mu=opts.mu;
loss=opts.loss;

K=length(Y);
N=cellfun('length',X);
[~,P]=size(X{1});

%c=1.5;

% Self-paced multi-task learning
%n=0;
%lambda=1;
%while n<K
% Solve for Omega with respect to W_A
%   temp=zeros(K);
%   temp(A,A)=1/n;
%   Omega = eye(K) - temp(A,A);
% Init W0 here
% opts.W0

% Solve for W, given Omega
[W,C] = MTLearner(X, Y,@grad,@func,@proj,opts);
%   lossvector=cell2mat(cellfun(@(x,y) mean(max(1-repmat(y,1,K).*(x*Wt),0))',X,Y,'UniformOutput',false));
%   tau=lossvector<lambda;
% Get Active Set
%   A=find(tau);
%   n=length(A);
%   lambda=lambda*c;
%end




    function [gW,gW0]=grad(W,W0)
        Wcell=mat2cell([W;W0],P+1,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                % Gradient of Hinge Loss
                Pr=cellfun(@(x,w,n) [x,ones(n,1)]*w,X,Wcell,Ncell,'UniformOutput',false);
                a=cellfun(@(p,y) p.*y<1,Pr,Y,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(x,y,a,n) (-[x,ones(n,1)]'*diag(a)*y),X,Y,a,Ncell,'UniformOutput',false)); % PxK matrix
                gW=temp(1:end-1,:)*diag(1./N)+2*mu*W;
                gW0=temp(end,:)*diag(1./N);
            case 'logit'
                %{
                gF = zeros(P, K);
                for i = 1:K
                    x=X{i}';
                    y=Y{i};
                    w=W(:,i);
                    m = length(y);
                    weight = ones(m, 1)/m;
                    weighty = weight.* y;
                    aa = -y.*(x'*w );
                    bb = max( aa, 0);
                    %funcVal = weight' * log(exp(-bb) + exp(aa-bb) + bb); %bug!!!
                    funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
                    pp = 1./ (1+exp(aa));
                    b = -weighty.*(1-pp);
                    grad_c = sum(b);
                    gF(:,i) = x * b;
                end
                gF=gF+mu*2*W;
                %}
                % Gradient of Logistic Loss
                Pr=cellfun(@(x,y,w,n) 1./(1+exp(-([x,ones(n,1)]*w).*y)),X,Y,Wcell,Ncell,'UniformOutput',false); % 1xK cell array
                temp=cell2mat(cellfun(@(x,y,a,n) (-[x,ones(n,1)]'*diag(1-a)*y),X,Y,Pr,Ncell,'UniformOutput',false)); % PxK matrix
                gW=temp(1:end-1,:)*diag(1./N)+2*mu*W;
                gW0=temp(end,:)*diag(1./N);
                
            otherwise
                % Gradient of Squared Error Loss
                temp=cell2mat(cellfun(@(x,y,w,n) (([x,ones(n,1)]'*[x,ones(n,1)])*w)-[x,ones(n,1)]'*y,X,Y,Wcell,Ncell,'UniformOutput',false)); % PxK matrix
                gW=temp(1:end-1,:)+2*mu*W;
                gW0=temp(end,:);
        end
    end
    function F=func(W,W0)
        Wcell=mat2cell([W;W0],P+1,ones(1,K));
        Ncell=num2cell(N);
        switch (loss)
            case 'hinge'
                temp=cellfun(@(x,w,y,n) mean(max(1-y.*([x,ones(n,1)]*w),0)),X,Wcell,Y,Ncell,'UniformOutput',false);
            case 'logit'
                %{
                F=0;
                for i = 1:K
                    x=X{i}';
                    y=Y{i};
                    w=W(:,i);
                    m = length(y);
                    weight = ones(m, 1)/m;
                    aa = -y.*(x'*w+0);
                    bb = max( aa, 0);
                    %funcVal = weight' * log(exp(-bb) + exp(aa-bb) + bb); % BUG!!!!
                    F = F+weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
                end
                %}
                temp=cellfun(@(x,w,y,n) mean(log(1+exp(-([x,ones(n,1)]*w).*y))),X,Wcell,Y,Ncell,'UniformOutput',false);
            otherwise % Default Least Square Loss
                % Func of Squared Error Loss
                temp=cellfun(@(x,w,y,n) 0.5*norm((y - [x,ones(n,1)]*w))^2,X,Wcell,Y,Ncell,'UniformOutput',false);
        end
        F=sum(cell2mat(temp))+mu*norm(W,'fro')^2;
        %F=F+mu*norm(W,'fro')^2;
    end



    function [W,ns]=proj(W,rho)
        ns=0;
        % For example: L1 penalty
        %W = sign(W).*max(0,abs(W)- rho/2);
        %ns = sum(sum(abs(W)));
    end
end

