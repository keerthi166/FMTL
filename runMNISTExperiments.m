

clear;
rng('default');

% Read Landmine data
dataset='mnist_1vs1_last';
load('data/mnist/split1_mnist_1000train.mat')
X=[digit_trainx;digit_testx];
Y=[digit_trainy;digit_testy];

classes=unique(Y);
nclass= length(classes);
K=(nclass-1)*nclass/2;

% Add intercept
%X = cellfun(@(x) [ones(size(x,1),1),x], X, 'uniformOutput', false);
% Change the labelspace from {0,1} to {-1,1}


Nrun=1;
% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'STL','MTFL'};%{'STL','MMTL','SPMMTL','MTFL','SPMTFL','MTML','SPMTML','MTASO','SPMTASO'}; % Choose subset: {'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'};
trainSizes=1000;

opts.dataset=dataset;
opts.loss='logit'; % Choose one: 'logit', 'least', 'hinge'
opts.debugMode=true;
opts.verbose=true;
opts.isHigherBetter=true;
opts.tol=1e-8;
opts.maxIter=100; % max iter for Accelerated Grad
opts.maxOutIter=50; % max iter for alternating optimization
opts.cv=false;

cv=[];

% Initilaization
result=cell(length(models),1);
for m=1:length(models)
    result{m}.score=zeros(Nrun,length(trainSizes));
    result{m}.taskScore=zeros(nclass,Nrun,length(trainSizes));
    if strncmpi(models{m},'SPM',3)
        result{m}.tau=cell(1,Nrun);
    end
    result{m}.runtime=zeros(1,length(trainSizes));
end



for nt=1:length(trainSizes)
    trainSize=trainSizes(nt);
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run Experiment
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Run Id - For Repeated Experiment
    fprintf('Train Size %d\n',trainSize);
    for rId=1:Nrun
        opts.rId=rId;
        if opts.verbose
            fprintf('Run %d (',rId);
        end
        %------------------------------------------------------------------------
        %                   Train-Test Split
        %------------------------------------------------------------------------
        
        
        % Split Data into train and test
        split = cvpartition(Y,'HoldOut',size(Y,1)-trainSize);
        Xrawtrain=X(split.training,:);
        Yrawtrain=Y(split.training);
        
        % Training set
        % One-vs-One setting
        Xtrain=cell(1,K);
        Ytrain=cell(1,K);
        tt=1;
        for class1=1:length(classes)-1
            cobs1=Yrawtrain==class1;
            for class2=class1+1:length(classes)
                cobs2=Yrawtrain==class2;
                Xtrain{tt}=[Xrawtrain(cobs1,:);Xrawtrain(cobs2,:)];
                Ytrain{tt}=[ones(sum(cobs1),1);-1*ones(sum(cobs2),1)];
                
                tt=tt+1;
            end
        end
        
        
        % Normalize Data if needed
        %[Xtrain,~,meanX,stdX] = normalizeMultitaskData(Xtrain);
        % Normalize Test Data
        %[Xtest,~,~,~] = normalizeMultitaskData(Xtest,[],meanX,stdX);
        
        
        %------------------------------------------------------------------------
        %                   Cross Validation
        %------------------------------------------------------------------------
        
        
        %load(sprintf('cv/%s_cv_%0.2f_%d.mat',dataset,trainSize,1));
        if opts.cv && isempty(cv)
            if opts.verbose
                fprintf('CV');
            end
            opts.method='cv';
            opts.h=5;
            param_range=[1e-3,1e-2,1e-1,1e-0,1e+1,1e+2,1e+3];
            lambda_range=[1e-2,0.1,0.2,1,2,5,10];
            lambdaselect_range=[5,10,15,20,25,30,35];
            
            cv.stl.perform_mat = zeros(length(param_range),1);
            cv.mmtl.perform_mat = zeros(length(param_range),1);
            cv.mtfl.perform_mat = zeros(length(param_range),1);
            cv.mtml.perform_mat = zeros(length(param_range),1);
            cv.mtaso.perform_mat = zeros(length(param_range),1);
            
            cv.spmmtl.perform_mat = zeros(length(param_range),length(lambda_range));
            cv.spmtfl.perform_mat = zeros(length(param_range),length(lambda_range));
            cv.spmtml.perform_mat = zeros(length(param_range),length(lambda_range));
            cv.spmtaso.perform_mat = zeros(length(param_range),length(lambda_range));
            
            cv_split = cvpartition(Yrawtrain,'KFold',kFold);
            if opts.verbose
                fprintf('[');
            end
            for cv_idx=1:kFold
                if opts.verbose
                    fprintf('-');
                end
                te_idx = test(cv_split,cv_idx);
                tr_idx = training(cv_split,cv_idx);
                cv_Xrawtrain=Xrawtrain(tr_idx,:);
                cv_Yrawtrain=Yrawtrain(tr_idx);
                cv_Xvalid=Xrawtrain(te_idx,:);
                cv_Yvalid=Yrawtrain(te_idx);
                
                cv_Xtrain=cell(1,K);
                cv_Ytrain=cell(1,K);
                tt=1;
                for class1=1:length(classes)-1
                    cobs1=cv_Yrawtrain==class1;
                    for class2=class1+1:length(classes)
                        cobs2=cv_Yrawtrain==class2;
                        cv_Xtrain{tt}=[cv_Xrawtrain(cobs1,:);cv_Xrawtrain(cobs2,:)];
                        cv_Ytrain{tt}=[ones(sum(cobs1),1);-1*ones(sum(cobs2),1)];
                        
                        tt=tt+1;
                    end
                end
                
                for p1_idx = 1: length(param_range)
                    [W,C] = STLearner(cv_Xtrain, cv_Ytrain, param_range(p1_idx), opts);
                    cv.stl.perform_mat(p1_idx) = cv.stl.perform_mat(p1_idx) + eval_1vs1( cv_Xvalid,cv_Yvalid,W,C,classes);
                    [W,C] = MMTLearner(cv_Xtrain, cv_Ytrain,param_range(p1_idx),opts);
                    cv.mmtl.perform_mat(p1_idx) = cv.mmtl.perform_mat(p1_idx) + eval_1vs1( cv_Xvalid,cv_Yvalid,W,C,classes);
                    [W,C] = MTFLearner(cv_Xtrain, cv_Ytrain,param_range(p1_idx),opts);
                    cv.mtfl.perform_mat(p1_idx) = cv.mtfl.perform_mat(p1_idx) + eval_1vs1( cv_Xvalid,cv_Yvalid,W,C,classes);
                    [W,C] = MTMLearner(cv_Xtrain, cv_Ytrain,param_range(p1_idx),opts);
                    cv.mtml.perform_mat(p1_idx) = cv.mtml.perform_mat(p1_idx) + eval_1vs1( cv_Xvalid,cv_Yvalid,W,C,classes);
                    [W,C] = MTASOLearner(cv_Xtrain, cv_Ytrain,param_range(p1_idx),opts);
                    cv.mtaso.perform_mat(p1_idx) = cv.mtaso.perform_mat(p1_idx) + eval_1vs1( cv_Xvalid,cv_Yvalid,W,C,classes);
                    for p2_idx = 1: length(lambda_range)
                        [W,C] = SPMMTLearner(cv_Xtrain, cv_Ytrain,param_range(p1_idx),lambda_range(p2_idx),opts);
                        cv.spmmtl.perform_mat(p1_idx,p2_idx) = cv.spmmtl.perform_mat(p1_idx,p2_idx) + eval_1vs1( cv_Xvalid,cv_Yvalid,W,C,classes);
                        [W,C] = SPMTFLearner(cv_Xtrain, cv_Ytrain,param_range(p1_idx),lambda_range(p2_idx),opts);
                        cv.spmtfl.perform_mat(p1_idx,p2_idx) = cv.spmtfl.perform_mat(p1_idx,p2_idx) + eval_1vs1( cv_Xvalid,cv_Yvalid,W,C,classes);
                        [W,C] = SPMTMLearner(cv_Xtrain, cv_Ytrain,param_range(p1_idx),lambdaselect_range(p2_idx),opts);
                        cv.spmtml.perform_mat(p1_idx,p2_idx) = cv.spmtml.perform_mat(p1_idx,p2_idx) + eval_1vs1( cv_Xvalid,cv_Yvalid,W,C,classes);
                        [W,C] = SPMTASOLearner(cv_Xtrain, cv_Ytrain,param_range(p1_idx),lambda_range(p2_idx),opts);
                        cv.spmtaso.perform_mat(p1_idx,p2_idx) = cv.spmtaso.perform_mat(p1_idx,p2_idx) + eval_1vs1( cv_Xvalid,cv_Yvalid,W,C,classes);
                    end
                end
                
                
            end
            
            cv.stl.perform_mat = cv.stl.perform_mat./kFold;
            [~,best_idx] = max(cv.stl.perform_mat);
            cv.stl.mu=param_range(best_idx);
            cv.mmtl.perform_mat = cv.mmtl.perform_mat./kFold;
            [~,best_idx] = max(cv.mmtl.perform_mat);
            cv.mmtl.rho_sr=param_range(best_idx);
            cv.mtfl.perform_mat = cv.mtfl.perform_mat./kFold;
            [~,best_idx] = max(cv.mtfl.perform_mat);
            cv.mtfl.rho_fr=param_range(best_idx);
            cv.mtml.perform_mat = cv.mtml.perform_mat./kFold;
            [~,best_idx] = max(cv.mtml.perform_mat);
            cv.mtml.rho_fr=param_range(best_idx);
            cv.mtaso.perform_mat = cv.mtaso.perform_mat./kFold;
            [~,best_idx] = max(cv.mtaso.perform_mat);
            cv.mtaso.rho_fr=param_range(best_idx);
            
            cv.spmmtl.perform_mat = cv.spmmtl.perform_mat./kFold;
            [row,col] = find(cv.spmmtl.perform_mat == max(cv.spmmtl.perform_mat(:)));
            cv.spmmtl.rho_sr=param_range(row(end));
            cv.spmmtl.lambda=lambda_range(col(end));
            cv.spmtfl.perform_mat = cv.spmtfl.perform_mat./kFold;
            [row,col] = find(cv.spmtfl.perform_mat == max(cv.spmtfl.perform_mat(:)));
            cv.spmtfl.rho_sr=param_range(row(end));
            cv.spmtfl.lambda=lambda_range(col(end));
            cv.spmtml.perform_mat = cv.spmtml.perform_mat./kFold;
            [row,col] = find(cv.spmtml.perform_mat == max(cv.spmtml.perform_mat(:)));
            cv.spmtml.rho_sr=param_range(row(end));
            cv.spmtml.lambda=lambdaselect_range(col(end));
            cv.spmtaso.perform_mat = cv.spmtaso.perform_mat./kFold;
            [row,col] = find(cv.spmtaso.perform_mat == max(cv.spmtaso.perform_mat(:)));
            cv.spmtaso.rho_sr=param_range(row(end));
            cv.spmtaso.lambda=lambda_range(col(end));
            if opts.verbose
                fprintf(']');
            end
            save(sprintf('cv/%s_cv_%0.2f_%d.mat',dataset,trainSize,rId),'cv');
            if opts.verbose
                fprintf(':DONE,');
            end
        end
        %------------------------------------------------------------------------
        %                   Experiment
        %------------------------------------------------------------------------
        if opts.verbose
            fprintf('Exp[');
        end
        for m=1:length(models)
            model=models{m};
            opts.method=model;
            tic
            switch model
                case 'STL'
                    % Single Task Learner
                    cv.stl.mu=0.1;
                    [W,C] = STLearner(Xtrain, Ytrain,cv.stl.mu,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'MMTL'
                    % Mean multi-task learner
                    R=eye (K) - ones (K) / K;
                    opts.Omega=R*R';
                    cv.mmtl.rho_sr=0.1;
                    [W,C] = MMTLearner(Xtrain, Ytrain,cv.mmtl.rho_sr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMMTL'
                    % Self-paced Mean multi-task learner
                    lambda=1;
                    cv.spmmtl.rho_sr=0.1;
                    cv.spmmtl.lambda=0.1;
                    [W,C,tau] = SPMMTLearner(Xtrain, Ytrain,cv.spmmtl.rho_sr,cv.spmmtl.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                    result{m}.tau{rId}=tau;
                case 'MTFL'
                    % Multi-task Feature Learner
                    cv.mtfl.rho_fr=0.1;
                    [W,C, invD] = MTFLearner(Xtrain, Ytrain,cv.mtfl.rho_fr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMTFL'
                    % Self-paced Multi-task Feature Learner
                    lambda=1;
                    cv.spmtfl.rho_sr=0.1;
                    cv.spmtfl.lambda=0.1;
                    [W,C,invD,tau] = SPMTFLearner(Xtrain, Ytrain,cv.spmtfl.rho_sr,cv.spmtfl.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                    result{m}.tau{rId}=tau;
                case 'MTML'
                    % Manifold-based multi-task learner
                    cv.mtml.rho_fr=0.1;
                    [W,C] = MTMLearner(Xtrain, Ytrain,cv.mtml.rho_fr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMTML'
                    % Self-pased Manifold-based multi-task learner
                    cv.spmtml.rho_sr=0.1;
                    cv.spmtml.lambda=5;
                    [W,C] = SPMTMLearner(Xtrain, Ytrain,cv.spmtml.rho_sr,cv.spmtml.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'MTASO'
                    % multi-task learner with Alternating Structure
                    % Optimization
                    opts.h=10;
                    cv.mtaso.rho_fr=0.1;
                    [W,C,theta] = MTASOLearner(Xtrain, Ytrain,cv.mtaso.rho_fr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMTASO'
                    % multi-task learner with Alternating Structure
                    % Optimization
                    opts.h=10;
                    cv.spmtaso.rho_sr=0.1;
                    cv.spmtaso.lambda=0.1;
                    [W,C,theta] = SPMTASOLearner(Xtrain, Ytrain,cv.spmtaso.rho_sr,cv.spmtaso.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'MTRL'
                    % Multi-task Relationship Learner
                    %opts.rho_l1=0;
                    opts.rho_sr=1e-2;
                    [W,C] = MTRLearner(Xtrain, Ytrain,opts);
                    if opts.verbose
                        fprintf('*');
                    end
            end
            result{m}.runtime(nt)=result{m}.runtime(nt)+toc;
            result{m}.model=model;
            result{m}.loss=opts.loss;
            result{m}.opts=opts;
            
            %------------------------------------------------------------------------
            %                   Evaluation
            %------------------------------------------------------------------------
            % Test set
            % Evaluation: Compute Accuracy
            Xtest=X(split.test,:);
            Ytest=Y(split.test);
            [result{m}.score(rId,nt),result{m}.taskScore(:,rId,nt)] = eval_1vs1( Xtest,Ytest,W,C,classes );
            %{
            Yhat=zeros(length(Ytest),K);
            tt=1;
            for class1=1:length(classes)-1
                for class2=class1+1:length(classes)
                    temp=sign(Xtest*W(:,tt)+C(tt));
                    Yhat(temp==1,tt)=class1;
                    Yhat(temp==-1,tt)=class2;
                    tt=tt+1;
                end
            end
            
            
            [~,Ypred] = max(hist(Yhat', nclass));
            Ypred=Ypred';
            if(size(Yhat, 2) == 1)
                Ypred = Yhat';
            end
            
            corr=sum(Ypred==Ytest);
            result{m}.score(rId,nt)=corr/length(Ytest);
            result{m}.taskScore(:,rId,nt)=zeros(nclass,1);
            for t=1:nclass
                Ypred_t=Ypred(Ytest==classes(t));
                Y_t=Ytest(Ytest==classes(t));
                
                corr=sum(Ypred_t==Y_t);
                result{m}.taskScore(t,rId,nt)=corr/length(Y_t);
            end
            %}
            
            if(opts.debugMode)
                fprintf('Method: %s, RunId: %d, AUC: %f \n',opts.method,rId,result{m}.score(rId,nt));
            end
        end
        if opts.verbose
            fprintf(']:DONE)\n');
        end
    end
    %%% Per TrainSize Stats
    if opts.verbose
        fprintf('Results: \n');
    end
    for m=1:length(models)
        result{m}.meanScore=mean(result{m}.score(:,nt));
        result{m}.stdScore=std(result{m}.score(:,nt));
        result{m}.meanTaskScore=mean(result{m}.taskScore(:,:,nt),2);
        result{m}.stdTaskScore=std(result{m}.taskScore(:,:,nt),0,2);
        result{m}.runtime(nt)=result{m}.runtime(nt)/Nrun;
        fprintf('Method: %s, Mean %s: %f, Std %s: %f Runtime: %0.4f\n', result{m}.model,'Accuracy',result{m}.meanScore,'Accuracy',result{m}.stdScore,result{m}.runtime(nt));
    end
    save(sprintf('results/%s_results_%0.2f.mat',dataset,trainSize),'result');
end




