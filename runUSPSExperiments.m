

clear;
rng('default');

% Read Landmine data
load('data/usps/split1_usps_1000train.mat')
X=[digit_trainx;digit_validx;digit_testx];
Y=[digit_trainy;digit_validy;digit_testy];

classes=unique(Y);
nclass= length(classes);
K=(nclass-1)*nclass/2;

% Add intercept
%X = cellfun(@(x) [ones(size(x,1),1),x], X, 'uniformOutput', false);
% Change the labelspace from {0,1} to {-1,1}


Nrun=10;
% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'STL','MMTL','SPMMTL','MTFL','SPMTFL'}; % Choose subset: {'STL','ITL','MMTL','MTFL','MTRL'};
trainSizes=0.75; %20:20:300;

opts.loss='hinge'; % Choose one: 'logit', 'least', 'hinge'
opts.debugMode=false;
opts.verbose=true;
opts.tol=1e-5;
opts.maxIter=100; % max iter for Accelerated Grad
opts.maxOutIter=25; % max iter for alternating optimization

% Initilaization
result=cell(length(models),1);
for m=1:length(models)
    result{m}.score=zeros(Nrun,length(trainSizes));
    result{m}.taskScore=zeros(nclass,Nrun,length(trainSizes));
    result{m}.runtime=zeros(1,length(trainSizes));
end



for nt=1:length(trainSizes)
    Ntrain=trainSizes(nt);
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run Experiment
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Run Id - For Repeated Experiment
    for rId=1:Nrun
        
        % Split Data into train and test
        split = cvpartition(Y,'HoldOut',1-Ntrain);
        Xrawtrain=X(split.training,:);
        Yrawtrain=Y(split.training);
        
        
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
        
        
        % Do Cross Validation for models
        %opts.mu=0;
        %opts.rho_l1=0;
        %opts.rho_sr=0.1;
        %opts.rho_fr=0.1;
        %{
            cvDebugFlag=false;
            if isempty(mu)
                    if(opts.debugMode)
                        fprintf('Performing CV for models .... \n');
                        opts.debugMode=false;
                        cvDebugFlag=true;
                    end
                    [lambda,alpha,~] = CrossValidation2Param( Xtrain, Ytrain, 'batchMTLearner', opts, lambdaSearchSpace, alphaSearchSpace, kFold, 'eval_MTL', true);
                    if cvDebugFlag
                        opts.debugMode=true;
                    end
            end
        %}
        for m=1:length(models)
            model=models{m};
            opts.method=model;
            tic
            switch model
                case 'STL'
                    % Single Task Learner
                    opts.mu=1e-2;
                    %opts.rho_l1=0;
                    [W,C] = STLearner(Xtrain, Ytrain,opts);
                case 'MMTL'
                    % Mean multi-task learner
                    opts.mu=0;
                    %opts.rho_l1=0;
                    opts.rho_sr=1e-2;
                    R=eye (K) - ones (K) / K;
                    Omega=R*R';
                    [W,C] = StructMTLearner(Xtrain, Ytrain,Omega,opts);
                case 'SPMMTL'
                    % Mean multi-task learner
                    opts.mu=0;
                    %opts.rho_l1=0;
                    opts.rho_sr=1e-2;
                    lambda=0.5;
                    [W,C] = SPMMTLearner(Xtrain, Ytrain,lambda,opts);
                case 'MTFL'
                    % Multi-task Feature Learner
                    %opts.rho_l1=0;
                    opts.rho_fr=1e-2;
                    [W,C] = MTFLearner(Xtrain, Ytrain,opts);
                case 'SPMTFL'
                    % Multi-task Feature Learner
                    %opts.rho_l1=0;
                    opts.rho_fr=1e-2;
                    lambda=0.5;
                    %lambda=0.0005;
                    %lambda=5;
                    [W,C,invD,tau] = SPMTFLearner(Xtrain, Ytrain,lambda,opts);
                case 'MTRL'
                    % Multi-task Relationship Learner
                    %opts.rho_l1=0;
                    opts.rho_sr=1e-2;
                    [W,C] = MTRLearner(Xtrain, Ytrain,opts);
            end
            result{m}.runtime(nt)=result{m}.runtime(nt)+toc;
            result{m}.model=model;
            result{m}.loss=opts.loss;
            result{m}.opts=opts;
            
            
            % Evaluation: Compute Accuracy
            Xtest=X(split.test,:);
            Ytest=Y(split.test);
            
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
            
            if(opts.debugMode)
                fprintf('Method: %s, Ntrain: %d, RunId: %d, AUC: %f \n',opts.method,Ntrain,rId,result{m}.score(rId,nt));
            end
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
    
end





