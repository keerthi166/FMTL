

clear;
rng('default');

% Read Landmine data
dataset='landmine';
load('data/landmine/landmine-19tasks.mat')
N=cellfun(@(x) size(x,1),X);
K= length(Y);

% Add intercept
%X = cellfun(@(x) [ones(size(x,1),1),x], X, 'uniformOutput', false);
% Change the labelspace from {0,1} to {-1,1}
Y=cellfun(@(y) 2*y-1,Y,'UniformOutput',false);


Nrun=10;
% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'STL','MMTL','SPMMTL','MTFL','SPMTFL','MTML','SPMTML','MTASO','SPMTASO'}; % Choose subset: {'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'};

trainSizes=80; %[30,40,80,160];

opts.loss='logit'; % Choose one: 'logit', 'least', 'hinge'
opts.scoreType='perfcurve'; % Choose one: 'perfcurve', 'class', 'mse', 'nmse'
opts.isHigherBetter=true;
opts.debugMode=false;
opts.verbose=true;
opts.tol=1e-5;
opts.maxIter=100; % max iter for Accelerated Grad
opts.maxOutIter=50; % max iter for alternating optimization
opts.cv=true;

cv=[];

% Initilaization
result=cell(length(models),1);
for m=1:length(models)
    result{m}.score=zeros(Nrun,length(trainSizes));
    result{m}.taskScore=zeros(K,Nrun,length(trainSizes));
    if strncmpi(models{m},'SPM',3)
        result{m}.tau=cell(1,Nrun);
    end
    result{m}.runtime=zeros(1,length(trainSizes));
end



for nt=1:length(trainSizes)
    Ntrain=trainSizes(nt);
    if Ntrain>=min(N)
        error('Invalid number of observations in the training set');
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run Experiment
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Run Id - For Repeated Experiment
    fprintf('Train Size %d\n',Ntrain);
    for rId=1:Nrun
        opts.rId=rId;
        if opts.verbose
            fprintf('Run %d (',rId);
        end
        % Split Data into train and test
        split=cellfun(@(y,n) cvpartition(y,'HoldOut',n-Ntrain),Y,num2cell(N),'UniformOutput',false);
        
        Xtrain=cellfun(@(x,split_t) {x(split_t.training,:)}, X, split);
        Ytrain=cellfun(@(y,split_t) {y(split_t.training,:)}, Y,split);
        Xtest=cellfun(@(x,split_t) {x(split_t.test,:)}, X, split);
        Ytest=cellfun(@(y,split_t) {y(split_t.test,:)}, Y,split);
        
        %------------------------------------------------------------------------
        %                   Cross Validation
        %------------------------------------------------------------------------
        %load(sprintf('cv/%s_cv_%0.2f.mat',dataset,Ntrain));
        if (isempty(cv) && opts.cv)
            
            %------------------------------------------------------------------------
            %                   Cross Validation
            %------------------------------------------------------------------------
            if opts.verbose
                fprintf('CV');
            end
            lambda_range=[1e-3,1e-2,1e-1,1e-0,1e+1,1e+2];
            param_range=[1e-3,1e-2,1e-1,1e-0,1e+1,1e+2,1e+3];
            
            
            cvDebugFlag=false;
            if (opts.debugMode)
                opts.debugMode=false;
                cvDebugFlag=true;
            end
            [cv.stl.mu,cv.stl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'STLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
            [cv.mmtl.rho_sr,cv.mmtl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MMTLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
            [cv.mtfl.rho_fr,cv.mtfl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTFLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
            [cv.mtml.rho_fr,cv.mtml.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTMLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
            [cv.mtaso.rho_fr,cv.mtaso.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTASOLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
            
            
            [cv.spmmtl.rho_sr,cv.spmmtl.lambda,cv.spmmtl.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMMTLearner', opts, param_range,lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
            [cv.spmtfl.rho_fr,cv.spmtfl.lambda,cv.spmtfl.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTFLearner', opts, param_range,lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
            [cv.spmtml.rho_fr,cv.spmtml.lambda,cv.spmtml.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTMLearner', opts, param_range,(10:5:15),kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
            [cv.spmtaso.rho_fr,cv.spmtaso.lambda,cv.spmtaso.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTASOLearner', opts, param_range,lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
            
            
            save(sprintf('cv/%s_cv_%0.2f.mat',dataset,Ntrain),'cv');
            if cvDebugFlag
                opts.debugMode=true;
            end
            
            if opts.verbose
                fprintf(':DONE,');
            end
        end
        
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
                    [W,C] = STLearner(Xtrain, Ytrain,cv.stl.mu,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'MMTL'
                    % Mean multi-task learner
                    R=eye (K) - ones (K) / K;
                    opts.Omega=R*R';
                    [W,C] = StructMTLearner(Xtrain, Ytrain,cv.mmtl.rho_sr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMMTL'
                    % Self-paced Mean multi-task learner
                    lambda=5;
                    [W,C,tau] = SPMMTLearner(Xtrain, Ytrain,cv.spmmtl.rho_sr,cv.spmmtl.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                    result{m}.tau{rId}=tau;
                case 'MTFL'
                    % Multi-task Feature Learner
                    [W,C, invD] = MTFLearner(Xtrain, Ytrain,cv.mtfl.rho_fr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMTFL'
                    % Self-paced Multi-task Feature Learner
                    %opts.rho_l1=0;
                    lambda=5;
                    [W,C,invD,tau] = SPMTFLearner(Xtrain, Ytrain,cv.spmtfl.rho_fr,cv.spmtfl.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                    result{m}.tau{rId}=tau;
                case 'MTML'
                    % Manifold-based multi-task learner
                    %cv.mtml.rho_fr=1;
                    [W,C] = MTMLearner(Xtrain, Ytrain,cv.mtml.rho_fr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMTML'
                    % Self-pased Manifold-based multi-task learner
                    %cv.spmtml.rho_fr=1;
                    %cv.spmtml.lambda=25;
                    [W,C] = SPMTMLearner(Xtrain, Ytrain,cv.spmtml.rho_fr,cv.spmtml.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'MTASO'
                    % multi-task learner with Alternating Structure
                    % Optimization
                    %cv.mtaso.rho_fr=0.1;
                    opts.h=2;
                    [W,C,theta] = MTASOLearner(Xtrain, Ytrain,cv.mtaso.rho_fr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMTASO'
                    % multi-task learner with Alternating Structure
                    % Optimization
                    %cv.spmtaso.rho_fr=0.1;
                    %cv.spmtaso.lambda=0.1;
                    opts.h=2;
                    [W,C,theta] = SPMTASOLearner(Xtrain, Ytrain,cv.spmtaso.rho_fr,cv.spmtaso.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'MTRL'
                    % Multi-task Relationship Learner
                    %opts.rho_l1=0;
                    opts.rho_sr=0.1;
                    [W,C] = MTRLearner(Xtrain, Ytrain,opts);
            end
            result{m}.runtime(nt)=result{m}.runtime(nt)+toc;
            result{m}.model=model;
            result{m}.loss=opts.loss;
            result{m}.scoreType=opts.scoreType;
            result{m}.opts=opts;
            
            % Compute Area under the ROC curve & Accuracy
            [result{m}.score(rId,nt),result{m}.taskScore(:,rId,nt)]=eval_MTL(Ytest, Xtest, W, C,[], opts.scoreType);
            if(opts.debugMode)
                fprintf('Method: %s, Ntrain: %d, RunId: %d, AUC: %f \n',opts.method,Ntrain,rId,result{m}.score(rId,nt));
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
        fprintf('Method: %s, Mean %s: %f, Std %s: %f Runtime: %0.4f\n', result{m}.model,opts.scoreType,result{m}.meanScore,opts.scoreType,result{m}.stdScore,result{m}.runtime(nt));
    end
    save(sprintf('results/%s_results_%0.2f.mat',dataset,Ntrain),'result');
end





