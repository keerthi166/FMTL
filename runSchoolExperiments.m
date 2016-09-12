

clear;
rng('default');

% Read School data
dataset='school';
load('data/school/school_data/school_b.mat')

K= length(task_indexes);

Nrun=10;
% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'STL','MMTL','SPMMTL','MTFL','SPMTFL','MTML','SPMTML','MTASO','SPMTASO'}; % Choose subset: {'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'};
trainSize=0.60; % 10% 20% 30%

X=cell(1,K);
Y=cell(1,K);

task_indexes(end+1)=length(y)+1;
for tt=1:K
    X{tt}=x(1:end-1,task_indexes(tt):task_indexes(tt+1)-1)';
    Y{tt}=y(task_indexes(tt):task_indexes(tt+1)-1);
end
% Add intercept
X = cellfun(@(x) [ones(size(x,1),1),x], X, 'uniformOutput', false);
N=cellfun(@(x) size(x,1),X);
clear task_indexes

opts.dataset=dataset;
opts.loss='least'; % Choose one: 'logit', 'least', 'hinge'
opts.scoreType='rmse'; % Choose one: 'perfcurve', 'class', 'mse', 'nmse'
opts.isHigherBetter=false;
opts.debugMode=false;
opts.verbose=true;
opts.tol=1e-5;
opts.maxIter=100;
opts.maxOutIter=50;
opts.cv=true;

cv=[];

% Initilaization
result=cell(length(models),1);
for m=1:length(models)
    result{m}.score=zeros(Nrun,1);
    result{m}.taskScore=zeros(K,Nrun);
    if strncmpi(models{m},'SPM',3)
        result{m}.tau=cell(1,Nrun);
    end
    result{m}.runtime=0;
end

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
    
    %{
    % Split Data into train and test
    split=cellfun(@(n) cvpartition(n,'HoldOut',1-trainSize),num2cell(N),'UniformOutput',false);
    
    Xtrain=cellfun(@(x,split_t) {x(split_t.training,:)}, X, split);
    Ytrain=cellfun(@(y,split_t) {y(split_t.training,:)}, Y,split);
    Xtest=cellfun(@(x,split_t) {x(split_t.test,:)}, X, split);
    Ytest=cellfun(@(y,split_t) {y(split_t.test,:)}, Y,split);
    %}
    
    Xtrain=cell(1,K);
    Ytrain=cell(1,K);
    Xtest=cell(1,K);
    Ytest=cell(1,K);
    % load the default splits used in their experiments
    load(strcat('school_',num2str(rId),'_indexes'));
    tempTrX = x(1:27,tr)';
    tempTrY = y(tr);
    tempTstX = x(1:27,tst)';
    tempTstY = y(tst);
    tr_indexes(end+1)=length(tr)+1;
    tst_indexes(end+1)=length(tst)+1;
    for tt=1:K
        Xtrain{tt}=tempTrX(tr_indexes(tt):tr_indexes(tt+1)-1,:);
        Ytrain{tt}=tempTrY(tr_indexes(tt):tr_indexes(tt+1)-1);
        Xtest{tt}=tempTstX(tst_indexes(tt):tst_indexes(tt+1)-1,:);
        Ytest{tt}=tempTstY(tst_indexes(tt):tst_indexes(tt+1)-1);
    end
    clear tempTrX tempTrY tempTstX tempTstY tst tst_indexes;
    %}
    
    % Normalize Data if needed
    %[Xtrain,~,meanX,stdX] = normalizeMultitaskData(Xtrain);
    % Normalize Test Data
    %[Xtest,~,~,~] = normalizeMultitaskData(Xtest,[],meanX,stdX);
    
    
    %load(sprintf('cv/%s_cv_%0.2f.mat',dataset,trainSize));
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
        [cv.spmtml.rho_fr,cv.spmtml.lambda,cv.spmtml.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTMLearner', opts, param_range,(20:10:50),kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.spmtaso.rho_fr,cv.spmtaso.lambda,cv.spmtaso.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTASOLearner', opts, param_range,lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        
        
        save(sprintf('cv/%s_cv_%0.2f.mat',dataset,trainSize),'cv');
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
                lambda=100;
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
                lambda=100;
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
                if opts.verbose
                    fprintf('*');
                end
            case 'MTDict'
                % Multi-task Dictionary Learner
                opts.rho_l1=3;
                opts.rho_fr=1;
                kappa=2;
                [W,C] = MTDictLearner(Xtrain, Ytrain,kappa,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTFactor'
                % Multi-task BiFactor Relationship Learner
                opts.rho_l1=0;
                opts.rho_fr1=0.0001;
                opts.rho_fr2=1;
                kappa=2;
                [W,C, Sigma, Omega] = BiFactorMTLearner(Xtrain, Ytrain,kappa,opts);
                if opts.verbose
                    fprintf('*');
                end
        end
        result{m}.runtime=result{m}.runtime+toc;
        result{m}.model=model;
        result{m}.loss=opts.loss;
        result{m}.scoreType=opts.scoreType;
        result{m}.opts=opts;
        
        % Compute Area under the ROC curve & Accuracy
        [result{m}.score(rId),result{m}.taskScore(:,rId)]=eval_MTL(Ytest, Xtest, W, C,[], opts.scoreType);
        if(opts.debugMode)
            fprintf('Method: %s, RunId: %d, %s: %f \n',opts.method,rId,opts.scoreType,result{m}.score(rId));
        end
    end
    if opts.verbose
        fprintf(']:DONE)\n');
    end
end
%%% Per TrainSize Stats
%%% Per TrainSize Stats
if opts.verbose
    fprintf('Results: \n');
end
for m=1:length(models)
    result{m}.meanScore=mean(result{m}.score);
    result{m}.stdScore=std(result{m}.score);
    result{m}.meanTaskScore=mean(result{m}.taskScore,2);
    result{m}.stdTaskScore=std(result{m}.taskScore,0,2);
    result{m}.runtime=result{m}.runtime/Nrun;
    fprintf('Method: %s, Mean %s: %f, Std %s: %f Runtime: %0.4f\n', result{m}.model,opts.scoreType,result{m}.meanScore,opts.scoreType,result{m}.stdScore,result{m}.runtime);
end

save(sprintf('results/%s_results_%0.2f.mat',dataset,trainSize),'result');



