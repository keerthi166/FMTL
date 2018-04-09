

clear;
rng('default');
addpath(genpath('.'))
% Read School data
dataset='school';
load('data/school/school_data/school_b.mat')

K= length(task_indexes);

Nrun=5;
% CV Settings
kFold=3; % 5 fold cross validation


% Model Settings
models={'ITL','STL','MTFL','SHAMO','MTDict','MTFactor','TriFactor'};
trainSize=0.6; % 10% 20% 30%

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
opts.debugMode=true;
opts.verbose=true;
opts.tol=1e-6;
opts.maxIter=100;
opts.maxOutIter=25;
opts.cv=true;

kappa=2;
kappa1=2;
kappa2=2;

fprintf('Kappa:%f Kappa1:%f Kappa2:%f',kappa,kappa1,kappa2);


cv=[];
cvTime=0;

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
fprintf('Train Size %d, date: %s\n',trainSize,date);
for rId=1:Nrun
    opts.rId=rId;
    if opts.verbose
        fprintf('Run %d (',rId);
    end
    %------------------------------------------------------------------------
    %                   Train-Test Split
    %------------------------------------------------------------------------
    
    
    % Split Data into train and test
    split=cellfun(@(n) cvpartition(n,'HoldOut',1-trainSize),num2cell(N),'UniformOutput',false);
    
    Xtrain=cellfun(@(x,split_t) {x(split_t.training,:)}, X, split);
    Ytrain=cellfun(@(y,split_t) {y(split_t.training,:)}, Y,split);
    Xtest=cellfun(@(x,split_t) {x(split_t.test,:)}, X, split);
    Ytest=cellfun(@(y,split_t) {y(split_t.test,:)}, Y,split);
    %}
    %{
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
    
    %cv=[];
    %load(sprintf('cv/%s_cv_%0.2f_%d.mat',dataset,0.60,1));
    %load(sprintf('cv/%s_cv_%0.2f_sparse.mat',dataset,0.20));
    
    if (isempty(cv) && opts.cv)
        tic
        %------------------------------------------------------------------------
        %                   Cross Validation
        %------------------------------------------------------------------------
        cvModels={'itl','stl','mtfl','shamo','mtdict','mtfactor','trifactor'};
        
        if opts.verbose
            fprintf('CV');
        end
        
        lambda_range=[1e-3,1e-2,1e-1,1e-0,1e+1,1e+2,1e+3];
        %lambda_range=0.1;
        %param_range=[1e-3,1e-2,1e-1,1e-0,1e+1,1e+2,1e+3];
        param_range= [1e-3,0.005,1e-2,0.05,1e-1,0.2,0.3,0.4];
        
        opts.method='cv';
        opts.h=kappa;
        
        opts.kappa=kappa;
        opts.kappa1=kappa1;
        opts.kappa2=kappa2;
        
        
        cvDebugFlag=false;
        if (opts.debugMode)
            opts.debugMode=false;
            cvDebugFlag=true;
        end
        cvpool=gcp;
        if isempty(cvpool)
            cvpool=parpool();
        end
        %cvpool.addAttachedFiles({'utils/mtl_dictlearn/'});
        addAttachedFiles(cvpool,{'cv','model','solver','utils'});
        cvpool.addAttachedFiles({'cv/CrossValidation1Param.m','cv/CrossValidation2Param.m'});       
        ab=cell(1,length(cvModels));
        parfor m=1:length(cvModels)
            cvModel=cvModels{m};
            cvOpts=opts;
            switch cvModel
                case 'itl'
                    [mu,perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'STLearner', cvOpts, lambda_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    stl=struct('mu',mu,'perfMat',perfMat);
                    ab{m}=stl;
                case 'stl'
                    cvOpts.kappa=1;
                    [mu,perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'SharedMTLearner', cvOpts, lambda_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    itl=struct('mu',mu,'perfMat',perfMat);
                    ab{m}=itl;
                
                case 'mmtl'
                    [rho_sr,perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MMTLearner', cvOpts, lambda_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    mmtl=struct('rho_sr',rho_sr,'perfMat',perfMat);
                    ab{m}=mmtl;
                case 'mtfl'
                    [rho_fr,perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTFLearner', cvOpts, lambda_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    mtfl=struct('rho_fr',rho_fr,'perfMat',perfMat);
                    ab{m}=mtfl;
                case 'mtrl'
                    [rho_sr,perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTRLearner', cvOpts, lambda_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    mtrl=struct('rho_sr',rho_sr,'perfMat',perfMat);
                    ab{m}=mtrl;
                case 'shamo'
                    [rho_fr,perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'SharedMTLearner', cvOpts, param_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    shamo=struct('rho_fr',rho_fr,'perfMat',perfMat);
                    ab{m}=shamo;
                case 'cmtl'
                    [rho_fr,perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTByClusteringLearner', cvOpts, param_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    cmtl=struct('rho_fr',rho_fr,'perfMat',perfMat);
                    ab{m}=cmtl;
                case 'mtdict'
                    addpath(genpath('utils/mtl_dictlearn'));
                    [rho_fr,rho_l1,perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'MTDictLearner', cvOpts, lambda_range, param_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    mtdict=struct('rho_fr',rho_fr,'rho_l1',rho_l1,'perfMat',perfMat);
                    ab{m}=mtdict;
                case 'mtfactor'
                    cvOpts.rho_l1=0;
                    [rho_fr1,rho_fr2,perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'BiFactorMTLearner', cvOpts, lambda_range, param_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    mtfactor=struct('rho_fr1',rho_fr1,'rho_fr2',rho_fr2,'perfMat',perfMat);
                    ab{m}=mtfactor;
                case 'trifactor'
                    cvOpts.rho_l1=0;
                    [rho_fr1,rho_fr2,perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'TriFactorMTLearner', cvOpts, lambda_range, param_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    trifactor=struct('rho_fr1',rho_fr1,'rho_fr2',rho_fr2,'perfMat',perfMat);
                    ab{m}=trifactor;
                case 'matnorm'
                    cvOpts.sparseOmega=true;
                    cvOpts.sparseSigma=true;
                    cvOpts.rho_l1=0.1;
                    [rho_fr,perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MatrixNormalMTLearner', cvOpts, lambda_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    matnorm=struct('rho_fr',rho_fr,'perfMat',perfMat);
                    ab{m}=matnorm;
                case 'sparsetrifactor'
                    cvOpts.sparseOmega=true;
                    cvOpts.sparseSigma=true;
                    cvOpts.rho_l1=0.1;
                    [rho_fr1,rho_fr2,perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'TriFactorMTLearner', cvOpts, lambda_range, param_range,kFold, 'eval_MTL', cvOpts.isHigherBetter,cvOpts.scoreType);
                    sparsetrifactor=struct('rho_fr1',rho_fr1,'rho_fr2',rho_fr2,'perfMat',perfMat);
                    ab{m}=sparsetrifactor;
                    
            end
            
        end
        delete(cvpool);
        for m=1:length(cvModels)
            cvModel=cvModels{m};
            cv.(cvModel)=ab{m};
        end
        
        
        
        save(sprintf('cv/%s_cv_%0.2f_%d.mat',dataset,trainSize,rId),'cv');
        if cvDebugFlag
            opts.debugMode=true;
        end
        cvTime=cvTime + toc;
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
                %cv.itl.mu=0.1;
                opts.kappa=1;
                opts.rho_l1=0;
                [W,C,clusters] = SharedMTLearner(Xtrain, Ytrain,cv.itl.mu,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'ITL'
                % Single Task Learner
                %cv.stl.mu=0.1;
                [W,C] = STLearner(Xtrain, Ytrain,cv.stl.mu,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MMTL'
                % Mean multi-task learner
                R=eye (K) - ones (K) / K;
                opts.Omega=R*R';
                %cv.mmtl.rho_sr=0.1;
                [W,C] = StructMTLearner(Xtrain, Ytrain,cv.mmtl.rho_sr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTFL'
                % Multi-task Feature Learner
                [W,C, invD] = MTFLearner(Xtrain, Ytrain,cv.mtfl.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
                
            case 'MTRL'
                % Multi-task Relationship Learner
                [W,C, Omega] = MTRLearner(Xtrain, Ytrain,cv.mtrl.rho_sr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTDict'
                % Multi-task Dictionary Learner
                opts.kappa=kappa;
                %cv.mtdict.rho_fr=0.1;
                %cv.mtdict.rho_l1=10;
                [W,C,F,G] = MTDictLearner(Xtrain, Ytrain,cv.mtdict.rho_fr,cv.mtdict.rho_l1,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'SHAMO'
                %cv.shamo.rho_fr=0.1;
                opts.kappa=kappa;
                opts.rho_l1=0;
                [W,C,clusters] = SharedMTLearner(Xtrain, Ytrain,cv.shamo.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'CMTL'
                % Multi-task learning by Clustering (Barzillai)
                cv.cmtl.rho_fr=1;
                [W,C,Omega] = MTByClusteringLearner(Xtrain, Ytrain,cv.cmtl.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTFactor'
                % Multi-task BiFactor Relationship Learner
                %cv.mtfactor.rho_fr1=0.1;
                %cv.mtfactor.rho_fr2=1;
                opts.kappa=kappa;
                opts.rho_l1=0;
                %opts.LF=LF;
                [W,C,F,G,Sigma, Omega] = BiFactorMTLearner(Xtrain, Ytrain,cv.mtfactor.rho_fr1,cv.mtfactor.rho_fr2,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'TriFactor'
                % Multi-task TriFactor Relationship Learner
                opts.kappa1=kappa1;
                opts.kappa2=kappa2;
                %cv.trifactor.rho_fr1=0.1;
                %cv.trifactor.rho_fr2=1;
                opts.rho_l1=0;
                [W,C,F,S,G,Sigma, Omega] = TriFactorMTLearner(Xtrain, Ytrain,cv.trifactor.rho_fr1,cv.trifactor.rho_fr2,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'SparseMatrixNorm'
                % Matrix Normal Multi-task Learner
                %cv.matnorm.rho_fr=0.1;
                opts.rho_l1=0.01;
                [W,C,Sigma, Omega] = MatrixNormalMTLearner(Xtrain, Ytrain,cv.matnorm.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'SparseTriFactor'
                % Multi-task TriFactor Relationship Learner
                opts.kappa1=kappa1;
                opts.kappa2=kappa2;
                opts.sparseOmega=true;
                opts.sparseSigma=true;
                
                %cv.sparsetrifactor.rho_fr1=0.1;
                %cv.sparsetrifactor.rho_fr2=0.1;
                opts.rho_l1=0.01;
                [W,C,F,S,G,Sigma, Omega] = TriFactorMTLearner(Xtrain, Ytrain,cv.sparsetrifactor.rho_fr1,cv.sparsetrifactor.rho_fr2,opts);
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
    %{
    if (isempty(cv) && opts.cv)
        
        %------------------------------------------------------------------------
        %                   Cross Validation
        %------------------------------------------------------------------------
        if opts.verbose
            fprintf('CV');
        end
        lambda_range=[1,10,25,50,75,100];
        param_range=[1e-3,1e-2,1e-1,1e-0,1e+1,1e+2,1e+3];

        opts.method='cv';
        opts.h=5;
        
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
        [cv.spmtml.rho_fr,cv.spmtml.lambda,cv.spmtml.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTMLearner', opts, param_range,[10:5:30],kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
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
                opts.h=5;
                [W,C,theta] = MTASOLearner(Xtrain, Ytrain,cv.mtaso.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'SPMTASO'
                % multi-task learner with Alternating Structure
                % Optimization
                %cv.spmtaso.rho_fr=0.1;
                %cv.spmtaso.lambda=0.1;
                opts.h=5;
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
                kappa=5;
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
        if(opts.verbose)
            fprintf('Method: %s, RunId: %d, %s: %f \n',opts.method,rId,opts.scoreType,result{m}.score(rId));
        end
    end
    %}
    if opts.verbose
        fprintf(']:DONE)\n');
    end
end
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
if opts.verbose
    fprintf('Total CV Time: %f \n', cvTime);
end
save(sprintf('results/%s_results_%0.2f.mat',dataset,trainSize),'result');



