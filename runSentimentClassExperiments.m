
%%% Sentiment Regression Tasks

clear;
rng('default');

% Read Sentiment data
dataset='sentiment';
%load('data/sentiment/sentiment_analysis.mat')
load('data/sentiment/reduced_sentiment_data.mat')



nsplits=2;
nth=2;

taskIds=unique(tasks);
nDomains=length(taskIds);
X=cell(1,nsplits*nDomains);
Y=cell(1,nsplits*nDomains);
lt=1;
for tt=1:nDomains
    s1=randperm(250);
    s2=250+randperm(250);
    s4=500+randperm(250);
    s5=750+randperm(250);
    data=feat(tasks==taskIds(tt),cv_featsel);
    val=target(tasks==taskIds(tt))';
    switch nsplits
        case 1
            % Use each domain as a single task
            X{lt}=data;
            Y{lt}=2*(val>3)-1;
            trainSize=250;
        case 2
            % Split each domain to 2 and use 2 thresholds
            
            
            
            t1Idx=[s1(1:120),s2(1:120),s4(1:120),s5(1:120)];
            t2Idx=[s1(121:240),s2(121:240),s4(121:240),s5(121:240)];
            X{lt}=data(t1Idx,:);
            Y{lt}=2*(val(t1Idx)==5)-1;
            X{lt+1}=data(t2Idx,:);
            Y{lt+1}=2*(val(t2Idx)==1)-1;
            trainSize=120; % Number of training instances
            
        case 3
            % Split each domain to 3 and use 3 thresholds
            
            t1Idx=[s1(1:80),s2(1:80),s4(1:80),s5(1:80)];
            t2Idx=[s1(81:160),s2(81:160),s4(81:160),s5(81:160)];
            t3Idx=[s1(161:240),s2(161:240),s4(161:240),s5(161:240)];
            X{lt}=data(t1Idx,:);
            Y{lt}=2*(val(t1Idx)==5)-1;
            X{lt+1}=data(t2Idx,:);
            Y{lt+1}=2*(val(t2Idx)>3)-1;
            X{lt+2}=data(t3Idx,:);
            Y{lt+2}=2*(val(t3Idx)==1)-1;
            trainSize=80; % Number of training instances
        case 4
            % Split each domain to 4 and use 2 thresholds
            
            t1Idx=[s1(1:60),s2(1:60),s4(1:60),s5(1:60)];
            t2Idx=[s1(61:120),s2(61:120),s4(61:120),s5(61:120)];
            t3Idx=[s1(121:180),s2(121:180),s4(121:180),s5(121:180)];
            t4Idx=[s1(181:240),s2(181:240),s4(181:240),s5(181:240)];
            X{lt}=data(t1Idx,:);
            Y{lt}=2*(val(t1Idx)==5)-1;
            X{lt+1}=data(t2Idx,:);
            Y{lt+1}=2*(val(t2Idx)==1)-1;
            X{lt+2}=data(t3Idx,:);
            Y{lt+2}=2*(val(t3Idx)==5)-1;
            X{lt+3}=data(t4Idx,:);
            Y{lt+3}=2*(val(t4Idx)==1)-1;
            trainSize=60; % Number of training instances
        case 6
            % Split each domain to 6 and use 2/3 thresholds
            
            t1Idx=[s1(1:40),s2(1:40),s4(1:40),s5(1:40)];
            t2Idx=[s1(41:80),s2(41:80),s4(41:80),s5(41:80)];
            t3Idx=[s1(81:120),s2(81:120),s4(81:120),s5(81:120)];
            t4Idx=[s1(121:160),s2(121:160),s4(121:160),s5(121:160)];
            t5Idx=[s1(161:200),s2(161:200),s4(161:200),s5(161:200)];
            t6Idx=[s1(201:240),s2(201:240),s4(201:240),s5(201:240)];
            if nth==2
                X{lt}=data(t1Idx,:);
                Y{lt}=2*(val(t1Idx)==5)-1;
                X{lt+1}=data(t2Idx,:);
                Y{lt+1}=2*(val(t2Idx)==1)-1;
                X{lt+2}=data(t3Idx,:);
                Y{lt+2}=2*(val(t3Idx)==5)-1;
                X{lt+3}=data(t4Idx,:);
                Y{lt+3}=2*(val(t4Idx)==1)-1;
                X{lt+4}=data(t5Idx,:);
                Y{lt+4}=2*(val(t5Idx)==5)-1;
                X{lt+5}=data(t6Idx,:);
                Y{lt+5}=2*(val(t6Idx)==1)-1;
            else
                X{lt}=data(t1Idx,:);
                Y{lt}=2*(val(t1Idx)==5)-1;
                X{lt+1}=data(t2Idx,:);
                Y{lt+1}=2*(val(t2Idx)>3)-1;
                X{lt+2}=data(t3Idx,:);
                Y{lt+2}=2*(val(t3Idx)==1)-1;
                X{lt+3}=data(t4Idx,:);
                Y{lt+3}=2*(val(t4Idx)==5)-1;
                X{lt+4}=data(t5Idx,:);
                Y{lt+4}=2*(val(t5Idx)>3)-1;
                X{lt+5}=data(t6Idx,:);
                Y{lt+5}=2*(val(t6Idx)==1)-1;
            end
            trainSize=40; % Number of training instances
        case 9
            % Split each domain to 9 and use 3 thresholds
            
            t1Idx=[s1(1:26),s2(1:26),s4(1:26),s5(1:26)];
            t2Idx=[s1(27:52),s2(27:52),s4(27:52),s5(27:52)];
            t3Idx=[s1(53:78),s2(53:78),s4(53:78),s5(53:78)];
            t4Idx=[s1(79:104),s2(79:104),s4(79:104),s5(79:104)];
            t5Idx=[s1(105:130),s2(105:130),s4(105:130),s5(105:130)];
            t6Idx=[s1(131:156),s2(131:156),s4(131:156),s5(131:156)];
            t7Idx=[s1(157:182),s2(157:182),s4(157:182),s5(157:182)];
            t8Idx=[s1(183:208),s2(183:208),s4(183:208),s5(183:208)];
            t9Idx=[s1(209:234),s2(209:234),s4(209:234),s5(209:234)];
            
            X{lt}=data(t1Idx,:);
            Y{lt}=2*(val(t1Idx)==5)-1;
            X{lt+1}=data(t2Idx,:);
            Y{lt+1}=2*(val(t2Idx)>3)-1;
            X{lt+2}=data(t3Idx,:);
            Y{lt+2}=2*(val(t3Idx)==1)-1;
            X{lt+3}=data(t4Idx,:);
            Y{lt+3}=2*(val(t4Idx)==5)-1;
            X{lt+4}=data(t5Idx,:);
            Y{lt+4}=2*(val(t5Idx)>3)-1;
            X{lt+5}=data(t6Idx,:);
            Y{lt+5}=2*(val(t6Idx)==1)-1;
            X{lt+6}=data(t7Idx,:);
            Y{lt+6}=2*(val(t7Idx)==5)-1;
            X{lt+7}=data(t8Idx,:);
            Y{lt+7}=2*(val(t8Idx)>3)-1;
            X{lt+8}=data(t9Idx,:);
            Y{lt+8}=2*(val(t9Idx)==1)-1;
            trainSize=26; % Number of training instances
    end
    lt=lt+nsplits;
    
end
clear feat target data val s1 s2 s4 s5
N=cellfun(@(x) size(x,1),X);

K=length(Y);
% Model Settings
models={'SparseMatrixNorm','SparseTriFactor'};%
%{'ITL','STL','MTFL','SHAMO','CMTL','MTDict','MTFactor','TriFactor'};%{'STL','MMTL','SPMMTL','MTFL','SPMTFL','MTML','SPMTML','MTASO','SPMTASO'}; % Choose subset: {'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'};





Nrun=5;
% CV Settings
kFold = 3; % 5 fold cross validation

K= length(X);
N=cellfun(@(x) size(x,1),X);

% Add intercept
X = cellfun(@(x) [ones(size(x,1),1),x], X, 'uniformOutput', false);
N=cellfun(@(x) size(x,1),X);

opts.dataset=dataset;
opts.loss='least'; % Choose one: 'logit', 'least', 'hinge'
opts.scoreType='fmeasure'; % Choose one: 'perfcurve', 'class', 'mse', 'nmse'
opts.isHigherBetter=true;
opts.debugMode=false;
opts.verbose=true;
opts.tol=1e-6;
opts.maxIter=100;
opts.maxOutIter=25;
opts.cv=true;

%TypeA: (8,10,8),(10,12,10)
%TypeB: (22,12,22)
kappa=15;
kappa1=12;
kappa2=15;

fprintf('kappa: %d, kappa1: %d, kappa2: %d nsplits:%d nthr:%d\n',kappa,kappa1,kappa2,nsplits,nth);

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
    split=cellfun(@(y,n) cvpartition(y,'HoldOut',n-trainSize),Y,num2cell(N),'UniformOutput',false);
    
    Xtrain=cellfun(@(x,split_t) {x(split_t.training,:)}, X, split);
    Ytrain=cellfun(@(y,split_t) {y(split_t.training,:)}, Y,split);
    Xtest=cellfun(@(x,split_t) {x(split_t.test,:)}, X, split);
    Ytest=cellfun(@(y,split_t) {y(split_t.test,:)}, Y,split);
    
    
    % Normalize Data if needed
    %[Xtrain,~,meanX,stdX] = normalizeMultitaskData(Xtrain);
    % Normalize Test Data
    %[Xtest,~,~,~] = normalizeMultitaskData(Xtest,[],meanX,stdX);
    
    %load(sprintf('cv/%s_cv_%0.2f_%d.mat',dataset,120,1));
    %cv=[];
    if (isempty(cv) && opts.cv)
        %------------------------------------------------------------------------
        %                   Cross Validation
        %------------------------------------------------------------------------
        if opts.verbose
            fprintf('CV');
        end
        lambda_range=[1e-3,1e-2,1e-1,1e-0,1e+1,1e+2,1e+3];
        param_range=  [1e-3,1e-2,1e-1,1e-0,1e+1,1e+2,1e+3];
        
        opts.method='cv';
        opts.h=kappa;
        
        opts.kappa=kappa;
        opts.kappa1=kappa1;
        opts.kappa2=kappa2;
        
        cv.split=split;
        cvDebugFlag=false;
        if (opts.debugMode)
            opts.debugMode=false;
            cvDebugFlag=true;
        end
        %{
        [cv.stl.mu,cv.stl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'STLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        kk=opts.kappa;
        opts.kappa=1;
        [cv.itl.mu,cv.itl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'SharedMTLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        opts.kappa=kk;
        %[cv.mmtl.rho_sr,cv.mmtl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MMTLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.mtfl.rho_fr,cv.mtfl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTFLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        %[cv.mtrl.rho_sr,cv.mtrl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTRLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        
        [cv.shamo.rho_fr,cv.shamo.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'SharedMTLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.cmtl.rho_fr,cv.cmtl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTByClusteringLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.mtdict.rho_fr,cv.mtdict.rho_l1,cv.mtdict.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'MTDictLearner', opts, lambda_range, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        opts.rho_l1=0;
        [cv.mtfactor.rho_fr1,cv.mtfactor.rho_fr2,cv.mtfactor.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'BiFactorMTLearner', opts, lambda_range, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.trifactor.rho_fr1,cv.trifactor.rho_fr2,cv.trifactor.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'TriFactorMTLearner', opts, lambda_range, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        %}
        
        opts.sparseOmega=true;
        opts.sparseSigma=true;
        opts.rho_l1=0.1;
        [cv.matnorm.rho_fr,cv.matnorm.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MatrixNormalMTLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.sparsetrifactor.rho_fr1,cv.sparsetrifactor.rho_fr2,cv.sparsetrifactor.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'TriFactorMTLearner', opts, lambda_range, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        opts.rho_l1=0;
        opts.sparseOmega=false;
        opts.sparseSigma=false;
        %}
        
        %[cv.mtml.rho_fr,cv.mtml.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTMLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        %[cv.mtaso.rho_fr,cv.mtaso.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTASOLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        %[cv.spmmtl.rho_sr,cv.spmmtl.lambda,cv.spmmtl.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMMTLearner', opts, param_range,lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        %[cv.spmtfl.rho_fr,cv.spmtfl.lambda,cv.spmtfl.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTFLearner', opts, param_range,lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        %[cv.spmtml.rho_fr,cv.spmtml.lambda,cv.spmtml.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTMLearner', opts, param_range,(10:5:25),kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        %[cv.spmtaso.rho_fr,cv.spmtaso.lambda,cv.spmtaso.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTASOLearner', opts, param_range,lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        
        
        save(sprintf('cv/%s_cv_%0.2f_%d.mat',dataset,trainSize,rId),'cv');
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
            case 'ITL'
                cv.itl.mu=1e-1;
                opts.kappa=1;
                opts.rho_l1=0;
                [W,C,clusters] = SharedMTLearner(Xtrain, Ytrain,cv.itl.mu,opts);
                if opts.verbose
                    fprintf('*');
                end
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
                %cv.mmtl.rho_sr=0.1;
                [W,C] = StructMTLearner(Xtrain, Ytrain,cv.mmtl.rho_sr,opts);
                if opts.verbose
                    fprintf('*');
                end
                %{
            case 'SPMMTL'
                % Self-paced Mean multi-task learner
                lambda=0.05;
                [W,C,tau] = SPMMTLearner(Xtrain, Ytrain,cv.spmmtl.rho_sr,cv.spmmtl.lambda,opts);
                if opts.verbose
                    fprintf('*');
                end
                result{m}.tau{rId}=tau;
                %}
            case 'MTFL'
                % Multi-task Feature Learner
                %cv.mtfl.rho_fr=0.01;
                [W,C, invD] = MTFLearner(Xtrain, Ytrain,cv.mtfl.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
                %{
            case 'SPMTFL'
                % Self-paced Multi-task Feature Learner
                %opts.rho_l1=0;
                lambda=0.05;
                [W,C,invD,tau] = SPMTFLearner(Xtrain, Ytrain,cv.spmtfl.rho_fr,cv.spmtfl.lambda,opts);
                if opts.verbose
                    fprintf('*');
                end
                result{m}.tau{rId}=tau;
            case 'MTML'
                % Manifold-based multi-task learner
                [W,C] = MTMLearner(Xtrain, Ytrain,cv.mtml.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'SPMTML'opts.kappa=3;
                % Self-pased Manifold-based multi-task learner
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
                opts.h=3;
                [W,C,theta] = SPMTASOLearner(Xtrain, Ytrain,cv.spmtaso.rho_fr,cv.spmtaso.lambda,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'CL'
                % Curriculum learner for multiple tasks
                cv.cl.rho_sr=0.01; % Use rho_sr based on the paper: 1/(2*sqrt(harmmean(N)));
                [W,C,taskOrder] = CLearner(Xtrain, Ytrain,cv.cl.rho_sr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case {'ELLA1','ELLA2','ELLA3','ELLA4'}
                % Curriculum learner for multiple tasks
                cv.ella.rho_fr=0.01;
                opts.h=3;
                opts.activeSelType=str2double(model(5));
                [W,C] = LifelongLearner(Xtrain, Ytrain,cv.ella.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
                %}
            case 'MTRL'
                % Multi-task Relationship Learner
                [W,C, Omega] = MTRLearner(Xtrain, Ytrain,cv.mtrl.rho_sr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTDict'
                % Multi-task Dictionary Learner
                opts.kappa=kappa;
                %cv.mtdict.rho_fr=1e-3;
                %cv.mtdict.rho_l1=10;
                [W,C,F,G] = MTDictLearner(Xtrain, Ytrain,cv.mtdict.rho_fr,cv.mtdict.rho_l1,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'SHAMO'
                %cv.shamo.rho_fr=1;
                opts.kappa=kappa;
                opts.rho_l1=0;
                [W,C,clusters] = SharedMTLearner(Xtrain, Ytrain,cv.shamo.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'CMTL'
                % Multi-task learning by Clustering (Barzillai)
                %cv.cmtl.rho_fr=0.01;
                [W,C,Omega] = MTByClusteringLearner(Xtrain, Ytrain,cv.cmtl.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTFactor'
                % Multi-task BiFactor Relationship Learner
                %cv.mtfactor.rho_fr1=1;
                %cv.mtfactor.rho_fr2=1000;
                opts.kappa=kappa;
                [W,C,F,G,Sigma, Omega] = BiFactorMTLearner(Xtrain, Ytrain,cv.mtfactor.rho_fr1,cv.mtfactor.rho_fr2,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'TriFactor'
                % Multi-task TriFactor Relationship Learner
                opts.kappa1=kappa1;
                opts.kappa2=kappa2;
                %cv.trifactor.rho_fr1=100;
                %cv.trifactor.rho_fr2=1000;
                [W,C,F,S,G,Sigma, Omega] = TriFactorMTLearner(Xtrain, Ytrain,cv.trifactor.rho_fr1,cv.trifactor.rho_fr2,opts);
                if opts.verbose
                    fprintf('*');
                end
             case 'SparseMatrixNorm'
                % Matrix Normal Multi-task Learner
                %cv.matnorm.rho_fr=0.1;
                opts.rho_l1=0.1;
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
                
                %cv.trifactor.rho_fr1=100;
                %cv.trifactor.rho_fr2=1000;
                opts.rho_l1=0.1;
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
        if(true)
            fprintf('Method: %s, RunId: %d, %s: %f \n',opts.method,rId,opts.scoreType,result{m}.score(rId));
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
    result{m}.meanScore=mean(result{m}.score);
    result{m}.stdScore=std(result{m}.score);
    result{m}.meanTaskScore=mean(result{m}.taskScore,2);
    result{m}.stdTaskScore=std(result{m}.taskScore,0,2);
    result{m}.runtime=result{m}.runtime/Nrun;
    fprintf('Method: %s, Mean %s: %f, Std %s: %f Runtime: %0.4f\n', result{m}.model,opts.scoreType,result{m}.meanScore,opts.scoreType,result{m}.stdScore,result{m}.runtime);
end

save(sprintf('results/%s_results_%0.2f.mat',dataset,trainSize),'result');



