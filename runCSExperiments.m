

clear;
rng('default');

% Read Computer Survey
dataset='cs';
load('data/cs/computer_survey_190.mat')

[N,K]= size(Y);
Y=mat2cell(Y,N,ones(1,K));

Nrun=1;
% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'MTDict'};%{'ITL','STL','MMTL','MTFL','MTRL','SHAMO','MTDict','MTFactor','TriFactor'};%{'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'}; %{'STL','MMTL','SPMMTL','MTFL','SPMTFL','MTML','SPMTML','MTASO','SPMTASO'}; % Choose subset: {'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'};

trainSize=8;

opts.dataset=dataset;
opts.loss='least'; % Choose one: 'logit', 'least', 'hinge'
opts.scoreType='rmse'; % Choose one: 'perfcurve', 'class', 'mse', 'nmse'
opts.isHigherBetter=false;
opts.debugMode=true;
opts.verbose=true;
opts.tol=1e-6;
opts.maxIter=100;
opts.maxOutIter=25;
opts.cv=false;

kappa=5;
kappa1=5;
kappa2=3;

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
    split=cvpartition(N,'HoldOut',N-trainSize);
    % Train Set
    Xtrain=cell(1,K);
    Xtest=cell(1,K);
    tempX=X(split.training,:);
    for tt=1:K
       Xtrain{tt}=tempX; 
    end
    Ytrain=cellfun(@(y) y(split.training),Y,'UniformOutput',false);
    % Test Set
    tempX=X(split.test,:);
    for tt=1:K
       Xtest{tt}=tempX; 
    end
    Ytest=cellfun(@(y) y(split.test),Y,'UniformOutput',false);
    
    %load(sprintf('cv/%s_cv_%0.2f_%d.mat',dataset,trainSize,1));
    %cv=[];
    if (isempty(cv) && opts.cv)
        
        %------------------------------------------------------------------------
        %                   Cross Validation
        %------------------------------------------------------------------------
        if opts.verbose
            fprintf('CV');
        end
        lambda_range=[1e-3,1e-2,1e-1,1e-0,1e+1,1e+2,1e+3];
        %param_range=[1e-3,1e-2,1e-1,1e-0,1e+1,1e+2,1e+3];
        param_range= [1e-3,0.005,1e-2,0.05,1e-1,0.2,0.3,0.4,1e-0,1e+1,1e+2,1e+3];
        
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
        [cv.stl.mu,cv.stl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'STLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        kk=opts.kappa;
        opts.kappa=1;
        [cv.itl.mu,cv.itl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'SharedMTLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        opts.kappa=kk;
        [cv.mmtl.rho_sr,cv.mmtl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MMTLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.mtfl.rho_fr,cv.mtfl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTFLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.mtrl.rho_sr,cv.mtrl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTRLearner', opts, lambda_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.shamo.rho_fr,cv.shamo.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'SharedMTLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        %[cv.cmtl.rho_fr,cv.cmtl.perfMat]=CrossValidation1Param( Xtrain,Ytrain, 'MTByClusteringLearner', opts, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.mtdict.rho_fr,cv.mtdict.rho_l1,cv.mtdict.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'MTDictLearner', opts, lambda_range, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        opts.rho_l1=0;
        [cv.mtfactor.rho_fr1,cv.mtfactor.rho_fr2,cv.mtfactor.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'BiFactorMTLearner', opts, lambda_range, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
        [cv.trifactor.rho_fr1,cv.trifactor.rho_fr2,cv.trifactor.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'TriFactorMTLearner', opts, lambda_range, param_range,kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);

        
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
                cv.itl.mu=0.1;
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
                cv.mmtl.rho_sr=0.1;
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
                cv.mtfl.rho_fr=0.1;
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
                cv.mtrl.rho_sr=0.1;
                [W,C, Omega] = MTRLearner(Xtrain, Ytrain,cv.mtrl.rho_sr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTDict'
                % Multi-task Dictionary Learner
                opts.kappa=kappa;
                cv.mtdict.rho_fr=0.1;
                cv.mtdict.rho_l1=10;
                [W,C,F,G] = MTDictLearner(Xtrain, Ytrain,cv.mtdict.rho_fr,cv.mtdict.rho_l1,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'SHAMO'
                %cv.shamo.rho_fr=0.1;
                opts.kappa=kappa;
                opts.rho_l1=0;
                cv.shamo.rho_fr=0.1;
                [W,C,clusters] = SharedMTLearner(Xtrain, Ytrain,cv.shamo.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'CMTL'
                % Multi-task learning by Clustering (Barzillai)
                cv.cmtl.rho_fr=0.1;
                [W,C,Omega] = MTByClusteringLearner(Xtrain, Ytrain,cv.cmtl.rho_fr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTFactor'
                % Multi-task BiFactor Relationship Learner
                cv.mtfactor.rho_fr1=1e-3;
                cv.mtfactor.rho_fr2=10;
                
                opts.kappa=kappa;
                opts.rho_l1=0;
                [W,C,F,G,Sigma, Omega] = BiFactorMTLearner(Xtrain, Ytrain,cv.mtfactor.rho_fr1,cv.mtfactor.rho_fr2,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'TriFactor'
                % Multi-task TriFactor Relationship Learner
                opts.kappa1=kappa1;
                opts.kappa2=kappa2;
                cv.trifactor.rho_fr1=0.1;
                cv.trifactor.rho_fr2=10;
                opts.rho_l1=0;
                [W,C,F,S,G,Sigma, Omega] = TriFactorMTLearner(Xtrain, Ytrain,cv.trifactor.rho_fr1,cv.trifactor.rho_fr2,opts);
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
    %------------------------------------------------------------------------
    %                   Cross Validation
    %------------------------------------------------------------------------
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

        opts.method='cv';
        opts.h=2;
        
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
        [cv.spmtml.rho_fr,cv.spmtml.lambda,cv.spmtml.perfMat]=CrossValidation2Param( Xtrain,Ytrain, 'SPMTMLearner', opts, param_range,(25:5:50),kFold, 'eval_MTL', opts.isHigherBetter,opts.scoreType);
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
                cv.stl.mu=0.1;
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
                lambda=0.5;
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
                lambda=0.1;
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
                [W,C, Omega] = MTRLearner(Xtrain, Ytrain,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTDict'
                % Multi-task Dictionary Learner
                opts.rho_l1=0.1;
                opts.rho_fr=0.1;
                kappa=3;
                [W,C,F,G] = MTDictLearner(Xtrain, Ytrain,kappa,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'MTFactor'
                % Multi-task BiFactor Relationship Learner
                opts.rho_l1=3;
                opts.rho_fr1=1e-5;
                
                opts.rho_fr2=1e-4;
                kappa=3;
                [W,C,F,G,Sigma, Omega] = BiFactorMTLearner(Xtrain, Ytrain,kappa,opts);
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
            fprintf('Method: %s, RunId: %d, %s: %f \n',opts.method,rId,opts.scoreType,syn_kang_result(rId));
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
    fprintf('Method: %s, Mean %s: %f, Std %s: %f Runtime: %0.4f\n', result{m}.model,opts.scoreType,result{m}.meanScore,opts.scoreType,result{m}.stdScore,result{m}.runtime);
end
save(sprintf('results/%s_results_%0.2f.mat',dataset,trainSize),'result');

