

clear;
rng('default');

% Read Synthetic data from Kang et. al.
load('data/synthetic/syn_3group_kang.mat')

K= nTask;

Nrun=10;
% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'STL','MMTL','SPMMTL','MTFL','SPMTFL'}; % Choose subset: {'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'};

trainSize=15;

X=cell(1,K);
Y=cell(1,K);


for tt=1:K
    
    X{tt}=[trainX(trainObsTaskMap==tt,:);testX(testObsTaskMap==tt,:)];
    Y{tt}=[trainY(trainObsTaskMap==tt);testY(testObsTaskMap==tt)];
end
% Add intercept
%X = cellfun(@(x) [ones(size(x,1),1),x], X, 'uniformOutput', false);
N=cellfun(@(x) size(x,1),X);


opts.loss='least'; % Choose one: 'logit', 'least', 'hinge'
opts.scoreType='rmse'; % Choose one: 'perfcurve', 'class', 'mse', 'nmse'
opts.debugMode=false;
opts.verbose=true;
opts.tol=1e-5;
opts.maxIter=100;
opts.maxOutIter=50;



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
    if opts.verbose
        fprintf('Run %d (',rId);
    end
    %------------------------------------------------------------------------
    %                   Train-Test Split
    %------------------------------------------------------------------------
    
    % Split Data into train and test
    split=cellfun(@(n) cvpartition(n,'HoldOut',n-trainSize),num2cell(N),'UniformOutput',false);
    % Train Set
    Xtrain=cellfun(@(x,split_t) {x(split_t.training,:)}, X, split);
    Ytrain=cellfun(@(y,split_t) {y(split_t.training,:)}, Y,split);
    % Test Set
    Xtest=cellfun(@(x,split_t) {x(split_t.test,:)}, X, split);
    Ytest=cellfun(@(y,split_t) {y(split_t.test,:)}, Y,split);
    
    % Normalize Data if needed
    %[Xtrain,~,meanX,stdX] = normalizeMultitaskData(Xtrain);
    % Normalize Test Data
    %[Xtest,~,~,~] = normalizeMultitaskData(Xtest,[],meanX,stdX);
    
    
    %------------------------------------------------------------------------
    %                   Cross Validation
    %------------------------------------------------------------------------
    %{
        opts.mu=0;
        opts.rho_l1=0;
        opts.rho_sr=0.1;
        opts.rho_fr=0.1;
        
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
                cv.mmtl.rho_sr=0.1;
                R=eye (K) - ones (K) / K;
                opts.Omega=R*R';
                [W,C] = StructMTLearner(Xtrain, Ytrain,cv.mmtl.rho_sr,opts);
                if opts.verbose
                    fprintf('*');
                end
            case 'SPMMTL'
                % Self-paced Mean multi-task learner
                cv.spmmtl.rho_sr=0.1;
                lambda=0.05;
                [W,C,tau] = SPMMTLearner(Xtrain, Ytrain,cv.spmmtl.rho_sr,lambda,opts);
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
                %opts.rho_l1=0;
                cv.spmtfl.rho_fr=0.1;
                lambda=0.05;
                [W,C,invD,tau] = SPMTFLearner(Xtrain, Ytrain,cv.spmtfl.rho_fr,lambda,opts);
                if opts.verbose
                    fprintf('*');
                end
                result{m}.tau{rId}=tau;
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




