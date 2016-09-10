

clear;
rng('default');

% Read School data
load('data/school/school_data/school_b.mat')

K= length(task_indexes);

Nrun=10;
% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'STL','MMTL','SPMMTL','MTFL','SPMTFL','MTDict'}; % Choose subset: {'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'};
trainSize=0.75; % 10% 20% 30%

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
    result{m}.runtime=0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run Id - For Repeated Experiment
for rId=1:Nrun
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
    
    % Do Cross Validation for models
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
    for m=1:length(models)
        model=models{m};
        opts.method=model;
        tic
        switch model
            case 'STL'
                % Single Task Learner
                opts.mu=0.1;
                %opts.rho_l1=0;
                [W,C] = STLearner(Xtrain, Ytrain,opts);
            case 'MMTL'
                % Mean multi-task learner
                opts.mu=0;
                %opts.rho_l1=0;
                opts.rho_sr=1;
                R=eye (K) - ones (K) / K;
                Omega=R*R';
                [W,C] = StructMTLearner(Xtrain, Ytrain,Omega,opts);
            case 'SPMMTL'
                % Mean multi-task learner
                opts.mu=0;
                %opts.rho_l1=0;
                opts.rho_sr=1;
                lambda=100;
                [W,C,tau] = SPMMTLearner(Xtrain, Ytrain,lambda,opts);
            case 'MTFL'
                % Multi-task Feature Learner
                %opts.rho_l1=0;
                opts.rho_fr=0.1;
                [W,C] = MTFLearner(Xtrain, Ytrain,opts);
            case 'SPMTFL'
                % Multi-task Feature Learner
                %opts.rho_l1=0;
                opts.rho_fr=0.1;
                lambda=100;
                [W,C,invD,tau] = SPMTFLearner(Xtrain, Ytrain,lambda,opts);
            case 'MTRL'
                % Multi-task Relationship Learner
                %opts.rho_l1=0;
                opts.rho_sr=0.1;
                [W,C] = MTRLearner(Xtrain, Ytrain,opts);
            case 'MTDict'
                % Multi-task Dictionary Learner
                opts.rho_l1=3;
                opts.rho_fr=1;
                kappa=2;
                [W,C] = MTDictLearner(Xtrain, Ytrain,kappa,opts);
            case 'MTFactor'
                % Multi-task BiFactor Relationship Learner
                opts.rho_l1=0;
                opts.rho_fr1=0.0001;
                opts.rho_fr2=1;
                kappa=2;
                [W,C, Sigma, Omega] = BiFactorMTLearner(Xtrain, Ytrain,kappa,opts);
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





