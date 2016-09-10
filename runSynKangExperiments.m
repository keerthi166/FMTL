


rng('default');

% Read Synthetic data from Kang et. al.
load('data/synthetic/syn_3group_kang.mat')

K= nTask;

Nrun=10;
% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'MMTL'};%{'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'}; % Choose subset: {'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'};

Ntrain=15;

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
opts.tol=1e-5;
opts.maxIter=100;
opts.maxOutIter=50;


result=cell(length(models),1);
for m=1:length(models)
    model=models{m};
    opts.method=model;
    
    
    
    % Initilaization
    syn_kang_result=zeros(Nrun,1);
    syn_kang_result_task=zeros(K,Nrun);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run Experiment
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Run Id - For Repeated Experiment
    for rId=1:Nrun
        
        % Split Data into train and test
        split=cellfun(@(n) cvpartition(n,'HoldOut',n-Ntrain),num2cell(N),'UniformOutput',false);
        
        Xtrain=cellfun(@(x,split_t) {x(split_t.training,:)}, X, split);
        Ytrain=cellfun(@(y,split_t) {y(split_t.training,:)}, Y,split);
        Xtest=cellfun(@(x,split_t) {x(split_t.test,:)}, X, split);
        Ytest=cellfun(@(y,split_t) {y(split_t.test,:)}, Y,split);
        
        
        
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
                opts.rho_sr=0.1;
                R=eye (K) - ones (K) / K;
                Omega=R*R';
                [W,C] = StructMTLearner(Xtrain, Ytrain,Omega,opts);
            case 'MTFL'
                % Multi-task Feature Learner
                %opts.rho_l1=0;
                opts.rho_fr=0.1;
                [W,C, invD] = MTFLearner(Xtrain, Ytrain,opts);
            case 'MTRL'
                % Multi-task Relationship Learner
                %opts.rho_l1=0;
                opts.rho_sr=0.1;
                [W,C, Omega] = MTRLearner(Xtrain, Ytrain,opts);
            case 'MTDict'
                % Multi-task Dictionary Learner
                opts.rho_l1=0.1;
                opts.rho_fr=0.1;
                kappa=3;
                [W,C,F,G] = MTDictLearner(Xtrain, Ytrain,kappa,opts);
            case 'MTFactor'
                % Multi-task BiFactor Relationship Learner
                opts.rho_l1=3;
                opts.rho_fr1=1e-5;
                opts.rho_fr2=1e-4;
                kappa=3;
                [W,C,F,G,Sigma, Omega] = BiFactorMTLearner(Xtrain, Ytrain,kappa,opts);
        end
        
        
        % Compute Area under the ROC curve & Accuracy
        [syn_kang_result(rId),syn_kang_result_task(:,rId)]=eval_MTL(Ytest, Xtest, W, C,[], opts.scoreType);
        if(opts.debugMode)
            fprintf('Method: %s, RunId: %d, %s: %f \n',opts.method,rId,opts.scoreType,syn_kang_result(rId));
        end
    end
    %%% Per TrainSize Stats
    fprintf('Method: %s, Mean %s: %f, Std %s: %f \n', opts.method,opts.scoreType,mean(syn_kang_result),opts.scoreType,std(syn_kang_result));
    
    % Store 2K+4 elements:
    result{m} = [mean(syn_kang_result),std(syn_kang_result),mean(syn_kang_result_task,2)', std(syn_kang_result_task,0,2)'];
    
    
end



