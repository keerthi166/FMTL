

clear;
rng('default');

% Read Landmine data
dataset='usps_1vsall';
load('data/usps/split1_usps_1000train.mat')
X=[digit_trainx;digit_testx];
Y=[digit_trainy;digit_testy];

classes=unique(Y);
nclass= length(classes);
K=nclass;

% Add intercept
%X = cellfun(@(x) [ones(size(x,1),1),x], X, 'uniformOutput', false);
% Change the labelspace from {0,1} to {-1,1}


Nrun=1;
% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'STL','MTFL','SPMTFL','MTML','SPMTML','MTASO','SPMTASO'}; % Choose subset: {'STL','MMTL','MTFL','MTRL','MTDict','MTFactor'};
trainSizes=1000;

opts.dataset=dataset;
opts.loss='logit'; % Choose one: 'logit', 'least', 'hinge'
opts.debugMode=false;
opts.scoreType='multiclass'; % Choose one: 'perfcurve', 'class', 'mse', 'nmse'
opts.verbose=true;
opts.isHigherBetter=true;
opts.tol=1e-5;
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
        % One-vs-All setting
        Xtrain=[];
        Ytrain=cell(1,K);
        for tt=1:K
            Xtrain=Xrawtrain;
            Ytrain{tt}=(((Yrawtrain==tt)*2)-1);
        end
        
        
        
        
        % Normalize Data if needed
        %[Xtrain,~,meanX,stdX] = normalizeMultitaskData(Xtrain);
        % Normalize Test Data
        %[Xtest,~,~,~] = normalizeMultitaskData(Xtest,[],meanX,stdX);
        
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
                    cv.stl.mu=1e-3;
                    [W,C] = STLearner(Xtrain, Ytrain,cv.stl.mu,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'MMTL'
                    % Mean multi-task learner
                    R=eye (K) - ones (K) / K;
                    opts.Omega=R*R';
                    cv.mmtl.rho_sr=1e-3;
                    [W,C] = StructMTLearner(Xtrain, Ytrain,cv.mmtl.rho_sr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMMTL'
                    % Self-paced Mean multi-task learner
                    lambda=1;
                    cv.spmmtl.rho_sr=1e-3;
                    cv.spmmtl.lambda=2;
                    [W,C,tau] = SPMMTLearner(Xtrain, Ytrain,cv.spmmtl.rho_sr,cv.spmmtl.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                    result{m}.tau{rId}=tau;
                case 'MTFL'
                    % Multi-task Feature Learner
                    cv.mtfl.rho_fr=1e-3;
                    [W,C, invD] = MTFLearner(Xtrain, Ytrain,cv.mtfl.rho_fr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMTFL'
                    % Self-paced Multi-task Feature Learner
                    lambda=1;
                    cv.spmtfl.rho_sr=1e-3;
                    cv.spmtfl.lambda=2;
                    [W,C,invD,tau] = SPMTFLearner(Xtrain, Ytrain,cv.spmtfl.rho_sr,cv.spmtfl.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                    result{m}.tau{rId}=tau;
                case 'MTML'
                    % Manifold-based multi-task learner
                    cv.mtml.rho_fr=1e-3;
                    [W,C] = MTMLearner(Xtrain, Ytrain,cv.mtml.rho_fr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMTML'
                    % Self-pased Manifold-based multi-task learner
                    cv.spmtml.rho_sr=1e-3;
                    cv.spmtml.lambda=5;
                    [W,C] = SPMTMLearner(Xtrain, Ytrain,cv.spmtml.rho_sr,cv.spmtml.lambda,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'MTASO'
                    % multi-task learner with Alternating Structure
                    % Optimization
                    opts.h=10;
                    cv.mtaso.rho_fr=1e-3;
                    [W,C,theta] = MTASOLearner(Xtrain, Ytrain,cv.mtaso.rho_fr,opts);
                    if opts.verbose
                        fprintf('*');
                    end
                case 'SPMTASO'
                    % multi-task learner with Alternating Structure
                    % Optimization
                    opts.h=10;
                    cv.spmtaso.rho_sr=1e-3;
                    cv.spmtaso.lambda=2;
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
            Yrawtest=Y(split.test);
            Ytest=cell(1,K);
            for tt=1:K
                Ytest{tt}=(((Yrawtest==tt)*2)-1);
            end
            [result{m}.score(rId,nt),result{m}.taskScore(:,rId,nt)] = eval_MTL(Ytest, Xtest, W, C,[], opts.scoreType);
            
            
            if(opts.verbose)
                fprintf('Method: %s, Ntrain: %d, RunId: %d, AUC: %f \n',opts.method,trainSize,rId,result{m}.score(rId,nt));
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




