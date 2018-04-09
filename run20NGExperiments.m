
% 20 Newsgroup data
clear;
rng('default');

% Read Sentiment data
dataset='newsgroup';
%load('data/sentiment/sentiment_analysis.mat')
load('data/20ng/reduced-newsgroup-data.mat')

np=10;
K=np;
temp=randperm(20)-1;
selGrps=temp(1:np*2);
selGrpsNames=taskInfo(selGrps+1,:);
XOrig=cell(1,np);
YOrig=cell(1,np);

ss=1;
for ii=1:np
    selPosId=target==selGrps(ss);
    selNegId=target==selGrps(ss+1);
    if ii==9
        disp ii
    end
    XOrig{ii}=[feat(selPosId,cv_featsel);feat(selNegId,cv_featsel)];
    YOrig{ii}=[ones(sum(selPosId),1);-1*ones(sum(selNegId),1)];
    ss=ss+2;
end
XOrig = cellfun(@(x) [ones(size(x,1),1),x], XOrig, 'uniformOutput', false);

P=size(XOrig{1},2);

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

Nrun=10;
trainSize=0.1;
kappa=5;
kappa1=15;
kappa2=5;

models={'MTDict','MTFactor','TriFactor'};
Fmats=cell(length(models),K);
Smats=cell(length(models),K);

for tt=1:K
    tt
    % Keep tt task as target domain
    srcTasks=setdiff(1:K,tt);
    X=XOrig(srcTasks);
    Y=YOrig(srcTasks);
    
    
    for m=1:length(models)
        model=models{m};
        opts.rho_l1=0;
        switch model
            case 'STL'
                F=eye(P);
                S=eye(P);
            case 'MTFactor'
                opts.kappa=kappa;
                cv.mtdict.rho_fr=1e-1;
                cv.mtdict.rho_l1=1e-1;
                [W,C,F,G] = MTDictLearner(X, Y,cv.mtdict.rho_fr,cv.mtdict.rho_l1,opts);
                S=eye(kappa);
            case 'MTDict'
                % Multi-task BiFactor Relationship Learner
                cv.mtfactor.rho_fr1=1e-1;
                cv.mtfactor.rho_fr2=1e-1;
                opts.kappa=kappa;
                [W,C,F,G,Sigma, Omega] = BiFactorMTLearner(X, Y,cv.mtfactor.rho_fr1,cv.mtfactor.rho_fr2,opts);
                
                S=eye(kappa);
            case 'TriFactor'
                % Multi-task TriFactor Relationship Learner
                opts.kappa1=kappa1;
                opts.kappa2=kappa2;
                cv.trifactor.rho_fr1=1e-1;
                cv.trifactor.rho_fr2=1e-1;
                [W,C,F,S,G,Sigma, Omega] = TriFactorMTLearner(X, Y,cv.trifactor.rho_fr1,cv.trifactor.rho_fr2,opts);
                
        end
        Fmats{m,tt}=F;
        Smats{m,tt}=S;
        
    end
    
end

 % Initilaization
    result=cell(length(models),1);
    for m=1:length(models)
        result{m}.score=zeros(Nrun,K);
    end
    
for rId=1:Nrun
    rId
    
    for tt=1:K
        % Keep tt task as target domain
        Xtarget=XOrig(tt);
        Ytarget=YOrig(tt);
        
        split=cvpartition(Ytarget{1},'HoldOut',round(length(Ytarget{1})-trainSize*length(Ytarget{1})));
        for m=1:length(models)
            model=models{m};
            
            F=Fmats{m,tt};
            S=Smats{m,tt};
            Xtrain=Xtarget{1}(split.training,:)*(F*S);
            Ytrain=Ytarget{1}(split.training);
            Xtest=Xtarget{1}(split.test,:)*(F*S);
            Ytest=Ytarget{1}(split.test);
            
            %wt= (Xtrain'*Xtrain+0.1)\(Xtrain'*Ytrain);
            [wt,c] = STLearner({Xtrain}, {Ytrain},0.1,opts);
            % Compute Area under the ROC curve & Accuracy
            [result{m}.score(rId,tt),~]=eval_MTL({Ytest}, {Xtest}, wt, c,[], opts.scoreType);
            if(opts.debugMode)
                fprintf('Method: %s, RunId:%d domain: %d, %s: %f \n',model,rId,tt,opts.scoreType,result{m}.score(rId,tt));
            end
        end
    end
end


%%% Per TrainSize Stats
if opts.verbose
    fprintf('Results: \n');
end

for m=1:length(models)
    result{m}.meanScore=mean(result{m}.score);
    result{m}.stdScore=std(result{m}.score);
    for tt=1:K-1
       fprintf('%0.2f (%0.2f) &',result{m}.meanScore(tt),result{m}.stdScore(tt)); 
    end
    fprintf('%0.2f (%0.2f) \n',result{m}.meanScore(K),result{m}.stdScore(K)); 
    %disp([result{m}.meanScore',std(result{m}.score)'])
    %fprintf('Method: %s, Mean %s: %f, Std %s: %f\n', result{m}.model,opts.scoreType,result{m}.meanScore,opts.scoreType,result{m}.stdScore);
end






