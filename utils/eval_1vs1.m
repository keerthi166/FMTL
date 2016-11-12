function [ score,taskScore ] = eval_1vs1( Xtest,Ytest,W,C,classes )
%EVAL_1VS1 Evaluation function for one-vs-one classification for Accuracy

nclass=length(classes);
K=(nclass-1)*nclass/2;
Yhat=zeros(length(Ytest),K);
tt=1;
for class1=1:length(classes)-1
    for class2=class1+1:length(classes)
        temp=sign(Xtest*W(:,tt)+C(tt));
        Yhat(temp==1,tt)=class1;
        Yhat(temp==-1,tt)=class2;
        tt=tt+1;
    end
end


[~,Ypred] = max(hist(Yhat', nclass));
Ypred=Ypred';
if(size(Yhat, 2) == 1)
    Ypred = Yhat';
end

corr=sum(Ypred==Ytest);
score=corr/length(Ytest);
taskScore=zeros(nclass,1);
for t=1:nclass
    Ypred_t=Ypred(Ytest==classes(t));
    Y_t=Ytest(Ytest==classes(t));
    
    corr=sum(Ypred_t==Y_t);
    taskScore(t)=corr/length(Y_t);
end

end

