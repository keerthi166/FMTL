function [ projX2,result] = computeManifoldProjections(X1,X2,K,nIter,runScriptPath,paramFilePath,rId)
% computeManifoldProjections computes Manifold using principle surfaces and
% principle curves from the given set of vectors X2
% X1 - Vector of data from which the manifold is generated
% X2 - Vector of data for which the projection to the manifold is to be
% computed
% K - number of neighbours in the Knn regression
% nIter - number of iterations in computing the manifold
% runScriptPath - Location where the R script is available
% paramFilePath - Location whether the intermediatory files are stored
% rId - randomId to be used for storing intermediate files

projX2=[]; 
dr='isomap'; % Dimiensionality reduction initalization
infile=sprintf('%s/input_%s.mat',paramFilePath,rId);
outfile=sprintf('%s/manifold_%s.mat',paramFilePath,rId);
logfile=sprintf('%s/log_%s.txt',paramFilePath,rId);

if exist(outfile,'file')
   delete(outfile); 
end

% Save input param files
save(infile,'X1','X2','K','nIter', 'dr');
rCmdStr=sprintf('Rscript %s/computeManifold.R %s %s > %s',runScriptPath,infile,outfile,logfile);
[status,result]=system(rCmdStr);
if status ~=0
    warning('Unable to compute manifold projections.');
    return;
end

% Load the output file
manifold=load(outfile);
projX2=manifold.proj';
end

