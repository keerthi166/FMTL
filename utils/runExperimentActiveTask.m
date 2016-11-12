%%
% Do an active task selection experiment on the landmine data
%
% Copyright (C) Paul Ruvolo and Eric Eaton 2013
%
% This file is part of ELLA.
%
% ELLA is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% ELLA is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with ELLA.  If not, see <http://www.gnu.org/licenses/>.
function [W,C,model] = runExperimentActiveTask(X,Y,kappa,rho_fr,opts)

useLogistic = true;
T = length(Y);


for t = 1 : T
    X{t}(:,end+1) = 1;
end
d = size(X{1},2);
activeSelType=2; % cv with type 2 by default
if isfield(opts,'activeSelType')
    activeSelType=opts.activeSelType;
end

rho_fr=exp(-10);
model = initModelELLA(struct('k',kappa,...
    'd',d,...
    'mu',exp(-12),...
    'lambda',rho_fr,...
    'ridgeTerm',exp(-5),...
    'initializeWithFirstKTasks',true,...
    'useLogistic',useLogistic,...
    'lastFeatureIsABiasTerm',true));
learned = logical(zeros(length(Y),1));
unlearned = find(~learned);
for t = 1 : T
    % change the last input to 1 for random, 2 for InfoMax, 3 for Diversity, 4 for Diversity++
    idx = selectTaskELLA(model,{X{unlearned}},{Y{unlearned}},activeSelType);
    model = addTaskELLA(model,X{unlearned(idx)},Y{unlearned(idx)},unlearned(idx));
    learned(unlearned(idx)) = true;
    unlearned = find(~learned);
    % encode the unlearned tasks
    for tprime = 1 : length(unlearned)
        model = addTaskELLA(model,X{unlearned(tprime)},Y{unlearned(tprime)},unlearned(tprime),true);
    end
    
end
W=zeros(d-1,T);
C=zeros(1,T);
for tt=1:T
temp = model.L*model.S(:,tt);
W(:,tt)=temp(1:d-1,1);
C(tt)=temp(d,1);
end
end
