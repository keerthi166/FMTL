function [ best_param1,best_param2,perform_mat] = CrossValidation2Param...
    ( X, Y, obj_func_str, obj_func_opts, param1_range,param2_range, cv_fold, eval_func_str, higher_better,errorType)
%% Function CROSSVALIDATION2PARAM
%   Model selection (cross validation) for 1 parameter
%       multi-task learning functions
%   For usage see test_script.m
%
%% INPUT
%   X:             input data
%   Y:             output data
%   obj_func_str:  1-parameter optimization algorithms
%   param1_range:   the range of the parameter1. array
%   param2_range:   the range of the parameter2. array
%   cv_fold:       cross validation fold
%   eval_func_str: evaluation function:
%       signature [performance_measure] = eval_func(Y_test, X_test, W_learnt)
%   higher_better: if the performance is better given
%           higher measurement (e.g., Accuracy, AUC)
%% OUTPUT
%   best_param1:  best parameter in the given parameter range 1
%   best_param2:  best parameter in the given parameter range 2
%   perform_mat: the average performance for every parameter in the
%                parameter ranges.
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%


eval_func = str2func(eval_func_str);
obj_func  = str2func(obj_func_str);


% compute sample size for each task
task_num = length(Y);
% performance vector
perform_mat = zeros(length(param1_range),length(param2_range));

% begin cross validation
fprintf('[');
if iscell(X)
    if strcmp(obj_func_opts.loss,'least')
        cv=cellfun(@(y) cvpartition(size(y,1),'KFold',cv_fold),Y,'UniformOutput',false);
    else
        cv=cellfun(@(y) cvpartition(y,'KFold',cv_fold),Y,'UniformOutput',false);
    end
else
    cv= cvpartition(size(X,1),'KFold',cv_fold); % Shared data representation
end
for cv_idx = 1: cv_fold
    fprintf('-');
    
    if iscell(X)
        % buid cross validation data splittings for each task.
        cv_Xtr = cell(1,task_num);
        cv_Xte = cell(1,task_num);
        
        cv_Ytr = cell(1,task_num);
        cv_Yte = cell(1,task_num);
        
        for t = 1: task_num
            te_idx = test(cv{t},cv_idx);
            tr_idx = training(cv{t},cv_idx);
            
            cv_Xtr{t} = X{t}(tr_idx, :);
            cv_Xte{t} = X{t}(te_idx, :);
            
            cv_Ytr{t} = Y{t}(tr_idx, :);
            cv_Yte{t} = Y{t}(te_idx, :);
        end
    else
        te_idx = test(cv,cv_idx);
        tr_idx = training(cv,cv_idx);
        cv_Xtr=X(tr_idx, :);
        cv_Xte = X(te_idx, :);
        
        cv_Ytr = cellfun(@(y) y(tr_idx),Y,'UniformOutput',false);
        cv_Yte = cellfun(@(y) y(te_idx),Y,'UniformOutput',false);
        
    end
    for p1_idx = 1: length(param1_range)
        for p2_idx = 1: length(param2_range)
            [W,C] = obj_func(cv_Xtr, cv_Ytr, param1_range(p1_idx),param2_range(p2_idx), obj_func_opts);
            err=eval_func(cv_Yte, cv_Xte, W,C,[],errorType);
            if isnan(err)
                err=0;
                %fprintf('%d-%d',p1_idx,p2_idx);
            end
            perform_mat(p1_idx,p2_idx) = perform_mat(p1_idx,p2_idx) + err;
        end
    end
end
perform_mat = perform_mat./cv_fold;
fprintf(']')


if( higher_better)
    %[~,best_idx] = max(perform_mat);
    [row,col] = find(perform_mat == max(perform_mat(:)));
else
    %[~,best_idx] = min(perform_mat);
    [row,col] = find(perform_mat == min(perform_mat(:)));
end

%best_param = param_range(best_idx);
best_param1=param1_range(row(end));
best_param2=param2_range(col(end));



end
