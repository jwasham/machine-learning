function [theta] = trainLinearReg(X, y, lambda)

% Trains linear regression given a dataset (X, y) and a
% regularization parameter lambda
%   [theta] = trainLinearReg(X, y, lambda) trains linear regression using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.

% Initialize Theta
initial_theta = zeros(size(X, 2), 1);

% Create "short hand" for the cost function to be minimized
costFunc = @(t) costFunction(X, y, t, lambda);

% Now, costFunc is a function that takes in only one argument
options = optimset('MaxIter', 200, 'GradObj', 'on');

% Minimize using fmincg
theta = fmincg(costFunc, initial_theta, options);

end