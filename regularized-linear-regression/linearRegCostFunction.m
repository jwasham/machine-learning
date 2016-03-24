function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hyp = X * theta; % 12x1 matrix

% --------- cost function ---------

% y is 12x1

sum_sq_error = sum((hyp - y).^2);

reg = ((lambda/(2*m)) * sum([0; theta(2:end)] .^ 2));

J = (1/(2*m) * sum_sq_error) + reg;

% ------------ gradient ------------

sum_error = X' * (hyp - y);

grad = ((1/m) * sum_error) + ((lambda/m) * [0; theta(2:end)]);

% =========================================================================

grad = grad(:);

end
