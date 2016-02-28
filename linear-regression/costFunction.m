function [J, grad] = costFunction(X, y, theta, lambda)

%Compute cost and gradient for regularized linear
%regression with multiple variables

%   [J, grad] = costFunction(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% Returns the following variables
J = 0;
grad = zeros(size(theta));

% =========================================================================

hyp = X * theta; % 12x1 matrix

% ------- linear regression ------

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