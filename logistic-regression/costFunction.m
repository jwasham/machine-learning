function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


hyp = sigmoid(X * theta);

J = -1 * 1/m * sum( y .* log(hyp) + ((1-y) .* log(1 - hyp)) );

%for j = 1:m
%    j_temp = sigmoid(X(j) * theta(j));
%    grad(j) = grad(j) + 1/m * sum((j_temp - y(j)) * X(:, j));
%endfor

grad = 1/m * X' * (sigmoid(X * theta) - y);

%for i = 1:m,
%    grad = grad + (y(i) - sigmoid(theta'*X(:,i)))* X(:,i);
%end;

%  ---- Inaccurate below ----

%sumError = 0;

%for i = 1:m
%    % hyp = sigmoid(X(i) * theta);
%    hyp = sigmoid(theta' * X(i));
%    sumError = sumError + ( (y(i) * log(hyp)) - ( (1 - y(i)) * log(1 - hyp)) );
%endfor

%J = -1 * 1/m * sumError;

% =============================================================

end
