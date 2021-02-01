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

H = X * theta; 
% Figure out why X is ahead of theta 
value_1 = sum((H-y).^2);
value_2 = sum((theta(2:end).^2));

J = (1/(2*m))*(value_1) + (lambda/(2*m))*(value_2);

grad = (1/m) * X'*(H-y) + (lambda/m) * [0;theta(2:end)];
% Why [0;theta(2:end)] rather than theta(2:end)
% =========================================================================
grad = grad(:);

end
