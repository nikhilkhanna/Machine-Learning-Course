function J = costFunctionJ(X, y, theta)

% x is the "design matrix" containing our training examples
% y is the class labels

m = size(X,1); % number of training examples
predictions = X * theta %predictions of  hypothesis
sqrErrors = (predictions-y).^2;
J = 1/(2*m) * sum(sqrErrors);