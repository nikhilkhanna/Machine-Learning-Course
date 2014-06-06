function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
% x is 5000 * 400
%theta 1 is 25*401
%theta 2 is 10*26

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.


X = [ones(m, 1) X];
% x is now 5000 * 401
Hidden1 = sigmoid(Theta1*X');
%hidden1 is 25*5000
Hidden1 = [ones(1, m); Hidden1];
%hidden1 is 26*5000

Output = sigmoid(Theta2*Hidden1);
%output = 10 * 5000
fprintf('output: %f', size(Output));

[maxVal maxInd] = max(Output',[],2);
p = maxInd;

%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
