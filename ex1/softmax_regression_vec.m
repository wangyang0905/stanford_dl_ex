function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  theta = [theta, zeros(n,1)];
  A = exp(X'*theta);
  I = sub2ind(size(A), 1:size(A,1), y);
  values = A(I);
  f = -sum(log(values'./sum(A,2)));
    
  p = bsxfun(@rdivide, A, sum(A,2));
  p0 = double(bsxfun(@eq, repmat(1:num_classes, m, 1), y'));
  diff = p0 - p;
  g = -X * diff(:,1:num_classes-1);
    
  g=g(:); % make gradient a vector for minFunc

