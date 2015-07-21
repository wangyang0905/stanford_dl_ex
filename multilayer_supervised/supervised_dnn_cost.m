function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%% Does all the work of cost / gradient computation

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% forward prop
%%% YOUR CODE HERE %%%
m = size(data,2);
hAct{1} = stack{1}.W * data + stack{1}.b * ones(1,m);
hAct{1} = 1./(1 + exp(-hAct{1}));
hAct{2} = stack{2}.W * hAct{1} + stack{2}.b * ones(1,m);
hAct{2} = bsxfun(@rdivide, exp(hAct{2}), sum(exp(hAct{2}),1));
pred_prob = hAct{2};


%% return here if only predictions desired.
if po
  cost = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
K = ei.output_dim;
ind = double(bsxfun(@eq, repmat([1:K]',1,m), labels'));
cost = -sum(sum(ind.*log(pred_prob),1),2);


%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
delta = -(ind-pred_prob);
gradStack{2}.W = delta * hAct{1}';
gradStack{2}.b = delta * ones(m,1);
delta = (stack{2}.W' * delta).*hAct{1}.*(1 - hAct{1});
gradStack{1}.W = delta * data';
gradStack{1}.b = delta * ones(m,1);


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
cost = cost + 0.5*ei.lambda*sum(sum(stack{1}.W.*stack{1}.W,1),2) ...
            + 0.5*ei.lambda*sum(sum(stack{2}.W.*stack{2}.W,1),2);
gradStack{1}.W = gradStack{1}.W + ei.lambda*stack{1}.W;       
gradStack{2}.W = gradStack{2}.W + ei.lambda*stack{2}.W;

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



