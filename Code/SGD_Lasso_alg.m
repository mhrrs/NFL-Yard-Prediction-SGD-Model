% import data
data = load('processed_dataset.mat');

% refine for nfl dataset
x = data.normalized_data(:, 1:14);
y = data.normalized_data(:, 15);
[n,d] = size(x);
perm = randperm(n);
x = x(perm,:); y = y(perm);
seed = 2; rand('state', seed); randn('state', seed);
lam = 1.5;
k = 10;

performance = lassoKCrossValidation(x, y, lam, k);
disp("PERFORMANCE VALUES:")
disp(performance)

% baseline lassoAlg vanillaGD function
function xh = lassoAlg(A,y,lam)     
    xnew = rand(size(A,2),1);
    xold = xnew + ones(size(xnew));
    loss = xnew - xold;
    thresh = 10e-3;

    while norm(loss) > thresh
        xold = xnew;
        for i = 1:length(xnew)
            a = A(:,i);
            p = (norm(a,2))^2;
            t = a*xnew(i) + y - A*xnew;
            q = a'*t;
            xnew(i) = (1/p) * sign(q) * max(abs(q)-lam, 0);
        end
        loss = xnew - xold;
    end
    xh = xnew;
end


% lassoAlg SGD function:
function xh = lassoAlgSGD(A, y, lam, learningRate, maxEpochs)     
    xnew = rand(size(A,2),1);
    xold = xnew + ones(size(xnew));
    thresh = 10e-3;
    n = size(A,2);

    for e = 1:maxEpochs
        for i = randperm(n) % Iterate over features in random order
            a = A(:,i);
            p = (norm(a,2))^2;
            t = a*xnew(i) + y - A*xnew;
            q = a'*t;
            gradient = (1/p) * sign(q) * max(abs(q)-lam, 0);
            xnew(i) = xnew(i) - learningRate * gradient;
        end
        
        % Check convergence
        loss = xnew - xold;
        if norm(loss)<thresh
            break
        end
        xold = xnew;

    end
    xh = xnew;
end



% this is the training function:
function avgPerformance = lassoKCrossValidation(A, y, lam, k)
    % partition the dataset
    indices = crossvalind('Kfold',length(y),k);
    
    performance = zeros(k,1);
    
    for i = 1:k
        %create boolean array to assign validation set to test
        test = (indices == i); train = ~test;
        A_train = A(train, :); y_train = y(train);
        A_test = A(test,:); y_test = y(test);

        % learning rate and epochs needs to be adjusted
        xh = lassoAlgSGD(A_train,y_train,lam, 1e-6, 1000);

        pred = A_test*xh;

        performance(i) = immse(y_test,pred);
        disp("K " + i + " complete")

    end
    disp(performance)
    avgPerformance = mean(performance);

end







