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

% baseline lassAlg function
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


function avgPerformance = lassoKCrossValidation(A, y, lam, k)
    % partition the dataset
    indices = crossvalind('Kfold',length(y),k);
    
    performance = zeros(k,1);
    
    for i = 1:k
        %create boolean array to assign validation set to test
        test = (indices == i); train = ~test;
        A_train = A(train, :); y_train = y(train);
        A_test = A(test,:); y_test = y(test);

        xh = lassoAlg(A_train,y_train,lam);

        pred = A_test*xh;

        performance(i) = immse(y_test,pred);

    end
    disp(performance)
    avgPerformance = mean(performance);

end







