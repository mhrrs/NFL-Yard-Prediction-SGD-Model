data = readtable('./data/processed_dataset.csv');

% subselect the columns we have pre-specified from the dataset
to_use = {'quarter' 'down' 'yardsToGo' 'defensiveTeamQt' ... 
'gameClockInSeconds' 'preSnapHomeScore' 'preSnapVisitorScore' ...
'signedScoreDiff' 'playResult' 'absoluteYardlineNumber' 'passPlay' ...
'offenseFormationQt' 'defendersInTheBox' 'heightInCm' 'weightInKg' ... 
'positionQt' 'ballCarrierAge'};
data_act = data(:,to_use); 
%{
% split the dataset into x and y 
x_use = {'quarter' 'down' 'yardsToGo' 'defensiveTeamQt' ... 
'gameClockInSeconds' 'preSnapHomeScore' 'preSnapVisitorScore' ...
'signedScoreDiff' 'absoluteYardlineNumber' 'passPlay' ...
'offenseFormationQt' 'defendersInTheBox' 'heightInCm' 'weightInKg' ... 
'positionQt' 'ballCarrierAge'};
x = data_act(:,x_use);
x = x{:, :}; 
%}

% Features to normalize
features_to_normalize = {'yardsToGo', 'gameClockInSeconds', 'preSnapHomeScore', ...
                         'preSnapVisitorScore', 'signedScoreDiff', 'absoluteYardlineNumber', ...
                         'heightInCm', 'weightInKg', 'ballCarrierAge'};

% Features not to normalize
features_to_not_normalize = {'quarter', 'down', 'defensiveTeamQt', 'passPlay', ...
                             'offenseFormationQt', 'defendersInTheBox', 'positionQt'};

% Extract features to be normalized
x_to_normalize = data_act(:, features_to_normalize);

% Normalize the selected features
x_normalized = varfun(@(x) (x - mean(x)) / std(x), x_to_normalize);

% Extract features not to be normalized
x_not_normalized = data_act(:, features_to_not_normalize);

% Combine normalized and non-normalized features
x = [x_not_normalized, x_normalized];
 
y = data_act(:,'playResult'); 
y = y{:, :}; 

% standardize our x variables (TODO: PROBABLY NEED TO STANDARDIZE Y TOO) 
% now hold on pal: x = normalize(x);
y_std = std(y);
y_mean = mean(y);
y = (y-y_mean) / y_std;

[n,d] = size(x);
seed = 2; rand('state', seed); randn('state', seed);
perm = randperm(n);
x = x(perm,:); 
y = y(perm);
lam = 1.5;
k = 15;

%REPLACE THIS ENTIRE SECTION WITH K_CROSS
x_sub = x(1:10000, :); 
y_sub = y(1:10000, :); 
y_test = y(11000:12000, :); 

coefficients = pcaRegression(x_sub, y_sub, 0.90); 

[msePerformance, rsePerformance, resids] = lassoKCrossValidation(x, y, lam, k, 2);
disp("y_std: " + y_std)
disp("MSE: " + msePerformance)
disp("RSE: " + rsePerformance*y_std)

% THIS CODE BLOCK IS SOLELY TO RETRIEVE TIMING INFO FOR VANILLA and SGD LASSO 
%pf = @() lassoKCrossValidation(x, y, lam, k, 1); 
%t_1 = timeit(pf); 
%pf = @() lassoKCrossValidation(x, y, lam, k, 2); 
%t_2 = timeit(pf); 

disp('Vanilla: ')
%disp(t_1)
disp("SGD: ")
%disp(t_2)


% -------------- Functions -----------------

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


function [msePerformance, rsePerformance, residuals] = lassoKCrossValidation(A, y, lam, k, t)
    % partition the dataset
    indices = crossvalind('Kfold',length(y),k);

    mse_performance = zeros(k,1);
    rse_performance = zeros(k,1);
    residuals = zeros(k,1);

    for i = 1:k
        %create boolean array to assign validation set to test
        test = (indices == i); train = ~test;
        A_train = A(train, :); y_train = y(train);
        A_test = A(test,:); y_test = y(test);

        % Convert A_train and A_test to type double if they are tables
        if istable(A_train)
            A_train = table2array(A_train);
        end
        if istable(A_test)
            A_test = table2array(A_test);
        end

        % vanilla lasso
        if t == 1
            xh = lassoAlg(A_train,y_train,lam);
        end

        % SGD lasso
        if t == 2
            xh = lassoAlgSGD(A_train, y_train, lam, 1e-6, 5000);
        end

        pred = A_test*xh;

        %mse
        mse_performance(i) = immse(y_test,pred);

        %rse
        residuals = y_test - pred;
        rss = sum(residuals.^2);
        n = length(y_test);
        p = size(A_train, 2);
        rse_performance(i) = sqrt(rss/(n-p-1));

        %residuals
        residuals = y_test - pred;

    end
    msePerformance = mean(mse_performance);
    rsePerformance = mean(rse_performance);

    %{
    plot(pred, residuals, 'o');
    xlabel('Predicted Values');
    ylabel('Residuals');
    title('Residual Plot');

    histogram(residuals);
    title('Histogram of Residuals');
    %}

end


function [exp_variance, eig_vec] = pcaFromScratch(X)
    %calculate the covariance matrix
    cov_x = cov(X); 

    %get the eigenvalues and eigenvectors
    [eig_vec, eig_val] = eig(cov_x); 

    %let's find out the explained variance
    %RESULT: WE WOULD NEED AT LEAST 10 OF THE EIGENVECTORS TO CAPTURE 90%
    %OF THE VARIANCE
    exp_variance = flip(nonzeros(eig_val / sum(eig_val(:)))); 

    %flip eig_vec to reflect eig_value order 
    eig_vec = flip(eig_vec); 
    
    %TODO: Create a Heatmap with the PCs 

    %TODO: Create a graph with the variances (SCREEPLOT) 
    figure
    pareto(exp_variance, 1); 
    xlabel('Principal Component'); 
    ylabel('Variance Explained (%)'); 
    title('Principal Component Analysis Results (Top 10 PCs)'); 

    %display explained variance
    display(exp_variance); 
end 

function coeff = pcaRegression(x_train, y_train, thresh) 
    % Convert A_train and A_test to type double if they are tables
    if istable(x_train)
        x_train = table2array(x_train);
    end
    if istable(y_train)
       y_train = table2array(y_train);
    end

    %get the priamry components
    [exp_variance, eig_vec] = pcaFromScratch(x_train); 
    
    %programatically find the # of PC's that explain the variance threshold
    %we've specified
    i = 0; 
    explanation = 0; 
    for i = 1:size(exp_variance, 1)
        explanation = explanation + exp_variance(i, :); 

        if explanation >= thresh
            break 
        end 
    end 

    % select the # of PC's using the found # above 
    sub_select_pc = eig_vec(:,1:i); 
    z = x_train * sub_select_pc; 

    %now use z as the input vector to simple lsr using the closed form
    %solution
    coeff = inv((z' * z)) * z' * y_train; 
end 