%% Runs our training data through linear regression to fit the data

%% Initialization
clear ; close all; clc

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from linRegData:
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('linRegData.mat');

% m = Number of examples
m = size(X, 1);

% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flow (y)');

%% =========== Part 2: Regularized Linear Regression Cost =============

theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Regularized Linear Regression Gradient =============

theta = [1 ; 1];
[J, grad] = costFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 4: Train Linear Regression =============

%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearRegression([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flow (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;