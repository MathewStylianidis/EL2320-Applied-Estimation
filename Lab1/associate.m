% This function performs the maximum likelihood association and outlier detection given a single measurement.
% Note that the bearing error lies in the interval [-pi,pi)
%           mu_bar(t)           3X1
%           sigma_bar(t)        3X3
%           z_i(t)              2X1
% Outputs: 
%           c(t)                1X1
%           outlier             1X1
%           nu^i(t)             2XN
%           S^i(t)              2X2XN
%           H^i(t)              2X3XN
function [c, outlier, nu, S, H] = associate(mu_bar, sigma_bar, z_i)
    % Import global variables
    global Q % measurement covariance matrix | 1X1
    global lambda_m % outlier detection threshold on mahalanobis distance | 1X1
    global map % map | 2Xn
           
    N = size(map, 2);
    H = zeros(2, 3, N);
    S = zeros(2, 2, N);
    nu = zeros(2, N);
    psi = zeros(1, N);
    D = zeros(N);

    for j=1:N
        % Calculate likelihood 
        z_j_hat = observation_model(mu_bar, j);
        H(:, :, j) = jacobian_observation_model(mu_bar, j, z_j_hat);
        S(:, :, j) = H(:, :, j) * sigma_bar * H(:, :, j)' + Q;
        % Calculate innovation and restrict angle to [-pi, pi)
        nu(:, j) = z_i - z_j_hat;
        nu(2, j) = mod(nu(2, j) + pi,  2 * pi) - pi;
        psi(j) = (det(2 * pi * S(:, :, j))^(-1 / 2)) * ...
           exp((-1 / 2) * nu(:, j)' * (S(:, :, j)^(-1)) * nu(:, j));
        % Calculate mahalanobis distance
        D(j) = nu(:, j)' * S(:, :, j)^(-1) * nu(:, j);
    end
    
    [~, c] = max(psi);
    if D(c) > lambda_m
        outlier = 1;
    else
        outlier = 0;
    end

end