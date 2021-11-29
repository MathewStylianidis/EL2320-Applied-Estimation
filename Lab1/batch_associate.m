% This function performs the maximum likelihood association and outlier detection.
% Note that the bearing error lies in the interval [-pi,pi)
%           mu_bar(t)           3X1
%           sigma_bar(t)        3X3
%           z(t)                2Xn
% Outputs: 
%           c(t)                1Xn
%           outlier             1Xn
%           nu_bar(t)           2nX1
%           H_bar(t)            2nX3
function [c, outlier, nu_bar, H_bar] = batch_associate(mu_bar, sigma_bar, z)

    % import global variables
    global map;
    global lambda_m;
    global Q;
    
    N = size(map, 2);
    obs_no = size(z, 2);
    
    % Pre-calculate 
    z_t_hat = zeros(2, N);
    H_t = zeros(2, 3, N);
    S_t_det = zeros(N);
    S_t_inv = zeros(2, 2, N);
    nu_bar = zeros(2 * N, 1);
    
    % For each landmark, pre-calculate z_hat, jacobians, and
    % covariances
    for j=1:N
        % Get z_hat
        z_t_hat(:, j) = observation_model(mu_bar, j);
        % Get jacobian
        H_t(:, :, j) = jacobian_observation_model(mu_bar, j, z_t_hat(:, j));
        % Calculate covariance
        S_t_j = H_t(:, :, j) * sigma_bar * H_t(:, :, j)' + Q;
        % Store covariance determinant and inverse
        S_t_det(j) = det(2 * pi * S_t_j);
        S_t_inv(:, :, j) = inv(S_t_j);
    end
    
    c = zeros(1, obs_no);
    outlier = zeros(1, obs_no);
    nu_bar = zeros(2 * obs_no, 1);
    H_bar = zeros(2 * obs_no, 3);
    for i=1:obs_no
        D = zeros(1, N);
        psi = zeros(1, N);
        % Calculate 2 x N innovation matrix for all landmarks 
        nu = bsxfun(@minus, z(:, i), z_t_hat);
        nu(2, :) = mod(nu(2, :) + pi, 2 * pi) - pi;
        for j=1:N
           D(1, j) = nu(:,j)' * S_t_inv(:, :, j) * nu(:, j);
           psi(1, j) = S_t_det(j)^(-1/2) * exp(-(1/2) * D(1, j));
        end
        [~, c(1,i)] = max(psi);
        outlier(1, i) = D(:, c(1, i)) > lambda_m;
        nu_bar(i * 2 - 1: i * 2, 1) = nu(:, c(1, i));
        H_bar(i * 2 - 1: i * 2, :) = H_t(:, :, c(1, i));
    end


           
end