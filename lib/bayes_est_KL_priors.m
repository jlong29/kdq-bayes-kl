function [Q1,Q2,KL_std] = bayes_est_KL_priors(data1,data2,alpha)
% This code calculates the bayesian estimator for the KL-divergence between
% Dirichlet distributions with a Dirichlet prior. This estimator utilizes the
% method found within:
% "Estimating functions of probability distributions from a finite set of
% samples" by Wolpert and Wolf in Physical Review E, 1995
%
% The output of this code is the first and second moment of the Bayesian
% estimator for the KL-divergence, Q(p), between two multinomimal samples 
% calculated as the posterior of the data P(Q(p) = q | n1,n2). The first 
% and second moments are according to this posterior distribution. The standard 
% deviation sqrt(Q2-Q1^2) is also provided as an error bar upon E[KL-divergence].
%
% This version of the code '_priors' also allows the user to specify a Dirichlet
% prior aside from the uniform distribution. This is the 'alpha' input
% parameter. This alpha sets all parameters for the Dirichlet prior to the same
% value.
%
% IMPORTANTLY: all quadratic terms have been linearized to keep things
% computationally tractable. This was done using the identity:
% sum_i~=j[u(i)v(i)U(j)V(j)] = sum_i[u(i)v(i)]*sum_j[U(j)V(j)] - sum_i[u(i)v(i)U(i)V(i)]
%
% This is code from the paper "A Statistical Description of Neural 
% Ensemble Dynamics," Long and Carmena 2011, submitted to  
% Frontiers in Computational Neuroscience.
% by: John D. Long II
% contact: jlong29@berkeley.edu or jlong29@gmail.com.
% Download at: http://code.google.com/p/kdq-bayes-kl
%
% INPUT:
% data1 - a vector of integer counts for each of m categories (or states)
% data2 - a vector of integer counts for each of m categories (or states)
% alpha - a real number setting the Dirichlet prior parameters to the same
%         value. Default: alpha = 1 (uniform prior). alpha < 1 biases output
%         toward non-uniform posterior and alpha > 1 biases output toward
%         uniform posterior. alpha must be < 0
%
% OUTPUT:
% Q1     - the first moment
% Q2     - the second moment
% KL_std - standard deviation of posterior distribution

% basic argument checks
if nargin < 2
	error('The user must supply 2 vectors of integer counts for calculating the KL-divergence')
end
if size(data1,1)~=size(data2,1) || size(data1,2) ~= size(data2,2)
	error('The size of data1 and data2 must be the same')
end
if nargin < 3 || isempty(alpha)
    alpha = 1;  % uniform prior
end
if alpha < 0
    error('alpha must be >0')
end

% Basic quantities
N1    = sum(data1);
N2    = sum(data2);
m     = length(data1);

% Let's pre-compute values for the psi functions to minimize function calls i.e.
% make a lookup table instead of using repeated function calls
temp  = [data1 data2];
max_n = max(temp(:));
Psi   = psi(alpha:max_n+alpha+2);
Psi2  = psi(1,alpha:max_n+alpha+2);
% NOTE: all calls to 'Psi' and 'Psi2' below have inputs that are adjusted to be
% indices of these lookup tables.

% first moment
Q1 = 0;

% pre-compute where possible
D        = (N1 + m*alpha);
PsiN1_1  = psi(N1+m*alpha+1);
PsiN2    = psi(N2+m*alpha);

if nargout < 2
        for i = 1:m
        % pre-compute where possible
        Ki1      = (data1(i)+alpha);
        Psi_i    = Psi(data1(i)+2);

        % first moment
        Q1   = Q1   + (Ki1/D)*(Psi_i-PsiN1_1) ...
                    - (Ki1/D)*(Psi(data2(i)+1)-PsiN2);
        end

else
    % second moment (for the KL-divergence, these take the form a^2 - 2ab + b^2 after integration)
    Q2 = 0;
    % Here we linearize the computation of the i~=j terms
    % define a^2 i~=j term as 'a' with sub-terms 1,2,3,4,5,6
    Q2a1 = 0;
    Q2a2 = 0;
    Q2a3 = 0;
    Q2a4 = 0;
    Q2a5 = 0;
    Q2a6 = 0;

    % define -2ab i~=j term as 'b' with sub-terms 1,2,3
    Q2b1 = 0;
    Q2b2 = 0;
    Q2b3 = 0;

    % define b^2 i~=j term as 'c' with sub-terms 1,2,3,4,5,6
    Q2c1 = 0;
    Q2c2 = 0;
    Q2c3 = 0;
    Q2c4 = 0;
    Q2c5 = 0;
    Q2c6 = 0;
    
    % pre-compute where possible
    PsiN1_2  = psi(N1+m*alpha+2);
    Psi2N1_2 = psi(1,N1+m*alpha+2);
    Psi2N2   = psi(1,N2+m*alpha);

    for i = 1:m
        % pre-compute where possible
        Kii12    = (data1(i)+alpha)*(data1(i)+alpha+1);
        Ki1      = (data1(i)+alpha);

        Psi_i    = Psi(data1(i)+2);

        dPsiN1ii = Psi(data1(i)+3)-PsiN1_2;
        dPsiN2jj = Psi(data2(i)+1)-PsiN2;

        % first moment
        Q1   = Q1   + (Ki1/D)*(Psi_i-PsiN1_1) ...
                    - (Ki1/D)*(Psi(data2(i)+1)-PsiN2);

        % second moment: i=j terms
        Q2 = Q2 + Kii12*(dPsiN1ii^2+(Psi2(data1(i)+3)-Psi2N1_2))...
             - 2* Kii12*(dPsiN1ii*(dPsiN2jj))...
                + Kii12*(dPsiN2jj^2+(Psi2(data2(i)+1)-Psi2N2));

        % second moment: i~=j terms, linearized
        Q2a1 = Q2a1 + Ki1*(Psi_i-PsiN1_2);
        Q2a2 = Q2a2 + Ki1*(Psi_i-PsiN1_2);
        Q2a3 = Q2a3 + Ki1^2*(Psi_i-PsiN1_2)^2;
        Q2a4 = Q2a4 + Ki1*Psi2N1_2;
        Q2a5 = Q2a5 + Ki1;
        Q2a6 = Q2a6 + Ki1^2*Psi2N1_2;
        
        Q2b1 = Q2b1 + Ki1*(Psi_i-PsiN1_2);
        Q2b2 = Q2b2 + Ki1*dPsiN2jj;
        Q2b3 = Q2b3 + Ki1^2*(Psi_i-PsiN1_2)*dPsiN2jj;

        Q2c1 = Q2c1 + Ki1*dPsiN2jj;
        Q2c2 = Q2c2 + Ki1*dPsiN2jj;
        Q2c3 = Q2c3 + Ki1^2*dPsiN2jj^2;
        Q2c4 = Q2c4 + Ki1*Psi2N2;
        Q2c5 = Q2c5 + Ki1;
        Q2c6 = Q2c6 + Ki1^2*Psi2N2;
    end

    % Calculate quadratic form
    Q2 = (1/(D*(N1+m*alpha+1)))*(Q2 + (Q2a1*Q2a2 - Q2a3 - (Q2a4*Q2a5 - Q2a6))...
                                - 2*(Q2b1*Q2b2 - Q2b3)...
                                  + (Q2c1*Q2c2 - Q2c3 - (Q2c4*Q2c5 - Q2c6)));

    % Calculate the standard deviation
    KL_std = sqrt(Q2-Q1^2);
end
