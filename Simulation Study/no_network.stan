functions {
    matrix to_triangular(vector y_basis, int K){
        matrix[K, K] y = rep_matrix(0, K, K);
        int pos = 1;
        for (i in 2:K){
            for (j in 1:(i-1)){
                y[i, j] = y_basis[pos];
                pos += 1;
            }
        }
        return y;
    }
}

data {
   int n;           // Sample size
   int p;           // Observation Dimension // number of locations
   int n_eta;           // number of fixed effects
   matrix[n, p] Y;  // Data
   matrix[n,p] X_1;  // population density
   matrix[n,p] X_2; //temperature
   matrix[n,p] Y_lag1; //AR term of period before
   
   
   real beta_m; // Prior hyperparameters mean
   real<lower=0> beta_s; //  Prior hyperparameter sigma
   real<lower=0> sigma2_a; 
   real<lower=0> sigma2_b; 
   
   vector[p] mu_0; // fixed mean of the random effects (= 0)
   //parameters for the variance of random effects
   real var_a;
   real var_b; 
   
   real lambda_m; //Prior mean for lambdas
   real<lower=0> lambda_s; //Prior sigma for lambdas
}

parameters {
   
   real beta_1; // population density
   real beta_2; //temperature
   real beta_lag1; //AR1 coefficient
   matrix[n_eta,p] eta;  // quarterly random effects 

   vector[(p*(p-1))/2] tilde_Rho_basis;
   vector<lower=0>[p] sqrt_theta_ii; // sqrt of variances
   
   real<lower=0> sigma2; // residual variance 
   real lambda0; // Network data coef 

}

transformed parameters {
    corr_matrix[p] Rho;
    matrix[p, p] tri_tilde_Rho;
    tri_tilde_Rho = to_triangular(tilde_Rho_basis, p);
    for (j in 1:p){
        for (k in 1:p){
            if(j==k){
                Rho[j,k] = 1;
            }
            if (k < j){
                Rho[j, k] = (tri_tilde_Rho[j, k]) * exp(-(lambda0));
            }
            if (k > j){
                Rho[j, k] = (tri_tilde_Rho[k,j]) * exp(-(lambda0));
            }
        }
    }
}

model {
  
   int count;
   int qtr;
   matrix[p, p] Theta; //Covariance matrix
   
   //beta ~ normal(beta_m, beta_s);
   target += normal_lpdf(beta_1 | beta_m, beta_s);
   target += normal_lpdf(beta_2 | beta_m, beta_s);
   target += normal_lpdf(beta_lag1 | beta_m, beta_s);
   
   //sigma2 ~ inv_gamma(sigma2_a, sigma2_b);
   target += inv_gamma_lpdf(sigma2 | sigma2_a, sigma2_b);
   
   //lambda0 ~ normal(lambda_m,lambda_s)
   target += normal_lpdf(lambda0|lambda_m, lambda_s);
   
   // Bayesian Graphical Lasso Models and Efficient Posterior Computation - Wang (2012)
   for(j in 1:p){
      for(k in 1:j){
         if(j == k){
             //sqrt_theta_ii[j] ~ inv_gamma(0.01, 0.01);
             target += inv_gamma_lpdf(sqrt_theta_ii[j] | var_a, var_b);
         } else{ 
            //Rho[j, k] ~ double_exponential(0, exp(-lambda0 - lambda_1 * A[j, k]));
            target += double_exponential_lpdf(tri_tilde_Rho[j, k] | 0, 1); //Is this filling up all components?
         }
      }
   }
   
   Theta = quad_form_diag(Rho, sqrt_theta_ii); //Transform correlation matrix back to covariance
   
   //eta ~ multi_normal_prec(0, Theta); 
   //mu_0 set to 0
   for (q in 1:n_eta){
     target += multi_normal_prec_lpdf(eta[q,]|mu_0,Theta);  //Eta goes by quarters, Data Y goes monthly
      //Last eta equals the negative sum of all others, so random effects are balanced out
   }
   

   // Y follows multi normal distribution
   qtr = 1;
   count = 0;
   for (i in 1:n){
      for(j in 1:p){
         //Y[i,j] ~ normal(X[i,] * beta + eta[qtr, j], sigma2);
         target += normal_lpdf(Y[i,j] |X_1[i,j] * beta_1 + X_2[i,j] * beta_2 
                                      + Y_lag1[i,j] * beta_lag1 + eta[qtr,j], sqrt(sigma2)); 
   }
   
   count += 1;
   if (count == (n/n_eta)){ 
     count = 0;
     qtr += 1;
   }
   
}
}