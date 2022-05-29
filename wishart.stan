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
   
   real df;
   matrix[p,p] I; 
}

parameters {
   
   real beta_1; // population density
   real beta_2; //temperature
   real beta_lag1; //AR1 coefficient
   matrix[n_eta,p] eta;  // quarterly random effects 

   cov_matrix[p] Theta;
   
   real<lower=0> sigma2; // residual variance 
}

model {
  
   int count;
   int qtr;
   
   //beta ~ normal(beta_m, beta_s);
   target += normal_lpdf(beta_1 | beta_m, beta_s);
   target += normal_lpdf(beta_2 | beta_m, beta_s);
   target += normal_lpdf(beta_lag1 | beta_m, beta_s);
   
   //sigma2 ~ inv_gamma(sigma2_a, sigma2_b);
   target += inv_gamma_lpdf(sigma2 | sigma2_a, sigma2_b);
   

   target += wishart_lpdf(Theta| df, I);
   
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