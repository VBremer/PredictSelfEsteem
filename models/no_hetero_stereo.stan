data {
	int<lower=1> N; // number of observations
	int<lower=1> C; // categories
	int<lower=1, upper=C> y[N]; // response variable
	int<lower=1> K; // number of predictors
	row_vector[K] x[N]; // X vectors
	vector<lower=0>[C-1] alpha;

	int<lower=1> N2; // number of test observations
	row_vector[K] x_pred[N2]; // X vectors
}
parameters {
	vector[K] beta; // beta parameter-slopes
	vector[K] delta; // beta parameters-slopes
	vector[C-1] alphaIntercept; //intercepts alpha
	simplex[C-1] gamma; // base for phi's
}
transformed parameters{
    vector<lower=0, upper=1>[C] phi; // according to Anh. et al 2009 // change the phis later into a loop when everything works
    phi[1] = 0;
    phi[2] = gamma[1];
    phi[3] = gamma[1] + gamma[2];
    phi[4] = gamma[1] + gamma[2] + gamma[3]; 
    phi[5] = gamma[1] + gamma[2] + gamma[3] + gamma[4];
    phi[6] = gamma[1] + gamma[2] + gamma[3] + gamma[4] + gamma[5];
    phi[7] = gamma[1] + gamma[2] + gamma[3] + gamma[4] + gamma[5] + gamma[6];
    phi[8] = gamma[1] + gamma[2] + gamma[3] + gamma[4] + gamma[5] + gamma[6] + gamma[7];
    phi[9] = gamma[1] + gamma[2] + gamma[3] + gamma[4] + gamma[5] + gamma[6] + gamma[7] + gamma[8];
    phi[10] = gamma[1] + gamma[2] + gamma[3] + gamma[4] + gamma[5] + gamma[6] + gamma[7] + gamma[8] + gamma[9];
}
model{
	vector[C] mu;
	vector[C] prob;

	gamma ~ dirichlet(alpha);

	delta[1] ~ normal(1, 10);	
	delta[2] ~ normal(-1, 10);
	delta[3] ~ normal(1, 10);
	delta[4] ~ normal(1, 10);
	delta[5] ~ normal(1, 10);
	
	beta ~ normal(delta, 10);

	alphaIntercept ~ normal(0,10); // equals .01 prec bc 1/10*10 = 0.01
	
	for(i in 1:N){
		mu[1] = exp(0 + phi[1]*(x[i]*beta));
		
		for(j in 2:C)
			mu[j] = exp(alphaIntercept[j-1] + phi[j] * (x[i] * beta));
		
		for(j in 1:C)
			prob[j] = mu[j]/sum(mu);
		
		y[i] ~ categorical(prob);
	}
}
generated quantities{
	vector[C] GeneratedMu;
	vector[N] logLik;
	vector[C] GeneratedProb;
	vector[C] mu_pred;
	vector[C] prob_pred;
	int<lower=1, upper=C> y_pred[N2]; // response variable
	real dev;
	dev = 0;
	
	for(i in 1:N){
		GeneratedMu[1] = exp(0 + phi[1]*(x[i]*beta));
		
		for(j in 2:C)
			GeneratedMu[j] = exp(alphaIntercept[j-1] + phi[j] * (x[i] * beta));
		
		for(j in 1:C)
			GeneratedProb[j] = GeneratedMu[j]/sum(GeneratedMu);

		logLik[i] = categorical_log(y[i], GeneratedProb);
		dev = dev + (-2) * categorical_log(y[i], GeneratedProb);
	}
	for(i in 1:N2){
		mu_pred[1] = exp(0 + phi[1]*(x_pred[i]*beta));
		
		for(j in 2:C)
			mu_pred[j] = exp(alphaIntercept[j-1] + phi[j] * (x_pred[i] * beta));
		
		for(j in 1:C)
			prob_pred[j] = mu_pred[j]/sum(mu_pred);
		
		y_pred[i] = categorical_rng(prob_pred);
	}
}
