data {
	int<lower=1> N; // number of observations
	int<lower=1> C; // categories
	int<lower=1, upper=C> y[N]; // response variable
	int<lower=1> K; // number of predictors
	row_vector[K] x[N]; // X vectors
	int<lower=1> n_users; // number of clients
	int<lower=1> users[N]; // clients

	int<lower=1> N2; // number of observations test set
	row_vector[K] x_pred[N2]; // X vectors
	int<lower=1, upper=n_users> users2[N2]; // clients
}
parameters {
	vector[K] beta[n_users]; // individual beta parameters-slopes
	vector[K] delta; // beta parameters-slopes
	ordered[C-1] alpha; // cutpoints
}
model{
	vector[N] mu;
	matrix[N,C] p;
	matrix[N,C] Q;
	vector[C] probs;

	alpha ~ normal(0,10); 

	delta[1] ~ normal(1, 10);	
	delta[2] ~ normal(-1, 10);
	delta[3] ~ normal(1, 10);
	delta[4] ~ normal(1, 10);
	delta[5] ~ normal(1, 10);

	for(i in 1:n_users)
		beta[i] ~ normal(delta, 10); // equals .01 prec bc 1/10*10 = 0.01	
	
	for(i in 1:N){
		mu[i] = x[i] * beta[users[i]];
		
		Q[i,1] = inv_logit(alpha[1] - mu[i]);        // equals logit on the left side (in this case: sigmoid instead of logit)
		p[i,1] = Q[i,1];
		probs[1] = p[i,1];

		for(j in 2:C-1){
			Q[i,j] = inv_logit(alpha[j] - mu[i]);
			p[i,j] = Q[i,j] - Q[i,j-1];
			probs[j] = p[i,j];
		}
		p[i,10] = 1 - Q[i,9];
		probs[10] = p[i,10];
		
		y[i] ~ categorical(probs);
	}
}
generated quantities{
	vector[N] GeneratedMu;
	vector[N] logLik;
	vector[C] GeneratedProbs;
	matrix[N,C] GeneratedP;
	matrix[N,C] GeneratedQ;
	vector[N2] mu_pred;
	matrix[N2,C] p_pred;
	matrix[N2,C] Q_pred;
	vector[C] probs_pred;
	int<lower=1, upper=C> y_pred[N2]; // response variable
	real dev;
	dev = 0;

	for(i in 1:N){
		GeneratedMu[i] = x[i] * beta[users[i]];
		
		GeneratedQ[i,1] = inv_logit(alpha[1] - GeneratedMu[i]);        
		GeneratedP[i,1] = GeneratedQ[i,1];
		GeneratedProbs[1] = GeneratedP[i,1];

		for(j in 2:C-1){
			GeneratedQ[i,j] = inv_logit(alpha[j] - GeneratedMu[i]);
			GeneratedP[i,j] = GeneratedQ[i,j] - GeneratedQ[i,j-1];
			GeneratedProbs[j] = GeneratedP[i,j];
		}
		GeneratedP[i,10] = 1 - GeneratedQ[i,9];
		GeneratedProbs[10] = GeneratedP[i,10];
		
		logLik[i] = categorical_log(y[i], GeneratedProbs);
		dev = dev + (-2) * categorical_log(y[i], GeneratedProbs);
	}
	for(i in 1:N2){
		mu_pred[i] = x_pred[i] * beta[users2[i]];
		
		Q_pred[i,1] = inv_logit(alpha[1] - mu_pred[i]);        
		p_pred[i,1] = Q_pred[i,1];
		probs_pred[1] = p_pred[i,1];

		for(j in 2:C-1){
			Q_pred[i,j] = inv_logit(alpha[j] - mu_pred[i]);
			p_pred[i,j] = Q_pred[i,j] - Q_pred[i,j-1];
			probs_pred[j] = p_pred[i,j];
		}
		p_pred[i,10] = 1 - Q_pred[i,9];
		probs_pred[10] = p_pred[i,10];
		
		y_pred[i] = categorical_rng(probs_pred);
	}
}
