# Background Survival Models

Survival analysis focuses on analyzing the time until an event occurs. Key concepts include:

## Survival Function
The survival function $S(t)$ represents the probability that an individual survives longer than time $t$:

$S(t) = P(T > t) = 1 - F(t)$

where $F(t)$ is the cumulative distribution function.

## Hazard Function
The hazard function $h(t)$ represents the instantaneous rate of failure at time $t$:

$h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t | T \geq t)}{\Delta t}$

## Relationship Between Functions
The hazard and survival functions are related:

$h(t) = -\frac{d}{dt}\log S(t)$

$S(t) = \exp(-\int_0^t h(u)du)$

## Censoring
Data is considered censored when the event has not occurred during the observation period. Common types are:

- Right censoring: Event occurs after observation ends
- Left censoring: Event occurs before observation begins
- Interval censoring: Event occurs within a time interval

## Common Models

Univariate model:

- **Kaplan-Meier**: 
  Non-parametric estimator of the survival function

Multivariate models: 
- **Cox Proportional Hazards**: 
  Semi-parametric model where covariates multiply the baseline hazard
- **Parametric Models**: 
  - Models where survival time follows a specific probability distribution
  - Include exponential, Weibull, and log-logistic distributions
- **Accelerated Failure Time (AFT)**: 
  - AFT models are a subset of parametric models where covariates act multiplicatively on time scale with _accelerated failure rate_ $\lambda$:

    $S(t) = S_{0}(\frac{t}{\lambda (x)})$

  - $\lambda$ as function of covariates: 

    $\log \lambda(x) = \beta_0 + \beta_i x_i $

  - Common distributions: Weibull, log-logistic, log-normal
  - ToDo: look into fragility terms that provide robustness to unmodelled covariates. 

## Log-logistic AFT model

Log-logistic is commonly used, especially for cancer survial data.
The reason is, that unlike the [Weibull distribution](#Weibull-AFT-model) it can model a non-montonous hazard function that can increase at earlier times and decrease at later times.  
Like log-normal, but heavier tails. 

## Weibull AFT model:

$H(t, x) = (\frac{t}{\lambda(x)})^\rho$,
  
where $\lambda$ is the _scale_ parameter 
and $\rho$ is  the _shape_  parameter. 
Then the hazard rate is 
$h(t) = \frac{\rho}{\lambda} (\frac{t}{\lambda})^{\rho-1}$. 

Weibull AFT has a nice interpretation:

- $k < 1 $ failure rate $f(t)$ decreases over time
- $k = 1 $ falure rate constant over time. 
  Here the Weibull distribution reduces to the exponential distribution.
- $k > 1 $ failure rate increases over time = sigmoidal hazard function. 


We can also model $\rho$ as function of the covariates (but note that relations become more complicated!):

$H(t, x) = (\frac{t}{\lambda(x)})^{\rho(x)}$.


## Notes

The Weibull distribution has probability density function:

$f(t) = \frac{\rho}{\lambda} (\frac{t}{\lambda})^{\rho-1} \exp(-(\frac{t}{\lambda})^\rho)$

where $\lambda$ is the _scale_ parameter 
and $\rho$ is the _shape_ parameter.   
    
### Failure rate and hazard relationship

$S(t) = 1 - F(t)$

$h(t) = \frac{f(t)}{1-F(t)}$

$h(t) = \frac{f(t)}{S(t)} = \frac{S'(t)}{S(t)} = -\frac{d}{dt}log(S(t))$

### Hazard and Cumulative Hazard Relationship

The cumulative hazard function $H(t)$ is the integral of the hazard function:

$H(t) = \int_0^t h(u)du$

For the Weibull model:
- Hazard: $h(t) = \frac{\rho}{\lambda} (\frac{t}{\lambda})^{\rho-1}$
- Cumulative hazard: $H(t) = (\frac{t}{\lambda})^\rho$

The relationship to survival function is:
$S(t) = \exp(-H(t))$

Therefore:
- $h(t) = \frac{d}{dt}H(t)$ (hazard is derivative of cumulative hazard)
- $H(t) = -\log(S(t))$ (cumulative hazard is negative log of survival)