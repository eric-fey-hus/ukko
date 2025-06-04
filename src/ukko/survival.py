import numpy as np
import pandas as pd
from scipy.stats import fisk as loglogistic 
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter, LogLogisticAFTFitter


def plot_KM(survival_data, figsize=(10, 3)):   
    # Create a KaplanMeierFitter object
    kmf = KaplanMeierFitter()

    # Fit the data to the Kaplan-Meier model
    # The 'durations' are the observed times, and 'event_observed' indicates if the event occurred
    kmf.fit(durations=survival_data['observed_time'], 
            event_observed=survival_data['event_observed'], 
            label='Overall Survival Curve')

    # Plot the survival function
    plt.figure(figsize=figsize)
    kmf.plot_survival_function(
        show_censors = True,
        censor_styles={'marker': '+', 'ms': 8, 'mew': 1.5}
    )
    plt.title('Kaplan-Meier Survival Curve')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.grid(False)
    plt.show()

def plot_loglogistic_hazard(shape_params, scale, max_time):
    """
    Plots the hazard function of the log-logistic distribution for given parameters.

    Args:
        shape_params (list): A list of shape parameters (c) to plot.
        scale (float): The scale parameter (alpha) for the log-logistic distribution.
        max_time (float): The maximum time value for the plot.
    """
    
    plt.figure(figsize=(10, 3))
    time_points = np.linspace(0.01, max_time, 500) # Avoid t=0 for hazard calculation

    for shape in shape_params:
        # Calculate PDF and CDF
        pdf_values = loglogistic.pdf(time_points, c=shape, scale=scale)
        cdf_values = loglogistic.cdf(time_points, c=shape, scale=scale)
        
        # Could to this:
        # Survival function S(t) = 1 - F(t)
        survival_function_values = 1 - cdf_values
        # but accodrding to scipy docs, sf is sometimes more accurate:
        survival_function_values = loglogistic.sf(time_points, c=shape, scale=scale) 

        # Hazard function h(t) = f(t) / S(t)
        # Handle potential division by zero if survival_function_values becomes very small
        hazard_values = np.divide(pdf_values, survival_function_values, 
                                  out=np.zeros_like(pdf_values), 
                                  where=survival_function_values != 0)
        
        plt.plot(time_points, hazard_values, label=f'Shape (β) = {shape}')

    plt.title('Log-Logistic Distribution Hazard Function')
    plt.xlabel('Time (t)')
    plt.ylabel('Hazard Rate h(t)')
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def generate_survival_data_LL(n_samples, n_features, n_informative_features, 
                              loglogistic_shape, loglogistic_scale, 
                              target_censoring_percentage, 
                              nonlinear=False):
    """
    Generates a synthetic survival dataset based on an Accelerated Failure Time (AFT) model
    with a Log-Logistic distribution.

    Uses scipy.stats.fisk.

    Args:
        n_samples (int): The number of samples (individuals) to generate.
        n_features (int): The total number of features for each sample.
        n_informative_features (int): The number of features that directly
                                      influence the survival time (accelerate or decelerate).
                                      These will have non-zero coefficients.
        loglogistic_shape (float): The shape parameter (c or \beta) for the log-logistic distribution.
                                   c > 0. When c > 1 unimodal.  
        loglogistic_scale (float): The baseline scale parameter (\alpha) for the log-logistic distribution.
                                   This is the 'median' survival time when covariates are zero.
        target_censoring_percentage (float): The desired percentage of events to be censored
                                             (e.g., 0.3 for 30% censoring). This value should
                                             be between 0.0 and 1.0.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated features,
                          true survival time, observed event status, and
                          observed time (min of true survival and censoring time).
        np.array: The true coefficients used for generating the data.
    """

    # 1. Generate features
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df_features = pd.DataFrame(X, columns=feature_names)

    # 2. Define coefficients for informative features (beta values for AFT model)
    # In AFT, positive coefficients typically mean longer survival (slower failure),
    # negative coefficients mean shorter survival (faster failure).
    # We'll use these coefficients to modify the log of the scale parameter.
    coefficients = np.zeros(n_features)
    # Coefficients will influence the log of the scale parameter
    coefficients[:n_informative_features] = np.random.uniform(0.1, 0.5, n_informative_features) 

    # 3. Calculate the linear predictor (beta * X)
    linear_predictor = np.dot(X, coefficients)

    # 4. Calculate the individual scale parameter for the Log-Logistic distribution
    # For AFT with log-logistic, log(T) = log(alpha) + beta*X + sigma*epsilon
    # Or, T = alpha * exp(beta*X) * exp(sigma*epsilon)
    # So, the effective scale (alpha_i) for each individual is:
    # alpha_i = baseline_scale * exp(linear_predictor)
    
    # A positive coefficient in AFT means a longer time to event (acceleration factor > 1)
    # So, exp(linear_predictor) acts as the acceleration factor.
    effective_scale = loglogistic_scale * np.exp(linear_predictor)
    #print(f"coefficients {coefficients}")
    #print(f"effective_scale linear {effective_scale}") 

    if nonlinear:
        #print(f"mean of linear_predictor: {np.mean(linear_predictor)}")
        # Define Gaussian function:
        def gaussian(x, a=1, b=1, c=1):
            """
            Gaussian function with parameters a (peak height), b (possition), and c (width).
            """
            return a * np.exp( - (x - b) ** 2 / (2 * c ** 2))
        def f(x):
            return gaussian(x, a=1, b=np.mean(linear_predictor), c=1)
        scale_factor = np.mean(np.exp(linear_predictor)) / np.mean(np.exp(f(linear_predictor)))
        #print(f"scale_factor: {scale_factor}")
        def f(x):
            return gaussian(x, a=scale_factor, b=np.mean(linear_predictor), c=1)
        effective_scale_lin = loglogistic_scale * np.exp(linear_predictor)
        effective_scale = loglogistic_scale * np.exp(f(linear_predictor))
        #print(f"mean: {np.mean(linear_predictor)}")
        #print(f"mean: {np.mean(f(linear_predictor))}")
        #print(f"mean of effective_scale_lin: {np.mean(effective_scale_lin)}")
        #print(f"mean of effective_scale: {np.mean(effective_scale)}")
        figure, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.scatter(effective_scale_lin, effective_scale, alpha=0.5)
        ax.set_xlabel('Effective Scale (linear)')
        ax.set_ylabel('Effective Scale (nonlinear)')


    # Ensure scale parameter is positive
    effective_scale = np.maximum(effective_scale, 1e-6) 

    # 5. Generate true survival times (T) from the Log-Logistic AFT model
    # loglogistic.rvs takes 'c' (shape) and 'scale' (alpha)
    true_survival_times = loglogistic.rvs(c=loglogistic_shape, 
                                          loc = 0, 
                                          scale=effective_scale, 
                                          size=n_samples)

    # 6. Initialize observed_time and event_observed
    observed_time = np.copy(true_survival_times) # Initially, observed time is the true survival time
    event_observed = np.ones(n_samples, dtype=int) # Initially, all events are observed (not censored)

    # 7. Apply censoring based on target_censoring_percentage
    num_censored_samples = int(n_samples * target_censoring_percentage)
    censored_indices = np.random.choice(n_samples, num_censored_samples, replace=False)
    event_observed[censored_indices] = 0
    # observed_time for censored events remains their true survival time as per previous request
    
    # 8. Max followup: Censor everything after 10*scale
    max_followup = 5*loglogistic_scale
    censored_indices = true_survival_times > max_followup
    observed_time[censored_indices] = max_followup
    event_observed[censored_indices] = 0
    

    # 8. Create DataFrame
    df_survival = pd.DataFrame({
        'survival_time': true_survival_times,
        'event_observed': event_observed,
        'observed_time': observed_time
    })

    df_final = pd.concat([df_features, df_survival], axis=1)

    return df_final, coefficients


