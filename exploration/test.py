import pymc as pm
import pandas as pd
import numpy as np
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import arviz as az
from pymc_marketing.mmm.transformers import geometric_adstock, hill_function
# ==========================================
# 1. Synthetic Data Generation
# ==========================================
# Simulating the challenge: Weekly Media Spend vs. Daily Noisy Tracking
np.random.seed(42)
weeks = 104
dates = pd.date_range(start='2023-01-01', periods=weeks, freq='W-MON')

# Independent Variables (Vivvix + Proxies)
# X1: TV Spend (Vivvix) - Pulsed flighting pattern
tv_spend = np.zeros(weeks)
tv_spend[10:20] = np.random.normal(1000, 100, 10) # Campaign 1
tv_spend[40:50] = np.random.normal(1200, 150, 10) # Campaign 2
tv_spend[80:90] = np.random.normal(1500, 200, 10) # Campaign 3

# X2: Search Volume (Google Trends) - Proxy for missing Digital/Social spend
# correlated with TV but has its own baseline
search_vol = np.random.normal(50, 5, weeks) + (tv_spend / 100) 

df_media = pd.DataFrame({
    'date': dates,
    'tv_spend': tv_spend,
    'search_vol': search_vol
})

# Normalize inputs (Critical for PyMC convergence)
# We use MaxAbsScaler logic here
df_media['tv_scaled'] = df_media['tv_spend'] / df_media['tv_spend'].max()
df_media['search_scaled'] = df_media['search_vol'] / df_media['search_vol'].max()

# Dependent Variable (YouGov)
# Simulating a "True" latent state that evolves slowly (Random Walk)
true_awareness = 0.30 # Starting baseline 30%
latent_state = [true_awareness]
for t in range(1, weeks):
    # Media impact (Adstocked & Saturated)
    media_lift = (0.05 * df_media['tv_scaled'].iloc[t]) + (0.02 * df_media['search_scaled'].iloc[t])
    # Random walk drift
    drift = np.random.normal(0, 0.005)
    new_state = latent_state[-1] + drift + media_lift
    latent_state.append(np.clip(new_state, 0, 1))




# Adding Sampling Noise (YouGov Methodology)
# YouGov sample sizes vary, creating heteroscedastic noise.
# We simulate daily noise then aggregate to weekly for the model.
# N = sample size per week (approx 1000 for stable brands)
n_samples = 1000
observed_awareness = np.random.binomial(n_samples, latent_state) / n_samples

df_model = df_media.copy()
df_model['observed_awareness'] = observed_awareness

# ==========================================
# 2. PyMC Model Definition
# ==========================================


def hill_saturation(x, K, S):
    """
    Hill Function for Diminishing Returns.
    K: Half-saturation point.
    S: Shape/Slope parameter.
    """
    return hill_function(x, S, K)

with pm.Model() as brand_model:
    
    # --- Data Containers ---
    # Allows for easy out-of-sample predictions later
    tv_data = pm.Data("tv_data", df_model['tv_scaled'].values)
    search_data = pm.Data("search_data", df_model['search_scaled'].values)
    y_data = pm.Data("y_data", df_model['observed_awareness'].values)
    
    # --- 1. Priors for Parameters ---
    
    # Adstock Decay (Alpha)
    # TV typically has higher retention (0.6-0.8), Digital/Search lower (0.1-0.3)
    alpha_tv = pm.Beta("alpha_tv", alpha=3, beta=3) 
    alpha_search = pm.Beta("alpha_search", alpha=1, beta=3) 
    
    # Saturation (K and S)
    K_tv = pm.Gamma("K_tv", alpha=3, beta=1) # Half-saturation point
    S_tv = pm.Gamma("S_tv", alpha=3, beta=1) # Slope
    
    # Media Effectiveness Coefficients (Beta)
    # HalfNormal ensures positive impact (ROI assumption)
    beta_tv = pm.HalfNormal("beta_tv", sigma=0.5)
    beta_search = pm.HalfNormal("beta_search", sigma=0.5)
    
    # --- 2. Variable Transformations ---
    
    # Apply Adstock
    tv_adstocked = geometric_adstock(tv_data, alpha_tv)
    search_adstocked = geometric_adstock(search_data, alpha_search)
    
    # Apply Saturation
    tv_saturated = hill_saturation(tv_adstocked, K_tv, S_tv)
    # We often assume Search is linear or has high saturation point, 
    # but let's keep it linear here for simplicity.
    
    # --- 3. Latent State (Brand Equity Trend) ---
    
    # Gaussian Random Walk to model "Baseline Awareness"
    # This captures the non-media driven brand health (Local Level Model)
    # sigma_drift controls how "flexible" the baseline is. 
    # Too high = overfits noise. Too low = rigid line.
    sigma_drift = pm.HalfNormal("sigma_drift", sigma=0.05)
    baseline_trend = pm.GaussianRandomWalk(
        "baseline_trend", 
        sigma=sigma_drift, 
        shape=len(dates),
        init_dist=pm.Normal.dist(mu=0.3, sigma=0.1) # Prior for initial awareness
    )
    
    # --- 4. System Equation ---
    
    # Expected Awareness (Mu)
    mu_awareness = pm.Deterministic(
        "mu_awareness",
        baseline_trend + 
        (beta_tv * tv_saturated) + 
        (beta_search * search_adstocked)
    )
    
    # --- 5. Likelihood ---
    
    # Observation Noise (Sampling Error)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.05)
    
    # We use Normal likelihood here for simplicity on the weekly aggregate.
    # For rigorous 0-1 modeling, use Beta or Logit-Normal.
    # observed = pm.Beta("observed", alpha=mu*phi, beta=(1-mu)*phi, observed=y_data)
    
    pm.Normal(
        "likelihood",
        mu=mu_awareness,
        sigma=sigma_obs,
        observed=y_data
    )

# ==========================================
# 3. Sampling
# ==========================================
if __name__ == "__main__":
    with brand_model:
        # Tuning samples help the NUTS sampler adapt to the geometry
        trace = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.98)
        
    # ==========================================
    # 4. Diagnostics & Plotting
    # ==========================================
    summary = az.summary(trace, var_names=["alpha_tv", "beta_tv", "K_tv", "sigma_drift"])
    print(summary)
    
    # Plotting the Decomposition
    post = az.extract(trace)
    mu_mean = post["mu_awareness"].mean(dim="sample")
    trend_mean = post["baseline_trend"].mean(dim="sample")
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, df_model['observed_awareness'], 'k.', label='YouGov Observed', alpha=0.5)
    plt.plot(dates, mu_mean, 'b-', label='Model Fit (Posterior Mean)')
    plt.plot(dates, trend_mean, 'g--', label='Latent Baseline (De-noised)')
    plt.plot(dates, latent_state, label='Actual Latent Brand Awareness', color='blue', alpha=0.6)
    plt.title("Brand Equity Decomposition: Media Lift vs. Baseline")
    plt.legend()
    plt.show()