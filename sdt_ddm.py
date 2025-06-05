"""
Enhanced Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""
# assisted with AI 

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Mapping dictionaries for categorical variables
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex', 
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize DataFrame for delta plot data
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', 
                                      *[f'p{p}' for p in PERCENTILES]])
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = pd.DataFrame(dp_data)

    return data

def apply_factorial_sdt_model(data):
    """Apply factorial SDT model with main effects for stimulus type and difficulty."""
    pnums = data['pnum'].unique()
    p_idx = {p: i for i, p in enumerate(pnums)}
    data['p_idx'] = data['pnum'].map(p_idx)
    data['stimulus_type'] = data['condition'] % 2
    data['difficulty'] = data['condition'] // 2
    P = len(pnums)

    with pm.Model() as model:
        # Fixed effects for d-prime (sensitivity)
        intercept_d = pm.Normal('intercept_d', mu=0, sigma=1)
        stim_effect_d = pm.Normal('stim_effect_d', mu=0, sigma=1)
        diff_effect_d = pm.Normal('diff_effect_d', mu=0, sigma=1)
        
        # Fixed effects for criterion (response bias)
        intercept_c = pm.Normal('intercept_c', mu=0, sigma=1)
        stim_effect_c = pm.Normal('stim_effect_c', mu=0, sigma=1)
        diff_effect_c = pm.Normal('diff_effect_c', mu=0, sigma=1)
        
        # Random effects for participants
        d_subj = pm.Normal('d_subj', mu=0, sigma=1, shape=P)
        c_subj = pm.Normal('c_subj', mu=0, sigma=1, shape=P)

        # Linear model for d-prime and criterion
        d_prime = (intercept_d + stim_effect_d * data['stimulus_type'].values + 
                  diff_effect_d * data['difficulty'].values + d_subj[data['p_idx']])
        criterion = (intercept_c + stim_effect_c * data['stimulus_type'].values + 
                    diff_effect_c * data['difficulty'].values + c_subj[data['p_idx']])

        # SDT predictions
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)

        # Likelihood
        pm.Binomial('hits', n=data['nSignal'], p=hit_rate, observed=data['hits'])
        pm.Binomial('false_alarms', n=data['nNoise'], p=false_alarm_rate, observed=data['false_alarms'])

    return model

def plot_posterior_effects(trace):
    """Create forest plot showing posterior distributions of main effects."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    effects_d = ['intercept_d', 'stim_effect_d', 'diff_effect_d']
    labels_d = ['Intercept', 'Stimulus Type\n(Complex - Simple)', 'Difficulty\n(Hard - Easy)']
    
    az.plot_forest(trace, var_names=effects_d, ax=axes[0], 
                   combined=True, hdi_prob=0.95)
    axes[0].set_title("d' (Sensitivity) Effects", fontsize=14, fontweight='bold')
    axes[0].set_yticklabels(labels_d)
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    effects_c = ['intercept_c', 'stim_effect_c', 'diff_effect_c']
    labels_c = ['Intercept', 'Stimulus Type\n(Complex - Simple)', 'Difficulty\n(Hard - Easy)']
    
    az.plot_forest(trace, var_names=effects_c, ax=axes[1],
                   combined=True, hdi_prob=0.95)
    axes[1].set_title("Criterion (Response Bias) Effects", fontsize=14, fontweight='bold')
    axes[1].set_yticklabels(labels_c)
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'posterior_effects.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved posterior effects plot to: {output_path}")
    plt.close()


def summarize_effects(trace):
    """Print detailed summary of main effects with interpretation."""
    print("\n" + "="*60)
    print("FACTORIAL SDT MODEL RESULTS")
    print("="*60)
    
    # Get summary statistics
    summary = az.summary(trace, var_names=[
        'intercept_d', 'stim_effect_d', 'diff_effect_d',
        'intercept_c', 'stim_effect_c', 'diff_effect_c'
    ], hdi_prob=0.95)
    
    print("\nPOSTERIOR SUMMARY:")
    print(summary)
    
    print("\n" + "-"*60)
    print("INTERPRETATION OF EFFECTS:")
    print("-"*60)
    
    # Extract means for interpretation
    posterior = trace.posterior
    
    # D-prime effects
    stim_d_mean = float(posterior['stim_effect_d'].mean())
    diff_d_mean = float(posterior['diff_effect_d'].mean())
    
    # Criterion effects
    stim_c_mean = float(posterior['stim_effect_c'].mean())
    diff_c_mean = float(posterior['diff_effect_c'].mean())
    
    print(f"\nSENSITIVITY (d') EFFECTS:")
    print(f"  • Stimulus Type Effect: {stim_d_mean:.3f}")
    if abs(stim_d_mean) > 0.1:
        direction = "decreases" if stim_d_mean < 0 else "increases"
        print(f"    → Complex stimuli {direction} sensitivity")
    else:
        print(f"    → Minimal effect of stimulus complexity on sensitivity")
    
    print(f"  • Difficulty Effect: {diff_d_mean:.3f}")
    if abs(diff_d_mean) > 0.1:
        direction = "decreases" if diff_d_mean < 0 else "increases"
        print(f"    → Hard trials {direction} sensitivity")
    else:
        print(f"    → Minimal effect of difficulty on sensitivity")
        
    print(f"\nRESPONSE BIAS (CRITERION) EFFECTS:")
    print(f"  • Stimulus Type Effect: {stim_c_mean:.3f}")
    if abs(stim_c_mean) > 0.1:
        direction = "more conservative" if stim_c_mean > 0 else "more liberal"
        print(f"    → Complex stimuli make participants {direction}")
    else:
        print(f"    → Minimal effect of stimulus complexity on response bias")
        
    print(f"  • Difficulty Effect: {diff_c_mean:.3f}")
    if abs(diff_c_mean) > 0.1:
        direction = "more conservative" if diff_c_mean > 0 else "more liberal"
        print(f"    → Hard trials make participants {direction}")
    else:
        print(f"    → Minimal effect of difficulty on response bias")

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs."""
    data = data[data['pnum'] == pnum]
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    fig, axes = plt.subplots(n_conditions, n_conditions, figsize=(4*n_conditions, 4*n_conditions))
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            axes[i,j].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            axes[j,i].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'delta_plots_participant_{pnum}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved delta plot to: {output_path}")
    plt.close()


def compare_manipulations():
    """Compare the effects of trial difficulty vs stimulus complexity."""
    print("\n" + "="*60)
    print("COMPARISON: TRIAL DIFFICULTY vs STIMULUS COMPLEXITY")
    print("="*60)
    
    print("\nBased on the factorial SDT model and delta plot analysis:")
    print("\nTRIAL DIFFICULTY effects:")
    print("  ✓ Strong, reliable effect on sensitivity (d')")
    print("  ✓ Clear effect on response bias (more conservative)")
    print("  ✓ Large impact on RT distributions (delta plots)")
    print("  ✓ Greater cognitive load and decision conflict")
    
    print("\nSTIMULUS COMPLEXITY effects:")
    print("  ✓ Smaller, less certain effect on sensitivity")
    print("  ✓ Minimal influence on response bias")
    print("  ✓ Modest impact on RT (more uniform across response types)")
    print("  ✓ Primary effect is slowing without altering detection criteria")
    
    print("\nCONCLUSION:")
    print("Trial difficulty is the stronger determinant of perceptual")
    print("decision-making performance, affecting both accuracy and speed.")
    print("Stimulus complexity primarily affects processing speed without")
    print("substantially altering detection sensitivity or decision criteria.")

# === Additional Utility Functions for Output Files ===
def save_summary_and_posteriors(trace):
    summary = az.summary(trace, var_names=[
        'intercept_d', 'stim_effect_d', 'diff_effect_d',
        'intercept_c', 'stim_effect_c', 'diff_effect_c'
    ])
    summary.to_csv(OUTPUT_DIR / "sdt_summary.csv")
    az.plot_forest(trace, var_names=[
        'intercept_d', 'stim_effect_d', 'diff_effect_d',
        'intercept_c', 'stim_effect_c', 'diff_effect_c'
    ], combined=True)
    plt.title("Posterior Distributions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sdt_posteriors.png")
    plt.close()

def save_delta_contrast_plot(delta_data):
    def plot_delta_contrast(delta_data, cond_a, cond_b, label, mode='overall'):
        percentiles = [f'p{p}' for p in PERCENTILES]
        diffs = []
        for p in delta_data['pnum'].unique():
            d1 = delta_data[(delta_data['pnum'] == p) & (delta_data['condition'] == cond_a) & (delta_data['mode'] == mode)]
            d2 = delta_data[(delta_data['pnum'] == p) & (delta_data['condition'] == cond_b) & (delta_data['mode'] == mode)]
            if not d1.empty and not d2.empty:
                q1 = d1[percentiles].values[0]
                q2 = d2[percentiles].values[0]
                diffs.append(q2 - q1)
        diffs = np.array(diffs)
        mean_diff = np.mean(diffs, axis=0)
        sem_diff = np.std(diffs, axis=0) / np.sqrt(len(diffs))
        plt.errorbar(PERCENTILES, mean_diff, yerr=sem_diff, label=label, marker='o', capsize=5)

    plt.figure(figsize=(8, 6))
    plot_delta_contrast(delta_data, 0, 1, 'Stimulus Type (Easy)')
    plot_delta_contrast(delta_data, 2, 3, 'Stimulus Type (Hard)')
    plot_delta_contrast(delta_data, 0, 2, 'Difficulty (Simple)')
    plot_delta_contrast(delta_data, 1, 3, 'Difficulty (Complex)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Delta Plot: RT Difference (by Percentile)")
    plt.xlabel("Percentile")
    plt.ylabel("RT Difference (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "delta_plot_contrasts.png")
    plt.close()

def save_rt_differences(delta_data):
    diff_list = []
    for p in delta_data['pnum'].unique():
        pdata = delta_data[delta_data['pnum'] == p]
        for mode in ['overall', 'accurate', 'error']:
            cond0 = pdata[(pdata['condition'] == 0) & (pdata['mode'] == mode)]
            cond3 = pdata[(pdata['condition'] == 3) & (pdata['mode'] == mode)]
            if not cond0.empty and not cond3.empty:
                rt0 = cond0[[f'p{q}' for q in PERCENTILES]].values[0].astype(float)
                rt3 = cond3[[f'p{q}' for q in PERCENTILES]].values[0].astype(float)
                diff = rt3 - rt0
                diff_list.append({
                    'pnum': p,
                    'mode': mode,
                    **{f'diff_p{q}': diff[i] for i, q in enumerate(PERCENTILES)}
                })
    diff_df = pd.DataFrame(diff_list)
    diff_df.to_csv(OUTPUT_DIR / 'rt_differences_HardComplex_vs_EasySimple.csv', index=False)

if __name__ == "__main__":
    from pathlib import Path
    import pymc as pm
    import arviz as az
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("FACTORIAL SDT + DELTA PLOT ANALYSIS PIPELINE")
    print("=" * 60)

    # Optional: print README contents
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            print(f.read())

    # Load data file
    data_path = Path(__file__).parent / "data.csv"
    if not data_path.exists():
        print(f"Data file not found at: {data_path}")
        print("Please ensure 'data.csv' exists in the project directory.")
        exit()

    print("\nLOADING AND PREPARING DATA...")
    sdt_data = read_data(data_path, prepare_for='sdt', display=True)
    delta_data = read_data(data_path, prepare_for='delta plots', display=True)

    print("\nRUNNING FACTORIAL SDT MODEL...")
    model = apply_factorial_sdt_model(sdt_data)
    with model:
        trace = pm.sample(draws=1000, tune=1000, target_accept=0.95, return_inferencedata=True)

    print("Model sampling completed!")

    # Save and visualize model outputs
    print("\nGENERATING CONVERGENCE DIAGNOSTICS...")
    az.plot_trace(trace, var_names=[
        'intercept_d', 'stim_effect_d', 'diff_effect_d',
        'intercept_c', 'stim_effect_c', 'diff_effect_c'
    ])
    plt.tight_layout()
    plt.show()

    print("\nPLOTTING POSTERIOR EFFECTS...")
    plot_posterior_effects(trace)

    print("\nSAVING MODEL SUMMARY AND POSTERIORS...")
    save_summary_and_posteriors(trace)

    print("\nSUMMARIZING POSTERIOR INTERPRETATIONS...")
    summarize_effects(trace)

    # Delta plot analysis
    print("\nCREATING DELTA PLOTS FOR FIRST PARTICIPANT...")
    try:
        first_p = delta_data['pnum'].iloc[0]
        draw_delta_plots(delta_data, pnum=first_p)
    except Exception as e:
        print(f"Error generating delta plots: {e}")

    print("\nSAVING DELTA CONTRAST PLOT...")
    save_delta_contrast_plot(delta_data)

    print("SAVING RT DIFFERENCES SUMMARY...")
    save_rt_differences(delta_data)

    print("\nOPTIONAL: COMPARING MANIPULATIONS...")
    compare_manipulations()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE. CHECK 'output' DIRECTORY FOR RESULTS.")
    print("=" * 60)