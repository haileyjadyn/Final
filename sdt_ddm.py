import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path

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

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def read_data(file_path, prepare_for='sdt', display=False):
    data = pd.read_csv(file_path)
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)

    if prepare_for == 'sdt':
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']

        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
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

    if prepare_for == 'delta plots':
        dp_data = []
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                if len(c_data) == 0:
                    continue

                for mode, subset in [('overall', c_data),
                                     ('accurate', c_data[c_data['accuracy'] == 1]),
                                     ('error', c_data[c_data['accuracy'] == 0])]:
                    if len(subset) >= 1:
                        percentiles = {f'p{p}': np.percentile(subset['rt'], p) for p in PERCENTILES}
                        dp_data.append({
                            'pnum': pnum,
                            'condition': condition,
                            'mode': mode,
                            **percentiles
                        })
        data = pd.DataFrame(dp_data)

    return data

def apply_hierarchical_sdt_model(data):
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())

    with pm.Model() as sdt_model:
        mean_d_prime = pm.Normal('mean_d_prime', mu=0.0, sigma=1.0, shape=C)
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=1.0)

        mean_criterion = pm.Normal('mean_criterion', mu=0.0, sigma=1.0, shape=C)
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=1.0)

        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=stdev_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=stdev_criterion, shape=(P, C))

        # Derived group-level effects for d-prime
        stimulus_type_effect = pm.Deterministic(
            'stimulus_type_effect',
            (mean_d_prime[1] + mean_d_prime[3]) / 2 - (mean_d_prime[0] + mean_d_prime[2]) / 2
        )

        trial_difficulty_effect = pm.Deterministic(
            'trial_difficulty_effect',
            (mean_d_prime[2] + mean_d_prime[3]) / 2 - (mean_d_prime[0] + mean_d_prime[1]) / 2
        )

        interaction_effect = pm.Deterministic(
            'interaction_effect',
            (mean_d_prime[3] - mean_d_prime[1]) - (mean_d_prime[2] - mean_d_prime[0])
        )

        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)

        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[data['pnum'] - 1, data['condition']], 
                   observed=data['hits'])

        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[data['pnum'] - 1, data['condition']], 
                   observed=data['false_alarms'])

    return sdt_model

def save_delta_contrast_plot(delta_data, contrast=3, baseline=0):
    contrast_data = delta_data[(delta_data['condition'] == contrast) & (delta_data['mode'] == 'overall')]
    baseline_data = delta_data[(delta_data['condition'] == baseline) & (delta_data['mode'] == 'overall')]

    deltas = []
    for pnum in contrast_data['pnum'].unique():
        c = contrast_data[contrast_data['pnum'] == pnum]
        b = baseline_data[baseline_data['pnum'] == pnum]
        if not c.empty and not b.empty:
            delta = [c[f'p{p}'].values[0] - b[f'p{p}'].values[0] for p in PERCENTILES]
            deltas.append(delta)

    if deltas:
        mean_delta = np.mean(deltas, axis=0)
        plt.figure()
        plt.plot(PERCENTILES, mean_delta, marker='o')
        plt.axhline(0, linestyle='--', color='gray')
        plt.title(f"Delta Plot (Contrast: {CONDITION_NAMES[contrast]} - Baseline: {CONDITION_NAMES[baseline]})")
        plt.xlabel("Percentile")
        plt.ylabel("RT Difference (s)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "delta_plot_contrasts.png")
        plt.close()

def main():
    data_path = Path("data.csv")
    if not data_path.exists():
        print("ERROR: data.csv not found.")
        return

    print("Reading data...")
    sdt_data = read_data(data_path, prepare_for='sdt', display=True)
    delta_data = read_data(data_path, prepare_for='delta plots', display=True)

    print("Building model...")
    model = apply_hierarchical_sdt_model(sdt_data)

    with model:
        print("Sampling...")
        trace = pm.sample(draws=1000, tune=1000, chains=2, target_accept=0.95, return_inferencedata=True)

    print("Saving convergence plots...")
    az.plot_trace(trace)
    plt.savefig(OUTPUT_DIR / "sdt_trace.png")
    plt.close()

    az.plot_posterior(trace)
    plt.savefig(OUTPUT_DIR / "sdt_posteriors.png")
    plt.close()

    summary = az.summary(trace)
    summary.to_csv(OUTPUT_DIR / "sdt_summary.csv")

    print("Generating delta contrast plot...")
    save_delta_contrast_plot(delta_data)

if __name__ == "__main__":
    main()
