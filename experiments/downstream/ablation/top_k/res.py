import polars as pl
import os
import json
from pathlib import Path

def save_results():
    NOTEBOOK_DIR = Path(os.path.abspath('')).resolve()
    PROJECT_ROOT = NOTEBOOK_DIR.parents[3]  # Fast_Data_Discovery
    PAPER_ROOT = PROJECT_ROOT.parent / 'Matryoshka'

    exp_path = NOTEBOOK_DIR / 'logs'
    base_scores = pl.read_csv(PROJECT_ROOT / 'experiments' / 'base_tables' / 'base_simple_scores.csv').rename({'': 'table'})
    aug_scores = pl.read_csv(exp_path / 'simple_results.csv').rename({'': 'experiment'})

    exp_dir = os.listdir(exp_path)
    runtime_res = {}
    for exp in exp_dir:
        if not os.path.isdir(exp_path / exp):
            continue
        comb_log = exp_path / exp / f'{exp}.log'
        with open(comb_log) as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i])
        df = pl.from_records(lines, orient='col')
        runtime = df.select(pl.col('runtime').sum()).to_series()[0]
        runtime_res[exp] = runtime
    runtime_df = pl.from_dicts([{'experiment': k, 'runtime': v} for k, v in runtime_res.items()])
    aug_scores = aug_scores.join(runtime_df, left_on='experiment', right_on='experiment', how='left')
    aug_scores = aug_scores.with_columns(
        pl.col('experiment').str.split('_').list.to_struct(upper_bound=3, fields=['lake', 'table', 'k']).struct.unnest()
    )
    aug_scores = aug_scores.join(base_scores, on='table', how='left', suffix='_base')

    aug_scores = aug_scores.with_columns((pl.col('rmse_base').fill_null(0) + pl.col('f1_weighted_base').fill_null(0)).round(3).alias('score_base'))
    aug_scores = aug_scores.with_columns((pl.col('rmse').fill_null(0) + pl.col('f1_weighted').fill_null(0)).round(3).alias('score'))
    aug_scores = aug_scores.with_columns(
        pl.when(pl.col('rmse').is_not_null())
        .then(pl.lit('rmse'))
        .otherwise(pl.lit('f1'))
        .alias('metric')
    )
    # Create base rows for each unique (lake, table) combination
    base_rows = (
        aug_scores
        .unique(subset=['lake', 'table'])
        .with_columns([
            pl.lit('base').alias('algorithm'),
            pl.col('score_base').alias('score'),
            pl.col('rmse_base').alias('rmse'),
            pl.col('f1_weighted_base').alias('f1'),
            pl.lit(0.0).alias('runtime'),
            (pl.col('lake') + '_' + pl.col('table') + '_base').alias('experiment')
        ])
        .select(aug_scores.columns)
    )
    aug_scores = pl.concat([aug_scores, base_rows])
    aug_scores = aug_scores.with_columns(
        pl.when(pl.col('table') == 'pageviews')
        .then((pl.col('score')/1e6).round(3))
        .otherwise(pl.col('score'))
        .alias('score')
    )

    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))
    import plots_style
    plots_style.apply_style()

    # Filter out base rows
    data = aug_scores.filter(pl.col('k') != 'base')

    # Relative improvement (direction-aware)
    data = data.with_columns(
        pl.when(pl.col('metric') == 'rmse')
        .then((pl.col('score_base') - pl.col('score')) / pl.col('score_base'))
        .otherwise((pl.col('score') - pl.col('score_base')) / pl.col('score_base'))
        .alias('rel_improvement')
    )

    # Aggregate across tables per k
    agg = data.group_by('k').agg([
        pl.col('rel_improvement').mean().alias('mean_improvement'),
        pl.col('runtime').mean().alias('mean_runtime'),
    ]).with_columns(pl.col('k').cast(pl.Int32)).sort('k')

    markers = {5: 'o', 10: 's', 20: '^', 50: 'D'}
    colors = {5: 'black', 10: 'dimgray', 20: '#4c72b0', 50: '#dd8452'}

    fig, ax = plt.subplots(figsize=(2.4, 1.6))
    for row in agg.iter_rows(named=True):
        k = row['k']
        ax.scatter(row['mean_runtime'], row['mean_improvement'] * 100,
                marker=markers.get(k, 'o'), color=colors.get(k, 'black'),
                label=f'$k$={k}', s=35, zorder=3, edgecolors='black', linewidths=0.4)

    ax.set_xlim(left=0)
    ax.set_xlabel('Runtime (s)')
    ax.set_ylabel('Score Gain (%)')
    ax.legend(ncol=1, columnspacing=0.6, handletextpad=0.2, borderpad=0.2, loc='right')
    plt.tight_layout(pad=0.3)
    plt.savefig(PAPER_ROOT / 'figures' / 'top_k_ablation.pdf')

if __name__ == "__main__":
    save_results()