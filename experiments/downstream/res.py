import polars as pl
import os
import json
from pathlib import Path


def save_results():
    NOTEBOOK_DIR = Path(os.path.abspath('')).resolve()
    PROJECT_ROOT = NOTEBOOK_DIR.parents[1]  # Fast_Data_Discovery

    exp_path = NOTEBOOK_DIR / 'logs5'
    base_scores = pl.read_csv(PROJECT_ROOT / 'experiments' / 'base_tables' / 'base_simple_scores.csv').rename({'': 'table'})
    aug_scores = pl.read_csv(exp_path / 'simple_results.csv').rename({'_duplicated_0': 'experiment'})
    aug_scores = aug_scores.with_columns(pl.col('experiment').fill_null(pl.col('')))

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
        pl.col('experiment').str.split('_').list.to_struct(upper_bound=3, fields=['lake', 'table', 'algorithm']).struct.unnest()
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

    import math

    TABLES_DIR = PROJECT_ROOT.parent / 'Matryoshka' / 'tables'

    # Dataset order matching datasets.tex; split into regression and classification
    regression_datasets = ['elections', 'realestate', 'energy', 'fire', 'imdb', 'jobs', 'pageviews', 'vgsales']
    classification_datasets = ['arrest', 'food', 'hospital', 'trees']
    all_datasets = regression_datasets + classification_datasets

    # Display names for columns
    dataset_display = {
        'elections': 'Elect.', 'realestate': 'R.~Estate', 'energy': 'Energy',
        'fire': 'Fire', 'imdb': 'IMDB', 'jobs': 'Jobs',
        'pageviews': 'P.~Views', 'vgsales': 'VG Sales',
        'arrest': 'Arrest', 'food': 'Food', 'hospital': 'Hosp.', 'trees': 'Trees',
    }

    # Algorithm display names
    algo_display = {
        'base': 'Base', 'arda': 'ARDA', 'autofeat': 'AutoFeat',
        'kitana': 'Kitana', 'qcr': 'QCR',
        'forward': 'Fwd (\\system)', 'backward': 'Bwd (\\system)',
    }

    def fmt_number(val, is_large=False):
        """Format a number: add thousand separators for large integers, otherwise round."""
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return '--'
        if is_large:
            return f'{int(round(val)):,}'.replace(',', '\\,')
        # For decimals, strip trailing zeros but keep at least 1 decimal
        s = f'{val:.3f}'.rstrip('0').rstrip('.')
        if '.' not in s:
            s += '.0'
        return s

    def highlight(val_str, rank):
        """Apply bold (best) or underline (second-best) formatting."""
        if rank == 1:
            return f'\\textbf{{{val_str}}}'
        elif rank == 2:
            return f'\\underline{{{val_str}}}'
        return val_str

    def compute_ranks(series, lower_is_better):
        """Return rank 1 (best) and 2 (second-best) for each value in the series."""
        vals = series.to_list()
        valid = [(v, i) for i, v in enumerate(vals) if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not valid:
            return [0] * len(vals)
        sorted_vals = sorted(set(v for v, _ in valid), reverse=not lower_is_better)
        best_val = sorted_vals[0]
        second_val = sorted_vals[1] if len(sorted_vals) > 1 else None
        ranks = []
        for v in vals:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                ranks.append(0)
            elif v == best_val:
                ranks.append(1)
            elif second_val is not None and v == second_val:
                ranks.append(2)
            else:
                ranks.append(0)
        return ranks

    def is_large_value(dataset, table_type):
        """Determine if a dataset column needs thousand separators."""
        if table_type == 'runtime':
            return False
        return dataset in ('realestate', 'fire', 'jobs')

    def generate_latex_table(pivoted_df, table_type, algo_order):
        """Generate a LaTeX table string from a pivoted Polars DataFrame.
        
        table_type: 'score' or 'runtime'
        """
        n_reg = len(regression_datasets)
        n_cls = len(classification_datasets)
        n_total = n_reg + n_cls
        
        # For scores, column names include metric suffix; for runtime, just dataset name
        if table_type == 'score':
            metric_map = dict(aug_scores.select('table', 'metric').unique().iter_rows())
            col_map = {}
            for ds in all_datasets:
                metric = metric_map.get(ds, '')
                col_name = f'{ds} {metric}'
                col_map[ds] = col_name
            lower_is_better = {ds: (metric_map.get(ds) == 'rmse') for ds in all_datasets}
        else:
            col_map = {ds: ds for ds in all_datasets}
            lower_is_better = {ds: True for ds in all_datasets}  # runtime: lower is always better
        
        # Sort by algo order
        df = pivoted_df.with_columns(
            pl.col('algorithm').cast(pl.Enum(algo_order))
        ).sort('algorithm')
        
        # Compute ranks per column
        ranks = {}
        for ds in all_datasets:
            col = col_map.get(ds)
            if col and col in df.columns:
                ranks[ds] = compute_ranks(df[col], lower_is_better[ds])
            else:
                ranks[ds] = [0] * len(df)
        
        # Build header
        lines = []
        lines.append('\\begin{table*}[t]')
        lines.append('    \\small')
        lines.append('    \\centering')
        lines.append('    \\setlength\\tabcolsep{3pt}')
        
        if table_type == 'score':
            lines.append('    \\caption{Downstream prediction quality. Regression tasks are evaluated with RMSE ($\\downarrow$), classification tasks with weighted F1 ($\\uparrow$). Best results are in \\textbf{bold}, second-best are \\underline{underlined}.}')
        else:
            lines.append('    \\caption{End-to-end runtime in seconds. Best results are in \\textbf{bold}, second-best are \\underline{underlined}.}')
        
        lines.append('    \\vspace{-0.3cm}')
        lines.append('')
        lines.append(f'    \\begin{{tabular}}{{l|{"r" * n_reg}|{"r" * n_cls}}}')
        lines.append('        \\toprule')
        
        if table_type == 'score':
            lines.append(f'        & \\multicolumn{{{n_reg}}}{{c|}}{{\\textit{{Regression (RMSE $\\downarrow$)}}}} & \\multicolumn{{{n_cls}}}{{c}}{{\\textit{{Classification (F1 $\\uparrow$)}}}} \\\\')
        else:
            lines.append(f'        & \\multicolumn{{{n_reg}}}{{c|}}{{\\textit{{Regression}}}} & \\multicolumn{{{n_cls}}}{{c}}{{\\textit{{Classification}}}} \\\\')
        
        lines.append(f'        \\cmidrule(lr){{2-{n_reg+1}}} \\cmidrule(l){{{n_reg+2}-{n_total+1}}}')
        
        # Column headers
        header = '        \\textbf{Method}'
        for ds in all_datasets:
            header += f'\n            & \\textbf{{{dataset_display[ds]}}}'
        header += ' \\\\'
        lines.append(header)
        lines.append('        \\midrule')
        
        # Data rows
        algos = df['algorithm'].to_list()
        system_algos = {'forward', 'backward'}
        
        for row_idx, algo in enumerate(algos):
            algo_str = str(algo)
            display_name = algo_display.get(algo_str, algo_str)
            
            # Add midrule before system methods
            if algo_str in system_algos and (row_idx == 0 or str(algos[row_idx - 1]) not in system_algos):
                lines.append('        \\midrule')
            
            cells = []
            for ds in all_datasets:
                col = col_map.get(ds)
                if col and col in df.columns:
                    val = df[col][row_idx]
                    large = is_large_value(ds, table_type)
                    val_str = fmt_number(val, is_large=large)
                    rank = ranks[ds][row_idx]
                    val_str = highlight(val_str, rank)
                else:
                    val_str = '--'
                cells.append(val_str)
            
            # Split into regression and classification for line wrapping
            reg_cells = ' & '.join(cells[:n_reg])
            cls_cells = ' & '.join(cells[n_reg:])
            lines.append(f'        {display_name}')
            lines.append(f'            & {reg_cells}')
            lines.append(f'            & {cls_cells} \\\\')
        
        lines.append('        \\bottomrule')
        lines.append('    \\end{tabular}')
        lines.append('')
        label = 'scores' if table_type == 'score' else 'runtime'
        lines.append(f'    \\label{{table:{label}}}')
        lines.append('\\end{table*}')
        
        return '\n'.join(lines)

    # --- Build the two pivoted dataframes ---
    score_pivoted = (
        aug_scores.with_columns(pl.col('table') + ' ' + pl.col('metric'))
        .pivot(index='algorithm', columns='table', values='score')
    )
    runtime_pivoted = (
        aug_scores.with_columns(pl.col('table'))
        .pivot(index='algorithm', columns='table', values='runtime')
        .with_columns(pl.exclude('algorithm').round(3))
        .filter(pl.col('algorithm') != 'base')
    )
    algorithm_order = ['base', 'arda', 'autofeat', 'kitana', 'qcr',  'forward', 'backward']
    # --- Generate and save ---
    score_tex = generate_latex_table(score_pivoted, 'score', algorithm_order)
    runtime_tex = generate_latex_table(runtime_pivoted, 'runtime', [a for a in algorithm_order if a != 'base'])

    (TABLES_DIR / 'performance.tex').write_text(score_tex + '\n')
    (TABLES_DIR / 'efficiency.tex').write_text(runtime_tex + '\n')


if __name__ == '__main__':
    save_results()