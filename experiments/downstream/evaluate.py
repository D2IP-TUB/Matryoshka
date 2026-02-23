import json
import os
import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from experiments.base_tables.base_scoring import AutoGluonTrainer, SimpleTrainer


if __name__ == "__main__":
    # experiment_dir = os.listdir('experiments/downstream/logs/')
    experiment_dir = ['gittables_pageviews_backward']
    trainers = {'simple': SimpleTrainer, 'autogluon': AutoGluonTrainer}
    for trainer_name in trainers:
        trainer_class = trainers[trainer_name]
        all_results = pd.DataFrame()
        for dir in experiment_dir:
            if not os.path.isdir(os.path.join('experiments/downstream/logs/', dir)):
                continue
            log_path = os.path.join('experiments/downstream/logs/', dir)
            lake, table_name, strategy = dir.split('_')
            try:
                X = pd.read_csv(os.path.join(log_path, f'augmented_{dir}.csv'))
            except FileNotFoundError:
                continue

            base_table_splits_path = os.path.join('experiments/base_tables/', table_name, 'splits.json')
            with open(base_table_splits_path, 'r') as f:
                splits = json.load(f)
            problem_type = splits[0]['target_type']
            if problem_type == 'continuous':
                problem_type = 'regression'
            target = splits[0]['target']
            query_col = splits[0]['query_col']
            if strategy in ['forward', 'backward', 'lasso']:
                X = X.drop(columns=query_col)

            feat_gen = AutoMLPipelineFeatureGenerator(enable_text_special_features=False, enable_text_ngram_features=False)
            X = feat_gen.fit_transform(X)
            print(X.columns)
            breakpoint()
            X = X.dropna(subset=[target])
            trainer = trainer_class(
                problem_type=problem_type,
                target_column=target,
                time_limit=1800,
                presets="good_quality",
                output_dir=f'{log_path}/{trainer_name}_model'
            )
            results = trainer.train(X)
            performance = trainer.evaluate(X)
            df = pd.DataFrame([performance], index=[dir])
            all_results = pd.concat([all_results, df], ignore_index=False)

        all_results.to_csv(f'experiments/downstream/logs/{trainer_name}_results.csv')