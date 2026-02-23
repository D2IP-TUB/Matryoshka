import json
import os
import sys
from dotenv import load_dotenv
load_dotenv()
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath:
    sys.path.extend(pythonpath.split(os.pathsep))

from experiments.base_tables.base_table_preprocessing import PreProcessor
from experiments.ml_tasks.models import RandomForestRegressorTuner, LinearRegressionTuner, LassoTuner, KNeighborsRegressorTuner
from sklearn.base import BaseEstimator


def find_best_model(bt_paths: list[str], task: str = 'regression', preprocessor_params: dict = {}, models: list[BaseEstimator] = None, save_json: bool = True, json_name: str = 'best_model.json'):
    best_models = {}
    for bt_path in bt_paths:
        BT_DIR = os.path.join(os.path.dirname(__file__), bt_path)

        preprocessor = PreProcessor(
            base_table_path=os.path.join(BT_DIR, f'{bt_path}.csv'),
            base_table_splits_path=os.path.join(BT_DIR, 'splits.json'),
        )
        table, query_col, target, _ = preprocessor.run(**preprocessor_params)

        X = table.drop([query_col, target]).to_numpy()
        y = table[target].to_numpy().reshape(-1)
        
        target_metric = ['neg_root_mean_squared_error']
        
        match task:
            case 'regression':
                if models is None:
                    models = [
                        LinearRegressionTuner(target_metric=target_metric),
                        RandomForestRegressorTuner(target_metric=target_metric),
                        LassoTuner(target_metric=target_metric),
                        KNeighborsRegressorTuner(target_metric=target_metric)
                    ]
            case '_':
                raise ValueError('Unsupported task')

        scores = {}
        for model in models:
            model.tune(X, y)
            scores[model.__class__.__name__] = model.fit_predict()
        
        best_model_name = min(scores, key=lambda k: scores[k]['neg_root_mean_squared_error'])
        best_model = next(model for model in models if model.__class__.__name__ == best_model_name)
        best_model_params = best_model.best_params_
        best_model_dict = {
            'model': best_model_name,
            'params': best_model_params,
            'scores': scores[best_model_name]
        }
        if save_json:
            with open(os.path.join(BT_DIR, json_name), 'w') as f:
                json.dump(best_model_dict, f, indent=4, default=str)
        best_models[bt_path] = best_model_dict

    return best_models


if __name__ == '__main__':
    if len(sys.argv) > 1:
        bt_paths = sys.argv[1:]
    else:
        raise RuntimeError('Base table folder path is required as the first positional argument. For example: `python experiments/base_tables/initial_training.py airbnb`')
    
    find_best_model(bt_paths)