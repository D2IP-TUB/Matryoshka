cd ~/Fast_Data_Discovery && \
    source .venv/bin/activate && \
    export PYTHONPATH="." && \
    #python experiments/downstream/run_cat_features.py && \
    #python experiments/downstream/ablation/runtime/run.py && \
    python experiments/downstream/ablation/lda_linreg/run.py && \
    python experiments/downstream/ablation/n_jobs/run.py && \
    python experiments/downstream/ablation/top_k/run.py
