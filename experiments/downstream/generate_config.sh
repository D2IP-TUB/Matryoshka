#!/bin/bash

# Experiment Configuration Generator - Explicit Configs
# Comment out the configurations you don't need

OUTPUT_FILE="experiments/downstream/experiments.csv"

# Lakes
lakes=(
    "nyc"
    "cuk" 
    "gittables"
    # "lakebench"
)

# Tables
tables=(
    "arrest"
    "elections"
    "hospital"
    "jobs"
    "realestate"
    "food"
    "imdb"
    "pageviews"
    "vgsales"
    "fire"
    "energy"
    # "inspections"
    "trees"
)

# Algorithms
algorithms=(
    # "forward"
    # "backward"
    # "lasso"
    # "qcr"
    "arda"
    "kitana"
    "autofeat"
)

# Generate all configurations - comment out what you don't need
configs=(
    # NYC Lake
    "nyc energy forward"
    "nyc energy backward" 
    "nyc energy lasso"
    "nyc energy arda"
    "nyc energy qcr"
    "nyc energy kitana"
    "nyc energy autofeat"
    "nyc crime forward"
    "nyc crime backward"
    "nyc crime lasso" 
    "nyc crime arda"
    "nyc crime qcr"
    "nyc crime kitana"
    "nyc crime autofeat"
    # CUK Lake
    "cuk flight forward"
    "cuk flight backward"
    "cuk flight lasso"
    "cuk flight arda"
    "cuk flight qcr"
    "cuk flight kitana"
    "cuk flight autofeat"
    
    "cuk housing forward"
    "cuk housing backward"
    "cuk housing lasso"
    "cuk housing arda"
    "cuk housing qcr"
    "cuk housing kitana"
    "cuk housing autofeat"
    "cuk flood forward"
    "cuk flood backward"
    "cuk flood lasso"
    "cuk flood arda"
    "cuk flood qcr"
    "cuk flood kitana"
    "cuk flood autofeat"
    
    "cuk elections forward"
    "cuk elections backward"
    "cuk elections lasso"
    "cuk elections arda"
    "cuk elections qcr"
    "cuk elections kitana"
    "cuk elections autofeat"
    
    "cuk real_estate forward"
    "cuk real_estate backward"
    "cuk real_estate lasso"
    "cuk real_estate arda"
    "cuk real_estate qcr"
    "cuk real_estate kitana"
    "cuk real_estate autofeat"
    
    # GitTables Lake
    "gittables vgsales forward"
    "gittables vgsales backward"
    "gittables vgsales lasso"
    "gittables vgsales arda"
    "gittables vgsales qcr"
    "gittables vgsales kitana"
    "gittables vgsales autofeat"
    
    "gittables pageviews forward"
    "gittables pageviews backward"
    "gittables pageviews lasso"
    "gittables pageviews arda"
    "gittables pageviews qcr"
    "gittables pageviews kitana"
    "gittables pageviews autofeat"
    
    "gittables food forward"
    "gittables food backward"
    "gittables food lasso"
    "gittables food arda"
    "gittables food qcr"
    "gittables food kitana"
    "gittables food autofeat"
    
    "gittables imdb forward"
    "gittables imdb backward"
    "gittables imdb lasso"
    "gittables imdb arda"
    "gittables imdb qcr"
    "gittables imdb kitana"
    "gittables imdb autofeat"
    
    # LakeBench Lake
    "lakebench vgsales forward"
    "lakebench vgsales backward"
    "lakebench vgsales lasso"
    "lakebench vgsales arda"
    "lakebench vgsales qcr"
    "lakebench vgsales kitana"
    "lakebench vgsales autofeat"
    
    "lakebench pageviews forward"
    "lakebench pageviews backward"
    "lakebench pageviews lasso"
    "lakebench pageviews arda"
    "lakebench pageviews qcr"
    "lakebench pageviews kitana"
    "lakebench pageviews autofeat"
    
    "lakebench food forward"
    "lakebench food backward"
    "lakebench food lasso"
    "lakebench food arda"
    "lakebench food qcr"
    "lakebench food kitana"
    "lakebench food autofeat"
    
    "lakebench imdb forward"
    "lakebench imdb backward"
    "lakebench imdb lasso"
    "lakebench imdb arda"
    "lakebench imdb qcr"
    "lakebench imdb kitana"
    "lakebench imdb autofeat"
)

# Generate CSV
python experiments/downstream/experiment_config_generator.py --lakes ${lakes[@]} --tables ${tables[@]} --algorithms ${algorithms[@]} --output "$OUTPUT_FILE"

echo "Generated ${#configs[@]} configurations in $OUTPUT_FILE"