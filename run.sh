#!/bin/bash

set -e

show_main_menu() {
	clear
	echo "1. Data Preprocessing"
	echo "2. Models"
	echo "3. Exit"
	echo -n "Input: "
}

show_model_menu() {
	clear
	echo "1. Train Model"
	echo "2. Evaluate Model"
	echo "3. Return"
	echo -n "Input: "
}

# python src/data_preprocessing/data_preprocessing.py
# python src/models/label/lgbm_fl/train.py
# python src/models/cat/lgbm_gbdt/train.py
# python src/models/cat/balanced_rf/train.py
python src/models/cat/xgboost/train.py
