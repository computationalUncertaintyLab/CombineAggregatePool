#mcandrew

PYTHON ?= python3 -W ignore
STATES ?= 'all'

VENV_DIR := .forecast
VENV_PYTHON := $(VENV_DIR)/bin/python -W ignore

R ?= Rscript

run: format_ili_data score_component_models zip_forecasts build_ensemble

format_ili_data:
	cd ./analysisdata && $(PYTHON) fromEW2MW.py
	cd ./analysisdata && $(PYTHON) formatEpiData.py
	cd ./analysisdata && $(PYTHON) fromEW2Season.py

score_component_models:
	cd ./analysisdata && $(PYTHON) addForecastSupportInfo.py
	cd ./analysisdata && $(PYTHON) scoreComponentForecasts.py

zip_forecasts:
	cd ./analysisdata && gzip forecastsFormatted.csv
	cd ./analysisdata && gzip epidataFormated.csv

build_ensemble:
	cd ./build_ensemble_models && $(PYTHON) generate_runs_for_ensemble.py
	cd ./build_ensemble_models && $(PYTHON) generate_ensemble_runs.py
	mkdir -p ./build_ensemble_models/cluster_data
	cd ./build_ensemble_models && $(PYTHON) adaptive_plus__cluster__selection.py

score_ensemble:
	cd ./score_ensemble_models && $(PYTHON) pit.py
	cd ./score_ensemble_models && $(PYTHON) brier_threshold.py
	cd ./score_ensemble_models && $(PYTHON) combine_brier_scores.py


download_analysis_data:
	cd ./analysisdata && $(PYTHON) ./analysisdata/download_data.py

build_fig_1:
	cd viz/Fig1 && $(PYTHON) compute_average_pairwise_KL.py
	cd viz/Fig1 && $(PYTHON) variance_reduction.py

build_fig_2:
	cd viz/Fig2 && $(PYTHON) define_overlapping_solutions.py
	cd viz/Fig2 && $(PYTHON) nonident.py

build_fig_3:
	cd viz/Fig3 && $(PYTHON) compute_correlation_over_time.py
	cd viz/Fig3 && $(PYTHON) plot.py

build_fig_4:
	cd viz/Fig4 && $(PYTHON) plot.py

build_fig_5:
	cd viz/Fig5 && $(PYTHON) produce_dataset_of_scores.py
	cd viz/Fig5 && $(PYTHON) PIT_DATA.py
	cd viz/Fig5 && $(PYTHON) plot.py

build_fig_6:
	cd viz/Fig6 && $(PYTHON) plot.py

