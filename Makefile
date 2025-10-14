.PHONY: help eval eval-single report

# Default values for evaluation parameters
EVAL_MODEL ?= /model_hub/modelhub/121003/69300040/
DATASETS ?= hotpotqa 2wiki popqa triviaqa musiqueqa
RESULT ?= ./results/

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  help         Show this help message."
	@echo "  eval         Evaluate the model on all datasets defined in DATASETS."
	@echo "  eval-single  Evaluate the model on specific dataset(s) (set DATASETS to select)."
	@echo "  report       Generate a report from the results in RESULT directory."
	@echo ""
	@echo "Variables (can be overridden):"
	@echo "  EVAL_MODEL   Path to the evaluation model. Default: $(EVAL_MODEL)"
	@echo "  DATASETS     Space-separated list of datasets to evaluate. Default: $(DATASETS)"
	@echo "  RESULT       Directory to write evaluation results. Default: $(RESULT)"
	@echo ""
	@echo "Examples:"
	@echo "  make eval"
	@echo "  DATASETS=hotpotqa make eval-single"
	@echo "  make report"
	@echo "  RESULT=./custom_results/ make report"

eval:
	@echo "==============================================="
	@echo "Running evaluation for all datasets: all datasets"
	@echo "Model path: $(EVAL_MODEL)"
	@echo "Results directory: $(RESULT)"
	@echo "==============================================="
	python eval.py --model-path "$(EVAL_MODEL)" --output-path "$(RESULT)"

eval-single:
	@echo "==============================================="
	@echo "Running evaluation for dataset(s): $(DATASETS)"
	@echo "Model path: $(EVAL_MODEL)"
	@echo "Results directory: $(RESULT)"
	@echo "==============================================="
	python eval.py --model-path $(EVAL_MODEL) --eval-dataset $(DATASETS) --output-path $(RESULT)

report:
	@echo "==============================================="
	@echo "Generating report from results in $(RESULT)"
	@echo "==============================================="
	python get_report.py --result-dir $(RESULT)