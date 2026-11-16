PYTHON ?= python3
MEASURE_DB ?= state/index-measure.sqlite3
EVAL_DB ?= data/evals-measure.sqlite3

.PHONY: help index-measure measure test clean-index

help:
	@echo "make index-measure  rebuild the frozen measurement index (no tests/, no README)"
	@echo "make measure        rebuild it, then run the eval suite against it"
	@echo "make test           unit tests (no network or model)"

# The baseline only stays comparable if this corpus is fixed: tests/ and
# README.md name the answers to the repo-grounded cases.
index-measure:
	$(PYTHON) -m agent.repo_index build --db $(MEASURE_DB) --root . \
		--exclude 'tests/*' --exclude 'README.md'

measure: index-measure
	$(PYTHON) -m agent.evals --repo-index $(MEASURE_DB) --db $(EVAL_DB)

test:
	$(PYTHON) -m unittest discover -s tests

clean-index:
	rm -f $(MEASURE_DB) $(EVAL_DB)
