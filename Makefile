# Inspired by: https://blog.mathieu-leplatre.info/tips-for-your-makefile-with-python.html
# 			   https://www.thapaliya.com/en/writings/well-documented-makefiles/

.DEFAULT_GOAL := help

PRE_COMMIT_HOOKS_IN_PIPELINE = \
	check-added-large-files check-ast check-json check-merge-conflict check-toml check-xml check-yaml \
	detect-private-key end-of-file-fixer trailing-whitespace ruff ruff-format yesqa poetry-check update-readme-title

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: install
install:  ## Install the package for development along with pre-commit hooks.
	poetry install --with dev --with test
	poetry run pre-commit install

.PHONY: clean
clean:  ## Clean up the project directory removing __pycache__, .coverage, test results, etc.
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf coverage.xml test-output.xml test-results.xml htmlcov .pytest_cache .ruff_cache
