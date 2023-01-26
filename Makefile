
# Pre commit hooks
install-pre-commit:
	pre-commit install
	pre-commit autoupdate

run-pre-commit:
	pre-commit run --all-files


# Styling
style:
	black .
	isort .


# Cleaning
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E "pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage


download-dataset:
	mkdir -p dataset
	wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -P dataset

