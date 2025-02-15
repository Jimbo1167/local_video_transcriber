.PHONY: setup clean install test venv install-torch

venv:
	python3.11 -m venv venv
	. venv/bin/activate && pip install --upgrade pip

activate:
	source venv/bin/activate

install-torch:
	. venv/bin/activate && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

install: venv install-torch
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -r requirements-dev.txt

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -delete
	find . -type f -name ".coverage" -delete

test:
	. venv/bin/activate && python -m pytest tests/

setup: clean venv install

# Help target to show available commands
help:
	@echo "Available commands:"
	@echo "  make setup  - Create virtual environment and install dependencies"
	@echo "  make venv   - Create virtual environment only"
	@echo "  make install- Install dependencies into virtual environment"
	@echo "  make test   - Run tests"
	@echo "  make clean  - Remove Python cache files and temporary files" 