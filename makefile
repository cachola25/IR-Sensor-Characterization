# Makefile for IR Sensor Characterization Project

# Default environment name
VENV_NAME=venv

# Commands
setup:
	@echo "[Checking if venv module is available]"
	@python3 -m venv --help > /dev/null || (echo "Missing venv module. Install it with your package manager (e.g., sudo apt install python3-venv)"; exit 1)
	@echo "[Setting up virtual environment]"
	python3 -m venv $(VENV_NAME)
	@echo "[Activating virtual environment and installing dependencies]"
	@. $(VENV_NAME)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "[Setup complete!]"

clean:
	@echo "[Removing virtual environment]"
	rm -rf $(VENV_NAME)

help:
	@echo "Available targets:"
	@echo "  setup     - Create virtual environment and install dependencies"
	@echo "  activate  - Activate the virtual environment"
	@echo "  clean     - Remove the virtual environment"
	@echo "  help      - Show this help message"

.PHONY: setup activate clean help