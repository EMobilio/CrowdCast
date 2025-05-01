VENV = crowdcastenv
PYTHON = ./$(VENV)/bin/python3
PIP = ./$(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

preprocess:
	$(PYTHON) preprocessing.py

visualize:
	$(PYTHON) exploration.py

train:
	$(PYTHON) train.py

clean:
	rm -rf $(VENV)

reinstall: clean install