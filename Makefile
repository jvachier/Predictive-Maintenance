install: 
	python -m pip install --upgrade pip &&\
		pip install -r requirements.txt

lint: 
	pylint --disable=R,C src/

black:
	python -m black src/

ruff:
	ruff check src/
	ruff check --fix src/
	ruff format src/