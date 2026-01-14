install: 
	uv sync

lint: 
	pylint --disable=R,C src/

black:
	python -m black src/

ruff:
	ruff check src/
	ruff check --fix src/
	ruff format src/