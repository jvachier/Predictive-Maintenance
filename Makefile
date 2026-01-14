install: 
	uv sync

lint: 
	uv run pylint --disable=R,C src/

black:
	uv run black src/

ruff:
	uv run ruff check src/
	uv run ruff check --fix src/
	uv run ruff format src/