install: 
	python -m pip install --upgrade pip &&\
		pip install -r requirements.txt

install poetry:
	python -m poetry==1.7.1 && \
	poetry install

lint: 
	pylint --disable=R,C src/

black:
	python -m black src/
	