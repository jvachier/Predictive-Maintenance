install: 
	python -m pip install --upgrade pip &&\
			pip install -r requirements.txt
lint: 
	pylint --disable=R,C src/