install: 
	python -m pip install --upgrade pip &&\
		pip install -r requirements.txt
lint: 
	pylint --disable=R,C,E0401,E0611,W0612 src/