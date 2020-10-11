install:
	pip install -r src/requirements.txt

generate_ins:
	python src/generate_instance.py

run:
	python src/optimaze.py

test:
	python -m pytest .