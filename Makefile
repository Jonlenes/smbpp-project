install:
	python -m pip install -r src/requirements.txt

generate_instance:
	python -m src.problem.instance_generator

run:
	python -m src.runner
