# Checks

lint:
		pylint easyfsl

test:
		pytest easyfsl

isort:
		isort easyfsl

isort-check:
		isort easyfsl --check

black:
		black easyfsl

black-check:
		black easyfsl --check

mypy:
		mypy easyfsl

# Install

dev-install:
		pip install -r dev_requirements.txt
