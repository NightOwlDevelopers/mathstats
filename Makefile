.PHONY: test install
install:
	pip install -e .
test:
	pytest -q
