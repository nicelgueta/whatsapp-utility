.PHONY: black black-check
black:
	python -m black wassap
black-check:
	python -m black wassap --check

.PHONY: interrogate
interrogate:
	python -m interrogate wassap

.PHONY: test
test:
	python -m pytest wassap -vv

.PHONY: check-commit
check-commit:
	make -j 3 black-check interrogate test