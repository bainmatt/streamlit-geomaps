# This script contains command line routines to streamline development.
#
# To run a recipe:
#
# 	$ make {recipe-name}
#
#
# To run all quality assurance steps during the development process:
#
# 	$ make qa-suite
#
# To upgrade the current environment and ensure forward compatibility, run:
#
# 	$ make upgrade-deps
#
#
# Makefile reference:
# https://makefiletutorial.com/#makefile-cookbook


.PHONY: doctests py-doctests pytests cov-report qa-suite update-pinned-requirements update-requirements-file upgrade-deps clean render


# -- Variables ---------------------------------------------------------------

# Default suffixes for updating sub-requirements files
suffixes ?= ""


# -- QA building blocks ------------------------------------------------------

doctests:
	python src/run_doctests.py


# Arg `mod` can be a path to a py module (relative to src/<PKG> dir) or a dir
py-doctests:
ifeq ($(mod),)
	@echo "\n\nNo subpackage or module specified. Running all doctests."
	coverage run -m pytest -vv src --doctest-modules
else
	coverage run -m pytest -vv src/$(mod) --doctest-modules
endif
	coverage report


pytests:
	coverage run -m pytest
	coverage report


cov-report:
	coverage html && open htmlcov/index.html


# -- QA suites ---------------------------------------------------------------

# Run in sequence: doctests with coverage via pytest, pytests, typing, linting
qa-suite:
	@echo "\n\n\n1 / 4  Running doctests..."
	@echo "=========================="
	coverage run -m pytest src --doctest-modules
	coverage report

	# @echo "\n\n\n2 / 4  Running pytests..."
	# @echo "========================="
	# coverage run -m pytest
	# coverage report

	@echo "\n\n\n3 / 4  Running type checking..."
	@echo "==============================="
	mypy src

	@echo "\n\n\n4 / 4  Running linting..."
	@echo "========================="
	flake8 src


# -- Environment management building blocks ----------------------------------

# Synchronize versions in sub-requirements files with the latest installations
# by exporting an up-to-date 'requirements_pip.txt' lock file and running
# `make update-requirements-file` on each specified sub-requirements file.
update-pinned-requirements:
	@echo "\n\n\ni / ii  Updating requirements lock file..."
	@echo "=================================================================="
	pip list --format=freeze > requirements_pip.txt

	@echo "\n\n\nii / ii  Updating requirements files..."
	@echo "=================================================================="
	@for suffix in $(suffixes); do \
		current_file=$$(echo requirements$${suffix}.txt); \
		if [ ! -f $${current_file} ]; then \
			echo "\nWarning: $${current_file} not found. Skipping.\n"; \
			continue; \
		fi; \
		make update-requirements-file SUFF=$${suffix}; \
	done
	@echo "All requirements synchronized.\n"


# Extract package names and versions from 'requirements.txt', then
# find lines with matching packages in a sub-requirements file and replace the
# version with that in 'requirements.txt' if it is outdated.
update-requirements-file:
	@echo "> Checking requirements$(SUFF).txt..."
	@echo "------------------------------------------------------------------"
	@packages=$$(cat requirements_pip.txt); \
	\
	for package in $$packages; do \
		package_name=$$(echo $${package} | cut -d '=' -f1); \
		package_version=$$(echo $${package} | cut -d '=' -f3); \
		matched_line=$$(grep "^$${package_name}=" requirements$(SUFF).txt); \
		\
		if [ ! -z "$${matched_line}" ]; then \
			matched_version=$$(echo $${matched_line} | cut -d '=' -f3); \
			\
			if [ "$${package_version}" != "$${matched_version}" ]; then \
				echo "$${package_name} version mismatch. Updating:" \
				"v$${matched_version} -> v$${package_version}"; \
				sed -i '' "s|^$$package_name=.*|$${package_name}==$${package_version}|" requirements$(SUFF).txt; \
			fi; \
		fi; \
	done
	@echo "done."
	@echo "------------------------------------------------------------------"


# -- Environment management suite --------------------------------------------

# Run this step to ensure forward compatibility.
# First update all dependencies in the current environment, then run CI suite:
# doctests, typing, linting, and build the documentation. If step 1 runs:
#
# (1) without error, make a lockfile and update any pinned requirements;
# (2) with error, print instructions to reverse the latest upgrades.
#
upgrade-deps:
	@echo "\n\n\nStep 1 / 3 : Upgrading dependencies..."
	@echo "------------------------------------------------------------------"
	conda env update --file environment.yml --prune

	@echo "\n\n\nStep 2 / 3 : Running continuous integration..."
	@echo "------------------------------------------------------------------"
	@if ! make qa-suite; then \
		conda list --revisions; \
		echo "\nWarning! Continuous integration failed."; \
		echo "Your package is currently not forward compatible."; \
		echo "Here's what to do until you make the necessary revisions:\n"; \
		echo "(1) Review the latest revision above (rev N)"; \
		echo "(2) Revert to the previous revision (rev n = N - 1):\n"; \
		echo "    conda install --revision n\n"; \
		echo "(3) Pin packages causing the error at their nth rev in:\n"; \
		echo "    (a) environment.yml"; \
		echo "    (b) the appropriate requirements_* file\n"; \
		exit 1; \
	fi

	@echo "\n\n\nStep 3 / 3 : Upgrading requirements lock files..."
	@echo "------------------------------------------------------------------"
	make update-pinned-requirements
	conda list --revisions


# -- Cleanup -----------------------------------------------------------------

clean:
	@echo "Cleaning up..."
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	# rm -rf .quarto
	# rm -rf _site
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete


# -- Render ------------------------------------------------------------------

render:
	quarto preview notebooks/
