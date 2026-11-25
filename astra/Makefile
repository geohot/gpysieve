CXX ?= g++
CXXFLAGS ?= -O3
PYTHON ?= python3
GP ?= gp

.PHONY: native test
native: chars join divisor_lift rank_divisors triple_orders

chars join: %: %.cpp
	$(CXX) $(CXXFLAGS) -std=c++20 $< -o $@

divisor_lift rank_divisors triple_orders: %: %.cpp
	$(CXX) $(CXXFLAGS) -std=c++20 $< -lgmpxx -lgmp -o $@

test: native
	@command -v "$(GP)" >/dev/null || { echo 'Install PARI/GP or set GP=/path/to/gp'; exit 1; }
	OPENBLAS_NUM_THREADS=1 GP="$(GP)" ASTRA_GP="$(GP)" $(PYTHON) -m unittest -q
