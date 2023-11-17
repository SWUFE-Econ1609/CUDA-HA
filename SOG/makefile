#NUMBER := $(shell seq 1 2 10)
NUMBER := 1 2 4 8 12 16 24 32
hello:
	@echo "Hello parallel!"

loop_og:
	for thread in ${NUMBER}; do \
	echo $${thread}; \
	julia -t $${thread} perf_og.jl; \
	done