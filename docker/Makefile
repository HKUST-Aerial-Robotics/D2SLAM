all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build all images"
	# @echo "   1. make pull             - pull all images"
	@echo "   1. make clean              - remove all images"
	@echo ""

build:
	@docker build  -t xuhao1/d2slam:pc -f ./Dockerfile ..

clean:
	@docker rmi -f xuhao1/d2slam:pc