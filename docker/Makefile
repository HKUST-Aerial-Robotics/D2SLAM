all: jetson_base jetson x86

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build all images"
	# @echo "   1. make pull             - pull all images"
	@echo "   1. make clean              - remove all images"
	@echo ""

x86:
	@docker build  -t d2slam:x86 -f ./Dockerfile.x86 ..

jetson:
	@docker build  -t d2slam:jetson -f ./Dockerfile.jetson ..

jetson_base:
	@docker build  -t d2slam:jetson_base_35.1.0 -f ./Dockerfile.jetson_base_35.1.0 --build-arg USE_PROC=20 ..

clean:
	@docker rmi -f xuhao1/d2slam:pc
	@docker rmi -f xuhao1/d2slam:jetson
	@docker rmi -f xuhao1/d2slam:jetson_base_35.1.0

upload_all: pc jetson_base jetson
	@docker tag d2slam:pc $(name)/d2slam:pc
	@docker tag d2slam:jetson $(name)/d2slam:jetson
	@docker tag d2slam:jetson_base_35.1.0 $(name)/d2slam:jetson_base_35.1.0
	@docker push $(name)/d2slam:pc
	@docker push $(name)/d2slam:jetson
	@docker push $(name)/d2slam:jetson_base_35.1.0

upload_base: jetson_base
	@docker tag d2slam:jetson_base_35.1.0 $(name)/d2slam:jetson_base_35.1.0
	@docker push $(name)/d2slam:jetson_base_35.1.0
