all: jetson_base jetson pc

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build all images"
	# @echo "   1. make pull             - pull all images"
	@echo "   1. make clean              - remove all images"
	@echo ""

pc:
	@docker build  -t d2slam:pc -f ./Dockerfile ..

jetson:
	@docker build  -t d2slam:jetson -f ./Dockerfile.jetson ..

jetson_base:
	@docker build  -t d2slam:jetson_base_35.1.0 -f ./Dockerfile.jetson_base_35.1.0 --build-arg USE_PROC=20 ..

clean:
	@docker rmi -f d2slam:pc
	@docker rmi -f d2slam:jetson
	@docker rmi -f d2slam:jetson_base_35.1.0

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
