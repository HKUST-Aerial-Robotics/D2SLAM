all: jetson_base jetson_orin pc

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build all images"
	# @echo "   1. make pull             - pull all images"
	@echo "   1. make clean              - remove all images"
	@echo ""

x86:
	@docker build  -t hkustswarm/d2slam:x86 -f ./Dockerfile.x86 ..

jetson_orin:
	@docker build  -t hkustswarm/d2slam:jetson_orin -f ./Dockerfile.jetson ..

jetson_base:
	@docker build  -t hkustswarm/d2slam:jetson_base_35.1.0 -f ./Dockerfile.jetson_base_35.1.0 --build-arg USE_PROC=20 ..

jetson_orin_base:
	@docker build  -t hkustswarm/d2slam:jetson_orin_base_35.3.1 -f ./Dockerfile.jetson_orin_base_35.3.1 --build-arg USE_PROC=20 ..

arm64_base:
	@docker build  -t hkustswarm/d2slam:arm64_base_ros1_noetic -f ./Dockerfile.arm64_base_ros1_noetic ..

arm64:
	@docker build  -t hkustswarm/d2slam:arm64_ros1_noetic -f ./Dockerfile.arm64_ros1_noetic ..

clean:
	@docker rmi -f d2slam:pc
	@docker rmi -f d2slam:jetson
	@docker rmi -f d2slam:jetson_base_35.1.0
	@docker rmi -f d2slam:jetson_orin_base_35.3.1
	@docker rmi -f d2slam:arm64_base_ros1_noetic
	@docker rmi -f d2slam:arm64_ros1_noetic

upload_all: pc jetson_base jetson
	@docker tag d2slam:pc $(name)/d2slam:pc
	@docker tag d2slam:jetson $(name)/d2slam:jetson
	@docker tag d2slam:jetson_base_35.1.0 $(name)/d2slam:jetson_base_35.1.0
	@docker push $(name)/d2slam:pc
	@docker push $(name)/d2slam:jetson
	@docker push $(name)/d2slam:jetson_base_35.1.0
	@docker push $(name)/d2slam:jetson_orin_base_35.3.1
	@docker push $(name)/d2slam:arm64_base_ros1_noetic
	@docker push $(name)/d2slam:arm64_ros1_noetic

upload_base: jetson_base
	@docker tag d2slam:jetson_base_35.1.0 $(name)/d2slam:jetson_base_35.1.0
	@docker push $(name)/d2slam:jetson_base_35.1.0
	@docker tag d2slam:jetson_orin_base_35.3.1 $(name)/d2slam:jetson_orin_base_35.3.1
	@docker push $(name)/d2slam:jetson_orin_base_35.3.1
	@docker tag d2slam:arm64_base_ros1_noetic $(name)/d2slam:arm64_base_ros1_noetic
	@docker push $(name)/d2slam:arm64_base_ros1_noetic