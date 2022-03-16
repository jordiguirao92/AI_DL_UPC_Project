build_containers:
	docker network create ai-project || true
	docker-compose build

service-up: build_containers
	docker-compose up --remove-orphans

service-down:
	docker-compose down -v

tensorboard: service-up
	docker-compose run --rm ai-image-denoising bash -c 'tensorboard --logdir=logs'

dive:
	dive build -t image-denoising .