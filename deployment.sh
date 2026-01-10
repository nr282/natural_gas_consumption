#Builds Docker and pushes it to the cloud.

sudo chmod 666 /var/run/docker.sock
docker system prune
docker rmi spectral_api
docker rmi 677276077251.dkr.ecr.us-east-2.amazonaws.com/spectral_technologies/predictigas
docker build -t spectral_api .
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 677276077251.dkr.ecr.us-east-2.amazonaws.com
docker tag spectral_api:latest 677276077251.dkr.ecr.us-east-2.amazonaws.com/spectral_technologies/predictigas
docker push 677276077251.dkr.ecr.us-east-2.amazonaws.com/spectral_technologies/predictigas