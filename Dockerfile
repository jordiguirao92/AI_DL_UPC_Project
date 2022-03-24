# set base image (host OS)
#python:3.8-slim 
#nvidia/cuda:11.3.1-base-ubuntu20.0
FROM nvidia/cuda:11.6.0-base-ubuntu20.04

# set the working directory in the container
WORKDIR /usr/app

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
# install dependencies
RUN pip install -r requirements.txt

# Copy code to the working directory
#COPY src/ .

# command to run on container start
#ENTRYPOINT ["python", "entrypoint.py"] Configured in the docker-compose