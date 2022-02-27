# set base image (host OS)
FROM python:3.8-slim

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