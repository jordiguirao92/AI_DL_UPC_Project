# Image Denoising

## Technologies used for the project :computer:
- Python
- Pytorch
- Minoconda
- Docker & Docker Compose
- Google Cloud
- Make

## Start the project :smiley:
Use the following steps to start the project.

### Create environment :book:

#### Option A - Using Miniconda
- Install miniconda in your computer. You can follow the steps in this link: https://docs.conda.io/en/latest/miniconda.html
- Create conda environment, only one time, if it has not been created yet: `conda create --name aidl-project python=3.8`
- Activate conda environment: `conda activate aidl-project` In order to deactivate conda environment you can use `conda deactivate`
- Install project dependencies: `pip install -r requirements.txt`
- If you are using GPU be sure that you have the drivers installed.
- Install cuda in order to use GPU `pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html`
- Now you have your project environment ready. Continue with the [preparation of dataset](#prepare-dataset-steps).

#### Option B - Using Docker & Docker Compose
- Install Docker in your computer. You can follow the steps in this link: https://docs.docker.com/desktop/#download-and-install
- Install Docker Compose in your computer. You can folloe the setps in this link: https://docs.docker.com/compose/install/
- If you use GPU install Docker requirements to let Docker use computer GPU in the container. You can install requirements using the following link: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
- Now you have your project environment ready. Continue with the [preparation of dataset](#prepare-dataset-steps).


### Prepare dataset :open_file_folder:
#### Option A - Dowload the prepared images
- Dowload the dataset from the following link: https://mega.nz/folder/Erg2EYiS#kVS2e-nAGL3etSqgeZ4lbw The images folder container the prepared dataset with the correct image name, also contain the validation, training and testing txt files.
- Copy the folder `images` in the `./dataset/`.
- Now, you have your dataset prepared.


#### Option B - Download original images
- Dowload the original images from the following link: https://www.eecs.yorku.ca/~kamel/sidd/dataset.php If you use this way, you need to download the images in different parts. We recomment to use the [option A](#option-a---dowload-the-prepared-images).
- Paste the original images in the folder `/dataset/original`.
- In orde to rename original images, slice, move and rename sliced images run the command: `python ./src/utils/prepare_dataset.py`.
- Now, you have your dataset prepared in the folder `/dataset/images`.



### Run the project :runner:

In the [parser file](./src/utils/parser.py) you can find the posible configurations of the training. Feel free to change the configurations.


#### Option A - Using conda environment
- Use the following commmand to start the model training: `python ./src/main.py`.
- As we have commented above you can run a training changing the configurations. You can change it in the code or by command. Find some examples:
    - Generator with Batchnorm and Sigmoid `python ./src/main.py --net generator --generator_norm batch --generator_last sigmoid`
    - Generator with Spectral and Tanh `python ./src/main.py --net generator --generator_norm spectral --generator_last tanh`
    - Generator with Instance and Tanh + Discriminator with LeakyRelu and Tanh, D=40 `python ./src/main.py --net gan --generator_norm instance --generator_last tanh --discriminator_last tanh --discriminator_activation leakyRelu --d_weight 40`

#### Option B - Using Docker-Compose
- In the following [file](./Dockerfile) you can find the image to build with docker compose.
- You can check the [docker compose configuration file](./docker-compose.yaml).
- We recommend that you set the training configuration changing the default configurations in [parser file](./src/utils/parser.py)
- Also, you can set the training command in the [docker compose configuration file](./docker-compose.yaml). You can find some commented examples.
- To run the training with docker-compose use the following command: `make service-up`
- To see the container running use: `docker ps`.
- In case you need to see the process logs, after the training use: `docker-compose logs -t -f --tail="all"`
- If you want to save the logs in a log file:
    1. Create the log file: `touch nameFile.log`
    2. Run: `docker-compose logs -t -f --tail="all" > nameFile.log`
    3. To print the logs file: `cat nameFile.log`
- In order to stop the contaner use: `make service-down`



After the training you can find the checkpoints [here](./checkpoints) and the training logs [here](./logs).


## Check logs with tensorboard :chart_with_upwards_trend:
We use tensorboard to check the logs. When you run a training you the logs of the process will appear in the folder `./logs`.
In order to check the logs in tensorboard, run the following command: `tensorboard --logdir=logs`

## Troubles and challenges :sweat:

### Instance requirements and configuration
Due to our inexperience we have had some doubts about instance configuration and requirements, as how much memory we need, which GPU is more suitable, instance configurations, etc.
### Google Cloud GPU
Google Cloud don’t let you add GPUs to your instance without authorization. We had needed to request for a GPU to Google Cloud. In some cases, the response is so fast, but in other ones not.
### Dataset selection and preparation
Find a good dataset that match with the project necessities sometimes is a hard task. In our case, we had the possibility to create our noise image dataset, but finally we decided to select a dataset with real noisy images in order to work in a real environment. One of the most important tasks is to prepare and adapt the dataset to our project
### Bad TXT dataset files
One of the big bugs during the project was that we created a TXT files with the clear and noisy images. Our dataset charges the image files in a wrong way. We were creating pairs with clearly and noise images, the results of our training were not properly correct. Due to this mistake, we needed to repeat all the model trainings. It was a hard experience.
### Transfer dataset to Google Cloud instance
We needed to do research about the best way to upload images to a Google Cloud instance in a fast way. Finally, we decided to use SCP. It works so fast.
### GPU Nvidia drivers & Cuda installation
We have had some difficulties during the GPU Nvidia drivers & Cuda installation. Finally, thanks to Google and our project mentor, our model works with cuda. 
### Docker configuration
Due to the problems with GPU drivers and cuda installation, we decided to use docker to create our development environment. Docker is very useful because you only need to prepare the image configuration one time, after that you can use it in different instances, avoiding all the configurations a requirements installations difficulty. Also, we have some difficulties to find the good configuration of our Dockerfile image. One of the challenges was to configure the Docker to use GPU.
### Linux/Windows
The team members have different OS to work, ones use Windows and others uses Linux. Python paths and other process works different in Windows and Linux. It was a little mess. Finally, thanks to the instance, it was solved.
### Find the best possible model
Due to our inexperience, we need to do a lot of trainings and testing different training configurations and hyperparameters in order to find the best model for the project.

## Development Learnings :thumbsup:

- Find a suitable dataset.
- Avoid to spend a lot of time in preparing the dataset.
- Try to use Docker. It will make it easier to deploy.
- Before start, try to define a realistic project plan.
- Create a good code structure. It will make it easier in front of changes.
- Separate big functions in small functions.
- Check that the dataset class is corrected defined.
- Select appropriate metrics to evaluate results.
- Defined a group of models to test. Iterate through results.
- Set up model without big changes to find the good performance.
- First try to get a good generator model alone, and after that try to improve it with a Discriminator.
- Better to train with a reduced dataset.
- Using a scheduler can provide better results.
- Understand what you are doing to take better decisions.
