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


### Prepare dataset steps :open_file_folder:
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


### Check logs with tensorboard :chart_with_upwards_trend:
We use tensorboard to check the logs. When you run a training you the logs of the process will appear in the folder `./logs`.
In order to check the logs in tensorboard, run the following command: `tensorboard --logdir=logs`



