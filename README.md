# Image Denoising

# AI_DL_UPC_Project
Final Project for the UPC Artificial Intelligence with Deep Learning Postgraduate Course 2021-2022 edition, authored by:
* Adrià Hdez
* Alessia Squitieri
* Jordi Guirao
* Nil Mira

Advised by professor Eva Mohedano

## Objective
Create a GANs for image denoising. First of all, we tried to prepare the images with artificial noise, but after analyzing it, we have decided to search for a dataset with real noised images, to do a better real approach in the model. We want to apply the model in real cases and not in artificial noisy cases. 

## Table of content
 1. Introduction
     * Why image denoising?
     * Choice of the model
     * General architecture 
 2. Milestones
 3. Methods 
     * Pix2pix
 4. Dataset
 5. Environment set up 
 6. Preliminary tests
 7. Experiments
 8. Metrics
 9. Conclusions and future goals

## Introduction

### _Why image denoising?_

Due to inherent physical limitations of various recording devices, denoising has been an important problem in image processing for many decades[1]. Noise refers to a basic signal distortion which hinders the process of image observation and information extraction[1]. There are different types of noise that can prevail according to the kind of image. As described by A. Buades et al. in A REVIEW OF IMAGE DENOISING ALGORITHMS, WITH A NEW ONE, the most frequently discussed noises are Additive White Gaussian Noise (AWGN), impulse noise (salt and pepper), quantisation noise, Poisson noise and speckle noise. Image denoising is aimed at recovering a clean image x from a noisy observation y which follows an image degradation model y = x + v[2].

### _Choice of the model_

Despite the incredible number of applications that Convolutional Neural Networks (CNNs) provide in the field of image prediction problems, they are not the best solution when dealing with image to image translation tasks. In fact, since CNNs try to minimize the distance between predicted and ground truth pixels, it will tend to produce blurry results[3]. 

### _General architecture_

In this context, a great solution has been provided by Generative Adversarial Networks (GANs). GANs, instead,  learn a loss that tries to classify if the output image is real or fake and, at the same time, they train a generative model to minimize this loss. For this reason, blurry images will not be tolerated, since they look obviously fake. In a general way, a GAN is formed by a generator (G) and a discriminator (D). The G is aimed to be trained to produce outputs that cannot be distinguished from “real” images by an adversarially trained D, which , instead, is trained to do as well as possible at detecting the generator’s “fakes”. The general structure of a GAN is shown in the figure below.  


![fig 1-Generator-and-Discriminator](https://learnopencv.com/wp-content/uploads/2021/07/Pix2Pix-Discriminator-working-as-patch-discriminator-1.jpg)

_Figure 1: general architecture of a cGAN, taken from:https://learnopencv.com/paired-image-to-image-translation-pix2pix/#discriminator_


In 2014, Conditional Generative Adversarial Nets by Mehdi Mirza and Simon Osindero was published. The main idea behind this GANs implementation is that both generator and discriminator are fed a class label  and conditioned on it. All other components are exactly the same as a typical  Generative Adversarial Networks. In order to accomplish our task, we decided to implement a pix2pix model, a specific type of conditional GAN (cGAN). 


## Milestones
* Obtain simulated images,
* Providing a working Pix2Pix mode,
* Optimizing final model through hyperparameters tuning,
*  Providing a well functioning and user friendly way to reproduce the model.


## Methods 

### _Pix2Pix Model_



![Pix2pix](https://learnopencv.com/wp-content/uploads/2021/07/pix-2-pix-GAN-a-type-of-CGAN.jpg) 

_Figure 2: Pix2pix model, image taken from the original paper._


A pix2pix model is a special type of cGAN in which the G follows a U-Net architecture and the D is a patchGAN. 

The U-Net architecture is shown in the figure above. It is a symmetric structure that consists of two major parts, the left part that is called contracting, which is formed by a general convolutional process, and the right part that is the expansive path, constituted by a transposed 2d convolutional layer[https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5]. 
The D is instead a PatchGAN, that only penalizes structure at the scale of patches. This discriminator tries to classify if each N ×N patch in an image is real or fake, described in [3]. The authors demonstrated that N can be much smaller than the full size of the image and still produce high quality results. This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied to arbitrarily large images.


![UNet](https://miro.medium.com/max/1400/1*J3t2b65ufsl1x6caf6GiBA.png)

_Figure 3: UNet architecture, image taken from: https://towardsdatascience.com/paper-summary-u-net-convolutional-networks-for-biomedical-image-segmentation-13f4851ccc5e_


## Dataset

Two different datasets were selected according to the goal,  both providing clean - noisy image pairs. The  PolyU-Real-World-Noisy-Images-Dataset was initially used in our Google Colag, to explore the general architecture of the model, while  the Smartphone Image Denoising Dataset was used for the final training. The final dataset contains 634 images of 4000x3000 pixels that were cropped to 12. So, we finally generated 7608 images, splitted into training, validation and testing sets. The training set was covered by 70% of the images, the validation by 15% and the test 15% too. At the beginning we tried to train our model by using the whole dataset but it was too computationally expensive. For this reason, and according to what the authors of the model suggested, we decided to train our model using a reduced dataset of 408 pares, obtaining good results. The evaluation dataset was reduced to 96 pares while the test set size was not modified, since we were not worried about the timing of the train and also to provide the same test set used previously. A second crop of 256 was applied, generating a final input of 256x256x3. The final dataset is available and we provide the following link to access and take a look to it:https://mega.nz/folder/Erg2EYiS#kVS2e-nAGL3etSqgeZ4lbw.  

![example of cleannoisy](https://miro.medium.com/max/1400/1*5bsoVIT2La_5-GDK6Vljyg.png)

_Figure 4: example of clean - noisy image pairs, taken by https://medium.com/analytics-vidhya/image-denoising-using-deep-learning-dc2b19a3fd54_


## Environment set up 

We used  [Google colab](https://colab.research.google.com/drive/1WSC55uiKy5inDzE7SJUzU66J27wMnInf?usp=sharing) to design and test our model, because it provides a single 12GB NVIDIA Tesla K80 GPU that can be used for up to 12 hours continuously. We also created a Google Cloud Deep Learning VM instance for longer training, using the following parameters:

Instance parameters  | 
:-------------: |
us-central1-a | 
n1-standard-8 | 
GPU: 1x Nvidia Tesla P4 | 
Disco SSD persistente 100GB | 
1 GPU | 


The whole model was developed in Python programming language, using the Pytorch library. Finally, Visual Studio code was selected as code editor and the whole development process was implemented using this GitHub repository. We also provide a Docker image to use our final model. 

## Preliminary tests

Our first approach, in order to better understand the pix2pix functionality, has been working on a Google Colab. We implemented a first model, based on the main characteristic we found from the original paper. Through the following link it is possibile access and have a look at our Google Colab. We used only 100 pairs of images to try the model, setting the following hyperparameters: 


Batch size  | Number of epochs | Test batch size | Learning rate | Log interval | Criterion |
:-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
1 | 50 | 1 | 1e-3 | 5 | L1 | 


Our model implementation was inspired by Pix2Pix:Image-to-Image Translation in PyTorch & TensorFlow and by the original paper[4]. As commented before, the model used was a pix2pix. For the data pre - processing part, we developed some functions to read, sort and crop the images. We used the OpenCV with the functions:

* `cv2.imread()` : The function imread loads an image from the specified file and returns; 1 is for NOT gray scale images

* `cv2.cvtColor()` : Converts an image from one color space to another. When code is cv2.COLOR_BGR2RGB, BGR is converted to RGB.

Then, we implemented the generator. As said before, the generator is a U-Net. The architecture is “U-shaped”, hence the name “U-Net”. The first half represents the contracting part and the second one the upsampling one. To implement the encoder part we used the following Pytorch functions:

* `Conv2D`: Applies a 2D convolution over an input signal composed of several input planes.
* `nn.ReLU()`: Applies the rectified linear unit function element-wise: ReLU(x)=(x)+=max(0,x)
* `nn.MaxPool2d`: Applies a 2D max pooling over an input signal composed of several input planes.
* `nn.BatchNorm2d`: Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .
* `nn.InstanceNorm2d`: Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization.
* `nn.utils.spectral_norm`: Spectral normalization stabilizes the training of discriminators (critics) in Generative Adversarial Networks (GANs) by rescaling the weight tensor with spectral norm of the weight matrix calculated using power iteration method. If the dimension of the weight tensor is greater than 2, it is reshaped to 2D in the power iteration method to get spectral norm. This is implemented via a hook that calculates spectral norm and rescales weight before every forward() call. See Spectral Normalization for Generative Adversarial Networks.
 
Basically, the encoder consists of a sequence of blocks for down-sampling operations. Each block includes several convolution layers, followed by max-pooling layers. After each down - sampling operation, the number of filters in the convolutional layers is doubled. In the end, the encoder outputs a learned feature map for the input image. The decoder, instead,  is designed for up-sampling. The decoder first utilizes a deconvolutional layer to up-sample the feature map generated by the encoder. The deconvolutional layer contains the transposed convolution operation, that is represented in Pytorch by:
* `ConvTranspose2d`: Applies a 2D transposed convolution operator over an input image composed of several input planes. This module can be seen as the gradient of Conv2d with respect to its input. It is also known as a fractionally-strided convolution or a deconvolution (although it is not an actual deconvolution operation as it does not compute a true inverse of convolution).
In the decoder part we do not apply the max pooling operation, because we are not trying to reduce the image. Finally, we create the last class in which the encoder and decoder are merged in a unique function (the U-Net generator one). 

Then, we define the Discriminator, that is a PathGAN. This type of discriminator tries to classify if each N ×N patch in an image is real or fake[4]. Moreover, in the pix2pix model the classification is conditioned. Infact, the discriminator takes both the source image and the target image as input and predicts the likelihood of whether the target image is real or a fake translation of the source image. 

![disc](https://www.researchgate.net/publication/339832261/figure/fig5/AS:867699496345602@1583887089690/The-PatchGAN-structure-in-the-discriminator-architecture.ppm)

_Figure 5: patchGAN discriminator illustration, taken from https://www.researchgate.net/figure/The-PatchGAN-structure-in-the-discriminator-architecture_fig5_339832261_ 

## Experiments 

To explore all the potentialities of our model we performed different types of tests. Our first implementations have been made in the Google Colab to give us a general idea of the model potentialities. Following the experiments performed by the authors of the pix2pix model, we also worked on the discriminator receptive field. The receptive field is the relationship between one output activation of the model to an area on the input image. In pix2pix the receptive field can be understood as the patch gan size. As written in the original paper “​​ … we design a discriminator architecture – which we term a PatchGAN – that only penalizes structure at the scale of patches. This discriminator tries to classify if each NxN patch in an image is real or fake. We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate output of D.” According to the paper, it can be calculated with the following equation: 

` receptive field = (output size - 1) * stride + kernel size `

We decided to perform the following test:

1. Evaluate the generator only;
2. EValuate the whole GAN with the PatchGAN suggested by the authors (70x70);
3. Evaluate the whole GAN changing the PatchGAN size. 

Name	| Test SSIM	| Test PSNR	| Test Loss	| Net | 	LR	 | Batch Size |	Epochs	 | Loss	| D_weight	| Gen Last |	Gen Norm |	Disc Last	 | Disc Norm	| Disc Act	| Disc Size | 	Dataset | 
:-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | 
GAN-training-20220420-200407	| 0.83	| 27.82 |	0.01	| GAN	| 0.0001| 	4 |	25	| L1	| 40	 | Sigmoid |	Batch	| Sigmoid	| Batch	| LeakyRelu	 | 14	| Reduced Github
GAN-training-20220422-180735	| 0.83	 |24.45 |	0.00 |	GAN	| Scheduler 3	| 4	| 25	| L1	| 5	| Tanh |	Spectral	| Tanh	| Batch	| LeakyRelu	| 15 |	Reduced Github
generator-training-20220420-195655 |	0.91 |	32.05 |	0.02	| Generator	| 0.0001 |	4	| 25	| L1| 	NA	| Tanh	| Spectral	| NA	| NA |	NA	 | NA |	Reduced Github
GAN-training-20220421-04226	| 0.88 |	31.34 |	0.00	| GAN	| 0.0001 + HardCode at epoch 18 to 0.00001 |	4	| 25	| L1	| 40	| Tanh	| Spectral	| Tanh	| Instance	| LeakyRelu	| 14 |	Reduced Github
GAN-training-20220421-205256	| 0.88	| 31.56	| 0.00	| GAN	| Scheduler  2	| 4	| 25	| L1	| 10	| Tanh	| Spectral |	Sigmoid |	Instance	| LeakyRelu	| 1	| Reduced Github
GAN-training-20220422-180332| 	0.87	| 30.48 |	0.00	| GAN	| Scheduler  3	| 4	| 25 |	L1	 |40	| Tanh	| Spectral |	Tanh	| Batch	| LeakyRelu |	15	| Reduced Github
GAN-training-20220422-191636 |	0.89 |	32.41 |	0.00	| GAN	 | Scheduler 3	| 4	| 25	| L1	| 40	| Tanh |	Spectral |	Sigmoid	 | Batch |	LeakyRelu	| 15 |	Reduced Github
GAN-training-20220422-201239	| 0.90 |	32.93 |	0.00 |	GAN	| Scheduler  4	| 4	| 25	| L1	| 40	| Tanh |	Spectral |	Sigmoid |	Batch	| LeakyRelu	 | 14	| Reduced Github

Once obtained our best model, we ran it using 100 epochs:

Name	| Test SSIM	| Test PSNR	| Test Loss	| Net | 	LR	 | Batch Size |	Epochs	 | Loss	| D_weight	| Gen Last |	Gen Norm |	Disc Last	 | Disc Norm	| Disc Act	| Disc Size | 	Dataset | 
:-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | 
GAN-training-20220422-201239| 0.90	| 34.62	| 0.00	| GAN	| Scheduler  4 |	4	| 100	| L1	| 40	| Tanh |	Spectral |	Sigmoid |	Batch	| LeakyRelu	 | 14 | Reduced Github


## Results 

In order to evaluate the models we used two different metrics, the Structural Similarity Index and the peak signal-to-noise ratio. The Structural Similarity Index (SSIM) is a perceptual metric that quantifies image quality degradation* caused by processing such as data compression or by losses in data transmission. It is a full reference metric that requires two images from the same image capture— a reference image and a processed image. The PSNR block computes the peak signal-to-noise ratio, in decibels, between two images. This ratio is used as a quality measurement between the original and a compressed image. The higher the PSNR, the better the quality of the compressed, or reconstructed image.
We initially trained only the generator, achieving very good results. So, our goal was implementing a complete model (generator and discriminator) having better performance than the generator only. In fact, adding the discriminator to the model can be challanging since it is not so easy find the right configuration between the generator and the discriminator. For this reason, our first complete models performed worse than the generator only. However, after different benchmarking we finally obtained a model achieving best results of the generator only. The **GAN-training-20220422-201239** reached best results. 

This is an example of how the generated image looks like using the **GAN-training-20220422-201239** mdoel:

![model_19_True_easy](https://user-images.githubusercontent.com/62135962/164976919-51514ad7-ba24-4b4b-a7f8-2c2f7df0032d.png)

_Figure 6: an example of generated image from our best model_ 


![model_19_False_easy](https://user-images.githubusercontent.com/62135962/164976983-6cc1486a-300f-42e4-a681-7cba11782560.png)
![model_1_False_easy](https://user-images.githubusercontent.com/62135962/164976986-8c68a6ac-d7a0-47e3-adaa-7267a7463415.png)

_Figure 6: comparison of two generated images from ours model. The first one is from our selected model, while the other is from our first model GAN-training-20220420-200407_ 

![model_19_False_easy](https://user-images.githubusercontent.com/62135962/164976983-6cc1486a-300f-42e4-a681-7cba11782560.png)
![model_2_False_easy](https://user-images.githubusercontent.com/62135962/164977337-e6855a41-a419-4e56-a8ef-674d8ee17373.png)

_Fogure 7: comparison of two generated images. The first one is from our selected model, while the second one is obtained from the generator only_ 

## Metrics

In light blue the best GAN model (GAN-training-20220422-201239), in orange the generator only model (generator-training-20220420-195655) and in dark blue the worse GAN model (GAN-training-20220420-200407):
![Schermata 2022-04-24 alle 16 31 53](https://user-images.githubusercontent.com/62135962/164981547-6d124582-16be-4712-b502-904f09d723ea.png)
![Schermata 2022-04-24 alle 16 31 56](https://user-images.githubusercontent.com/62135962/164981546-c524b4a2-48af-4f9b-af4d-e6f59e390ee2.png)

Since the GAN-training-20220422-201239 returned best results, we trained it using 100 epochs and as you can see from the plots below, the metrics improved: 
![Schermata 2022-04-24 alle 16 31 59](https://user-images.githubusercontent.com/62135962/164981545-d325e2d6-6990-46cf-9f50-34fd385534e9.png)

In green the results of the good GAN model using 100 epochs. 

All the models compared:
![Schermata 2022-04-24 alle 16 32 03](https://user-images.githubusercontent.com/62135962/164981543-1e8a2025-72a1-40f2-8135-4f84bc6b0387.png)
![Schermata 2022-04-24 alle 16 32 06](https://user-images.githubusercontent.com/62135962/164981538-56a58e09-53de-4bb8-b667-dd67d90145b0.png)

## Conclusion and future goals

Our project was aimed to reslve the image denoising issue, by using a pix2pix model. Having a look to the state of the art of this task, we noticed that the pix2pix model was not included. So far, looking at our results, we belive that cGAN and in particular pix2pix model could be included in a future benchmarking. It would be interesting to compare our model to the others reported in the state of the art for images denoising and see if, using the same conditions of the others models, we can reach the same results than we obtained now. We could not perform this comparison for a metter of time, but it would be in our future goals. 


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
- Fix the version of the the required project libraries.
- Select appropriate metrics to evaluate results.
- Defined a group of models to test. Iterate through results.
- Set up model without big changes to find the good performance.
- First try to get a good generator model alone, and after that try to improve it with a Discriminator.
- Better to train with a reduced dataset.
- Using a scheduler could provide better results.
- Understand what you are doing to take better decisions.
