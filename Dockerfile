FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
WORKDIR /work

RUN apt-get update
#RUN apt-get install -y python3.10 python3-pip
RUN apt-get install -y gcc git zip unzip wget curl htop
RUN apt upgrade -y openssl tar

RUN apt-get install -y tree
RUN apt-get install -y git vim wget
RUN apt-get install -y git-lfs

RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install torchinfo
RUN pip3 install pillow
RUN pip3 install ipykernel
RUN pip3 install kaggle
RUN pip3 install pandas
RUN pip3 install albumentations
RUN pip3 install scikit-learn
RUN pip3 install pyyaml
RUN pip3 install timm
RUN pip3 install regex
RUN pip3 install opencv-python
RUN pip3 install deepspeed # to accelerate training and save gpu memory cost
RUN pip3 install tensorboard

# RUN apt install ffmpeg libsm6 libxext6 -y

# RUN mkdir /work/dataset/
# RUN mkdir /work/dataset/CULane
# RUN mkdir /work/dataset/LLAMAS
# RUN mkdir /work/dataset/Tusimple


