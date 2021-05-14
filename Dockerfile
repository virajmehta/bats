FROM continuumio/miniconda
RUN apt-get update && apt-get install -y libxi-dev \
                                         libxcursor-dev \
                                         libxdamage-dev \
                                         libxcomposite-dev \
                                         x11-xserver-utils \
                                         libxinerama-dev \
                                         unzip \
                                         gcc \
                                         multiarch-support \
                                         libgl1-mesa-dev \
                                         libgl1-mesa-glx \
                                         libosmesa6-dev
RUN wget https://launchpad.net/~ubuntu-security-proposed/+archive/ubuntu/ppa/+build/7110687/+files/libgcrypt11_1.5.4-2ubuntu1.1_amd64.deb
RUN dpkg -i libgcrypt11_1.5.4-2ubuntu1.1_amd64.deb
COPY container/environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "batcave", "/bin/bash", "-c"]
# RUN pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN mkdir ~/.mujoco
RUN wget https://www.roboti.us/download/mujoco200_linux.zip
RUN unzip mujoco200_linux.zip
RUN mv mujoco200_linux ~/.mujoco/mujoco200
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
RUN pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html tensorboard
COPY container/mujoco-py mujoco-py
RUN pip install -e mujoco-py
RUN python -c "import torch"
RUN python -c "import graph_tool"
COPY container/mjkey.txt .
RUN mv mjkey.txt ~/.mujoco/
RUN ln -s ~/.mujoco/mujoco200/ ~/.mujoco/mujoco200_linux
RUN pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
RUN pip install ipdb
RUN mkdir /src
COPY . /src
RUN mkdir /src/experiments
