FROM tiagopeixoto/graph-tool
RUN pacman -Syu && pacman -S --noconfirm libxi\
                                         libxcursor\
                                         libxdamage\
                                         libxcomposite\
                                         libxinerama\
                                         unzip\
                                         gcc\
                                         mesa\
                                         libgcrypt\
                                         python\
                                         git\
                                         patchelf\
                                         python-pip\
                                         wget
RUN pip install pandas scipy scikit-learn tqdm ipdb gym torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html tensorboard
RUN mkdir mujoco
RUN wget https://www.roboti.us/download/mujoco200_linux.zip
RUN unzip mujoco200_linux.zip
RUN mv mujoco200_linux /mujoco/mujoco200
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/mujoco/mujoco200/bin
COPY container/mujoco-py mujoco-py
RUN pip install -e mujoco-py
COPY container/mjkey.txt .
RUN mv mjkey.txt mujoco/
RUN mkdir ~/.mujoco
RUN cp -r /mujoco/mujoco200 ~/.mujoco/mujoco200_linux
# RUN ln -s mujoco/mujoco200/ ~/.mujoco/mujoco200
# RUN ln -s mujoco/mujoco200/ ~/.mujoco/mujoco200_linux
# ENV MUJOCO_PY_MUJOCO_PATH /mujoco
# ENV MJLIB_PATH /mujoco/mujoco200_linux/bin/
# ENV MJKEY_PATH /mujoco/mjkey.txt
RUN echo $MJLIB_PATH
RUN pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
RUN mkdir /bats
COPY . /bats
RUN mkdir /bats/experiments
RUN rm /mujoco-py/mujoco_py/generated/mujocopy-buildlock
RUN chmod 777 -R /mujoco-py
CMD /bin/bash
