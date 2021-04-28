FROM continuumio/miniconda
RUN apt-get update && apt-get install -y libxi-dev libxcursor-dev libxdamage-dev libxcomposite-dev x11-xserver-utils libxinerama-dev
RUN conda create --name batcave -c conda-forge -c menpo graph-tool python=3.8 pandas scipy scikit-learn tqdm patchelf osmesa

