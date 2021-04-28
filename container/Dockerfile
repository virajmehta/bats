FROM continuumio/miniconda
RUN apt-get update && apt-get install -y libxi-dev libxcursor-dev libxdamage-dev libxcomposite-dev x11-xserver-utils libxinerama-dev
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "batcave", "/bin/bash", "-c"]
RUN pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -c "import torch"
RUN python -c "import graph_tool"
