# Start with Python base image
# ATTENTION: Modify the image below to your need - PyTorch, Tensorflow, NVidia, etc.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Setting a working directory explicitly
WORKDIR /user

# copy the contents of source files to the docker container
COPY ./src /user 

# install Python dependencies that are specified in the requirements.txt file
# ATTENTION: Make sure your code dependencies are listed in the requirements.txt file
#RUN pip install -r requirements.txt

RUN python -m pip install --no-cache-dir /user/

# launching the code when the docker container in this image is run
ENTRYPOINT [ "/bin/bash", "/user/run.sh" ]