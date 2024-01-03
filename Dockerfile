# Use an official Miniconda image as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN conda env create -f environment.yml -n adaptae

ENV PYTHONPATH="./:$PYTHONPATH"
