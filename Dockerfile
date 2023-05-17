FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN mkdir -p /my_test
WORKDIR /my_test
COPY ./requirements.txt /my_test

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install tensorrt==8.6.1