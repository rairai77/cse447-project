FROM python:3.11-slim
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# No dependencies needed - n-gram model uses only standard library (pickle, collections)
