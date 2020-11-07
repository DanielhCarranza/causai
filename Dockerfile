# FROM mancunian1792/causal_inference:latest
# # FROM fentechai/cdt:0.5.21


# COPY . /build
# WORKDIR /build
# # Install Python dependencies
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir notebook==5.*
# RUN python3.7 -m pip install -r requirements.txt 
# ENV PYTHONPATH="/mnt:${PYTHONPATH}" 

# RUN rm -rf /build

FROM jupyter/scipy-notebook:95ccda3619d0
RUN pip install --no-cache-dir notebook==5.*

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
    
# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}