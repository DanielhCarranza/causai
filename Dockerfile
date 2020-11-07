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
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook

# ARG NB_USER=causai
# ARG NB_UID=1000
# ENV USER ${NB_USER}
# ENV NB_UID ${NB_UID}
# ENV HOME /home/${NB_USER}

# RUN adduser --disabled-password \
#     --gecos "Default user" \
#     --uid ${NB_UID} \
#     ${NB_USER}
# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}

