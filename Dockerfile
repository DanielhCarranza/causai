FROM mancunian1792/causal_inference:latest
# Binder Config
RUN pip install --no-cache notebook
ENV HOME=/tmp
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}


# Repo Config 
COPY . /build
WORKDIR /build
# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir notebook==5.*
RUN python3.7 -m pip install -r requirements.txt 
ENV PYTHONPATH="/mnt:${PYTHONPATH}" 
WORKDIR /mnt
RUN rm -rf /build


# RUN pip install --no-cache --upgrade pip && \
#     pip install --no-cache notebook

