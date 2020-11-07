# FROM mancunian1792/causal_inference:latest
FROM fentechai/cdt:0.5.21


COPY . /build
WORKDIR /build
# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir notebook==5.*
RUN python3.7 -m pip install -r requirements.txt 
ENV PYTHONPATH="/mnt:${PYTHONPATH}" 

RUN rm -rf /build