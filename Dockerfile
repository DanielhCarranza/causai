# FROM mancunian1792/causal_inference:latest
FROM datascienceschool/rpython:803dbe97e778e576d9e7ae8d6f17cdb80c0a0a91e375c714cd9de33bd4c15658

# If you're behind a proxy url, enter it here.
#ENV http_proxy "<http_proxy_url>"
#ENV https_proxy "<https_proxy_url>"

COPY . /build
WORKDIR /build
# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir notebook==5.*
RUN python3.7 -m pip install -r requirements.txt 
ENV PYTHONPATH="/mnt:${PYTHONPATH}" 

RUN rm -rf /build