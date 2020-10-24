FROM mancunian1792/causal_inference:latest

# If you're behind a proxy url, enter it here.
#ENV http_proxy "<http_proxy_url>"
#ENV https_proxy "<https_proxy_url>"

COPY . /build
WORKDIR /build
# Install Python dependencies
RUN pip install --upgrade pip
RUN python3.7 -m pip install -r requirements.txt 
ENV PYTHONPATH="/mnt:${PYTHONPATH}" 
WORKDIR /mnt
RUN rm -rf /build