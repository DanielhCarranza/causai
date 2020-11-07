FROM mancunian1792/causal_inference:latest
# FROM divkal/cdt-py3.7:0.5.21
# If you're behind a proxy url, enter it here.
#ENV http_proxy "<http_proxy_url>"
#ENV https_proxy "<https_proxy_url>"

COPY . /build
WORKDIR /build

RUN python3.7 -m pip install -r requirements.txt
ENV PYTHONPATH="/mnt:${PYTHONPATH}" 
WORKDIR /mnt
RUN rm -rf /build