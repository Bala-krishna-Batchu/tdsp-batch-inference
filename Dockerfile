# Stage 1: Builder
FROM docker-prod.artifactory.tmna-devops.com/ubuntu:22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Update system and install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    openjdk-8-jdk-headless \
    && apt-get clean


# Stage 2: Final Runtime Image
FROM docker-prod.artifactory.tmna-devops.com/coc/chofer-golden-containers/language-os/amazonlinux2023-python3:2.7.0
USER root

RUN mkdir -p /usr/share/terminfo && \
    touch /etc/sagemaker-mms.properties && \
    mkdir -p /logs

# Set working directory
WORKDIR /opt/ml/code

# Copy application code
COPY requirements.txt /opt/ml/code/
COPY ./src /opt/ml/code/src/
COPY ./deploy /opt/ml/code/deploy/

# Copy required Python packages and Java runtime from builder stage
COPY --from=builder /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/java-8-openjdk-amd64

# Install requirements library
COPY dist/pip-packages /tmp/pip-packages

RUN pip install --no-cache-dir --no-index --no-build-isolation --find-links=/tmp/pip-packages/ /tmp/pip-packages/* 

# Set environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:/usr/local/bin:/usr/local/bin/python3.10:$PATH
ENV PYTHONPATH=/opt/ml/code:/opt/ml/code/src:/usr/local/lib/python3.10/site-packages:$PYTHONPATH

# Ensure Python is correctly linked
RUN ln -sf /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3

# Set Docker labels for multi-model support and port binding
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Ensure the script has the right permissions
RUN chmod +x /opt/ml/code/src/entrypoint.sh

# Define the entrypoint script
ENTRYPOINT ["/bin/bash", "/opt/ml/code/src/entrypoint.sh"]
