#!/bin/bash
# Read the ENTRYPOINT environment variable
entrypoint=$ENTRYPOINT


# Check the value of ENTRYPOINT and run the corresponding Python script
case $entrypoint in
    inference)
        echo "Running Python inferencing"
        python /opt/ml/code/src/inference_server.py
        ;;
    preprocessing)
        echo "Running Python preprocessing"
        python /opt/ml/code/src/preprocessing.py
        ;;
    *)
        echo "ENTRYPOINT is not set to a recognized value. Running default Python script..."
        python /opt/ml/code/src/inference_server.py
        ;;
esac