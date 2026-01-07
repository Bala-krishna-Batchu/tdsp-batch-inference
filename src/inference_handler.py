import os
from io import StringIO

import joblib
import pandas as pd
from sagemaker_inference import default_inference_handler, encoder

class InferenceHandler(default_inference_handler.DefaultInferenceHandler):
    def default_model_fn(self, model_dir, context=None):
        """Loads a model. For PyTorch, a default function to load a model cannot be provided.
        Users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.
            context (obj): the request context (default: None).

        Returns: A model.
        """     
        sub_dir = os.listdir()[0]
        try:
            model_file = os.path.join(context.system_properties.get("model_dir"), sub_dir)
            model = joblib.load(model_file)
            print(f"Loaded model from {model_file}")
            return model
        except AttributeError as e:
            raise AttributeError("Invalid context object or missing attributes.") from e
        except KeyError as e:
            raise KeyError("Key 'model_dir' not found in system properties.") from e
        except FileNotFoundError as e:
            raise FileNotFoundError("Model file not found.") from e
        except IsADirectoryError as e:
            raise IsADirectoryError("Expected a file, but found a directory.") from e
        except ValueError as e:
            raise ValueError("Error loading model: File may be corrupt or incompatible.") from e
        except joblib.MemoryError as e:
            raise MemoryError("Memory error while loading model. File may be too large.") from e

    def default_input_fn(self, input_data, content_type, context=None):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type
            context (obj): the request context (default: None).

        Returns: input_data
        """
        print("input_data: {}".format(input_data))
        return input_data

    def default_predict_fn(self, data, model, context=None):
        """A default predict_fn. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn
            context (obj): the request context (default: None).

        Returns: a prediction
        """
        df = pd.read_csv(StringIO(data), header=None, index_col=0)

        print("Dropping target column........")
        df_target_dropped = df.drop(df.columns[9], axis=1)
        print("Before going for predictions shape of data is", df.shape)
        try:
            res = model.predict(df_target_dropped)
        except ValueError:
            res = model.predict(df)

        print("prediction: {}".format(res))
        return res

    def default_output_fn(self, prediction, accept, context=None):
        """A default output_fn. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized
            context (obj): the request context (default: None).

        Returns: output data serialized
        """
        output = encoder.encode(prediction, accept)
        print("Output: {}".format(output))
        return output
