import io
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

# Get the absolute path of the parent directory
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_directory)
# Add the parent directory to sys.path
sys.path.append(parent_directory)

# Now you can import inference_handler
try:
    from src import inference_handler

except ImportError:
    import inference_handler


class TestInferenceHandler(unittest.TestCase):
    def setUp(self):
        self.inference_handler = inference_handler.InferenceHandler()
        self.context = Mock()
        self.context.system_properties = {"model_dir": "/test/dir"}

    @patch("os.path.join")
    @patch("joblib.load")
    def test_default_model_fn(self, mock_load, mock_path_join):
        mock_path_join.return_value = "/some/path"
        mock_load.return_value = Mock()

        model = self.inference_handler.default_model_fn("/test/dir", self.context)

        self.assertTrue(model)

    # mock_path_join.assert_called_with(self.context.system_properties.get("model_dir"), '/some/path')

    # mock_load.assert_called_with("/some/path")

    @patch("os.path.join")
    @patch("joblib.load")
    def test_default_model_fn_exception(self, mock_load, mock_path_join):
        mock_path_join.return_value = "/some/path"
        mock_load.side_effect = Exception("Error loading model")

        with self.assertRaises(Exception):
            self.inference_handler.default_model_fn("/test/dir", self.context)

    def test_default_input_fn(self):
        input_data = "sample input"
        content_type = "sample content type"

        result = self.inference_handler.default_input_fn(input_data, content_type, self.context)

        self.assertEqual(input_data, result)

    @patch("pandas.read_csv")
    def test_default_predict_fn(self, mock_read_csv):
        # Mocking the model's predict method
        model = MagicMock()
        model.predict.return_value = (
            "Test Prediction"  # Change this line to match the expected result
        )

        model.predict.return_value = (
            "Test Prediction"  # Change this line to match the expected result
        )

        # Mocking the CSV data
        csv_data = io.StringIO(
            """ index,feature1,feature2,Exited
                0,1.0,2.0,0
                1,3.0,4.0,1"""
        )
        mock_read_csv.return_value = pd.read_csv(csv_data)

        # Mocking the input data
        data = "index,feature1,feature2,Exited\n0,1.0,2.0,0\n1,3.0,4.0,1"

        # Instantiate the class and call the method
        result = self.inference_handler.default_predict_fn(data, model, self.context)

        self.assertEqual(result, "Test Prediction")

    @patch("src.inference_handler.encoder.encode")
    def test_default_output_fn(self, mock_encode):
        mock_encode.return_value = "Test Encoding"

        result = self.inference_handler.default_output_fn("Test Prediction", "accept", self.context)

        self.assertEqual(result, "Test Encoding")
        mock_encode.assert_called_with("Test Prediction", "accept")


if __name__ == "__main__":
    unittest.main()
