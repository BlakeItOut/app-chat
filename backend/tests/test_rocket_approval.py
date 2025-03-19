from unittest.mock import patch

import pytest
import requests

from arcade_rocket_approval.tools.purchase import start_application


class TestPurchase:
	def test_start_application_1(self):
		"""
		Test that start_application returns an empty string when the response data is not a dictionary.

		This test mocks the requests.post method to return a response with non-dictionary JSON data.
		It verifies that the start_application function handles this case correctly by returning an empty string.
		"""
		with patch("requests.post") as mock_post:
			mock_response = mock_post.return_value
			mock_response.json.return_value = ["Not a dictionary"]

			result = start_application()

			assert result == "", "Expected an empty string when response data is not a dictionary"

	def test_start_application_2(self):
		"""
		Test start_application when response_data is a dict but context_data is not a dict.

		This test mocks the requests.post response to return a JSON object where:
		- The response_data is a valid dictionary
		- The 'context' key in response_data is not a dictionary

		Expected behavior: The function should return an empty string.
		"""
		mock_response = {"context": "not a dictionary"}

		with patch("requests.post") as mock_post:
			mock_post.return_value.json.return_value = mock_response
			mock_post.return_value.raise_for_status.return_value = None

			result = start_application()

			assert result == "", "Expected an empty string when context_data is not a dictionary"

	def test_start_application_3(self):
		"""
		Test that start_application returns an empty string when rm_loan_id is not a string.
		This test covers the case where the API response is valid, but the rmLoanId is not a string.
		"""
		mock_response = {
			"context": {
				"rmLoanId": 12345  # Non-string value
			}
		}

		with patch("requests.post") as mock_post:
			mock_post.return_value.json.return_value = mock_response
			mock_post.return_value.raise_for_status.return_value = None

			result = start_application()

			assert result == "", "Expected an empty string when rmLoanId is not a string"

	def test_start_application_4(self):
		"""
		Tests the start_application function when the API response is successful and contains valid data.

		This test verifies that the function correctly extracts and returns the rmLoanId
		when the API response contains a valid dictionary structure with the expected data.
		"""
		mock_response = {"context": {"rmLoanId": "12345"}}

		with patch("requests.post") as mock_post:
			mock_post.return_value.json.return_value = mock_response
			mock_post.return_value.raise_for_status.return_value = None

			result = start_application()

			assert result == "12345"
			mock_post.assert_called_once()

	def test_start_application_invalid_json_response(self):
		"""
		Test that start_application handles invalid JSON responses by returning an empty string.
		"""
		with patch("requests.post") as mock_post:
			mock_post.return_value.json.side_effect = ValueError("Invalid JSON")
			result = start_application()
			assert result == ""

	def test_start_application_missing_context(self):
		"""
		Test that start_application handles responses without 'context' key by returning an empty string.
		"""
		with patch("requests.post") as mock_post:
			mock_post.return_value.json.return_value = {"data": "No context"}
			result = start_application()
			assert result == ""

	def test_start_application_missing_rm_loan_id(self):
		"""
		Test that start_application handles missing 'rmLoanId' in context by returning an empty string.
		"""
		with patch("requests.post") as mock_post:
			mock_post.return_value.json.return_value = {"context": {"other": "data"}}
			result = start_application()
			assert result == ""

	def test_start_application_network_error(self):
		"""
		Test that start_application handles network errors gracefully by returning an empty string.
		"""
		with patch("requests.post") as mock_post:
			mock_post.side_effect = requests.exceptions.RequestException("Network error")
			result = start_application()
			assert result == ""

	def test_start_application_non_dict_context(self):
		"""
		Test that start_application handles non-dictionary 'context' value by returning an empty string.
		"""
		with patch("requests.post") as mock_post:
			mock_post.return_value.json.return_value = {"context": "Not a dictionary"}
			result = start_application()
			assert result == ""

	def test_start_application_non_dict_response(self):
		"""
		Test that start_application handles non-dictionary responses by returning an empty string.
		"""
		with patch("requests.post") as mock_post:
			mock_post.return_value.json.return_value = ["Not a dictionary"]
			result = start_application()
			assert result == ""

	def test_start_application_non_string_rm_loan_id(self):
		"""
		Test that start_application handles non-string 'rmLoanId' value by returning an empty string.
		"""
		with patch("requests.post") as mock_post:
			mock_post.return_value.json.return_value = {"context": {"rmLoanId": 12345}}
			result = start_application()
			assert result == ""
