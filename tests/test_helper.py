import unittest
from unittest.mock import MagicMock, patch, mock_open
import os

# Assuming the functions are in a file named helper.py in the same directory
# Adjust import path if necessary
from src.helper import extract_text_from_pdf, get_past_judgement_heading, past_judgement_link, count_tokens

class TestHelperFunctions(unittest.TestCase):

    def test_extract_text_from_pdf_happy_path(self):
        """Tests extract_text_from_pdf with a mock PDF and multiple pages."""
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "This is page 1 content."
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "This is page 2 content."

        mock_pdf_document = MagicMock()
        mock_pdf_document.return_value.__enter__.return_value = [mock_page1, mock_page2]
        mock_pdf_document.return_value.__len__.return_value = 2

        with patch('fitz.open', mock_pdf_document):
            mock_pdf_file = MagicMock()
            mock_pdf_file.read.return_value = b'pdf_data'
            extracted_text = extract_text_from_pdf(mock_pdf_file)
            self.assertEqual(extracted_text, "This is page 1 content.This is page 2 content.")

    def test_extract_text_from_pdf_empty_pdf(self):
        """Tests extract_text_from_pdf with a mock empty PDF."""
        mock_pdf_document = MagicMock()
        mock_pdf_document.return_value.__enter__.return_value = []
        mock_pdf_document.return_value.__len__.return_value = 0

        with patch('fitz.open', mock_pdf_document):
            mock_pdf_file = MagicMock()
            mock_pdf_file.read.return_value = b'empty_pdf_data'
            extracted_text = extract_text_from_pdf(mock_pdf_file)
            self.assertEqual(extracted_text, "")

    def test_extract_text_from_pdf_special_characters(self):
        """Tests extract_text_from_pdf with a mock PDF containing special characters."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Content with special characters: © é ü ñ"

        mock_pdf_document = MagicMock()
        mock_pdf_document.return_value.__enter__.return_value = [mock_page]
        mock_pdf_document.return_value.__len__.return_value = 1

        with patch('fitz.open', mock_pdf_document):
            mock_pdf_file = MagicMock()
            mock_pdf_file.read.return_value = b'special_char_pdf_data'
            extracted_text = extract_text_from_pdf(mock_pdf_file)
            self.assertEqual(extracted_text, "Content with special characters: © é ü ñ")

    def test_extract_text_from_pdf_error_handling(self):
        """Tests extract_text_from_pdf when fitz.open raises an error."""
        with patch('fitz.open', side_effect=RuntimeError("PDF corruption")):
            mock_pdf_file = MagicMock()
            mock_pdf_file.read.return_value = b'corrupted_data'
            with self.assertRaises(RuntimeError):
                extract_text_from_pdf(mock_pdf_file)

    def test_get_past_judgement_heading_happy_path(self):
        """Tests get_past_judgement_heading with text containing multiple headings."""
        text = "Some text [Case1 vs Case2 on 12 January, 2023] more text [AnotherCase vs OtherParty on 01 February, 2024] end."
        headings = get_past_judgement_heading(text)
        self.assertEqual(headings, ["Case1 vs Case2 on 12 January, 2023", "AnotherCase vs OtherParty on 01 February, 2024"])

    def test_get_past_judgement_heading_no_headings(self):
        """Tests get_past_judgement_heading with text containing no headings."""
        text = "Some text without any relevant headings here."
        headings = get_past_judgement_heading(text)
        self.assertEqual(headings, [])

    def test_get_past_judgement_heading_malformed_headings(self):
        """Tests get_past_judgement_heading with malformed headings."""
        text = "[Malformed vs Heading] [Proper vs One on 01 March, 2020] [No Date vs Case]"
        headings = get_past_judgement_heading(text)
        self.assertEqual(headings, ["Proper vs One on 01 March, 2020"])

    def test_get_past_judgement_heading_empty_string(self):
        """Tests get_past_judgement_heading with an empty string."""
        text = ""
        headings = get_past_judgement_heading(text)
        self.assertEqual(headings, [])

    def test_past_judgement_link_happy_path(self):
        """Tests past_judgement_link with text containing multiple links."""
        text = "Link 1: [Full Document](https://example.com/doc1) Link 2: [Full Document](https://test.org/doc2)"
        links = past_judgement_link(text)
        self.assertEqual(links, ["https://r.jina.ai/https://example.com/doc1", "https://r.jina.ai/https://test.org/doc2"])

    def test_past_judgement_link_no_links(self):
        """Tests past_judgement_link with text containing no links."""
        text = "No full document links here."
        links = past_judgement_link(text)
        self.assertEqual(links, [])

    def test_past_judgement_link_malformed_links(self):
        """Tests past_judgement_link with malformed links."""
        text = "[Full Document] (example.com/bad) [Full Document](https://good.com/doc)"
        links = past_judgement_link(text)
        self.assertEqual(links, ["https://r.jina.ai/https://good.com/doc"])

    def test_past_judgement_link_empty_string(self):
        """Tests past_judgement_link with an empty string."""
        text = ""
        links = past_judgement_link(text)
        self.assertEqual(links, [])

    @patch('tiktoken.encoding_for_model')
    def test_count_tokens_happy_path(self, mock_encoding_for_model):
        """Tests count_tokens with a simple string."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoding_for_model.return_value = mock_encoder
        
        text = "hello world"
        tokens = count_tokens(text)
        self.assertEqual(tokens, 5)

    @patch('tiktoken.encoding_for_model')
    def test_count_tokens_empty_string(self, mock_encoding_for_model):
        """Tests count_tokens with an empty string."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = []
        mock_encoding_for_model.return_value = mock_encoder

        text = ""
        tokens = count_tokens(text)
        self.assertEqual(tokens, 0)

    @patch('tiktoken.encoding_for_model')
    def test_count_tokens_long_string(self, mock_encoding_for_model):
        """Tests count_tokens with a long string."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(100))
        mock_encoding_for_model.return_value = mock_encoder

        text = "a very long string that would have many tokens if properly encoded" * 10
        tokens = count_tokens(text)
        self.assertEqual(tokens, 100)

    @patch('tiktoken.encoding_for_model')
    def test_count_tokens_special_characters(self, mock_encoding_for_model):
        """Tests count_tokens with a string containing special characters."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [10, 20, 30]
        mock_encoding_for_model.return_value = mock_encoder

        text = "你好世界! 😊"
        tokens = count_tokens(text)
        self.assertEqual(tokens, 3)

if __name__ == '__main__':
    unittest.main()