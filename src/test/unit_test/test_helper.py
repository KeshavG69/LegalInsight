import unittest
from unittest.mock import patch, MagicMock
import os
import io
import fitz # PyMuPDF
from PIL import Image

# Mock the environment variables and imports that might cause issues outside the Streamlit context
patch.dict(os.environ, {
    "LANGCHAIN_API_KEY": "test_langchain_api_key",
    "PINECONE_API_KEY": "test_pinecone_api_key",
    "TOKENIZERS_PARALLELISM": "false",
}).start()

# Mock Streamlit specific imports if they are used at import time in helper.py
# If helper.py directly uses st.secrets at import time, this mock needs to be here.
# Otherwise, it can be patched within test methods as needed.
class MockStSecrets:
    def __getitem__(self, key):
        return f"mock_{key}"

patch('streamlit.secrets', MockStSecrets()).start()
patch('streamlit as st', MagicMock()).start()


# Dynamically import helper after os.environ is patched
import sys
sys.path.insert(0, './src') # Add src directory to path
from helper import (
    get_summary,
    get_link,
    scrape_jina_ai,
    get_past_judgement_heading,
    past_judgement_link,
    get_similar_cases_summary,
    strategy,
    extract_text_from_pdf,
    raptor_retriever,
    raptor,
    prediction,
    count_tokens,
    raptor_retriever_pinecone
)

# Mock external dependencies for helper functions
mock_llm = MagicMock()
mock_chat_prompt_template = MagicMock()
mock_requests = MagicMock()
mock_chromadb = MagicMock()
mock_recursive_character_text_splitter = MagicMock()
mock_pinecone = MagicMock()
mock_pinecone_vectorstore = MagicMock()
mock_serverless_spec = MagicMock()
mock_tiktoken = MagicMock()
mock_rag_chain = MagicMock()

# Patching at the module level for convenience
with patch('helper.llm', mock_llm), \
     patch('helper.ChatPromptTemplate', mock_chat_prompt_template), \
     patch('helper.requests', mock_requests), \
     patch('helper.chromadb', mock_chromadb), \
     patch('helper.RecursiveCharacterTextSplitter', mock_recursive_character_text_splitter), \
     patch('helper.Pinecone', mock_pinecone), \
     patch('helper.PineconeVectorStore', mock_pinecone_vectorstore), \
     patch('helper.ServerlessSpec', mock_serverless_spec), \
     patch('helper.tiktoken', mock_tiktoken), \
     patch('helper.fitz', MagicMock()) as mock_fitz:
    
    class TestHelper(unittest.TestCase):

        def setUp(self):
            # Reset mocks before each test
            mock_llm.reset_mock()
            mock_chat_prompt_template.reset_mock()
            mock_requests.reset_mock()
            mock_chromadb.reset_mock()
            mock_recursive_character_text_splitter.reset_mock()
            mock_pinecone.reset_mock()
            mock_pinecone_vectorstore.reset_mock()
            mock_serverless_spec.reset_mock()
            mock_tiktoken.reset_mock()
            mock_rag_chain.reset_mock()
            mock_fitz.reset_mock()

            # Configure common mock returns where applicable
            mock_chat_prompt_template.from_messages.return_value = MagicMock(
                __or__=MagicMock(return_value=MagicMock(stream=MagicMock(return_value=["chunk1", "chunk2"])))
            )
            mock_chat_prompt_template.from_template.return_value = MagicMock(
                __or__=MagicMock(return_value=MagicMock(invoke=MagicMock(return_value="crime_test")))
            )
            mock_llm.return_value = MagicMock()
            mock_requests.get.return_value = MagicMock(text="mock response")
            mock_chromadb.PersistentClient.return_value = MagicMock()
            mock_chromadb.PersistentClient.return_value.list_collections.return_value = []
            mock_recursive_character_text_splitter.return_value.split_text.return_value = ["leaf1", "leaf2"]
            
            # Mock fitz for PDF extraction
            mock_fitz_doc = MagicMock()
            mock_fitz.open.return_value = mock_fitz_doc
            mock_fitz_doc.__len__.return_value = 1
            mock_page = MagicMock()
            mock_fitz_doc.load_page.return_value = mock_page
            mock_page.get_text.return_value = "Extracted PDF text"
            
            # Mock Pinecone related calls
            mock_pinecone.return_value = MagicMock()
            mock_pinecone.return_value.list_indexes.return_value = []
            mock_pinecone.return_value.describe_index.return_value = MagicMock(status={"ready": true})
            mock_pinecone_vectorstore.return_value = MagicMock()
            mock_pinecone_vectorstore.from_texts.return_value = null
            mock_pinecone_vectorstore.as_retriever.return_value = MagicMock()

            # Mock tiktoken
            mock_tiktoken_encoding = MagicMock()
            mock_tiktoken_encoding.encode.return_value = [1, 2, 3, 4, 5]
            mock_tiktoken.encoding_for_model.return_value = mock_tiktoken_encoding

        def test_get_summary(self):
            docs = "chargesheet details"
            result_chunks = list(get_summary(docs))
            self.assertEqual(result_chunks, ["chunk1", "chunk2"])
            mock_chat_prompt_template.from_messages.assert_called_once()
            mock_llm.assert_called_once() # Represents llm instance usage
        
        def test_get_summary_empty_docs(self):
            docs = ""
            result_chunks = list(get_summary(docs))
            self.assertEqual(result_chunks, ["chunk1", "chunk2"])
            mock_chat_prompt_template.from_messages.assert_called_once()

        def test_get_link(self):
            docs = "chargesheet details"
            link = get_link(docs)
            self.assertEqual(link, "https://indiankanoon.org/search/?formInput=crime_test+doctypes:judgments")
            mock_chat_prompt_template.from_template.assert_called_once()
            mock_llm.assert_called_once()
            
        def test_get_link_empty_docs(self):
            docs = ""
            link = get_link(docs)
            self.assertEqual(link, "https://indiankanoon.org/search/?formInput=crime_test+doctypes:judgments")
            mock_chat_prompt_template.from_template.assert_called_once_with(unittest.mock.ANY) # Checks if called with a template

        def test_scrape_jina_ai(self):
            url = "example.com/document"
            mock_requests.get.return_value.text = "scraped content"
            content = scrape_jina_ai(url)
            self.assertEqual(content, "scraped content")
            mock_requests.get.assert_called_once_with("https://r.jina.ai/" + url)

        def test_scrape_jina_ai_network_error(self):
            url = "example.com/document"
            mock_requests.get.side_effect = Exception("Network Error")
            with self.assertRaises(Exception):
                scrape_jina_ai(url)

        def test_get_past_judgement_heading(self):
            text = "[Case A vs. Case B on 1 Jan, 2023] Some text [Case C vs. Case D on 2 Feb, 2024]"
            headings = get_past_judgement_heading(text)
            self.assertEqual(headings, ["Case A vs. Case B on 1 Jan, 2023", "Case C vs. Case D on 2 Feb, 2024"])

        def test_get_past_judgement_heading_no_match(self):
            text = "No headings here."
            headings = get_past_judgement_heading(text)
            self.assertEqual(headings, [])
            
        def test_get_past_judgement_heading_empty_text(self):
            text = ""
            headings = get_past_judgement_heading(text)
            self.assertEqual(headings, [])

        def test_past_judgement_link(self):
            text = '[Full Document](https://doc1.com) some text [Full Document](https://doc2.com)'
            links = past_judgement_link(text)
            self.assertEqual(links, ['https://r.jina.ai/https://doc1.com', 'https://r.jina.ai/https://doc2.com'])

        def test_past_judgement_link_no_match(self):
            text = 'No links here.'
            links = past_judgement_link(text)
            self.assertEqual(links, [])
            
        def test_past_judgement_link_empty_text(self):
            text = ''
            links = past_judgement_link(text)
            self.assertEqual(links, [])

        def test_get_similar_cases_summary(self):
            judgement_link = "http://example.com/judgement"
            mock_requests.get.return_value.text = "judgement content"
            
            # Mock count_tokens for normal case
            mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 1000 # Simulating small token count

            result_chunks = list(get_similar_cases_summary(judgement_link))
            self.assertEqual(result_chunks, ["chunk1", "chunk2"])
            mock_requests.get.assert_called_once_with(judgement_link)
            mock_chat_prompt_template.from_template.assert_called_once()
            mock_llm.assert_called_once()
            
        def test_get_similar_cases_summary_document_too_long(self):
            judgement_link = "http://example.com/long_judgement"
            mock_requests.get.return_value.text = "long judgement content"
            
            # Mock count_tokens to return a value greater than 131000
            mock_tiktoken.encoding_for_model.return_value.encode.return_value = [1] * 200000 

            result = list(get_similar_cases_summary(judgement_link))
            self.assertEqual(result, ["Document too long. Please refer to the original document at the above link "])
            mock_requests.get.assert_called_once_with(judgement_link)
            mock_llm.assert_not_called() # LLM should not be called if document is too long

        def test_strategy(self):
            docs = "chargesheet details"
            result_chunks = list(strategy(docs))
            self.assertEqual(result_chunks, ["chunk1", "chunk2"])
            mock_chat_prompt_template.from_messages.assert_called_once()
            mock_llm.assert_called_once()
            
        def test_strategy_empty_docs(self):
            docs = ""
            result_chunks = list(strategy(docs))
            self.assertEqual(result_chunks, ["chunk1", "chunk2"])
            mock_chat_prompt_template.from_messages.assert_called_once()


        def test_extract_text_from_pdf(self):
            mock_pdf_file = io.BytesIO(b"PDF content") # Mock a PDF file-like object
            text = extract_text_from_pdf(mock_pdf_file)
            self.assertEqual(text, "Extracted PDF text")
            mock_fitz.open.assert_called_once_with(stream=b"PDF content", filetype="pdf")
            mock_fitz.open.return_value.load_page.assert_called_once_with(0)
            mock_fitz.open.return_value.load_page.return_value.get_text.assert_called_once()
            
        def test_extract_text_from_pdf_empty_file(self):
            mock_pdf_file = io.BytesIO(b"") 
            text = extract_text_from_pdf(mock_pdf_file)
            self.assertEqual(text, "") # Expect empty text if PDF is empty
            mock_fitz.open.assert_called_once_with(stream=b"", filetype="pdf")

        @patch('helper.VectorStore') # Mocking base class if used in helper
        @patch('helper.Chroma')  # Mocking Chroma for raptor_retriever
        @patch('helper.recursive_embed_cluster_summarize')
        @patch('helper.embd')
        def test_raptor_retriever_new_index(self, mock_embd, mock_recursive_embed_cluster_summarize, mock_chroma, mock_vector_store):
            docs_text = "document text"
            index_name = "new_index"
            
            mock_chroma.return_value = MagicMock()
            mock_chroma.return_value.add_texts.return_value = null
            mock_chroma.return_value.as_retriever.return_value = "mock_retriever"

            mock_chromadb.PersistentClient.return_value.list_collections.return_value = [MagicMock(name="existing_index")] # Mock existing index
            mock_recursive_embed_cluster_summarize.return_value = {1: [null, {"summaries": MagicMock(tolist=MagicMock(return_value=["summary1"]))}]}

            retriever = raptor_retriever(docs_text, index_name)

            mock_chromadb.PersistentClient.assert_called_once_with(path="./chromadb")
            mock_recursive_character_text_splitter.assert_called_once()
            mock_recursive_embed_cluster_summarize.assert_called_once()
            mock_chroma.assert_called_once() # Chroma constructor called
            mock_chroma.return_value.add_texts.assert_called_once()
            self.assertEqual(retriever, "mock_retriever")

        @patch('helper.VectorStore') # Mocking base class if used in helper
        @patch('helper.Chroma')  # Mocking Chroma for raptor_retriever
        @patch('helper.recursive_embed_cluster_summarize')
        @patch('helper.embd')
        def test_raptor_retriever_existing_index(self, mock_embd, mock_recursive_embed_cluster_summarize, mock_chroma, mock_vector_store):
            docs_text = "document text"
            index_name = "existing_index"
            
            mock_chroma.return_value = MagicMock()
            mock_chroma.return_value.as_retriever.return_value = "mock_retriever"

            mock_chromadb.PersistentClient.return_value.list_collections.return_value = [MagicMock(name="existing_index")]
            
            retriever = raptor_retriever(docs_text, index_name)

            mock_chromadb.PersistentClient.assert_called_once_with(path="./chromadb")
            mock_recursive_character_text_splitter.assert_not_called()
            mock_recursive_embed_cluster_summarize.assert_not_called()
            mock_chroma.assert_called_once() # Chroma constructor called
            mock_chroma.return_value.add_texts.assert_not_called()
            self.assertEqual(retriever, "mock_retriever")
            
        @patch('helper.RunnableLambda')
        @patch('helper.RunnablePassthrough')
        @patch('helper.format_docs', return_value="formatted docs")
        def test_raptor(self, mock_format_docs, mock_runnable_passthrough, mock_runnable_lambda):
            retriever = MagicMock()
            question = "What is the capital of France?"
            
            # Reset mock_chat_prompt_template for this specific test to control its __or__ behavior
            mock_chat_prompt_template.from_template.return_value = MagicMock(
                __or__=MagicMock(return_value=MagicMock(stream=MagicMock(return_value=["answer chunk1", "answer chunk2"])))
            )

            result_chunks = list(raptor(retriever, question))

            self.assertEqual(result_chunks, ["answer chunk1", "answer chunk2"])
            mock_chat_prompt_template.from_template.assert_called_once()
            retriever.assert_called_once() # Retriever should be called as part of the chain
            mock_format_docs.assert_called_once()
            mock_llm.assert_called_once() # Represents llm instance usage in the stream

        def test_prediction(self):
            docs = "document for prediction"
            mock_chat_prompt_template.from_messages.return_value = MagicMock(
                __or__=MagicMock(return_value=MagicMock(invoke=MagicMock(return_value="GUILTY")))
            )
            
            result = prediction(docs)
            self.assertEqual(result, "GUILTY")
            mock_chat_prompt_template.from_messages.assert_called_once()
            mock_llm.assert_called_once()
            
        def test_prediction_empty_docs(self):
            docs = ""
            mock_chat_prompt_template.from_messages.return_value = MagicMock(
                __or__=MagicMock(return_value=MagicMock(invoke=MagicMock(return_value="INNOCENT")))
            )
            result = prediction(docs)
            self.assertEqual(result, "INNOCENT")
            mock_chat_prompt_template.from_messages.assert_called_once()

        @patch('helper.time.sleep')
        @patch('helper.PineconeVectorStore')
        @patch('helper.Pinecone')
        @patch('helper.recursive_embed_cluster_summarize')
        @patch('helper.embd')
        def test_raptor_retriever_pinecone_new_index(self, mock_embd, mock_recursive_embed_cluster_summarize, mock_pinecone, mock_pinecone_vectorstore_class, mock_sleep):
            docs_text = "document text for pinecone"
            index_name = "new-pinecone-index"

            mock_pinecone_instance = MagicMock()
            mock_pinecone.return_value = mock_pinecone_instance
            mock_pinecone_instance.list_indexes.return_value = []
            mock_pinecone_instance.describe_index.return_value = MagicMock(status={"ready": true})

            mock_recursive_embed_cluster_summarize.return_value = {1: [null, {"summaries": MagicMock(tolist=MagicMock(return_value=["pinecone_summary1"]))}]}

            mock_pinecone_vectorstore_instance = MagicMock()
            mock_pinecone_vectorstore_class.return_value = mock_pinecone_vectorstore_instance
            mock_pinecone_vectorstore_class.from_texts.return_value = null
            mock_pinecone_vectorstore_instance.as_retriever.return_value = "mock_pinecone_retriever"

            retriever = raptor_retriever_pinecone(docs_text, index_name)

            mock_pinecone.assert_called_once()
            mock_pinecone_instance.list_indexes.assert_called_once()
            mock_recursive_character_text_splitter.assert_called_once()
            mock_recursive_embed_cluster_summarize.assert_called_once()
            mock_pinecone_instance.create_index.assert_called_once_with(
                name=index_name, dimension=384, metric="cosine", spec=mock_serverless_spec.return_value
            )
            mock_pinecone_instance.describe_index.assert_called_once_with(index_name)
            mock_pinecone_vectorstore_class.assert_called_once()
            mock_pinecone_vectorstore_class.from_texts.assert_called_once()
            self.assertEqual(retriever, "mock_pinecone_retriever")
            
        @patch('helper.time.sleep')
        @patch('helper.PineconeVectorStore')
        @patch('helper.Pinecone')
        @patch('helper.recursive_embed_cluster_summarize')
        @patch('helper.embd')
        def test_raptor_retriever_pinecone_existing_index(self, mock_embd, mock_recursive_embed_cluster_summarize, mock_pinecone, mock_pinecone_vectorstore_class, mock_sleep):
            docs_text = "document text for pinecone"
            index_name = "existing-pinecone-index"

            mock_pinecone_instance = MagicMock()
            mock_pinecone.return_value = mock_pinecone_instance
            mock_pinecone_instance.list_indexes.return_value = [{"name": "existing-pinecone-index"}]
            
            mock_pinecone_vectorstore_instance = MagicMock()
            mock_pinecone_vectorstore_class.return_value = mock_pinecone_vectorstore_instance
            mock_pinecone_vectorstore_instance.as_retriever.return_value = "mock_pinecone_retriever_existing"

            retriever = raptor_retriever_pinecone(docs_text, index_name)

            mock_pinecone.assert_called_once()
            mock_pinecone_instance.list_indexes.assert_called_once()
            mock_recursive_character_text_splitter.assert_not_called()
            mock_recursive_embed_cluster_summarize.assert_not_called()
            mock_pinecone_instance.create_index.assert_not_called()
            mock_pinecone_instance.describe_index.assert_not_called()
            mock_pinecone_vectorstore_class.assert_called_once_with(index_name=index_name, embedding=mock_embd)
            mock_pinecone_vectorstore_class.from_texts.assert_not_called()
            self.assertEqual(retriever, "mock_pinecone_retriever_existing")


        def test_count_tokens(self):
            text = "This is a test sentence."
            mock_tiktoken_encoding = mock_tiktoken.encoding_for_model.return_value
            mock_tiktoken_encoding.encode.return_value = [1, 2, 3, 4, 5] # Simulating 5 tokens
            
            tokens = count_tokens(text)
            self.assertEqual(tokens, 5)
            mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4")
            mock_tiktoken_encoding.encode.assert_called_once_with(text)

        def test_count_tokens_empty_string(self):
            text = ""
            mock_tiktoken_encoding = mock_tiktoken.encoding_for_model.return_value
            mock_tiktoken_encoding.encode.return_value = [] 
            
            tokens = count_tokens(text)
            self.assertEqual(tokens, 0)
            mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4")
            mock_tiktoken_encoding.encode.assert_called_once_with(text)


if __name__ == '__main__':
    unittest.main()