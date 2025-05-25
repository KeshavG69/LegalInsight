import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
import pandas as pd
import os

# Mocking environment variables and Streamlit interactions before importing the module
patch.dict(os.environ, {
    "LANGCHAIN_API_KEY": "test_langchain_api_key",
    "TOGETHER_API_KEY": "test_together_api_key",
    "PINECONE_API_KEY": "test_pinecone_api_key",
    "TOKENIZERS_PARALLELISM": "false",
}).start()

class MockStSecrets:
    def __getitem__(self, key):
        return f"mock_{key}"

patch('streamlit.secrets', MockStSecrets()).start()
patch('streamlit as st', MagicMock()).start()

# Dynamically import raptor_helper after patching environment variables
import sys
sys.path.insert(0, './src')  # Add src directory to path
from raptor_helper import (
    global_cluster_embeddings,
    local_cluster_embeddings,
    get_optimal_clusters,
    GMM_cluster,
    perform_clustering,
    embed,
    embed_cluster_texts,
    fmt_txt,
    embed_cluster_summarize_texts,
    recursive_embed_cluster_summarize,
    format_docs,
)

# Mock external dependencies for raptor_helper functions
mock_umap = MagicMock()
mock_gaussian_mixture = MagicMock()
mock_huggingface_embeddings = MagicMock()
mock_chat_prompt_template = MagicMock()
mock_str_output_parser = MagicMock()
mock_llm = MagicMock() # Assuming llm is directly imported, typically it's an instance

# Patching at the module level for convenience
with patch('raptor_helper.umap.UMAP', mock_umap), \
     patch('raptor_helper.GaussianMixture', mock_gaussian_mixture), \
     patch('raptor_helper.HuggingFaceEmbeddings', mock_huggingface_embeddings), \
     patch('raptor_helper.ChatPromptTemplate', mock_chat_prompt_template), \
     patch('raptor_helper.StrOutputParser', mock_str_output_parser), \
     patch('raptor_helper.llm', mock_llm):  # Mock the llm instance

    class TestRaptorHelper(unittest.TestCase):

        def setUp(self):
            # Reset mocks before each test
            mock_umap.reset_mock()
            mock_gaussian_mixture.reset_mock()
            mock_huggingface_embeddings.reset_mock()
            mock_chat_prompt_template.reset_mock()
            mock_str_output_parser.reset_mock()
            mock_llm.reset_mock()

            # Global mock configurations
            # Mock UMAP fit_transform
            mock_umap.return_value.fit_transform.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

            # Mock GaussianMixture
            mock_gaussian_mixture.return_value.fit.return_value = null
            mock_gaussian_mixture.return_value.bic.return_value = np.array([100, 50, 150]) # Optimal at 2 components
            mock_gaussian_mixture.return_value.predict_proba.return_value = np.array([[0.9, 0.1], [0.1, 0.9]])
            mock_gaussian_mixture.return_value.n_components = 2 # Default for optimal clusters

            # Mock HuggingFaceEmbeddings
            mock_huggingface_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

            # Mock ChatPromptTemplate and StrOutputParser for summarization chains
            mock_str_output_parser.return_value = MagicMock()
            mock_chain_instance = MagicMock()
            mock_chain_instance.invoke.return_value = "mocked summary"
            mock_chat_prompt_template.from_template.return_value.__or__.return_value = mock_chain_instance

            # Mock the embd object that is used globally
            patcher = patch('raptor_helper.embd', mock_huggingface_embeddings.return_value)
            self.mock_embd = patcher.start()
            self.addCleanup(patcher.stop)


        def test_global_cluster_embeddings(self):
            embeddings = np.array([[1, 2], [3, 4], [5, 6]])
            dim = 2
            result = global_cluster_embeddings(embeddings, dim)
            mock_umap.assert_called_once_with(n_neighbors=2, n_components=dim, metric="cosine")
            mock_umap.return_value.fit_transform.assert_called_once_with(embeddings)
            self.asserttrue(np.array_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))) # Adjusted for mock_umap

        def test_local_cluster_embeddings(self):
            embeddings = np.array([[1, 2], [3, 4]])
            dim = 2
            num_neighbors = 2
            result = local_cluster_embeddings(embeddings, dim, num_neighbors)
            mock_umap.assert_called_once_with(n_neighbors=num_neighbors, n_components=dim, metric="cosine")
            mock_umap.return_value.fit_transform.assert_called_once_with(embeddings)
            self.asserttrue(np.array_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))) # Adjusted for mock_umap

        def test_get_optimal_clusters(self):
            embeddings = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            # Mock BIC values to ensure optimal is selected
            mock_gaussian_mixture.return_value.bic.side_effect = lambda x: np.array([100, 50, 150]) # 2 is optimal
            
            optimal_clusters = get_optimal_clusters(embeddings, max_clusters=3)
            self.assertEqual(optimal_clusters, 2) # Corresponds to index 1 (n_clusters = 2)
            self.assertEqual(mock_gaussian_mixture.call_count, 3) # Called for n_clusters 1, 2, 3

        def test_GMM_cluster(self):
            embeddings = np.array([[1, 2], [3, 4]])
            threshold = 0.5
            
            # For this test, we need get_optimal_clusters to return a specific value
            with patch('raptor_helper.get_optimal_clusters', return_value=2) as mock_get_optimal_clusters:
                labels, n_clusters = GMM_cluster(embeddings, threshold)
                mock_gaussian_mixture.assert_called_once_with(n_components=2, random_state=unittest.mock.ANY)
                mock_gaussian_mixture.return_value.fit.assert_called_once_with(embeddings)
                # Based on mock_gaussian_mixture.return_value.predict_proba.return_value
                self.assertEqual(labels, [[0], [1]])
                self.assertEqual(n_clusters, 2)

        def test_perform_clustering_insufficient_data(self):
            embeddings = np.array([[1, 2]])  # Less than dim + 1
            dim = 2
            threshold = 0.5
            result = perform_clustering(embeddings, dim, threshold)
            self.assertEqual(len(result), 1)
            self.asserttrue(np.array_equal(result[0], np.array([0])))
            mock_umap.assert_not_called()
            mock_gaussian_mixture.assert_not_called()
            
        def test_perform_clustering_sufficient_data(self):
            embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
            dim = 2
            threshold = 0.5

            # Mock global and local clustering parts
            with (
                patch('raptor_helper.global_cluster_embeddings', return_value=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])),
                patch('raptor_helper.GMM_cluster') as mock_gmm_cluster,
                patch('raptor_helper.local_cluster_embeddings', return_value=np.array([[0.1, 0.2], [0.3, 0.4]])),
            ):
                # Mock calls for global and local GMM_cluster
                # Global:
                mock_gmm_cluster.side_effect = [ 
                    ([np.array([0]), np.array([1]), np.array([0]), np.array([1])], 2), # Global clusters
                    ([np.array([0])], 1), # Local clusters for first global_cluster_embeddings_ (size 2, but just one cluster)
                    ([np.array([0])], 1)  # Local clusters for second global_cluster_embeddings_ (size 2, but just one cluster)
                ]

                result = perform_clustering(embeddings, dim, threshold)
                self.assertEqual(len(result), 4) # Should return labels for all 4 initial embeddings
                
                # More detailed assertions would require knowing the exact internal logic,
                # but we can check that GMM_cluster was called at least 3 times.
                self.assertEqual(mock_gmm_cluster.call_count, 3)

        def test_embed(self):
            texts = ["text1", "text2"]
            embeddings = embed(texts)
            self.mock_embd.embed_documents.assert_called_once_with(texts)
            self.asserttrue(np.array_equal(embeddings, np.array([[0.1, 0.2], [0.3, 0.4]]))) # Based on mock_huggingface_embeddings

        def test_embed_cluster_texts(self):
            texts = ["text1", "text2"]
            with patch('raptor_helper.embed', return_value=np.array([[0.1, 0.2], [0.3, 0.4]])),
                 patch('raptor_helper.perform_clustering', return_value=[np.array([0]), np.array([1])]):
                df = embed_cluster_texts(texts)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertEqual(len(df), 2)
                self.assertIn("text", df.columns)
                self.assertIn("embd", df.columns)
                self.assertIn("cluster", df.columns)
                self.asserttrue(np.array_equal(df["text"].tolist(), texts))
                self.asserttrue(np.array_equal(df["cluster"].tolist(), [np.array([0]), np.array([1])]))

        def test_fmt_txt(self):
            df = pd.DataFrame({"text": ["part1", "part2", "part3"]})
            formatted_text = fmt_txt(df)
            self.assertEqual(formatted_text, "part1--- ---\n --- ---part2--- ---\n --- ---part3")
            
        def test_fmt_txt_empty_df(self):
            df = pd.DataFrame({"text": []})
            formatted_text = fmt_txt(df)
            self.assertEqual(formatted_text, "")

        def test_embed_cluster_summarize_texts(self):
            texts = ["doc1", "doc2", "doc3"]
            level = 1

            # Mock embed_cluster_texts to return a predictable DataFrame
            mock_df_clusters = pd.DataFrame({
                "text": texts,
                "embd": [[0.1,0.1],[0.2,0.2],[0.3,0.3]],
                "cluster": [np.array([0]), np.array([1]), np.array([0])] # Two clusters
            })
            
            with patch('raptor_helper.embed_cluster_texts', return_value=mock_df_clusters) as mock_ect:
                df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

                mock_ect.assert_called_once_with(texts)
                self.assertIsInstance(df_clusters, pd.DataFrame)
                self.assertIsInstance(df_summary, pd.DataFrame)

                # Expect 2 summaries (one for each unique cluster)
                self.assertEqual(len(df_summary), 2)
                self.assertEqual(df_summary["summaries"].tolist(), ["mocked summary", "mocked summary"])
                self.asserttrue(np.array_equal(df_summary["level"].tolist(), [level, level]))
                self.assertEqual(set(df_summary["cluster"].tolist()), {0, 1})
                
                # Check that the LLM chain was invoked twice (once per cluster)
                self.assertEqual(mock_chat_prompt_template.from_template.return_value.__or__.return_value.invoke.call_count, 2)

        def test_recursive_embed_cluster_summarize_no_recursion(self):
            texts = ["doc1"]
            level = 1
            n_levels = 3

            # Mock embed_cluster_summarize_texts to return a single cluster
            mock_df_clusters_level1 = pd.DataFrame({
                "text": texts,
                "embd": [[0.1,0.1]],
                "cluster": [np.array([0])]
            })
            mock_df_summary_level1 = pd.DataFrame({
                "summaries": ["summary_level1"],
                "level": [level],
                "cluster": [0]
            })
            
            with patch('raptor_helper.embed_cluster_summarize_texts', return_value=(mock_df_clusters_level1, mock_df_summary_level1)) as mock_ecst:
                results = recursive_embed_cluster_summarize(texts, level, n_levels)

                mock_ecst.assert_called_once_with(texts, level)
                self.assertEqual(len(results), 1)
                self.assertIn(1, results)
                self.assertEqual(results[1][1]["summaries"].tolist(), ["summary_level1"])

        def test_recursive_embed_cluster_summarize_with_recursion(self):
            texts = ["doc1", "doc2", "doc3"]
            level = 1
            n_levels = 2

            # Mock objects for level 1
            mock_df_clusters_level1 = pd.DataFrame({
                "text": texts,
                "embd": [[0.1,0.1],[0.2,0.2],[0.3,0.3]],
                "cluster": [np.array([0]), np.array([1]), np.array([0])] # Two clusters
            })
            mock_df_summary_level1 = pd.DataFrame({
                "summaries": ["summary_level1_cluster0", "summary_level1_cluster1"],
                "level": [level, level],
                "cluster": [0, 1]
            })
            
            # Mock objects for level 2
            mock_df_clusters_level2 = pd.DataFrame({
                "text": ["summary_level1_cluster0", "summary_level1_cluster1"],
                "embd": [[0.4,0.4],[0.5,0.5]],
                "cluster": [np.array([0]), np.array([0])] # One cluster at level 2
            })
            mock_df_summary_level2 = pd.DataFrame({
                "summaries": ["summary_level2_cluster0"],
                "level": [level + 1],
                "cluster": [0] # Single cluster after recursion
            })

            with patch('raptor_helper.embed_cluster_summarize_texts') as mock_ecst:
                # Configure side_effect for multiple calls
                mock_ecst.side_effect = [
                    (mock_df_clusters_level1, mock_df_summary_level1),  # First call (level 1)
                    (mock_df_clusters_level2, mock_df_summary_level2)   # Second call (level 2)
                ]
                
                results = recursive_embed_cluster_summarize(texts, level, n_levels)

                self.assertEqual(mock_ecst.call_count, 2)
                self.assertEqual(len(results), 2)
                self.assertIn(1, results)
                self.assertIn(2, results)
                self.assertEqual(results[1][1]["summaries"].tolist(), ["summary_level1_cluster0", "summary_level1_cluster1"])
                self.assertEqual(results[2][1]["summaries"].tolist(), ["summary_level2_cluster0"])


        def test_format_docs(self):
            mock_doc1 = MagicMock()
            mock_doc1.page_content = "Content of doc 1"
            mock_doc2 = MagicMock()
            mock_doc2.page_content = "Content of doc 2"
            docs = [mock_doc1, mock_doc2]
            
            formatted_string = format_docs(docs)
            self.assertEqual(formatted_string, "Content of doc 1\n\nContent of doc 2")
            
        def test_format_docs_empty_list(self):
            docs = []
            formatted_string = format_docs(docs)
            self.assertEqual(formatted_string, "")


if __name__ == '__main__':
    unittest.main()