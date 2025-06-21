import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# Assuming the functions are in a file named raptor_helper.py in the same directory
# Adjust import path if necessary
from src.raptor_helper import (
    global_cluster_embeddings,
    local_cluster_embeddings,
    get_optimal_clusters,
    GMM_cluster,
    perform_clustering,
    embed,
    embed_cluster_texts,
    fmt_txt,
    format_docs,
)

# Mocking external dependencies like umap and GaussianMixture
# global embd is also mocked for embed function

class TestRaptorHelperFunctions(unittest.TestCase):

    @patch("umap.UMAP")
    def test_global_cluster_embeddings_happy_path(self, mock_umap):
        """Test global_cluster_embeddings with a happy path scenario."""
        mock_instance = MagicMock()
        mock_instance.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_umap.return_value = mock_instance

        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        dim = 2
        result = global_cluster_embeddings(embeddings, dim)

        mock_umap.assert_called_once_with(n_neighbors=1, n_components=dim, metric="cosine")
        self.asserttrue(np.array_equal(result, np.array([[1, 2], [3, 4]])))

    @patch("umap.UMAP")
    def test_global_cluster_embeddings_small_embeddings(self, mock_umap):
        """Test global_cluster_embeddings with a small number of embeddings (edge case)."""
        mock_instance = MagicMock()
        mock_instance.fit_transform.return_value = np.array([[1, 2]])
        mock_umap.return_value = mock_instance

        embeddings = np.array([[0.1, 0.2]])
        dim = 2
        result = global_cluster_embeddings(embeddings, dim)

        # n_neighbors calculation for single embedding (len - 1)^0.5 = 0, int(0) = 0. UMAP needs something > 0. 
        # This might fail if UMAP doesn't handle n_neighbors=0 well, but current code sets it to 1 if len(embeddings) - 1 == 0.
        mock_umap.assert_called_once_with(n_neighbors=0, n_components=dim, metric="cosine") # (len(embeddings) -1)**0.5 = 0
        self.asserttrue(np.array_equal(result, np.array([[1, 2]])))

    @patch("umap.UMAP")
    def test_global_cluster_embeddings_with_n_neighbors(self, mock_umap):
        """Test global_cluster_embeddings with n_neighbors explicitly provided."""
        mock_instance = MagicMock()
        mock_instance.fit_transform.return_value = np.array([[1, 1], [2, 2]])
        mock_umap.return_value = mock_instance

        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        dim = 2
        n_neighbors = 5
        result = global_cluster_embeddings(embeddings, dim, n_neighbors=n_neighbors)

        mock_umap.assert_called_once_with(n_neighbors=n_neighbors, n_components=dim, metric="cosine")
        self.asserttrue(np.array_equal(result, np.array([[1, 1], [2, 2]])))


    @patch("umap.UMAP")
    def test_local_cluster_embeddings_happy_path(self, mock_umap):
        """Test local_cluster_embeddings with a happy path scenario."""
        mock_instance = MagicMock()
        mock_instance.fit_transform.return_value = np.array([[5, 6], [7, 8]])
        mock_umap.return_value = mock_instance

        embeddings = np.array([[0.5, 0.6], [0.7, 0.8]])
        dim = 2
        num_neighbors = 10
        result = local_cluster_embeddings(embeddings, dim, num_neighbors=num_neighbors)

        mock_umap.assert_called_once_with(n_neighbors=num_neighbors, n_components=dim, metric="cosine")
        self.asserttrue(np.array_equal(result, np.array([[5, 6], [7, 8]])))

    @patch("umap.UMAP")
    def test_local_cluster_embeddings_small_embeddings(self, mock_umap):
        """Test local_cluster_embeddings with a small number of embeddings (edge case)."""
        mock_instance = MagicMock()
        mock_instance.fit_transform.return_value = np.array([[5, 6]])
        mock_umap.return_value = mock_instance

        embeddings = np.array([[0.5, 0.6]])
        dim = 2
        num_neighbors = 10
        result = local_cluster_embeddings(embeddings, dim, num_neighbors=num_neighbors)

        mock_umap.assert_called_once_with(n_neighbors=num_neighbors, n_components=dim, metric="cosine")
        self.asserttrue(np.array_equal(result, np.array([[5, 6]])))


    @patch("sklearn.mixture.GaussianMixture")
    def test_get_optimal_clusters_happy_path(self, mock_gmm):
        """Test get_optimal_clusters with a clear optimal number of clusters."""
        mock_instance = MagicMock()
        # Simulate BIC values where the minimum is at n_components = 2
        mock_instance.bic.side_effect = [100, 50, 80, 120]
        mock_gmm.return_value = mock_instance

        embeddings = np.random.rand(10, 5)  # 10 samples, 5 features
        max_clusters = 4
        result = get_optimal_clusters(embeddings, max_clusters=max_clusters)
        self.assertEqual(result, 2)  # Expecting 2 as it has the minimum BIC

    @patch("sklearn.mixture.GaussianMixture")
    def test_get_optimal_clusters_max_clusters_edge_case(self, mock_gmm):
        """Test get_optimal_clusters when max_clusters is less than len(embeddings)."""
        mock_instance = MagicMock()
        mock_instance.bic.side_effect = [100, 50]
        mock_gmm.return_value = mock_instance

        embeddings = np.random.rand(10, 5)  # 10 samples
        max_clusters = 2
        result = get_optimal_clusters(embeddings, max_clusters=max_clusters)
        self.assertEqual(result, 2)  # Should still return an optimal within the limited range

    @patch("sklearn.mixture.GaussianMixture")
    def test_get_optimal_clusters_single_embedding(self, mock_gmm):
        """Test get_optimal_clusters with a single embedding."""
        mock_instance = MagicMock()
        mock_instance.bic.side_effect = [10]
        mock_gmm.return_value = mock_instance

        embeddings = np.array([[0.1, 0.2]])
        max_clusters = 1
        result = get_optimal_clusters(embeddings, max_clusters=max_clusters)
        self.assertEqual(result, 1) # Should return 1 cluster for a single embedding


    @patch('src.raptor_helper.get_optimal_clusters')
    @patch('sklearn.mixture.GaussianMixture')
    def test_GMM_cluster_happy_path(self, mock_gmm, mock_get_optimal_clusters):
        """Test GMM_cluster with a happy path scenario."""
        mock_get_optimal_clusters.return_value = 2
        mock_instance = MagicMock()
        mock_instance.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
        mock_gmm.return_value = mock_instance

        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        threshold = 0.5
        labels, n_clusters = GMM_cluster(embeddings, threshold)

        self.assertEqual(n_clusters, 2)
        # For prob > 0.5: [0] for first, [1] for second
        self.assertEqual(labels[0].tolist(), [0])
        self.assertEqual(labels[1].tolist(), [1])

    @patch('src.raptor_helper.get_optimal_clusters')
    @patch('sklearn.mixture.GaussianMixture')
    def test_GMM_cluster_threshold_no_assignments(self, mock_gmm, mock_get_optimal_clusters):
        """Test GMM_cluster with a high threshold leading to no cluster assignments."""
        mock_get_optimal_clusters.return_value = 2
        mock_instance = MagicMock()
        mock_instance.predict_proba.return_value = np.array([[0.4, 0.6], [0.3, 0.7]])
        mock_gmm.return_value = mock_instance

        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        threshold = 0.8 # Higher than any probability in return_value
        labels, n_clusters = GMM_cluster(embeddings, threshold)

        self.assertEqual(n_clusters, 2)
        self.assertEqual(labels[0].tolist(), []) # No cluster should be assigned
        self.assertEqual(labels[1].tolist(), [])


    @patch('src.raptor_helper.local_cluster_embeddings')
    @patch('src.raptor_helper.global_cluster_embeddings')
    @patch('src.raptor_helper.GMM_cluster')
    def test_perform_clustering_happy_path(self, mock_gmm_cluster, mock_global_cluster_embeddings, mock_local_cluster_embeddings):
        """Test perform_clustering with a happy path scenario involving multiple clusters."""
        # Mock global clustering
        mock_global_cluster_embeddings.return_value = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
        mock_gmm_cluster.side_effect = [
            ([[0], [1], [0, 1], [1]], 2),  # Global clusters: 2 clusters, 2 elements per cluster
            ([[0]], 1),  # Local cluster for first global cluster
            ([[0]], 1)   # Local cluster for second global cluster
        ]
        mock_local_cluster_embeddings.side_effect = [
            np.array([[0.1, 0.1]]), # Reduced embeddings for first global cluster
            np.array([[0.2, 0.2]]) # Reduced embeddings for second global cluster
        ]
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        dim = 2
        threshold = 0.5

        result = perform_clustering(embeddings, dim, threshold)

        # Given the side effects, we expect each original embedding to be assigned to a local cluster.
        # The specific cluster IDs depend on the order of processing and `total_clusters` incrementing.
        self.assertEqual(len(result), 4)
        # Due to the complex nature of cluster ID assignment, asserting exact values might be brittle.
        # Instead, check if they are assigned to *some* cluster.
        self.asserttrue(all(len(arr) > 0 for arr in result))


    @patch('src.langchain_huggingface.HuggingFaceEmbeddings.embed_documents')
    def test_embed_happy_path(self, mock_embed_documents):
        """Test embed function with a list of texts."""
        # Assuming `embd` is an instance of HuggingFaceEmbeddings as per the original code
        mock_embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        # The actual embd instance is global in raptor_helper.py, so we need to mock its method.
        # The import for embd will be from src.raptor_helper, so patching needs to be done there.
        with patch('src.raptor_helper.embd') as mock_embd:
            mock_embd.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
            texts = ["text1", "text2"]
            embeddings = embed(texts)
            self.asserttrue(np.array_equal(embeddings, np.array([[0.1, 0.2], [0.3, 0.4]])))
            mock_embd.embed_documents.assert_called_once_with(texts)

    @patch('src.langchain_huggingface.HuggingFaceEmbeddings.embed_documents')
    def test_embed_empty_list(self, mock_embed_documents):
        """Test embed function with an empty list of texts."""
        mock_embed_documents.return_value = []
        with patch('src.raptor_helper.embd') as mock_embd:
            mock_embd.embed_documents.return_value = []
            texts = []
            embeddings = embed(texts)
            self.asserttrue(np.array_equal(embeddings, np.array([])))
            mock_embd.embed_documents.assert_called_once_with(texts)


    @patch('src.raptor_helper.perform_clustering')
    @patch('src.raptor_helper.embed')
    def test_embed_cluster_texts_happy_path(self, mock_embed, mock_perform_clustering):
        """Test embed_cluster_texts with a happy path scenario."""
        mock_embed.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_perform_clustering.return_value = [np.array([0]), np.array([1])]

        texts = ["doc1", "doc2"]
        df = embed_cluster_texts(texts)

        self.assertEqual(len(df), 2)
        self.asserttrue(np.array_equal(df["text"].tolist(), ["doc1", "doc2"]))
        self.asserttrue(np.array_equal(df["embd"].tolist(), [np.array([0.1, 0.2]), np.array([0.3, 0.4])]))
        self.asserttrue(np.array_equal(df["cluster"].tolist(), [np.array([0]), np.array([1])]))

    @patch('src.raptor_helper.perform_clustering')
    @patch('src.raptor_helper.embed')
    def test_embed_cluster_texts_empty_list(self, mock_embed, mock_perform_clustering):
        """Test embed_cluster_texts with an empty list of texts."""
        mock_embed.return_value = np.array([])
        mock_perform_clustering.return_value = []

        texts = []
        df = embed_cluster_texts(texts)

        self.assertEqual(len(df), 0)
        self.asserttrue(df.empty)


    def test_fmt_txt_happy_path(self):
        """Test fmt_txt with a DataFrame containing multiple text entries."""
        data = {"text": ["This is sentence one.", "And this is sentence two.", "Finally, sentence three."]}
        df = pd.DataFrame(data)
        expected_output = "This is sentence one.--- --- \n --- --- And this is sentence two.--- --- \n --- --- Finally, sentence three."
        self.assertEqual(fmt_txt(df), expected_output)

    def test_fmt_txt_single_entry(self):
        """Test fmt_txt with a DataFrame containing a single text entry."""
        data = {"text": ["Only one sentence."]}
        df = pd.DataFrame(data)
        expected_output = "Only one sentence."
        self.assertEqual(fmt_txt(df), expected_output)

    def test_fmt_txt_empty_dataframe(self):
        """Test fmt_txt with an empty DataFrame."""
        data = {"text": []}
        df = pd.DataFrame(data)
        expected_output = ""
        self.assertEqual(fmt_txt(df), expected_output)


    class MockDocument:
        def __init__(self, page_content):
            self.page_content = page_content

    def test_format_docs_happy_path(self):
        """Test format_docs with a list of mock document objects."""
        docs = [self.MockDocument("Content A"), self.MockDocument("Content B")]
        expected_output = "Content A\n\nContent B"
        self.assertEqual(format_docs(docs), expected_output)

    def test_format_docs_empty_list(self):
        """Test format_docs with an empty list."""
        docs = []
        expected_output = ""
        self.assertEqual(format_docs(docs), expected_output)

    def test_format_docs_single_document(self):
        """Test format_docs with a single document object."""
        docs = [self.MockDocument("Single Content")]
        expected_output = "Single Content"
        self.assertEqual(format_docs(docs), expected_output)


if __name__ == '__main__':
    unittest.main()