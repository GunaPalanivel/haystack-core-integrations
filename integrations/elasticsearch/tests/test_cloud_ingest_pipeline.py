# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
#
# These tests connect to a managed Elastic Cloud cluster and exercise real ingest pipelines
# that generate embeddings at index time using Elasticsearch inference processors.
#
# Required environment variables (tests are skipped automatically when absent):
#
#   For both TestIngestPipelineDense and TestIngestPipelineSparse:
#     ELASTICSEARCH_URL   - cluster endpoint, e.g. https://my-cluster.es.io:443
#     ELASTIC_API_KEY     - base64-encoded API key (id:secret)
#
# Optional (sensible defaults are used when not set):
#
#   For TestIngestPipelineDense:
#     ELASTICSEARCH_DENSE_INFERENCE_ID   - deployed dense inference endpoint
#                                          (default: ".multilingual-e5-small-elasticsearch")
#     ELASTICSEARCH_DENSE_EMBEDDING_DIMS - output dimension of that model
#                                          (default: "384")
#
#   For TestIngestPipelineSparse:
#     ELASTICSEARCH_INFERENCE_ID         - deployed sparse inference endpoint
#                                          (default: ".elser-2-elastic", Elastic's hosted ELSER
#                                          service which does not consume local ML node capacity)
#
# Example (bash):
#   export ELASTICSEARCH_URL="https://my-cluster.es.io:443"
#   export ELASTIC_API_KEY="<base64-id:secret>"
#   pytest -m integration tests/test_cloud_ingest_pipeline.py

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.elasticsearch import (
    ElasticsearchEmbeddingRetriever,
    ElasticsearchInferenceSparseRetriever,
)


def _get_dense_query_embedding(client, inference_id: str, text: str) -> list[float]:
    """Call the ES inference API to embed a query string using the same model as the ingest pipeline."""
    response = client.inference.inference(inference_id=inference_id, input=[text])
    return response["text_embedding"][0]["embedding"]


@pytest.mark.integration
class TestIngestPipelineDense:
    """
    End-to-end integration tests for ElasticsearchDocumentStore with an ingest pipeline
    that generates dense embeddings at index time.

    The fixture creates a real ES ingest pipeline (inference processor → embedding field)
    on Elastic Cloud. Documents are written without pre-computed embeddings; the pipeline
    fills the embedding field before the document is committed to the index.
    """

    def test_indexed_document_has_embedding_filled_by_pipeline(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        store.write_documents([Document(id="doc-1", content="The Eiffel Tower is located in Paris.")])

        store.client.indices.refresh(index=store._index)
        raw = store.client.get(index=store._index, id="doc-1")

        # ES inference pipelines do NOT write dense_vector data to _source; only to the vector index.
        assert raw["_source"].get("embedding") is None

        # The vector IS in the index: a KNN search using the same model finds the document.
        query_embedding = _get_dense_query_embedding(store.client, inference_id, "Eiffel Tower Paris")
        result = store.client.search(
            index=store._index,
            knn={"field": "embedding", "query_vector": query_embedding, "k": 1, "num_candidates": 10},
        )
        assert result["hits"]["total"]["value"] == 1, "pipeline did not populate the 'embedding' field"

    def test_embedding_retriever_finds_most_relevant_document(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=1)

        store.write_documents(
            [
                Document(id="1", content="The Eiffel Tower is a famous landmark in Paris, France."),
                Document(id="2", content="The Amazon River flows through the South American rainforest."),
            ]
        )

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "famous tower in France")
        result = retriever.run(query_embedding=query_embedding)

        assert len(result["documents"]) == 1
        assert "Eiffel" in result["documents"][0].content

    def test_embedding_retriever_respects_top_k(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=2)

        store.write_documents(
            [
                Document(id="1", content="Python is a popular high-level programming language."),
                Document(id="2", content="Java is widely used in enterprise software development."),
                Document(id="3", content="Rust focuses on memory safety and systems programming."),
            ]
        )

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "programming language")
        result = retriever.run(query_embedding=query_embedding)

        assert 0 < len(result["documents"]) <= 2

    def test_embedding_retriever_with_metadata_filter(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=5)

        store.write_documents(
            [
                Document(id="1", content="Berlin is the capital of Germany.", meta={"lang": "en"}),
                Document(id="2", content="Berlin ist die Hauptstadt von Deutschland.", meta={"lang": "de"}),
            ]
        )

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "capital of Germany")
        result = retriever.run(
            query_embedding=query_embedding,
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Berlin is the capital of Germany."

    def test_multiple_documents_are_all_indexed_with_embeddings(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        docs = [Document(id=f"doc-{i}", content=f"Document number {i} about various topics.") for i in range(5)]
        store.write_documents(docs)

        store.client.indices.refresh(index=store._index)
        assert store.count_documents() == 5

        for doc in docs:
            raw = store.client.get(index=store._index, id=doc.id)
            # ES inference pipelines do NOT write dense_vector data to _source; only to the vector index.
            assert raw["_source"].get("embedding") is None

        # Verify all documents have embeddings via KNN search — finds docs only if vectors exist.
        query_embedding = _get_dense_query_embedding(store.client, inference_id, "document about topics")
        result = store.client.search(
            index=store._index,
            knn={"field": "embedding", "query_vector": query_embedding, "k": 5, "num_candidates": 50},
        )
        assert result["hits"]["total"]["value"] == 5, "not all documents were indexed with embeddings"

    def test_retrieved_documents_carry_score(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=2)

        store.write_documents(
            [
                Document(id="1", content="Mount Everest is the highest mountain on Earth."),
                Document(id="2", content="The Pacific Ocean is the largest ocean on Earth."),
            ]
        )

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "tallest mountain")
        result = retriever.run(query_embedding=query_embedding)

        assert len(result["documents"]) > 0
        for doc in result["documents"]:
            assert doc.score is not None
            assert doc.content is not None

    @pytest.mark.asyncio
    async def test_async_write_documents_via_pipeline(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=1)

        await store.write_documents_async(
            [
                Document(id="async-1", content="Rome is the capital of Italy."),
                Document(id="async-2", content="Tokyo is the capital of Japan."),
            ]
        )

        store.client.indices.refresh(index=store._index)
        raw = store.client.get(index=store._index, id="async-1")
        # ES inference pipelines do NOT write dense_vector data to _source; only to the vector index.
        assert raw["_source"].get("embedding") is None

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "capital of Italy")
        result = retriever.run(query_embedding=query_embedding)

        assert len(result["documents"]) == 1
        assert "Rome" in result["documents"][0].content


@pytest.mark.integration
class TestIngestPipelineSparse:
    """
    End-to-end integration tests for ElasticsearchDocumentStore with an ingest pipeline
    that generates ELSER sparse embeddings at index time.

    The fixture creates a real ES ingest pipeline (ELSER inference processor → sparse_vec field)
    on Elastic Cloud. Documents are written without pre-computed sparse embeddings; the pipeline
    fills the sparse_vec field before the document is committed to the index.

    Run with: pytest -m integration
    """

    def test_indexed_document_has_sparse_embedding_filled_by_pipeline(self, ingest_pipeline_sparse_document_store):
        store, _ = ingest_pipeline_sparse_document_store
        store.write_documents([Document(id="doc-1", content="The Eiffel Tower is located in Paris.")])

        store.client.indices.refresh(index=store._index)
        raw = store.client.get(index=store._index, id="doc-1")

        # ES inference pipelines do NOT write sparse_vector data to _source; only to the inverted index.
        assert raw["_source"].get("sparse_vec") is None

        # The data IS in the index and retrievable via the fields API.
        result = store.client.search(
            index=store._index,
            fields=["sparse_vec"],
            query={"ids": {"values": ["doc-1"]}},
        )
        hit = result["hits"]["hits"][0]
        assert "sparse_vec" in hit.get("fields", {}), "pipeline did not write to the 'sparse_vec' field"

    def test_inference_sparse_retriever_finds_most_relevant_document(self, ingest_pipeline_sparse_document_store):
        store, inference_id = ingest_pipeline_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=1)

        store.write_documents(
            [
                Document(id="1", content="The Eiffel Tower is a famous landmark in Paris, France."),
                Document(id="2", content="The Amazon River flows through the South American rainforest."),
            ]
        )

        result = retriever.run(query="famous tower in France")

        assert len(result["documents"]) == 1
        assert "Eiffel" in result["documents"][0].content

    def test_inference_sparse_retriever_respects_top_k(self, ingest_pipeline_sparse_document_store):
        store, inference_id = ingest_pipeline_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=2)

        store.write_documents(
            [
                Document(id="1", content="Python is a popular high-level programming language."),
                Document(id="2", content="Java is widely used in enterprise software development."),
                Document(id="3", content="Rust focuses on memory safety and systems programming."),
            ]
        )

        result = retriever.run(query="programming language")

        assert 0 < len(result["documents"]) <= 2

    def test_inference_sparse_retriever_with_metadata_filter(self, ingest_pipeline_sparse_document_store):
        store, inference_id = ingest_pipeline_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=5)

        store.write_documents(
            [
                Document(id="1", content="Berlin is the capital of Germany.", meta={"lang": "en"}),
                Document(id="2", content="Berlin ist die Hauptstadt von Deutschland.", meta={"lang": "de"}),
            ]
        )

        result = retriever.run(
            query="capital of Germany",
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Berlin is the capital of Germany."

    def test_multiple_documents_are_all_indexed_with_sparse_embeddings(self, ingest_pipeline_sparse_document_store):
        store, _ = ingest_pipeline_sparse_document_store
        docs = [Document(id=f"doc-{i}", content=f"Document number {i} about various topics.") for i in range(5)]
        store.write_documents(docs)

        store.client.indices.refresh(index=store._index)
        assert store.count_documents() == 5

        for doc in docs:
            raw = store.client.get(index=store._index, id=doc.id)
            # ES inference pipelines do NOT write sparse_vector data to _source; only to the inverted index.
            assert raw["_source"].get("sparse_vec") is None

        # Verify all documents have sparse_vec populated in the index via the fields API.
        doc_ids = [doc.id for doc in docs]
        result = store.client.search(
            index=store._index,
            fields=["sparse_vec"],
            query={"ids": {"values": doc_ids}},
            size=len(docs),
        )
        for hit in result["hits"]["hits"]:
            assert "sparse_vec" in hit.get("fields", {}), f"doc {hit['_id']} missing sparse_vec in index"

    def test_retrieved_documents_carry_score(self, ingest_pipeline_sparse_document_store):
        store, inference_id = ingest_pipeline_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=2)

        store.write_documents(
            [
                Document(id="1", content="Mount Everest is the highest mountain on Earth."),
                Document(id="2", content="The Pacific Ocean is the largest ocean on Earth."),
            ]
        )

        result = retriever.run(query="tallest mountain in the world")

        assert len(result["documents"]) > 0
        for doc in result["documents"]:
            assert doc.score is not None
            assert doc.content is not None

    @pytest.mark.asyncio
    async def test_async_write_documents_via_pipeline(self, ingest_pipeline_sparse_document_store):
        store, inference_id = ingest_pipeline_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=1)

        await store.write_documents_async(
            [
                Document(id="async-1", content="Rome is the capital of Italy."),
                Document(id="async-2", content="Tokyo is the capital of Japan."),
            ]
        )

        store.client.indices.refresh(index=store._index)
        result = retriever.run(query="capital of Italy")

        assert len(result["documents"]) == 1
        assert "Rome" in result["documents"][0].content
