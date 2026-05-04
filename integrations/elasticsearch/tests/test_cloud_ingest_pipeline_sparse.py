# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchInferenceSparseRetriever


@pytest.mark.integration
class TestElasticSearchIngestPipelineSparse:
    """
    End-to-end integration tests for ElasticsearchDocumentStore with an ingest pipeline
    that generates ELSER sparse embeddings at index time.

    The fixture creates a real ES ingest pipeline (ELSER inference processor → sparse_vec field)
    on Elastic Cloud. Documents are written without pre-computed sparse embeddings; the pipeline
    fills the sparse_vec field before the document is committed to the index.
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
