"""A source's per-document licences, read off the manifest beside its shards where its card says so."""

import pyarrow as pa
import pyarrow.parquet as pq

from lsa import assets


def test_a_per_document_source_reads_each_documents_licence_off_its_manifest(tmp_path):
    pq.write_table(
        pa.table({"doc_id": ["a", "b"], "upstream": ["org/x"] * 2, "licence": ["CC-BY-4.0", "CC0-1.0"]}),
        tmp_path / "manifest.parquet",
    )
    card = assets.SourceCard(name="x", path=str(tmp_path), dataset="org/x", licence="per-document")

    assert card.document_licences() == {"a": "CC-BY-4.0", "b": "CC0-1.0"}
