from ingestion.parser import ParsedSection
from ingestion.chunker import chunk_section, chunk_document


def test_chunk_section():
    # Setup section with text
    section = ParsedSection(
        section_type="MD&A",
        title="Management Discussion",
        content="This is the first sentence. This is the second sentence. "
        * 30,  # generate a longer paragraph to trigger splits
        page_start=5,
        page_end=6,
    )

    chunks = chunk_section(
        section=section,
        doc_id="doc1",
        section_id="sec1",
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) > 1
    # Check elements of first chunk
    first = chunks[0]
    assert first.source_doc_id == "doc1"
    assert first.section_type == "MD&A"
    assert first.section_id == "sec1"
    assert first.page == 5
    assert first.sequence == 0
    assert first.id == "doc1_MD&A_0"
    assert first.text.strip()


def test_chunk_document():
    sections = [
        ParsedSection(
            section_type="EARNINGS",
            title="Overview",
            content="Earnings were high. Profits increased.",
            page_start=1,
            page_end=1,
        ),
        ParsedSection(
            section_type="FINANCIALS",
            title="Statements",
            content="Revenue was 100M. Expenses were 80M.",
            page_start=2,
            page_end=3,
        ),
    ]

    all_chunks = chunk_document(
        sections=sections,
        doc_id="doc2",
        chunk_size=50,
        overlap=5,
    )

    # Sequence numbers should be sequential across sections
    assert len(all_chunks) >= 2
    for i, chunk in enumerate(all_chunks):
        assert chunk.sequence == i
        assert chunk.source_doc_id == "doc2"
