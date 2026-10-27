"""
Knowledge Base Loader & RAG Indexing for ZentroMart VoiceOps 2026.

Parses, chunks, and indexes the 21 Markdown policy files from kb/:
- Preserves YAML frontmatter metadata (doc_id, doc_type, status, effective_from, title).
- Respects precedence rules:
    1. Policy (status: current)
    2. Agent guide
    3. Help article
    4. FAQ
    5. Marketing
    6. Meta
- Ignores superseded files (status: superseded) during default indexing.
- Extracts unique section IDs formatted as `## [SECTION-ID] Heading`.
- Implements hybrid lexical (TF-IDF + keyword) retrieval with precedence-aware re-ranking.
- Provides confidence thresholding to strictly detect out-of-scope / unsupported questions.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Precedence hierarchy mapping (lower number = higher precedence)
PRECEDENCE_ORDER = {
    "policy": 1,
    "agent_guide": 2,
    "help": 3,
    "faq": 4,
    "marketing": 5,
    "meta": 6,
}


@dataclass
class KBSection:
    """Represents a chunked section of a knowledge base document."""
    section_id: str
    doc_id: str
    title: str
    doc_type: str
    priority: int
    heading: str
    content: str
    full_text: str
    effective_from: str = ""
    status: str = "current"
    language: str = "en"
    file_name: str = ""
    is_superseded: bool = False


class KnowledgeBaseLoader:
    """Manages document ingestion, chunking, indexing, and precedence-aware retrieval."""

    def __init__(self, kb_dir: str | Path | None = None) -> None:
        if kb_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.kb_dir = base_dir / "kb"
        else:
            self.kb_dir = Path(kb_dir)

        # Active indexed sections (excluding superseded files)
        self.sections: List[KBSection] = []
        self.sections_by_id: Dict[str, KBSection] = {}

        # Superseded sections kept for historical reference (e.g. pre-Aug 2026)
        self.superseded_sections: Dict[str, KBSection] = {}

        # All sections combined
        self.all_sections_by_id: Dict[str, KBSection] = {}

        # Document metadata store
        self.documents_metadata: Dict[str, Dict[str, str]] = {}

        # Vectorizer and matrix for retrieval
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self.load_documents()

    def _parse_frontmatter(self, text: str) -> Tuple[Dict[str, str], str]:
        """Extract YAML frontmatter and remaining body text."""
        metadata = {}
        body = text
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if match:
            fm_text = match.group(1)
            body = text[match.end():]
            for line in fm_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip()] = v.strip().strip('"\'')
        return metadata, body

    def load_documents(self) -> None:
        """Parse all markdown files in kb/ and build search indexes."""
        self.sections.clear()
        self.sections_by_id.clear()
        self.superseded_sections.clear()
        self.all_sections_by_id.clear()

        md_files = sorted(self.kb_dir.glob("*.md"))

        for file_path in md_files:
            content = file_path.read_text(encoding="utf-8")
            metadata, body = self._parse_frontmatter(content)

            doc_id = metadata.get("doc_id", file_path.stem)
            doc_type = metadata.get("doc_type", "policy")
            status = metadata.get("status", "current")
            title = metadata.get("title", file_path.stem)
            effective_from = metadata.get("effective_from", "")
            language = metadata.get("language", "en")

            priority = PRECEDENCE_ORDER.get(doc_type, 5)
            is_superseded = (status == "superseded")

            self.documents_metadata[doc_id] = {
                "file_name": file_path.name,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "status": status,
                "title": title,
                "effective_from": effective_from,
                "language": language,
                "priority": str(priority),
            }

            # Split document body into sections by `## [SECTION-ID] Heading`
            # Pattern matches the section delimiter and captures id and heading
            section_pattern = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$", re.MULTILINE)
            matches = list(section_pattern.finditer(body))

            if not matches:
                # Document has no bracketed sections; index whole body if needed
                continue

            for i, match in enumerate(matches):
                sec_id = match.group(1).strip()
                heading = match.group(2).strip()

                start_idx = match.end()
                end_idx = matches[i + 1].start() if (i + 1) < len(matches) else len(body)
                sec_content = body[start_idx:end_idx].strip()

                full_search_text = (
                    f"Document: {title} ({doc_type})\n"
                    f"Section [{sec_id}]: {heading}\n"
                    f"{sec_content}"
                )

                section_obj = KBSection(
                    section_id=sec_id,
                    doc_id=doc_id,
                    title=title,
                    doc_type=doc_type,
                    priority=priority,
                    heading=heading,
                    content=sec_content,
                    full_text=full_search_text,
                    effective_from=effective_from,
                    status=status,
                    language=language,
                    file_name=file_path.name,
                    is_superseded=is_superseded,
                )

                self.all_sections_by_id[sec_id] = section_obj

                if is_superseded:
                    # Respect precedence: ignore superseded files in active retrieval
                    self.superseded_sections[sec_id] = section_obj
                else:
                    self.sections.append(section_obj)
                    self.sections_by_id[sec_id] = section_obj

        self._build_vector_index()

    def _build_vector_index(self) -> None:
        """Build TF-IDF vector index over all active sections."""
        if not self.sections:
            return

        corpus = [s.full_text for s in self.sections]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w-]+\b",
            lowercase=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def get_section(self, section_id: str) -> Optional[KBSection]:
        """Direct lookup of section by its exact section ID (e.g. 'RET-2')."""
        clean_id = section_id.strip("[] ")
        return self.all_sections_by_id.get(clean_id)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.08,
        include_superseded: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant sections for a user query respecting document precedence.
        
        Args:
            query: User's question or search query.
            top_k: Maximum number of sections to return.
            min_score: Threshold below which results are excluded (guard against hallucination).
            include_superseded: Whether to search superseded policy versions.
        
        Returns:
            List of section dictionaries sorted by precedence-weighted score.
        """
        if not self.sections or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Direct section ID mention check in query (e.g. "[RET-2]" or "RET-2")
        explicit_matches = re.findall(r"\b([A-Z0-9]+-[A-Z0-9\.]+)\b", clean_query)
        exact_results = []
        for sec_id in explicit_matches:
            sec = self.all_sections_by_id.get(sec_id)
            if sec:
                exact_results.append({
                    "section_id": sec.section_id,
                    "doc_id": sec.doc_id,
                    "doc_type": sec.doc_type,
                    "priority": sec.priority,
                    "title": sec.title,
                    "heading": sec.heading,
                    "content": sec.content,
                    "score": 1.5,  # High boost for exact ID lookup
                    "is_superseded": sec.is_superseded,
                })

        # 2. Vector search over active sections
        query_vec = self.vectorizer.transform([clean_query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        scored_candidates = []
        for idx, base_score in enumerate(sims):
            if base_score < min_score:
                continue

            sec = self.sections[idx]

            # Keyword bonus for critical operational terms
            q_lower = clean_query.lower()
            bonus = 0.0
            heading_lower = sec.heading.lower()
            content_lower = sec.content.lower()

            # Check overlap with key entities
            for term in ["wallet", "refund", "damage", "replacement", "49", "festive",
                         "assured", "warranty", "pickup", "exchange", "10,000", "cancel",
                         "verification", "ticket", "human", "pincode", "cod", "plus"]:
                if term in q_lower and (term in heading_lower or term in content_lower):
                    bonus += 0.05

            # Apply Precedence Weighting (Policy > Agent Guide > Help > FAQ > Marketing)
            # Priority 1 (policy) gets highest multiplier; Priority 5 (marketing) gets lower
            precedence_multiplier = 1.0 + (6 - sec.priority) * 0.08
            final_score = (base_score + bonus) * precedence_multiplier

            scored_candidates.append({
                "section_id": sec.section_id,
                "doc_id": sec.doc_id,
                "doc_type": sec.doc_type,
                "priority": sec.priority,
                "title": sec.title,
                "heading": sec.heading,
                "content": sec.content,
                "score": float(final_score),
                "is_superseded": sec.is_superseded,
            })

        # If include_superseded requested, also score superseded sections
        if include_superseded and self.superseded_sections:
            for sec in self.superseded_sections.values():
                sup_vec = self.vectorizer.transform([sec.full_text])
                sup_sim = float(cosine_similarity(query_vec, sup_vec)[0][0])
                if sup_sim >= min_score:
                    scored_candidates.append({
                        "section_id": sec.section_id,
                        "doc_id": sec.doc_id,
                        "doc_type": sec.doc_type,
                        "priority": 10,  # superseded has lowest priority
                        "title": sec.title,
                        "heading": sec.heading,
                        "content": sec.content,
                        "score": sup_sim * 0.7,
                        "is_superseded": True,
                    })

        # Combine exact matches with scored results
        seen_ids = set()
        final_list = []

        for item in exact_results + sorted(scored_candidates, key=lambda x: x["score"], reverse=True):
            sid = item["section_id"]
            if sid not in seen_ids:
                seen_ids.add(sid)
                final_list.append(item)
            if len(final_list) >= top_k:
                break

        return final_list
