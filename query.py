"""
=============================================================================
scripts/query.py — Terminal RAG with intent-driven hard filtering (v6.1)
=============================================================================

WHAT CHANGED vs v6
------------------
v6 pushed an intent filter into Qdrant. Two problems showed up in the
golden-dataset run:

  (A) The 3B model could not execute the conditional prompt rules. It
      emitted "I could not find that information..." AND then answered,
      and once leaked the literal placeholder "<jurisdiction>".
  (B) In loose mode the single top-40 retrieval let IEC chunks (closer
      in embedding space) starve ALL pt_legislation chunks, even when
      the jurisdiction filter was correct.
  (C) Real engineering questions ("select the cross-section of MV
      cables") triggered no intent at all and drifted to handbooks.

v6.1 fixes:

  A. Single-rule prompt. No conditional self-diagnosis. The system
     already knows what it retrieved; the model just answers or gives
     ONE refusal sentence. No <placeholder> tokens.

  B. Dual-pass quota retrieval for jurisdiction questions. One pass
     retrieves ONLY the jurisdiction (guarantees legislation chunks),
     a second pass retrieves ONLY iec_standard (supporting normative
     context). Merge + dedupe. Neither side can starve the other.

  C. Engineering-sizing lexicon added to detect_intent so cable /
     ampacity / earthing questions prioritise normative sources.

  + New --model flag to switch the generation model (e.g. 14B) for
    the 3B-vs-14B benchmark, without editing the file.

PIPELINE
--------
    1. Detect intent (PT / BR / HR / EU / IEC number / sizing terms).
    2. If jurisdiction intent  -> DUAL-PASS QUOTA retrieval.
       Else                    -> single filtered retrieval (as v6).
    3. If too few chunks       -> warn + retry without filter.
    4. Boost IEC chunks whose filename matches the requested number.
    5. Top rerank_in candidates -> BGE-v2-m3 reranker -> top-N.
    6. Grouped context -> single-rule prompt -> generation model.

CLI
---
    python scripts\\query.py "..."                       # default (3B)
    python scripts\\query.py "..." --no-filter           # v4 behaviour
    python scripts\\query.py "..." --strict              # only target juris
    python scripts\\query.py "..." --min-after-filter 8  # higher threshold
    python scripts\\query.py "..." --debug               # full audit
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import sys, os, re, time, argparse, warnings, logging
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']      = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY']          = '1'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))
from classification import classify_source, TYPE_ORDER  # noqa: E402

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilters, MetadataFilter, FilterCondition, FilterOperator,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from qdrant_client import QdrantClient


# =============================================================================
# ENCODING REPAIRS
# =============================================================================
ENCODING_FIXES = [
    ('\u221eC', '\u00b0C'),
    ('\u221eF', '\u00b0F'),
]
def fix_encoding(text: str) -> str:
    for bad, good in ENCODING_FIXES:
        text = text.replace(bad, good)
    return text


# =============================================================================
# INTENT DETECTION
# =============================================================================
def detect_intent(query: str) -> dict:
    q = query.lower()
    priority_types         = []
    priority_jurisdictions = []
    notes                  = []

    # IEC + number capture
    iec_match  = re.search(r'\biec\s*(\d{3,5})\b', q)
    iec_number = iec_match.group(1) if iec_match else None
    if iec_match:
        priority_types.append("iec_standard")
        notes.append(f"Question references IEC {iec_number}. Hard-filter to "
                     f"iec_standard source type.")
    elif re.search(r'\biec[\s\-]', q):
        priority_types.append("iec_standard")
        notes.append("Question references IEC standards generally. "
                     "Hard-filter to iec_standard source type.")
    elif re.search(r'\bstandard\b|\bnorma\b(?!l)', q):
        if "iec_standard" not in priority_types:
            priority_types.append("iec_standard")
        notes.append("Question is about standards generally. "
                     "Prioritize normative source types.")

    # ---- (C) Engineering-sizing questions imply normative content ----
    # These questions ("select the cross-section of MV cables", "ampacity",
    # "burial depth", "trefoil"...) almost never say "IEC" or "standard",
    # so without this they trigger no intent and drift to handbooks.
    # NOTE (thesis trade-off): for a pure sizing question with NO
    # jurisdiction this routes to the iec_only branch, which excludes
    # documents like IEEE 80. Acceptable for cable sizing; documented as
    # a known limitation of keyword-based intent.
    if re.search(
        r'cross[\s\-]?section|ampacity|current[\s\-]?rating|'
        r'\bderating\b|cable\s+sizing|conductor\s+siz|'
        r'earth(?:ing)?\s+resistance|burial\s+depth|\btrefoil\b',
        q,
    ):
        if "iec_standard" not in priority_types:
            priority_types.append("iec_standard")
        notes.append("Engineering sizing question. Prioritize normative "
                     "(iec_standard) sources.")

    # Portugal
    if re.search(
        r'\bportugal\b|portugu[êe]s|portuguesa|\brtiebt\b|\brsiut\b|'
        r'\bersee?\b|\bdgeg\b|decreto[\s\-]lei|\bportaria\b|\bdespacho\b|'
        r'di[áa]rio\s+da\s+rep[úu]blica|\bren\b|\bror\b|'
        r'rede\s+(?:de\s+)?transporte|\brnt\b|\brnd\b',
        q,
    ):
        priority_jurisdictions.append("PT")
        if "pt_legislation" not in priority_types:
            priority_types.append("pt_legislation")
        notes.append("Question concerns Portuguese context. Hard-filter to "
                     "jurisdiction=PT (loose: + iec_standard).")

    # Brazil
    if re.search(r'\bbrasil\b|brazilian\b|\baneel\b|\bons\b|\bprodist\b', q):
        priority_jurisdictions.append("BR")
        if "br_legislation" not in priority_types:
            priority_types.append("br_legislation")
        notes.append("Question concerns Brazilian context. Hard-filter to "
                     "jurisdiction=BR (loose: + iec_standard).")

    # Croatia
    if re.search(r'croat\w+|\bhrvatska\b|narodne\s+novine|\bhera\b', q):
        priority_jurisdictions.append("HR")
        if "hr_legislation" not in priority_types:
            priority_types.append("hr_legislation")
        notes.append("Question concerns Croatian context. Hard-filter to "
                     "jurisdiction=HR (loose: + iec_standard).")

    # EU
    if re.search(
        r'\beu\b|european\s+union|european\s+commission|'
        r'directiva|directive|regulamento\s+eu|\brfg\b',
        q,
    ):
        priority_jurisdictions.append("EU")
        if "eu_legislation" not in priority_types:
            priority_types.append("eu_legislation")
        notes.append("Question concerns EU context. Hard-filter to "
                     "jurisdiction=EU (loose: + iec_standard).")

    # Book / handbook
    if re.search(r'\bbook\b|\bhandbook\b|\btextbook\b|livro\b', q):
        if "book" not in priority_types:
            priority_types.append("book")
        notes.append("Question asks about book/handbook content. "
                     "No hard filter applied for this signal alone.")

    return {
        "priority_types":         priority_types,
        "priority_jurisdictions": priority_jurisdictions,
        "iec_number":             iec_number,
        "notes":                  notes,
    }


# =============================================================================
# INTENT → QDRANT METADATA FILTER  (used for NON-quota paths)
# =============================================================================
def build_intent_filter(intent: dict, strict: bool):
    """
    Translate intent into a LlamaIndex MetadataFilters object that Qdrant
    evaluates server-side. Returns (filter_or_None, description_str).
    """
    juris_list = intent.get("priority_jurisdictions") or []
    types_list = intent.get("priority_types") or []
    iec_only   = ("iec_standard" in types_list) and not juris_list

    if not juris_list and not iec_only:
        return None, "no filter (no jurisdiction or IEC intent detected)"

    if juris_list:
        juris_filters = [
            MetadataFilter(key="jurisdiction", value=j,
                           operator=FilterOperator.EQ)
            for j in juris_list
        ]
        if strict:
            if len(juris_filters) == 1:
                desc = f"jurisdiction == {juris_list[0]} (strict)"
                return (
                    MetadataFilters(filters=juris_filters,
                                    condition=FilterCondition.AND),
                    desc,
                )
            desc = f"jurisdiction in {juris_list} (strict)"
            return (
                MetadataFilters(filters=juris_filters,
                                condition=FilterCondition.OR),
                desc,
            )

        # Loose: jurisdictions OR iec_standard
        iec_filter = MetadataFilter(
            key="source_type", value="iec_standard",
            operator=FilterOperator.EQ,
        )
        all_filters = juris_filters + [iec_filter]
        desc = (f"jurisdiction in {juris_list} OR "
                f"source_type == iec_standard (loose)")
        return (
            MetadataFilters(filters=all_filters,
                            condition=FilterCondition.OR),
            desc,
        )

    # iec_only branch
    desc = "source_type == iec_standard"
    return (
        MetadataFilters(
            filters=[MetadataFilter(
                key="source_type", value="iec_standard",
                operator=FilterOperator.EQ,
            )],
            condition=FilterCondition.AND,
        ),
        desc,
    )


# =============================================================================
# IEC NUMBER SCORE ADJUSTMENT (post-filter)
# =============================================================================
def adjust_iec_scores(nodes, iec_number, boost: float, penalty: float,
                      verbose: bool = False):
    """
    When the query mentions a specific IEC number (e.g. "60287"):
      - Boost  iec_standard chunks whose filename contains that number.
      - Penalise iec_standard chunks whose filename contains a different
        4-5 digit IEC family number.
      - Non-IEC chunks untouched.
    """
    if not iec_number:
        return nodes

    for n in nodes:
        md = n.node.metadata
        if md.get("source_type") != "iec_standard":
            continue

        fname     = (md.get("file_name", "") or "").lower()
        compact   = fname.replace(" ", "").replace("-", "").replace("_", "")
        original  = float(n.score or 0.0)
        delta     = 0.0
        reasons   = []

        if iec_number in compact:
            delta += boost
            reasons.append(f"+{boost:.2f}/IEC_match={iec_number}")
        else:
            # Look for any 4-5 digit IEC family number in the filename;
            # if a *different* one is present, penalise.
            other = re.search(r'\b(\d{4,5})\b', fname)
            if other and other.group(1) != iec_number:
                delta -= penalty
                reasons.append(f"-{penalty:.2f}/IEC_other={other.group(1)}")

        if delta != 0.0:
            md["score_original"] = original
            md["score_delta"]    = delta
            md["score_reasons"]  = " ".join(reasons)
            n.score = original + delta
            if verbose:
                print(f"   adjust {fname[:50]:<50} "
                      f"{original:.3f} -> {n.score:.3f} "
                      f"({md['score_reasons']})", flush=True)

    nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
    return nodes


# =============================================================================
# CONTEXT GROUPING
# =============================================================================
def build_grouped_context(reranked) -> str:
    grouped = defaultdict(list)
    for i, n in enumerate(reranked, 1):
        t = n.node.metadata.get("source_type", "unknown")
        grouped[t].append((i, n))

    parts = []
    for t in TYPE_ORDER:
        if t not in grouped:
            continue
        parts.append(f"=== {t.upper()} ===")
        for i, n in grouped[t]:
            md = n.node.metadata
            header = (
                f"[Source {i}] "
                f"file={md.get('file_name','?')} "
                f"page={md.get('page_label','?')} "
                f"jurisdiction={md.get('jurisdiction','unknown')}"
            )
            parts.append(f"{header}\n{n.node.get_content()}")
        parts.append("")
    return "\n".join(parts).strip()


# =============================================================================
# PROMPT TEMPLATE  — (A) single-rule, no conditional self-diagnosis
# =============================================================================
PROMPT_TEMPLATE = """You are a technical assistant for wind farm electrical \
engineering, standards, and cable sizing. Always answer in English, whatever \
the language of the question or the sources.

CONTEXT is a set of chunks. Each chunk is tagged [Source N] with its type \
(iec_standard, pt_legislation, book, document, ...) and jurisdiction.

The CONTEXT has been pre-filtered and reranked. Your job is to extract the \
answer from it. Do not invent content that is not in the CONTEXT.

RETRIEVAL CONFIDENCE: {retrieval_confidence}

YOUR OUTPUT RULE (pick exactly one):

>>> If RETRIEVAL CONFIDENCE = HIGH or MEDIUM:
    You MUST answer the question from the CONTEXT. Cite each fact with [#N].
    Quote numbers, formulas and clauses exactly as written. Be concrete.
    Do NOT say "I could not find that information" — the system has already \
verified that relevant material is in the CONTEXT. If the CONTEXT contains \
material that is on-topic but does not give a complete answer, give the \
partial answer that the CONTEXT supports and clearly state what is missing.
    For MEDIUM only, open with: "Based on the retrieved sources (partial \
match):"

>>> If RETRIEVAL CONFIDENCE = LOW:
    Reply with EXACTLY this one sentence and nothing else: "I could not \
find that information in the knowledge base."

Other rules (apply to HIGH and MEDIUM):
- Use only facts from the CONTEXT. No outside knowledge.
- A standard (iec_standard) or legislation outranks a book or paper. Do not \
present a book/paper claim as a normative requirement.
- If two sources disagree on a value, list both with [#N] and flag it.
- Do not announce which sections are empty, do not narrate your reasoning, \
do not apologize.

CONTEXT:
{context_str}

QUESTION: {query_str}

ANSWER:"""


def format_intent_hints(intent: dict) -> str:
    """Kept for --debug printing only; no longer injected into the prompt."""
    if not intent["notes"]:
        return ("  (no specific intent detected — answering from all "
                "available sources, grouped by type)")
    lines = [f"  - {note}" for note in intent["notes"]]
    if intent["priority_types"]:
        lines.append(
            f"  - Priority source types: {', '.join(intent['priority_types'])}"
        )
    if intent["priority_jurisdictions"]:
        lines.append(
            f"  - Priority jurisdictions: "
            f"{', '.join(intent['priority_jurisdictions'])}"
        )
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Terminal RAG with intent-driven hard filtering."
    )
    p.add_argument("question", nargs="*", help="Question to ask")
    p.add_argument("--model", default="qwen2.5:3b-instruct-q4_K_M ")
    
    p.add_argument("--top-k", type=int, default=40,
                   help="Initial BGE-M3 retrieval with filter (default 40)")
    p.add_argument("--top-k-fallback", type=int, default=12,
                   help="Retrieval size used in the no-filter fallback "
                        "(default 12)")
    p.add_argument("--rerank-in", type=int, default=12,
                   help="How many candidates feed the cross-encoder "
                        "(default 12)")
    p.add_argument("--top-n", type=int, default=5,
                   help="Final results after reranker (default 5)")
    p.add_argument("--min-after-filter", type=int, default=5,
                   help="If filter returns fewer than N chunks, trigger "
                        "no-filter fallback (default 5)")
    p.add_argument("--type", default=None, help="Hard filter by source type")
    p.add_argument("--jurisdiction", default=None,
                   help="Hard filter by jurisdiction")
    p.add_argument("--no-filter", action="store_true",
                   help="Disable intent-driven hard filter (v4 mode)")
    p.add_argument("--strict", action="store_true",
                   help="Strict filter: jurisdiction only, no IEC fallback")
    p.add_argument("--debug", action="store_true",
                   help="Print score adjustments + full chunks")
    p.add_argument("--collection", default="kb")
    p.add_argument("--no-intent", action="store_true",
                   help="Skip intent detection entirely")
    p.add_argument("--boost-iec", type=float, default=0.40)
    p.add_argument("--penalty-iec", type=float, default=0.30)
    return p.parse_args()


# =============================================================================
# RETRIEVAL HELPERS
# =============================================================================
def retrieve_with_filter(index, query, top_k, filters, label=""):
    t0 = time.time()
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    )
    nodes = retriever.retrieve(query)
    if label:
        print(f"   {label}: {len(nodes)} nodes in {time.time()-t0:.1f}s")
    return nodes


def retrieve_with_quota(index, query, intent, args):
    """
    (B) Jurisdiction question: TWO retrievals, then RESERVED SLOTS in the
    rerank pool so neither side can starve the other.

    v6.1 BUG (fixed here): the merge re-sorted everything by raw embedding
    score. PT legislation scores lower than IEC standards in embedding
    space for technical queries, so every PT chunk was truncated by
    nodes[:rerank_in] BEFORE the cross-encoder ever saw it. The dual-pass
    retrieval was pointless because the merge threw the PT side away.

    Fix: do NOT re-sort the union by score. Reserve the first rerank_in
    slots — half guaranteed to the jurisdiction pass, half to
    iec_standard — keeping each pass in its own Qdrant score order. The
    cross-encoder then judges actual relevance on a pool that is
    guaranteed to CONTAIN jurisdiction candidates. Leftovers follow,
    score-sorted, only as fallback backfill.
    """
    juris_list = intent.get("priority_jurisdictions") or []

    jfilters = [
        MetadataFilter(key="jurisdiction", value=j,
                       operator=FilterOperator.EQ)
        for j in juris_list
    ]
    jfilter = MetadataFilters(
        filters=jfilters,
        condition=FilterCondition.OR if len(jfilters) > 1
        else FilterCondition.AND,
    )
    juris_nodes = retrieve_with_filter(
        index, query, args.top_k, jfilter, label="jurisdiction pass"
    )

    iec_filter = MetadataFilters(
        filters=[MetadataFilter(key="source_type", value="iec_standard",
                                operator=FilterOperator.EQ)],
        condition=FilterCondition.AND,
    )
    iec_nodes = retrieve_with_filter(
        index, query, max(args.top_k // 2, 12), iec_filter,
        label="iec pass"
    )

    # Dedupe across passes, preserving each pass's own Qdrant order
    # (a chunk seen in the jurisdiction pass is not repeated in IEC).
    seen = set()
    def _dedupe(lst):
        out = []
        for n in lst:
            nid = n.node.node_id
            if nid in seen:
                continue
            seen.add(nid)
            out.append(n)
        return out
    juris_nodes = _dedupe(juris_nodes)
    iec_nodes   = _dedupe(iec_nodes)

    # Reserved split for the rerank pool: half jurisdiction, half IEC.
    cap     = args.rerank_in
    j_quota = max(1, cap // 2)
    i_quota = cap - j_quota

    head = juris_nodes[:j_quota] + iec_nodes[:i_quota]

    # If one side was short, backfill the head from whatever remains so
    # the rerank pool is still full.
    if len(head) < cap:
        leftover = juris_nodes[j_quota:] + iec_nodes[i_quota:]
        leftover.sort(key=lambda x: x.score or 0.0, reverse=True)
        head += leftover[: cap - len(head)]

    in_head = set(id(n) for n in head)
    rest = [n for n in (juris_nodes + iec_nodes) if id(n) not in in_head]
    rest.sort(key=lambda x: x.score or 0.0, reverse=True)
    merged = head + rest

    n_pt  = sum(1 for n in head
                if n.node.metadata.get("jurisdiction") in juris_list)
    n_iec = len(head) - n_pt
    print(f"   quota split into rerank pool: "
          f"{n_pt} jurisdiction + {n_iec} iec_standard "
          f"(of {len(head)} slots)")

    desc = (f"quota: jurisdiction {juris_list} (reserved {j_quota}) "
            f"+ iec_standard (reserved {i_quota}), slot-protected")
    return merged, desc


# =============================================================================
# MAIN
# =============================================================================
def main():
    args  = parse_args()
    query = " ".join(args.question) or "Summarize the main topics of the documents."

    print(f">> Question: {query}", flush=True)

    # ---- Intent ----
    intent = (
        {"priority_types": [], "priority_jurisdictions": [],
         "iec_number": None, "notes": []}
        if args.no_intent else detect_intent(query)
    )
    if intent["notes"]:
        print(">> Detected intent:")
        for note in intent["notes"]:
            print(f"   - {note}")
    elif not args.no_intent:
        print(">> No specific intent detected.")

    if args.type:
        print(f">> Hard filter (CLI override): source_type={args.type}")
    if args.jurisdiction:
        print(f">> Hard filter (CLI override): jurisdiction={args.jurisdiction}")
    if args.no_filter:
        print(">> --no-filter: intent-driven filtering DISABLED.")

    # ---- Services ----
    Settings.embed_model = OllamaEmbedding(
        model_name='bge-m3', base_url='http://localhost:11434'
    )
    print(f">> Generation model: {args.model}", flush=True)
    Settings.llm = Ollama(
        model=args.model,
        base_url='http://localhost:11434',
        request_timeout=1800.0,
        temperature=0.1,
        context_window=8192,
        additional_kwargs={'num_ctx': 8192, 'num_predict': 1536},
    )
    vstore = QdrantVectorStore(
        client=QdrantClient(url='http://localhost:6333', timeout=60),
        collection_name=args.collection,
    )
    index = VectorStoreIndex.from_vector_store(vstore)

    print(">> Loading reranker...", flush=True)
    reranker = SentenceTransformerRerank(
        model='BAAI/bge-reranker-v2-m3',
        top_n=args.top_n,
    )

    # ---- Build intent filter (used by NON-quota paths) ----
    intent_filter = None
    filter_desc   = "no filter"
    if not args.no_filter and not (args.type or args.jurisdiction):
        intent_filter, filter_desc = build_intent_filter(intent, args.strict)

    # CLI overrides become additional filters
    cli_filters = []
    if args.type:
        cli_filters.append(MetadataFilter(
            key="source_type", value=args.type, operator=FilterOperator.EQ))
    if args.jurisdiction:
        cli_filters.append(MetadataFilter(
            key="jurisdiction", value=args.jurisdiction,
            operator=FilterOperator.EQ))
    if cli_filters:
        intent_filter = MetadataFilters(filters=cli_filters,
                                        condition=FilterCondition.AND)
        filter_desc = " AND ".join(
            f"{f.key} == {f.value}" for f in cli_filters
        )

    t_start = time.time()

    # ---- 1. Retrieval ----------------------------------------------------
    # Quota path only for loose jurisdiction questions with no CLI override
    # and no --strict / --no-filter. Everything else keeps v6 behaviour.
    use_quota = (
        not args.no_filter
        and not args.strict
        and not (args.type or args.jurisdiction)
        and bool(intent.get("priority_jurisdictions"))
    )

    used_fallback = False

    if use_quota:
        print(f"\n>> Retrieval: dual-pass quota "
              f"(jurisdiction {intent['priority_jurisdictions']} "
              f"+ iec_standard)")
        nodes, filter_desc = retrieve_with_quota(index, query, intent, args)
        if len(nodes) < args.min_after_filter:
            print(f"\n>> WARNING: quota returned only {len(nodes)} chunks "
                  f"(< --min-after-filter={args.min_after_filter}).")
            print(f">> FALLBACK: retrying without filter at "
                  f"top-k={args.top_k_fallback}.", flush=True)
            nodes = retrieve_with_filter(
                index, query, args.top_k_fallback, None,
                label="fallback returned",
            )
            used_fallback = True
    else:
        print(f"\n>> Retrieval filter: {filter_desc}")
        print(f">> Retrieving top-{args.top_k} from Qdrant "
              f"({'with filter' if intent_filter else 'no filter'})...",
              flush=True)
        nodes = retrieve_with_filter(
            index, query, args.top_k, intent_filter, label="returned"
        )
        if intent_filter and len(nodes) < args.min_after_filter:
            print(f"\n>> WARNING: filter returned only {len(nodes)} chunks "
                  f"(< --min-after-filter={args.min_after_filter}).")
            print(f">> FALLBACK: retrying without filter at "
                  f"top-k={args.top_k_fallback}.", flush=True)
            nodes = retrieve_with_filter(
                index, query, args.top_k_fallback, None,
                label="fallback returned",
            )
            used_fallback = True

    if not nodes:
        print(">> No chunks available — cannot answer.")
        sys.exit(0)

    # ---- 2. Fix encoding + ensure classification ----
    for n in nodes:
        n.node.text = fix_encoding(n.node.text)
        md = n.node.metadata
        if md.get("source_type") in (None, "", "unknown"):
            md.update(classify_source(md.get("file_name", ""), n.node.text))

    # ---- 3. IEC number score adjustment ----
    # NOTE: adjust_iec_scores() re-sorts the whole list by raw score at
    # the end. On the quota path the list is already slot-protected
    # (reserved jurisdiction seats at the head); re-sorting here would
    # resurrect the v6.1 bug and truncate PT chunks again. So the IEC
    # boost only runs on NON-quota paths. (Quota fires on jurisdiction
    # questions; the specific-IEC-number boost targets IEC-number
    # questions — the two intents rarely coincide, so little is lost.)
    if intent.get("iec_number") and not args.no_intent and not use_quota:
        print("\n>> Applying IEC-number score adjustments:")
        nodes = adjust_iec_scores(
            nodes, intent["iec_number"],
            boost=args.boost_iec,
            penalty=args.penalty_iec,
            verbose=args.debug,
        )
    elif intent.get("iec_number") and use_quota:
        print("\n>> (IEC-number boost skipped on quota path to preserve "
              "reserved jurisdiction slots.)")

    # ---- 4. Keep top rerank_in ----
    pre_rerank = nodes[: args.rerank_in]
    print(f"\n>> Top {len(pre_rerank)} candidates entering reranker:")
    for i, n in enumerate(pre_rerank, 1):
        md = n.node.metadata
        delta_str = ""
        if md.get("score_delta", 0):
            delta_str = f"  (orig={md['score_original']:.3f} {md['score_reasons']})"
        print(
            f"  [{i:>2}] {md.get('file_name','?')[:50]:<50} | "
            f"{md.get('source_type','?'):<15}/{md.get('jurisdiction','?'):<13} | "
            f"score={n.score or 0:.3f}{delta_str}"
        )

    # ---- 5. Reranker ----
    print(f"\n>> Reranking to top-{args.top_n} with BGE-v2-m3...", flush=True)
    t1 = time.time()
    reranked = reranker.postprocess_nodes(pre_rerank, query_str=query)
    print(f"   reranked in {time.time()-t1:.1f}s")

    print("\n>> Final sources after reranker:")
    for i, n in enumerate(reranked, 1):
        md = n.node.metadata
        print(
            f"  [#{i}] {md.get('file_name','?')} "
            f"p.{md.get('page_label','?')} | "
            f"{md.get('source_type','?')} / {md.get('jurisdiction','?')} "
            f"({md.get('classification_confidence','?')}) | "
            f"rerank={n.score or 0:.3f}"
        )

    if args.debug:
        print("\n>> Intent hints (debug):")
        print(format_intent_hints(intent))
        print("\n>> Full content of reranked chunks:")
        for i, n in enumerate(reranked, 1):
            print(f"\n--- [#{i}] ---")
            print(n.node.get_content())

    # ---- 6. Prompt + LLM ----
    # Confidence tier from top rerank score. Thresholds calibrated against
    # empirical golden-dataset runs: vague PT queries with the correct
    # documents in the top-5 typically score 0.10-0.20 at rerank top-1
    # (e.g. Manual de Ligacoes p.7 on the "niveis de tensao" question
    # scored 0.150). Treating those as LOW caused blanket refusals; they
    # are MEDIUM and the answer should be partial-with-caveat.
    top_rerank = reranked[0].score if reranked else 0.0
    if top_rerank >= 0.40:
        retrieval_confidence = "HIGH"
    elif top_rerank >= 0.08:
        retrieval_confidence = "MEDIUM"
    else:
        retrieval_confidence = "LOW"
    print(f"\n>> Retrieval confidence tier: {retrieval_confidence} "
          f"(top rerank = {top_rerank:.3f})")

    context_str = build_grouped_context(reranked)
    full_prompt = PROMPT_TEMPLATE.format(
        retrieval_confidence=retrieval_confidence,
        context_str=context_str,
        query_str=query,
    )

    print("\n>> Sending to LLM...", flush=True)
    t2 = time.time()
    resp = Settings.llm.complete(full_prompt)
    elapsed = time.time() - t2
    total   = time.time() - t_start

    if used_fallback:
        print("\n>> [!] FALLBACK_NO_FILTER: this answer was produced WITHOUT "
              "the intent-based filter because retrieval returned too few "
              "chunks. Source diversity may exceed the question's "
              "jurisdiction scope.\n")

    print(f">> Answer (LLM {elapsed:.1f}s, total {total:.1f}s):")
    print(str(resp))

    # ---- Summary ----
    top_score   = reranked[0].score if reranked else 0
    type_counts = defaultdict(int)
    jur_counts  = defaultdict(int)
    for n in reranked:
        type_counts[n.node.metadata.get("source_type", "unknown")] += 1
        jur_counts[n.node.metadata.get("jurisdiction", "unknown")] += 1

    print("\n>> Retrieval summary:")
    print(f"   filter applied: {filter_desc}"
          + ("  (FALLBACK triggered)" if used_fallback else ""))
    print(f"   generation model: {args.model}")
    print(f"   top rerank score: {top_score:.3f}")
    print("   source types:   " +
          ", ".join(f"{k}={v}" for k, v in type_counts.items()))
    print("   jurisdictions:  " +
          ", ".join(f"{k}={v}" for k, v in jur_counts.items()))
    # NOTE: rerank score measures embedding proximity, NOT answer
    # faithfulness. The tier label here is the SAME tier passed to the
    # prompt (thresholds 0.08 / 0.40), so the >> Retrieval confidence
    # tier line and this line always agree.
    if retrieval_confidence == "LOW":
        print("   [!] Low retrieval confidence "
              "(prompt instructed model to refuse).")
    elif retrieval_confidence == "MEDIUM":
        print("   [!] Medium retrieval confidence "
              "— verify the answer manually against the cited sources.")
    else:  # HIGH
        print("   [ok] High retrieval confidence "
              "(retrieval only — does not certify answer faithfulness).")


if __name__ == "__main__":
    main()