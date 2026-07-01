

# =============================================================================
# IMPORTAÇÕES
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
# REPARAÇÃO DE CODIFICAÇÃO
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
# DETEÇÃO DE INTENÇÃO
# =============================================================================
def detect_intent(query: str) -> dict:
    q = query.lower()
    priority_types         = []
    priority_jurisdictions = []
    notes                  = []

    # Captura de IEC + número
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

    # ---- (C) Perguntas de dimensionamento implicam conteúdo normativo ----
    # Estas perguntas ("select the cross-section of MV cables", "ampacity",
    # "burial depth", "trefoil"...) quase nunca dizem "IEC" nem "standard",
    # pelo que sem isto não acionam nenhuma intenção e derivam para handbooks.
    # NOTA (compromisso da tese): para uma pergunta de dimensionamento puro SEM
    # jurisdição, isto encaminha para o ramo iec_only, que exclui documentos
    # como a IEEE 80. Aceitável para dimensionamento de cabos; documentado como
    # limitação conhecida da intenção baseada em palavras-chave.
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

    # Brasil
    if re.search(r'\bbrasil\b|brazilian\b|\baneel\b|\bons\b|\bprodist\b', q):
        priority_jurisdictions.append("BR")
        if "br_legislation" not in priority_types:
            priority_types.append("br_legislation")
        notes.append("Question concerns Brazilian context. Hard-filter to "
                     "jurisdiction=BR (loose: + iec_standard).")

    # Croácia
    if re.search(r'croat\w+|\bhrvatska\b|narodne\s+novine|\bhera\b', q):
        priority_jurisdictions.append("HR")
        if "hr_legislation" not in priority_types:
            priority_types.append("hr_legislation")
        notes.append("Question concerns Croatian context. Hard-filter to "
                     "jurisdiction=HR (loose: + iec_standard).")

    # UE
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

    # Livro / handbook
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
# INTENÇÃO → FILTRO DE METADADOS QDRANT  (usado nos caminhos SEM quota)
# =============================================================================
def build_intent_filter(intent: dict, strict: bool):
    """
    Traduz a intenção num objeto MetadataFilters do LlamaIndex que o Qdrant
    avalia no lado do servidor. Devolve (filtro_ou_None, string_descrição).
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

        # Loose: jurisdições OR iec_standard
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

    # ramo iec_only
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
# AJUSTE DE PONTUAÇÃO POR NÚMERO IEC (pós-filtro)
# =============================================================================
def adjust_iec_scores(nodes, iec_number, boost: float, penalty: float,
                      verbose: bool = False):
    """
    Quando a pergunta menciona um número IEC específico (ex. "60287"):
      - Reforça os chunks iec_standard cujo nome de ficheiro contém esse número.
      - Penaliza os chunks iec_standard cujo nome de ficheiro contém um número
        de família IEC diferente (4-5 dígitos).
      - Os chunks não-IEC ficam intactos.
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
            # Procurar qualquer número de família IEC (4-5 dígitos) no nome do
            # ficheiro; se estiver presente um *diferente*, penalizar.
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
# AGRUPAMENTO DE CONTEXTO
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
# MODELO DE PROMPT  — (A) regra única, sem autodiagnóstico condicional
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
    """Mantido apenas para impressão com --debug; já não é injetado no prompt."""
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
# AUXILIARES DE RECUPERAÇÃO
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
    (B) Pergunta com jurisdição: DUAS recuperações e depois SLOTS RESERVADOS no
    conjunto de rerank, para que nenhum dos lados esfomeie o outro.

    BUG v6.1 (corrigido aqui): a fusão reordenava tudo pela pontuação de
    embedding em bruto. A legislação PT pontua mais baixo do que as normas IEC
    no espaço de embedding para perguntas técnicas, pelo que todos os chunks PT
    eram truncados por nodes[:rerank_in] ANTES de o cross-encoder os ver sequer.
    A recuperação em duas passagens era inútil porque a fusão deitava fora o
    lado PT.

    Correção: NÃO reordenar a união pela pontuação. Reservar os primeiros
    rerank_in slots — metade garantida à passagem da jurisdição, metade a
    iec_standard — mantendo cada passagem na sua própria ordem de pontuação do
    Qdrant. O cross-encoder avalia então a relevância real num conjunto que
    garantidamente CONTÉM candidatos da jurisdição. Os restantes seguem,
    ordenados por pontuação, apenas como preenchimento de reserva.
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

    # Desduplicar entre passagens, preservando a ordem Qdrant de cada passagem
    # (um chunk visto na passagem da jurisdição não se repete no IEC).
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

    # Divisão reservada para o conjunto de rerank: metade jurisdição, metade IEC.
    cap     = args.rerank_in
    j_quota = max(1, cap // 2)
    i_quota = cap - j_quota

    head = juris_nodes[:j_quota] + iec_nodes[:i_quota]

    # Se um dos lados ficou curto, preencher a cabeça com o que sobra para o
    # conjunto de rerank continuar cheio.
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

    # ---- Intenção ----
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

    # ---- Serviços ----
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

    # ---- Construir filtro de intenção (usado nos caminhos SEM quota) ----
    intent_filter = None
    filter_desc   = "no filter"
    if not args.no_filter and not (args.type or args.jurisdiction):
        intent_filter, filter_desc = build_intent_filter(intent, args.strict)

    # Os overrides de CLI tornam-se filtros adicionais
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

    # ---- 1. Recuperação ----
    # Caminho de quota apenas para perguntas de jurisdição em modo loose, sem
    # override de CLI e sem --strict / --no-filter. Tudo o resto mantém o
    # comportamento v6.
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

    # ---- 2. Corrigir codificação + garantir classificação ----
    for n in nodes:
        n.node.text = fix_encoding(n.node.text)
        md = n.node.metadata
        if md.get("source_type") in (None, "", "unknown"):
            md.update(classify_source(md.get("file_name", ""), n.node.text))

    # ---- 3. Ajuste de pontuação por número IEC ----
    # NOTA: adjust_iec_scores() reordena a lista inteira pela pontuação em bruto
    # no fim. No caminho de quota a lista já está protegida por slots (lugares de
    # jurisdição reservados na cabeça); reordenar aqui ressuscitaria o bug v6.1 e
    # voltaria a truncar os chunks PT. Por isso o reforço IEC só corre nos
    # caminhos SEM quota. (A quota dispara em perguntas de jurisdição; o reforço
    # por número IEC específico visa perguntas de número IEC — as duas intenções
    # raramente coincidem, pelo que pouco se perde.)
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

    # ---- 4. Manter os melhores rerank_in ----
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
    # Nível de confiança a partir da pontuação de rerank do topo. Limiares
    # calibrados com execuções empíricas do golden dataset: perguntas PT vagas
    # com os documentos corretos no top-5 pontuam tipicamente 0.10-0.20 no
    # rerank top-1 (ex. o Manual de Ligações p.7 na pergunta dos "níveis de
    # tensão" pontuou 0.150). Tratá-las como LOW causava recusas generalizadas;
    # são MEDIUM e a resposta deve ser parcial-com-ressalva.
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

    # ---- Resumo ----
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
    # NOTA: a pontuação de rerank mede proximidade de embedding, NÃO a fidelidade
    # da resposta. O rótulo de nível aqui é o MESMO nível passado ao prompt
    # (limiares 0.08 / 0.40), pelo que a linha ">> Retrieval confidence tier" e
    # esta linha concordam sempre.
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
