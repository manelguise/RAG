"""
scripts/ingest.py — Ingestor incremental de PDF.

Ingere dois corpora em duas coleções Qdrant distintas: o corpus público
(legislação, normas, manuais) em "kb" e o dossiê confidencial do projeto em
"kb_project". A segmentação é sensível à estrutura: a legislação é dividida por
artigo/anexo/capítulo e as normas e manuais por cláusula numerada; cada unidade
estrutural é mantida inteira desde que caiba no orçamento de embedding, sendo
apenas subdividida quando o excede. Documentos sem estrutura detetável recorrem
ao segmentador de frases de tamanho fixo.

A ingestão é incremental por hash estável, com fallback de OCR, reparação de
codificação, continuidade entre páginas (no modo fixo), índices de payload,
salvaguardas anti-duplicação, varrimento de órfãos e backoff exponencial.
"""

# =============================================================================
# IMPORTAÇÕES
# =============================================================================
import os, sys, re, time, json, hashlib, argparse, warnings, logging, gc, bisect
import datetime as dt
from pathlib import Path
from collections import defaultdict
from logging.handlers import RotatingFileHandler

warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']    = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY']        = '1'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from classification import classify_source  # noqa: E402

import fitz                                                # PyMuPDF
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
# --- Dois corpora -> duas coleções ---------------------------------------
DOCS_PUBLIC          = "docs_full"        # público: legislação, normas, manuais
DOCS_PROJECT         = "docs_project"     # confidencial: dossiê do projeto
COLLECTION_PUBLIC    = "kb"
COLLECTION_PROJECT   = "kb_project"

STATE_FILE           = Path("scripts") / "ingest_state.json"
LOG_FILE             = Path("scripts") / "ingest.log"

EMBED_BATCH_SIZE     = 8
INSERT_BATCH_CHUNKS  = 512

# --- Chunking -------------------------------------------------------------
# O segmentador de frases é usado (a) como fallback para o documento inteiro
# quando não se deteta estrutura e (b) para subdividir qualquer unidade
# estrutural maior do que CHUNK_SIZE tokens. As unidades estruturais abaixo
# deste tamanho são mantidas intactas, que é todo o propósito da segmentação
# sensível à estrutura.
CHUNK_SIZE           = 512        # tokens (alvo do segmentador de frases)
CHUNK_OVERLAP        = 64         # tokens
MIN_STRUCT_UNITS     = 3          # nº mínimo de fronteiras para tratar um
                                  # documento como "estruturado"
MIN_UNIT_CHARS       = 200        # unidades estruturais menores do que isto (ex.
                                  # uma linha de cabeçalho isolada) são fundidas
                                  # com a seguinte

OCR_DPI                 = 200
NATIVE_PAGE_MIN_CHARS   = 200
USEFUL_TOTAL_MIN        = 500
OCR_THRESHOLD_RATIO     = 0.5
PAGE_OVERLAP_CHARS      = 500     # continuidade suave entre páginas (só no modo fixo)
ENCODING_HEALTH_THRESH  = 0.10

BACKOFF_MAX_RETRIES     = 5
BACKOFF_BASE_DELAY      = 1.0

PAYLOAD_INDEXES = {
    "jurisdiction":  "keyword",
    "source_type":   "keyword",
    "file_name":     "keyword",
    "stable_hash":   "keyword",
}

EXTENSIONS              = {'.pdf', '.docx', '.md', '.txt', '.html'}
DEFAULT_SKIP            = set()


# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
def setup_logging(verbose: bool = False) -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger("ingest")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3,
                             encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

log = setup_logging()


# =============================================================================
# REPARAÇÃO DE CODIFICAÇÃO
# =============================================================================
ENCODING_FIXES = [
    ('\u221eC',   '\u00b0C'), ('\u221eF',   '\u00b0F'), ('\u221e ',   '\u00b0'),
    ('mC',        'µC'),
    ('Ã¡', 'á'), ('Ã©', 'é'), ('Ã­', 'í'), ('Ã³', 'ó'), ('Ãº', 'ú'),
    ('Ã ', 'à'), ('Ã¨', 'è'), ('Ãª', 'ê'), ('Ã´', 'ô'),
    ('Ã£', 'ã'), ('Ãµ', 'õ'), ('Ã§', 'ç'),
    ('Ã\u0081', 'Á'), ('Ã\u0089', 'É'), ('Ã\u008d', 'Í'),
    ('Ã\u0093', 'Ó'), ('Ã\u009a', 'Ú'), ('Ã\u0087', 'Ç'),
    ('Â°', '°'), ('Âª', 'ª'), ('Âº', 'º'),
    ('\ufb01', 'fi'), ('\ufb02', 'fl'),
    ('\u2018', "'"), ('\u2019', "'"), ('\u201c', '"'), ('\u201d', '"'),
    ('\u2013', '-'), ('\u2014', '-'), ('\u2026', '...'),
]


def fix_encoding(text: str) -> str:
    for bad, good in ENCODING_FIXES:
        text = text.replace(bad, good)
    return text


def encoding_health(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    bad = sum(
        1 for c in text
        if c == '\ufffd'
        or ('\ue000' <= c <= '\uf8ff')
        or (c < ' ' and c not in '\n\t\r')
    )
    bad += len(re.findall(r'Ã[\u0080-\u00ff]', text))
    return bad / max(n, 1)


# =============================================================================
# HASH ESTÁVEL
# =============================================================================
def stable_hash(path: Path) -> str:
    h = hashlib.sha256()
    fn_norm = re.sub(r'[^a-z0-9]+', '', path.name.lower())
    size_kb = round(path.stat().st_size / 1024)
    h.update(fn_norm.encode())
    h.update(f"|{size_kb}|".encode())
    try:
        suffix = path.suffix.lower()
        if suffix == '.pdf':
            with fitz.open(str(path)) as pdf:
                sample = pdf[0].get_text("text")[:2000] if len(pdf) > 0 else ""
        elif suffix in {'.md', '.txt'}:
            sample = path.read_text(encoding='utf-8', errors='ignore')[:2000]
        else:
            sample = ""
        h.update(sample.encode('utf-8', errors='ignore'))
    except Exception:
        with open(path, 'rb') as f:
            h.update(f.read(65536))
    return h.hexdigest()[:16]


# =============================================================================
# GESTÃO DE ESTADO
# =============================================================================
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            log.warning(f"Corrupt {STATE_FILE} — starting from empty state.")
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    tmp.replace(STATE_FILE)


# =============================================================================
# BACKOFF
# =============================================================================
RECOVERABLE_EXCEPTIONS = (
    ConnectionError, TimeoutError, OSError, ResponseHandlingException,
)

def is_recoverable(exc: Exception) -> bool:
    if isinstance(exc, RECOVERABLE_EXCEPTIONS):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        'connection', 'timeout', 'timed out', 'temporarily unavailable',
        'failed to connect', 'connection refused', 'ollama',
    ))


def with_backoff(fn, *args, label: str = "operation", **kwargs):
    delay = BACKOFF_BASE_DELAY
    for attempt in range(1, BACKOFF_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not is_recoverable(e) or attempt == BACKOFF_MAX_RETRIES:
                raise
            log.warning(f"{label}: {type(e).__name__} (attempt {attempt}/"
                        f"{BACKOFF_MAX_RETRIES}), retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= 2


# =============================================================================
# EXTRAÇÃO DE PDF
# =============================================================================
_OCR_ENGINE = None

def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        log.info("Loading RapidOCR engine (first use)...")
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _ocr_one_page(page) -> str:
    engine = _get_ocr_engine()
    matrix = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pixmap.tobytes("png")
    del pixmap
    result, _ = engine(png_bytes)
    if not result:
        return ""
    return "\n".join(item[1] for item in result if item and len(item) > 1)


def extract_pdf_pages(path: Path):
    """Gerador que produz (índice_página, texto, usou_ocr) por cada página."""
    with fitz.open(str(path)) as pdf:
        for i, page in enumerate(pdf):
            native_text = page.get_text("text") or ""
            stripped = native_text.strip()
            if len(stripped) >= NATIVE_PAGE_MIN_CHARS:
                yield (i + 1, native_text, False)
            else:
                try:
                    ocr_text = _ocr_one_page(page)
                except Exception as e:
                    log.error(f"OCR error on page {i+1}: "
                              f"{type(e).__name__}: {str(e)[:120]}")
                    ocr_text = ""
                if len(ocr_text.strip()) > len(stripped):
                    yield (i + 1, ocr_text, True)
                else:
                    yield (i + 1, native_text, False)


def extract_document(path: Path, stats: dict):
    """
    Lê um ficheiro para um único texto contínuo, acumulando estatísticas de
    páginas/carateres.

    Devolve (full_text, page_starts), onde page_starts é uma lista ordenada de
    (offset, número_da_página) que marca onde cada página começa dentro de
    full_text. Serve para atribuir um rótulo de página aproximado a cada chunk.
    """
    parts, page_starts, offset = [], [], 0
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        for page_num, text, used_ocr in extract_pdf_pages(path):
            text = fix_encoding(text)
            if not text.strip():
                continue
            health = encoding_health(text)
            if health > ENCODING_HEALTH_THRESH:
                log.warning(f"{path.name} page {page_num}: "
                            f"{health*100:.1f}% suspicious chars "
                            f"(possible encoding damage)")
            stats["total_chars"] += len(text)
            stats["ocr_pages" if used_ocr else "native_pages"] += 1
            page_starts.append((offset, page_num))
            parts.append(text)
            offset += len(text) + 1            # +1 para o "\n" da junção abaixo
    elif suffix in {".md", ".txt"}:
        text = fix_encoding(path.read_text(encoding='utf-8', errors='ignore'))
        if text.strip():
            stats["total_chars"] += len(text)
            stats["native_pages"] += 1
            page_starts.append((0, 1))
            parts.append(text)
    else:
        return "", []

    return "\n".join(parts), page_starts


def page_for_offset(offset: int, page_starts) -> int:
    """Devolve o número da página cujo intervalo contém `offset`."""
    if not page_starts:
        return 1
    keys = [o for o, _ in page_starts]
    idx = bisect.bisect_right(keys, offset) - 1
    idx = max(0, min(idx, len(page_starts) - 1))
    return page_starts[idx][1]


# =============================================================================
# SEGMENTAÇÃO SENSÍVEL À ESTRUTURA
# =============================================================================
# Padrões de fronteira. Afinados para texto legislativo português e numeração
# de cláusulas genérica; verificar os logs "chunking=" por documento no corpus
# e ajustar se uma família de documentos não estiver a ser detetada.
#
#   LEGIS_RE  — "Artigo 5.º", "Artigo 12.º-A", e ANEXO/CAPÍTULO/SECÇÃO/TÍTULO
#   CLAUSE_RE — "4 Scope", "4.1 ...", "4.1.2 ..." no início de linha (IEC/IEEE, manuais)
LEGIS_RE = re.compile(
    r'(?m)^[ \t]*(?:'
    r'Artigo[ \t]+\d+\.?[\u00ba\u00b0]?(?:[-\u2013][A-Za-z])?'
    r'|ANEXO\b|CAP[\u00cdI]TULO\b|SEC[\u00c7C][\u00c3A]O\b|T[\u00cdI]TULO\b'
    r')',
    re.IGNORECASE,
)
CLAUSE_RE = re.compile(r'(?m)^[ \t]*\d+(?:\.\d+){0,3}[ \t]+[A-Za-z\u00c0-\u00ff]')


def _classifier_hint(source_type: str) -> str:
    st = (source_type or "").lower()
    if any(k in st for k in ("legisl", "decreto", "portaria", "regulament",
                             "despacho", "diretiv", "lei", "ror", "rari")):
        return "legislation"
    if any(k in st for k in ("iec", "ieee", "norm", "standard", "iso", "en5")):
        return "norm"
    if any(k in st for k in ("manual", "operad", "operator", "guia",
                             "scada", "redes", "dit")):
        return "manual"
    return "fallback"


def choose_strategy(source_type: str, text: str) -> str:
    """
    Decide a estratégia de segmentação a partir da pista do classificador,
    confirmada (ou recuperada, quando a pista é 'fallback') pela estrutura
    efetivamente presente no texto. Devolve um de: legislation, norm, manual,
    fallback.
    """
    hint = _classifier_hint(source_type)
    if hint == "legislation" and len(LEGIS_RE.findall(text)) >= MIN_STRUCT_UNITS:
        return "legislation"
    if hint in ("norm", "manual") and len(CLAUSE_RE.findall(text)) >= MIN_STRUCT_UNITS:
        return hint
    # Pista ausente ou não confirmada pelo conteúdo: detetar a partir do próprio texto.
    if len(LEGIS_RE.findall(text)) >= MIN_STRUCT_UNITS:
        return "legislation"
    if len(CLAUSE_RE.findall(text)) >= MIN_STRUCT_UNITS:
        return "norm"
    return "fallback"


def _split_on_boundaries(text: str, pattern):
    """
    Corta `text` em cada linha que corresponde a `pattern`. Devolve uma lista de
    (label, segmento, offset_inicial), ou None se forem encontradas menos de
    MIN_STRUCT_UNITS fronteiras. A linha de cabeçalho encontrada torna-se o
    rótulo do segmento.
    """
    matches = list(pattern.finditer(text))
    if len(matches) < MIN_STRUCT_UNITS:
        return None
    units = []
    if matches[0].start() > MIN_UNIT_CHARS:        # preâmbulo substancial
        units.append((None, text[:matches[0].start()], 0))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        label = re.sub(r'\s+', ' ', m.group().strip())[:80]
        units.append((label, text[start:end], start))
    return units


def _merge_small_units(units):
    """Transporta uma unidade demasiado curta (tipicamente uma linha de
    cabeçalho isolada) para a frente e antepõe-na à unidade seguinte, mantendo
    o cabeçalho como rótulo dessa unidade."""
    out, carry = [], None
    for label, seg, off in units:
        if carry is not None:
            c_label, c_seg, c_off = carry
            seg = c_seg + "\n" + seg
            label = label or c_label
            off = c_off
            carry = None
        if len(seg.strip()) < MIN_UNIT_CHARS:
            carry = (label, seg, off)
        else:
            out.append((label, seg, off))
    if carry is not None:
        if out:
            l, s, o = out[-1]
            out[-1] = (l, s + "\n" + carry[1], o)
        else:
            out.append(carry)
    return out


def chunk_by_structure(full_text: str, source_type: str,
                       splitter: SentenceSplitter, mode: str):
    """
    Divide `full_text` em tuplos (label, texto_chunk, offset_inicial).

    mode == 'fixed'      -> sempre o segmentador de frases (comportamento antigo)
    mode == 'structure'  -> divide por artigo/cláusula; as unidades maiores do
                            que CHUNK_SIZE são subdivididas pelo segmentador de
                            frases; os documentos sem estrutura recorrem a 'fixed'.

    Devolve também o rótulo da estratégia efetivamente usada.
    """
    if mode == "fixed":
        chunks = [(None, t, o)
                  for t, o in _locate_chunks(splitter.split_text(full_text),
                                             full_text)]
        return "fixed", chunks

    strategy = choose_strategy(source_type, full_text)
    pattern = {"legislation": LEGIS_RE,
               "norm": CLAUSE_RE,
               "manual": CLAUSE_RE}.get(strategy)

    units = _split_on_boundaries(full_text, pattern) if pattern else None
    if not units:
        chunks = [(None, t, o)
                  for t, o in _locate_chunks(splitter.split_text(full_text),
                                             full_text)]
        return "fixed", chunks

    units = _merge_small_units(units)
    out = []
    for label, seg, off in units:
        subs = splitter.split_text(seg)
        if len(subs) <= 1:
            out.append((label, seg.strip(), off))
        else:
            # Relocalizar os sub-chunks dentro da unidade, mapeando de volta para full_text.
            for sub, rel in _locate_chunks(subs, seg):
                out.append((label, sub, off + rel))
    return strategy, out


def _locate_chunks(chunks, haystack):
    """
    Atribui um offset inicial a cada chunk, localizando-o em `haystack` com um
    cursor que avança (os chunks saem por ordem). Devolve uma lista de
    (texto_chunk, offset_inicial). Resistente a pequenas normalizações de
    espaços em branco porque procura sobre um prefixo do chunk sem espaços nas
    pontas.
    """
    located, cursor = [], 0
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        probe = c[:60].strip()
        pos = haystack.find(probe, cursor)
        if pos < 0:
            pos = haystack.find(probe)        # recorre a procura global
        if pos < 0:
            pos = cursor
        located.append((c, pos))
        cursor = pos + max(len(c), 1)
    return located


# =============================================================================
# CONSTRUÇÃO DE NÓS
# =============================================================================
EXCLUDED_EMBED_KEYS = [
    "doc_id", "ingested_at", "file_path", "classification_confidence",
    "stable_hash", "corpus", "chunking",
]
EXCLUDED_LLM_KEYS = [
    "doc_id", "ingested_at", "file_path", "stable_hash",
]


def build_nodes_for_file(path: Path, classification: dict, doc_id: str,
                         corpus: str, full_text: str, page_starts,
                         splitter: SentenceSplitter, mode: str, stats: dict):
    """Produz TextNodes para um ficheiro, um por cada chunk sensível à estrutura."""
    base_meta = {
        "source":       str(path),
        "file_name":    path.name,
        "file_path":    str(path),
        "doc_id":       doc_id,
        "stable_hash":  doc_id,
        "corpus":       corpus,
        "ingested_at":  dt.datetime.utcnow().isoformat(),
        **classification,
    }

    strategy, chunks = chunk_by_structure(full_text, classification["source_type"],
                                          splitter, mode)
    stats["chunking"] = strategy

    for label, text, off in chunks:
        if not text.strip():
            continue
        meta = dict(base_meta)
        meta["page_label"]       = str(page_for_offset(off, page_starts))
        meta["structural_label"] = label or ""
        meta["chunking"]         = strategy
        stats["n_chunks"] += 1
        node = TextNode(text=text, metadata=meta)
        node.excluded_embed_metadata_keys = EXCLUDED_EMBED_KEYS
        node.excluded_llm_metadata_keys   = EXCLUDED_LLM_KEYS
        yield node


# =============================================================================
# DETERMINAÇÃO DE ESTADO
# =============================================================================
def determine_status(stats: dict) -> str:
    total_pages = stats["native_pages"] + stats["ocr_pages"]
    if total_pages == 0 or stats["total_chars"] == 0:
        return "no_text"
    if stats["total_chars"] < USEFUL_TOTAL_MIN:
        return "low_yield"
    if stats["ocr_pages"] == 0:
        return "completed_native"
    if stats["native_pages"] == 0:
        return "completed_ocr"
    ratio_ocr = stats["ocr_pages"] / total_pages
    return "completed_ocr" if ratio_ocr >= OCR_THRESHOLD_RATIO else "completed_mixed"


# =============================================================================
# ÍNDICES DE PAYLOAD
# =============================================================================
def ensure_payload_indexes(client: QdrantClient, collection: str):
    if not client.collection_exists(collection):
        return
    try:
        info = client.get_collection(collection)
        existing = set((info.payload_schema or {}).keys())
    except Exception:
        existing = set()
    for field, schema in PAYLOAD_INDEXES.items():
        if field in existing:
            continue
        try:
            client.create_payload_index(collection_name=collection,
                                        field_name=field, field_schema=schema)
            log.info(f"[{collection}] created payload index on '{field}'.")
        except Exception as e:
            log.warning(f"[{collection}] could not create index on '{field}': "
                        f"{type(e).__name__}: {str(e)[:120]}")


# =============================================================================
# SALVAGUARDAS ANTI-DUPLICAÇÃO
# =============================================================================
from qdrant_client.http import models as qm  # noqa: E402


def delete_chunks_by_hash(client: QdrantClient, collection: str,
                          stable_hash_val: str) -> int:
    if not client.collection_exists(collection):
        return 0
    try:
        flt = qm.Filter(must=[qm.FieldCondition(
            key="stable_hash", match=qm.MatchValue(value=stable_hash_val))])
        n = client.count(collection_name=collection, count_filter=flt,
                         exact=True).count
        if n == 0:
            return 0
        client.delete(collection_name=collection,
                      points_selector=qm.FilterSelector(filter=flt), wait=True)
        return n
    except Exception as e:
        log.warning(f"delete_chunks_by_hash({stable_hash_val}): "
                    f"{type(e).__name__}: {str(e)[:120]}")
        return 0


def scan_orphan_chunks(client: QdrantClient, collection: str,
                       state: dict) -> dict:
    if not client.collection_exists(collection):
        return {}
    known_completed = {
        rec.get("hash") for rec in state.values()
        if rec.get("status", "").startswith("completed")
        and rec.get("collection") == collection
    }
    known_completed.discard(None)
    orphans = defaultdict(int)
    offset, n_scanned = None, 0
    while True:
        try:
            points, offset = client.scroll(
                collection_name=collection, limit=1024, offset=offset,
                with_payload=["stable_hash"], with_vectors=False)
        except Exception as e:
            log.warning(f"scan_orphan_chunks[{collection}]: scroll failed: "
                        f"{type(e).__name__}: {str(e)[:120]}")
            break
        if not points:
            break
        for pt in points:
            n_scanned += 1
            h = (pt.payload or {}).get("stable_hash")
            if h and h not in known_completed:
                orphans[h] += 1
        if offset is None:
            break
    log.debug(f"scan_orphan_chunks[{collection}]: scanned {n_scanned}, "
              f"found {len(orphans)} orphan hashes")
    return dict(orphans)


def prompt_orphan_cleanup(client: QdrantClient, collection: str,
                          orphans: dict, auto_clean: bool) -> int:
    if not orphans:
        return 0
    total = sum(orphans.values())
    log.warning(f"[{collection}] {len(orphans)} orphan hash(es), {total} chunks, "
                f"with no matching completed_* entry in state.")
    for h, n in sorted(orphans.items(), key=lambda kv: -kv[1])[:10]:
        log.warning(f"   hash={h}  chunks={n}")
    if auto_clean:
        proceed = True
        log.info("--auto-clean-orphans set — deleting without prompt.")
    else:
        print(f"\n[{collection}] Delete these orphan chunks now?")
        proceed = (input("Type YES to delete, anything else to keep: ").strip()
                   == "YES")
    if not proceed:
        log.info("Orphans kept.")
        return 0
    deleted = sum(delete_chunks_by_hash(client, collection, h) for h in orphans)
    log.info(f"[{collection}] deleted {deleted} orphan chunks.")
    return deleted


# =============================================================================
# INSERÇÃO COM BACKOFF
# =============================================================================
def safe_insert(index, batch, file_label: str):
    return with_backoff(index.insert_nodes, batch,
                        label=f"insert_nodes({file_label}, n={len(batch)})")


# =============================================================================
# INGESTÃO DE UM FICHEIRO (numa dada coleção/índice)
# =============================================================================
def ingest_file(path: Path, index, splitter, state: dict, args,
                client: QdrantClient, collection: str, corpus: str):
    rel = str(path)
    log.info(f"-> [{corpus}] {path.name} "
             f"({path.stat().st_size/1024/1024:.2f} MB)")

    digest = stable_hash(path)
    prev = state.get(rel)
    if (not args.force_rechunk and prev and prev.get("hash") == digest
            and prev.get("status", "").startswith("completed")):
        log.info(f"   skipped_unchanged (hash {digest}, prev={prev['status']})")
        return "skipped_unchanged"

    if not args.dry_run:
        n_wiped = delete_chunks_by_hash(client, collection, digest)
        if n_wiped > 0:
            log.warning(f"   wiped {n_wiped} pre-existing chunks "
                        f"with hash {digest} before re-ingesting")

    t0 = time.time()
    stats = {"native_pages": 0, "ocr_pages": 0, "total_chars": 0,
             "n_chunks": 0, "chunking": "?"}

    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".md", ".txt"}:
        log.warning(f"   unsupported extension {suffix} — skipped")
        return None

    try:
        # 1) Extrair o texto do documento inteiro + mapa de páginas.
        full_text, page_starts = extract_document(path, stats)
        if not full_text.strip():
            status = "no_text"
            state[rel] = {"hash": digest, "status": status,
                          "collection": collection, "corpus": corpus,
                          "ts": dt.datetime.utcnow().isoformat()}
            save_state(state)
            log.info(f"   {status}")
            return status

        # 2) Classificar (primeiro pelo nome do ficheiro, refinar pelo conteúdo se desconhecido).
        classification = classify_source(path.name, "")
        if classification["source_type"] == "unknown":
            better = classify_source(path.name, full_text[:4000])
            if better["source_type"] != "unknown":
                classification = better
                log.info(f"   classification refined from content: "
                         f"{better['source_type']}/{better['jurisdiction']}")

        # 3) Segmentação sensível à estrutura + 4) construir nós, inserindo em lotes.
        batch = []
        for node in build_nodes_for_file(path, classification, digest, corpus,
                                         full_text, page_starts, splitter,
                                         args.chunking, stats):
            batch.append(node)
            if not args.dry_run and len(batch) >= INSERT_BATCH_CHUNKS:
                safe_insert(index, batch, path.name)
                batch = []
                gc.collect()
        if not args.dry_run and batch:
            safe_insert(index, batch, path.name)
            gc.collect()

    except Exception as e:
        status = "failed_recoverable" if is_recoverable(e) else "failed_content"
        log.error(f"   X {status}: {type(e).__name__}: {str(e)[:200]}")
        state[rel] = {"hash": digest, "status": status,
                      "collection": collection, "corpus": corpus,
                      "error": f"{type(e).__name__}: {str(e)[:300]}",
                      "ts": dt.datetime.utcnow().isoformat()}
        save_state(state)
        return status

    status = determine_status(stats)
    elapsed = time.time() - t0
    log.info(f"   pages: native={stats['native_pages']} ocr={stats['ocr_pages']} "
             f"chars={stats['total_chars']} chunks={stats['n_chunks']} "
             f"chunking={stats['chunking']} status={status} ({elapsed:.1f}s)")
    log.info(f"   classified as: {classification['source_type']}/"
             f"{classification['jurisdiction']} "
             f"({classification['classification_confidence']})")

    state[rel] = {
        "hash": digest, "status": status, "collection": collection,
        "corpus": corpus, "chunking": stats["chunking"],
        "n_chunks": stats["n_chunks"], "native_pages": stats["native_pages"],
        "ocr_pages": stats["ocr_pages"], "total_chars": stats["total_chars"],
        "source_type": classification["source_type"],
        "jurisdiction": classification["jurisdiction"],
        "confidence": classification["classification_confidence"],
        "ts": dt.datetime.utcnow().isoformat(), "elapsed_sec": round(elapsed, 1),
    }
    save_state(state)
    return status


# =============================================================================
# VERIFICAÇÕES PRÉVIAS
# =============================================================================
def preflight() -> bool:
    import urllib.request, urllib.error
    try:
        QdrantClient(url='http://localhost:6333').get_collections()
        log.info("Pre-flight: Qdrant reachable.")
    except Exception as e:
        log.error(f"Pre-flight FAILED: Qdrant unreachable: {e}")
        return False
    try:
        with urllib.request.urlopen('http://localhost:11434/api/tags',
                                    timeout=5) as r:
            models = {m['name'] for m in json.loads(r.read().decode()).get('models', [])}
        if not (models & {'bge-m3:latest', 'bge-m3'}):
            log.error(f"Pre-flight FAILED: 'bge-m3' not in Ollama: {sorted(models)}")
            return False
        log.info(f"Pre-flight: Ollama reachable, bge-m3 available.")
    except Exception as e:
        log.error(f"Pre-flight FAILED: Ollama: {type(e).__name__}: {e}")
        return False
    return True


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Incremental ingestor v3 "
                                "(structure-aware chunking, dual collection).")
    # Aliases de corpus único, compatíveis com versões anteriores:
    p.add_argument("--docs", default=None,
                   help="Alias for --docs-public (back-compat)")
    p.add_argument("--collection", default=None,
                   help="Alias for --collection-public (back-compat)")
    # Corpus duplo:
    p.add_argument("--docs-public", default=DOCS_PUBLIC)
    p.add_argument("--docs-project", default=DOCS_PROJECT)
    p.add_argument("--collection-public", default=COLLECTION_PUBLIC)
    p.add_argument("--collection-project", default=COLLECTION_PROJECT)
    # Chunking:
    p.add_argument("--chunking", choices=["structure", "fixed"],
                   default="structure",
                   help="structure: split by article/clause (default); "
                        "fixed: legacy sentence splitter")
    p.add_argument("--force-rechunk", action="store_true",
                   help="Reprocess files even if hash unchanged "
                        "(needed after changing --chunking)")
    # Operacionais:
    p.add_argument("--reset", action="store_true",
                   help="Drop BOTH collections and wipe state (asks for YES)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only-failed", action="store_true")
    p.add_argument("--only-failed-recoverable", action="store_true")
    p.add_argument("--only-failed-content", action="store_true")
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-preflight", action="store_true")
    p.add_argument("--skip-orphan-scan", action="store_true")
    p.add_argument("--auto-clean-orphans", action="store_true")
    args = p.parse_args()
    # Resolver aliases.
    if args.docs:
        args.docs_public = args.docs
    if args.collection:
        args.collection_public = args.collection
    return args


def confirm_reset(collections) -> bool:
    print("\n" + "=" * 60)
    print("DESTRUCTIVE OPERATION: --reset will:")
    for c in collections:
        print(f"   - DROP collection '{c}' (all chunks lost)")
    print(f"   - DELETE {STATE_FILE} (all ingest progress lost)")
    print("This cannot be undone.")
    print("=" * 60)
    return input("Type YES (exactly, uppercase) to confirm: ").strip() == "YES"


# =============================================================================
# PROCESSAR UM CORPUS
# =============================================================================
def process_corpus(corpus_name: str, docs_dir: Path, collection: str,
                   client: QdrantClient, splitter, state: dict, args,
                   summary: dict):
    if not docs_dir.exists():
        log.info(f"[{corpus_name}] folder '{docs_dir}' not found — skipping.")
        return
    log.info("=" * 60)
    log.info(f"CORPUS '{corpus_name}'  dir={docs_dir}  collection={collection}")
    log.info("=" * 60)

    vstore = QdrantVectorStore(client=client, collection_name=collection)
    storage = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_vector_store(vstore, storage_context=storage)

    if client.collection_exists(collection):
        ensure_payload_indexes(client, collection)

    files = sorted(p for p in docs_dir.rglob('*')
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    skip_names = set(args.skip) | (set() if args.force else DEFAULT_SKIP)
    if skip_names:
        files = [f for f in files if f.name not in skip_names]

    # Varrimento de órfãos (por coleção).
    if (not args.skip_orphan_scan and not args.dry_run
            and client.collection_exists(collection)):
        log.info(f"[{collection}] scanning for orphan chunks...")
        orphans = scan_orphan_chunks(client, collection, state)
        if orphans:
            prompt_orphan_cleanup(client, collection, orphans,
                                  args.auto_clean_orphans)
        else:
            log.info(f"[{collection}] no orphans found.")

    # Filtros --only-failed.
    def st(f):
        return state.get(str(f), {}).get("status", "")
    if args.only_failed:
        files = [f for f in files if st(f).startswith("failed")]
    elif args.only_failed_recoverable:
        files = [f for f in files if st(f) == "failed_recoverable"]
    elif args.only_failed_content:
        files = [f for f in files if st(f) == "failed_content"]

    log.info(f"[{corpus_name}] files to process: {len(files)}")

    durations = []
    for i, f in enumerate(files, 1):
        eta = ""
        if durations:
            avg = sum(durations[-5:]) / min(len(durations), 5)
            eta = f", ETA {(len(files)-i+1)*avg/60:.1f} min"
        log.info(f"[{corpus_name}] [{i}/{len(files)}{eta}] {f.name}")
        t_file = time.time()
        try:
            status = ingest_file(f, index, splitter, state, args,
                                 client, collection, corpus_name)
        except KeyboardInterrupt:
            log.warning("Interrupted by user. State saved.")
            save_state(state)
            sys.exit(130)
        except Exception as e:
            status = "failed_recoverable" if is_recoverable(e) else "failed_content"
            log.error(f"Unexpected {status}: {type(e).__name__}: {e}")
            state[str(f)] = {"hash": "unknown", "status": status,
                             "collection": collection, "corpus": corpus_name,
                             "error": f"{type(e).__name__}: {str(e)[:300]}",
                             "ts": dt.datetime.utcnow().isoformat()}
            save_state(state)
        if status:
            summary[status] += 1
        durations.append(time.time() - t_file)
        if i == 1 and client.collection_exists(collection) and not args.dry_run:
            ensure_payload_indexes(client, collection)
        gc.collect()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)

    corpora = [
        ("public",  Path(args.docs_public),  args.collection_public),
        ("project", Path(args.docs_project), args.collection_project),
    ]
    collections = [c for _, _, c in corpora]

    if args.reset and not confirm_reset(collections):
        log.info("Reset aborted by user.")
        sys.exit(0)

    if not args.no_preflight and not preflight():
        log.error("Pre-flight failed. Fix services or use --no-preflight.")
        sys.exit(2)

    Settings.embed_model = OllamaEmbedding(
        model_name='bge-m3', base_url='http://localhost:11434',
        embed_batch_size=EMBED_BATCH_SIZE)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    client = QdrantClient(url='http://localhost:6333')

    if args.reset:
        log.info("Resetting collections and state...")
        for c in collections:
            if client.collection_exists(c):
                client.delete_collection(c)
        if STATE_FILE.exists():
            STATE_FILE.unlink()

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    state = load_state()

    log.info(f"Chunking mode: {args.chunking}"
             + ("  (force-rechunk ON)" if args.force_rechunk else ""))
    if args.dry_run:
        log.info("--dry-run: nothing will be written to Qdrant.")

    summary = defaultdict(int)
    t_start = time.time()
    for name, docs_dir, collection in corpora:
        process_corpus(name, docs_dir, collection, client, splitter,
                       state, args, summary)
    save_state(state)

    log.info("=" * 60)
    log.info(f"INGEST COMPLETE in {(time.time()-t_start)/60:.1f} min")
    log.info("=" * 60)
    for status, count in sorted(summary.items(), key=lambda x: -x[1]):
        log.info(f"  {status:<24} {count}")
    log.info(f"State file: {STATE_FILE}")
    log.info(f"Log file:   {LOG_FILE}")


if __name__ == "__main__":
    main()
