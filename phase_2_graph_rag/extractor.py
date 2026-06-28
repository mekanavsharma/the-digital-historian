"""High‑precision graph fact extractor – strict, relation‑first, period‑agnostic.

- Entities: only proper names (capitalised, multi‑word, not generic)
- Relations: dependency parsing + regex patterns + co‑occurrence fallback
- Metadata: propagated to every node and edge
- Canonicalisation: fuzzy merge of near‑duplicate names
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import spacy

# ----------------------------------------------------------------------
# Schema imports (with fallback)
# ----------------------------------------------------------------------
try:
    from .schema import NODE_LABELS, RELATION_TYPES
    from .utils import normalize_name, split_sentences
except ImportError:
    NODE_LABELS = {
        "person": "Person", "organization": "Organization", "place": "Place",
        "event": "Event", "year": "Year", "passage": "Passage", "source": "Source",
        "historian": "Historian", "battle": "Battle", "dynasty": "Dynasty",
        "movement": "Movement", "treaty": "Treaty", "kingdom": "Kingdom",
        "empire": "Empire", "religious_figure": "ReligiousFigure",
    }
    RELATION_TYPES = {
        "fought_against": "FOUGHT_AGAINST", "wrote_about": "WROTE_ABOUT",
        "occurred_in": "OCCURRED_IN", "succeeded_by": "SUCCEEDED_BY",
        "ruled": "RULED", "allied_with": "ALLIED_WITH",
        "contemporary_of": "CONTEMPORARY_OF", "member_of": "MEMBER_OF",
        "founded": "FOUNDED", "led": "LED", "associated_with": "ASSOCIATED_WITH",
        "related_to": "RELATED_TO", "married_to": "MARRIED_TO",
        "opposed": "OPPOSED", "rebelled_against": "REBELLED_AGAINST",
        "conquered": "CONQUERED", "captured": "CAPTURED", "annexed": "ANNEXED",
        "invaded": "INVADED", "besieged": "BESIEGED", "raided": "RAIDED",
        "expanded_into": "EXPANDED_INTO", "controlled": "CONTROLLED",
        "governed": "GOVERNED", "administered": "ADMINISTERED",
        "patronized": "PATRONIZED", "built": "BUILT", "commissioned": "COMMISSIONED",
        "converted": "CONVERTED", "tributary_to": "TRIBUTARY_TO",
        "vassal_of": "VASSAL_OF", "vanquished": "VANQUISHED", "defeated_by": "DEFEATED_BY",
    }
    def normalize_name(s): return re.sub(r"[^a-z0-9]+", "", str(s).lower())
    def split_sentences(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", t) if s.strip()]

# ----------------------------------------------------------------------
# Load spaCy (cached)
# ----------------------------------------------------------------------
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = None
    return _nlp

# ----------------------------------------------------------------------
# Canonical name cache (fuzzy merging)
# ----------------------------------------------------------------------
_CANONICAL_CACHE: Dict[str, str] = {}

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def get_canonical(name: str, existing_names: Iterable[str] = ()) -> str:
    name = " ".join(str(name).split()).strip(" .,:;()[]{}\"'“”’")
    if not name:
        return name
    key = normalize_name(name)
    if key in _CANONICAL_CACHE:
        return _CANONICAL_CACHE[key]
    best = name
    best_score = 0.0
    for existing in existing_names:
        if not existing:
            continue
        score = _ratio(normalize_name(name), normalize_name(existing))
        if score > best_score:
            best_score = score
            best = existing
    if best_score >= 0.85:
        _CANONICAL_CACHE[key] = best
        return best
    _CANONICAL_CACHE[key] = name
    return name

# ----------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------
def _flatten_dict(data: Any, prefix: str = "") -> Dict[str, Any]:
    flat = {}
    if not isinstance(data, dict):
        return flat
    for k, v in data.items():
        full = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, full))
        elif isinstance(v, (str, int, float, bool, type(None))):
            flat[full] = v
        elif isinstance(v, list):
            flat[full] = [str(x) if not isinstance(x, (str, int, float, bool)) else x for x in v]
        else:
            flat[full] = str(v)
    return flat

def _sanitise_props(props: Dict[str, Any]) -> Dict[str, Any]:
    sanitised = {}
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            sanitised[k] = v
        elif isinstance(v, list):
            sanitised[k] = [str(item) if not isinstance(item, (str, int, float, bool)) else item for item in v]
        elif isinstance(v, dict):
            sanitised[k] = json.dumps(v, ensure_ascii=False)
        else:
            sanitised[k] = str(v)
    return sanitised

# ----------------------------------------------------------------------
# Label inference
# ----------------------------------------------------------------------
def infer_label(name: str, context: str = "", ner_label: Optional[str] = None, relation: Optional[str] = None, role: Optional[str] = None) -> str:
    s = " ".join(str(name).split()).strip(" .,:;()[]{}\"'“”’")
    if not s:
        return "Event"
    lower = s.lower()
    ctx = context.lower()

    if re.fullmatch(r"\b(1[0-9]{3}|20[0-2][0-9])\b", s):
        return "Year"

    # Strong keyword signals (period‑agnostic)
    if any(k in lower for k in ("battle", "war", "siege", "campaign", "skirmish")):
        return "Battle"
    if any(k in lower for k in ("movement", "andolan", "satyagraha", "rebellion", "uprising", "revolt")):
        return "Movement"
    if any(k in lower for k in ("treaty", "pact", "accord", "agreement")):
        return "Treaty"
    if any(k in lower for k in ("kingdom", "sultanate")):
        return "Kingdom"
    if "empire" in lower:
        return "Empire"
    if any(k in lower for k in ("dynasty", "house", "lineage")):
        return "Dynasty"
    if any(k in lower for k in ("temple", "mosque", "church", "cathedral", "gurudwara")):
        return "Temple"
    if any(k in lower for k in ("fort", "castle", "citadel", "garh", "qila")):
        return "Fort"
    if any(k in lower for k in ("inscription", "edict", "pillar", "prasasti")):
        return "Inscription"
    if any(k in lower for k in ("chronicle", "treatise", "itihasa", "purana", "grantha", "manuscript")):
        return "Text"
    if any(k in lower for k in ("organization", "association", "party", "congress", "league", "sabha", "samiti", "sangh")):
        return "Organization"
    if any(k in lower for k in ("newspaper", "journal", "periodical")):
        return "Newspaper"
    if any(k in lower for k in ("legislation", "law", "act", "decree")):
        return "Legislation"
    if any(k in lower for k in ("king", "emperor", "sultan", "raja", "maharaja", "nawab", "peshwa", "shah", "caliph", "samrat", "queen")):
        return "Ruler"
    if any(k in lower for k in ("saint", "sant", "guru", "acharya", "sheikh", "pir", "maulana", "swami", "monk", "abbot")):
        return "ReligiousFigure"

    if ner_label:
        nl = ner_label.upper()
        if nl == "PERSON":
            return "Person"
        if nl in {"ORG", "NORP"}:
            return "Organization"
        if nl in {"GPE", "LOC", "FAC"}:
            return "Place"
        if nl == "EVENT":
            return "Event"
        if nl == "WORK_OF_ART":
            return "Text"

    # Relation‑role hints
    if relation and role == "target" and relation in {"FOUNDED", "LED", "MEMBER_OF", "ASSOCIATED_WITH"}:
        if any(k in lower for k in ("movement", "organization", "party", "association", "sabha")):
            return "Organization" if "movement" not in lower else "Movement"

    if len(s.split()) >= 2 and s[0].isupper():
        if lower.endswith(("pur", "nagar", "abad", "garh", "gad", "qila", "killa", "durg", "fort", "city", "town", "village")):
            return "Place"
        return "Person"

    return "Event"

# ----------------------------------------------------------------------
# Verb → relation mapping (covers both modern and ancient)
# ----------------------------------------------------------------------
VERB_RELATION_MAP = {
    # Warfare
    "fight": "fought_against",
    "defeat": "fought_against",
    "conquer": "conquered",
    "capture": "captured",
    "annex": "annexed",
    "invade": "invaded",
    "besiege": "besieged",
    "raid": "raided",
    "vanquish": "vanquished",
    # Alliance / diplomacy
    "ally": "allied_with",
    "support": "allied_with",
    "unite": "allied_with",
    "join": "member_of",       # careful: join can be member_of or joined
    "belong": "member_of",
    "associate": "associated_with",
    "affiliate": "associated_with",
    "connect": "associated_with",
    "link": "associated_with",
    # Rule / governance
    "rule": "ruled",
    "govern": "governed",
    "administer": "administered",
    "control": "controlled",
    "reign": "ruled",
    # Succession
    "succeed": "succeeded_by",
    "follow": "succeeded_by",
    # Foundation / creation
    "found": "founded",
    "establish": "founded",
    "create": "founded",
    "set up": "founded",
    "build": "built",
    "commission": "commissioned",
    "construct": "built",
    # Leadership
    "lead": "led",
    "head": "led",
    "command": "led",
    "preside": "presided_over",
    # Personal / social
    "marry": "married_to",
    "wed": "married_to",
    "oppose": "opposed",
    "rebel": "rebelled_against",
    "protest": "protested",
    "demand": "demanded",
    "sign": "signed",
    "imprison": "imprisoned",
    "assassinate": "assassinated",
    "negotiate": "negotiated",
    "write": "wrote_about",
    "describe": "wrote_about",
    "mention": "mentioned_in",
    "convert": "converted",
    "patronize": "patronized",
    "influence": "influenced",
    "contemporary": "contemporary_of",
    # Temporal
    "occur": "occurred_in",
    "happen": "occurred_in",
}

# ----------------------------------------------------------------------
# Entity filtering – strict: only proper names, no generic nouns
# ----------------------------------------------------------------------
_GENERIC_NOUNS = {
    "party", "group", "army", "government", "country", "city", "state", "kingdom",
    "empire", "province", "region", "district", "river", "mountain", "fort", "temple",
    "mosque", "church", "cathedral", "school", "college", "university", "association",
    "committee", "society", "council", "league", "union", "club", "band", "gang",
    "tiger", "elephant", "horse", "man", "woman", "boy", "girl", "people", "person",
    "officer", "soldier", "minister", "ruler", "king", "queen", "emperor", "sultan",
    "raja", "maharaja", "nawab", "shah", "caliph", "viceroy", "governor", "general",
    "admiral", "captain", "colonel", "major", "sergeant", "constable", "inspector",
    "deputy", "assistant", "clerk", "messenger", "spy", "agent", "conspirator",
    "revolutionary", "freedom fighter", "patriot", "martyr", "leader", "head",
    "president", "chairman", "secretary", "treasurer", "member", "supporter",
    "follower", "ally", "enemy", "opponent", "rival", "successor", "predecessor",
    "ancestor", "descendant", "son", "daughter", "brother", "sister", "father", "mother",
    "husband", "wife", "friend", "colleague", "associate", "partner", "companion",
    "lavatory", "water", "port-hole", "quay", "gendarme", "yard", "distance",
}

def _is_valid_entity(text: str) -> bool:
    """Return True if the phrase looks like a proper named entity."""
    s = text.strip()
    if not s or len(s) < 2:
        return False
    # Must start with uppercase letter (if multi‑word, at least first is upper)
    if not s[0].isupper():
        return False
    # If it's a single word, it must be longer than 2 and not a generic noun
    if len(s.split()) == 1:
        if s.lower() in _GENERIC_NOUNS:
            return False
        if len(s) <= 2:
            return False
        return True
    # Multi‑word: at least one word (not the first) is capitalised
    words = s.split()
    if not any(w[0].isupper() for w in words[1:]):
        return False
    # If all words are generic nouns, reject
    if all(w.lower() in _GENERIC_NOUNS for w in words):
        return False
    # Reject if starts with "the", "a", "an"
    if words[0].lower() in {"the", "a", "an"}:
        return False
    return True

# ----------------------------------------------------------------------
# Relation extraction – dependency parsing + regex fallback
# ----------------------------------------------------------------------
def extract_relations_from_sent(sent) -> List[Tuple[str, str, str]]:
    """Return (subject_text, relation_type, object_text)."""
    rels = []
    for token in sent:
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        lemma = token.lemma_.lower()
        if lemma not in VERB_RELATION_MAP:
            continue
        rel_type = VERB_RELATION_MAP[lemma]

        subjects = []
        objects = []

        # Subject: nsubj, nsubjpass, csubj
        for child in token.children:
            if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
                span = child.subtree
                subj_text = " ".join(t.text for t in span).strip()
                if subj_text and _is_valid_entity(subj_text):
                    subjects.append(subj_text)

        # Objects: dobj, obj, attr, oprd, and pobj after prep
        for child in token.children:
            if child.dep_ in {"dobj", "obj", "attr", "oprd"}:
                span = child.subtree
                obj_text = " ".join(t.text for t in span).strip()
                if obj_text and _is_valid_entity(obj_text):
                    objects.append(obj_text)
            elif child.dep_ == "prep":
                prep = child.text.lower()
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        span = pobj.subtree
                        obj_text = " ".join(t.text for t in span).strip()
                        if obj_text and _is_valid_entity(obj_text):
                            # If it's a passive agent (after "by")
                            if prep == "by" and any(c.dep_ == "auxpass" for c in token.children):
                                subjects.append(obj_text)
                            else:
                                objects.append(obj_text)

        # Passive handling: if auxpass and we have no subjects, try to find agent
        is_passive = any(child.dep_ == "auxpass" for child in token.children) or any(child.dep_ == "nsubjpass" for child in token.children)
        if is_passive and not subjects:
            for child in token.children:
                if child.dep_ == "prep" and child.text.lower() == "by":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            agent_text = " ".join(t.text for t in pobj.subtree).strip()
                            if agent_text and _is_valid_entity(agent_text):
                                subjects.append(agent_text)

        if not subjects or not objects:
            continue

        for s in subjects:
            for o in objects:
                if normalize_name(s) != normalize_name(o):
                    rels.append((s, rel_type, o))

    return rels

# ----------------------------------------------------------------------
# Regex fallback for common historical patterns
# ----------------------------------------------------------------------
_RELATION_PATTERNS = [
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+founded\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "founded"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+was\s+a\s+member\s+of\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "member_of"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+joined\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "joined"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+led\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "led"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+associated\s+with\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "associated_with"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+ruled\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "ruled"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+defeated\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "fought_against"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+fought\s+against\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "fought_against"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+allied\s+with\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "allied_with"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+was\s+succeeded\s+by\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "succeeded_by"),
    (r"(?P<a>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})\s+married\s+(?P<b>[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,5})", "married_to"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in _RELATION_PATTERNS]

def regex_relations(sentence: str) -> List[Tuple[str, str, str]]:
    out = []
    for pattern, rel in _COMPILED_PATTERNS:
        for m in pattern.finditer(sentence):
            a = m.group("a").strip() if "a" in m.groupdict() else ""
            b = m.group("b").strip() if "b" in m.groupdict() else ""
            if a and b and _is_valid_entity(a) and _is_valid_entity(b) and a != b:
                out.append((a, rel, b))
    return out

# ----------------------------------------------------------------------
# Co‑occurrence fallback (only for Person ↔ Organization)
# ----------------------------------------------------------------------
def cooccurrence_associations(doc) -> List[Tuple[str, str, str]]:
    """If no direct relation found, connect people to orgs in the same sentence."""
    entities = []
    for ent in doc.ents:
        txt = ent.text.strip()
        if txt and _is_valid_entity(txt):
            label = infer_label(txt, doc.text, ner_label=ent.label_)
            entities.append((txt, label, ent.label_))
    # Also add noun chunks that passed the filter (but may not be NER)
    for chunk in doc.noun_chunks:
        txt = chunk.text.strip()
        if txt and _is_valid_entity(txt):
            # avoid duplicates
            if not any(txt == e[0] for e in entities):
                label = infer_label(txt, doc.text)
                entities.append((txt, label, None))

    people = [t for t, lbl, _ in entities if lbl == "Person"]
    orgs = [t for t, lbl, _ in entities if lbl == "Organization"]
    movements = [t for t, lbl, _ in entities if lbl == "Movement"]
    dynasties = [t for t, lbl, _ in entities if lbl == "Dynasty"]
    all_org_like = orgs + movements + dynasties

    triples = []
    for p in people:
        for o in all_org_like:
            if p != o:
                triples.append((p, "associated_with", o))
    return triples

# ----------------------------------------------------------------------
# Core extraction function
# ----------------------------------------------------------------------
def extract_facts_from_record(record: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    # Normalise input
    content = record.get("content") or record.get("page_content") or ""
    meta = record.get("metadata") or {}
    flat_meta = _flatten_dict(meta)
    chunk_id = flat_meta.get("chunk_id") or record.get("chunk_id") or ""
    historian = flat_meta.get("historian") or record.get("historian") or ""
    volume = flat_meta.get("volume") or flat_meta.get("volume_title") or flat_meta.get("source_title") or ""
    chapter = flat_meta.get("chapter") or ""
    page = flat_meta.get("page") or record.get("page")

    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []

    def add_node(label: str, name: str, **props):
        if not name:
            return
        canonical = get_canonical(name, (n.get("name") for n in nodes.values()))
        key = normalize_name(canonical)
        if key not in nodes:
            nodes[key] = {"label": label, "name": canonical, "properties": {}}
        nodes[key]["properties"].update(_sanitise_props(props))
        aliases = nodes[key]["properties"].setdefault("aliases", [])
        if canonical not in aliases:
            aliases.append(canonical)
        if name != canonical and name not in aliases:
            aliases.append(name)
        if chunk_id:
            src_chunks = nodes[key]["properties"].setdefault("source_chunk_ids", [])
            if chunk_id not in src_chunks:
                src_chunks.append(chunk_id)

    def add_edge(src_label: str, src_name: str, relation: str, tgt_label: str, tgt_name: str, **props):
        if not src_name or not tgt_name:
            return
        src_canon = get_canonical(src_name, (n.get("name") for n in nodes.values()))
        tgt_canon = get_canonical(tgt_name, (n.get("name") for n in nodes.values()))
        add_node(src_label, src_canon)
        add_node(tgt_label, tgt_canon)
        rel_upper = relation.upper()
        edge = {
            "source_label": src_label,
            "source_name": src_canon,
            "relation": rel_upper,
            "target_label": tgt_label,
            "target_name": tgt_canon,
            "properties": _sanitise_props(props)
        }
        for existing in edges:
            if (existing["source_label"] == edge["source_label"] and
                existing["source_name"] == edge["source_name"] and
                existing["relation"] == edge["relation"] and
                existing["target_label"] == edge["target_label"] and
                existing["target_name"] == edge["target_name"]):
                existing["properties"].update(edge["properties"])
                return
        edges.append(edge)

    # Structural nodes
    if chunk_id:
        add_node("Passage", chunk_id, chunk_id=chunk_id, volume=volume, chapter=chapter, page=page)
    if volume:
        source_name = volume if not chapter else f"{volume} :: {chapter}"
        add_node("Source", source_name, source_title=volume, chapter=chapter, page=page)
        if chunk_id:
            add_edge("Source", source_name, "mentioned_in", "Passage", chunk_id, chunk_id=chunk_id, page=page)
    if historian:
        add_node("Historian", historian, aliases=[historian])
        if chunk_id:
            add_edge("Historian", historian, "wrote_about", "Passage", chunk_id, chunk_id=chunk_id, page=page)

    # Years
    year_pattern = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
    years = [int(y) for y in year_pattern.findall(content) if 1000 <= int(y) <= 2099]
    for y in set(years):
        add_node("Year", str(y), value=y)

    nlp = get_nlp()
    if nlp is not None and content:
        doc = nlp(content)

        # 1. Entities from NER + proper noun chunks (already filtered)
        # We'll just rely on the add_node calls later; we only need to extract relations.

        # 2. Relation extraction per sentence
        for sent in doc.sents:
            rels = extract_relations_from_sent(sent)
            if not rels:
                rels = regex_relations(sent.text)
            for src, rel, tgt in rels:
                src_label = infer_label(src, context=sent.text, relation=rel, role="source")
                tgt_label = infer_label(tgt, context=sent.text, relation=rel, role="target")
                add_edge(src_label, src, rel, tgt_label, tgt,
                         chunk_id=chunk_id or None, page=page,
                         volume=volume or None, chapter=chapter or None,
                         context=sent.text)

            # If still no relations, fallback to co‑occurrence (Person ↔ Organization)
            if not rels:
                for src, rel, tgt in cooccurrence_associations(sent):
                    src_label = infer_label(src, context=sent.text, relation=rel, role="source")
                    tgt_label = infer_label(tgt, context=sent.text, relation=rel, role="target")
                    add_edge(src_label, src, rel, tgt_label, tgt,
                             chunk_id=chunk_id or None, page=page,
                             volume=volume or None, chapter=chapter or None,
                             context=sent.text)

    # 3. Ground events to years
    event_labels = {"Battle", "Movement", "Event", "Treaty", "Dynasty", "Empire", "Kingdom", "Inscription", "Text"}
    for node in list(nodes.values()):
        if node["label"] in event_labels:
            for y in set(years):
                add_edge(node["label"], node["name"], "occurred_in", "Year", str(y),
                         chunk_id=chunk_id or None, page=page)

    # 4. Historian provenance
    if historian:
        for node in list(nodes.values()):
            if node["label"] not in {"Historian", "Passage", "Source", "Year"}:
                add_edge("Historian", historian, "wrote_about", node["label"], node["name"],
                         chunk_id=chunk_id or None, page=page)

    return list(nodes.values()), edges

# ----------------------------------------------------------------------
# Batch helpers (unchanged)
# ----------------------------------------------------------------------
def extract_facts_from_records(records: List[Dict]) -> List[Dict]:
    results = []
    for rec in records:
        nodes, edges = extract_facts_from_record(rec)
        chunk_id = rec.get("chunk_id") or ""
        results.append({"record_id": chunk_id, "nodes": nodes, "edges": edges})
    return results

def build_fact_table(batch_records, extracted_records):
    rows = []
    for idx, rec in enumerate(extracted_records):
        record_id = rec.get("record_id")
        if not record_id and batch_records and idx < len(batch_records):
            record_id = batch_records[idx].get("chunk_id") or f"rec-{idx}"
        elif not record_id:
            record_id = f"rec-{idx}"
        for n in rec.get("nodes", []):
            rows.append({
                "chunk_id": record_id,
                "historian": n.get("properties", {}).get("historian"),
                "source_label": n.get("label"),
                "source_name": n.get("name"),
                "relation": None,
                "target_label": None,
                "target_name": None,
                "year": n.get("properties", {}).get("value"),
                "volume": None,
                "chapter": None,
                "page": None,
                "text": None,
            })
        for e in rec.get("edges", []):
            rows.append({
                "chunk_id": record_id,
                "historian": e.get("properties", {}).get("historian"),
                "source_label": e.get("source_label"),
                "source_name": e.get("source_name"),
                "relation": e.get("relation"),
                "target_label": e.get("target_label"),
                "target_name": e.get("target_name"),
                "year": e.get("properties", {}).get("year"),
                "volume": e.get("properties", {}).get("volume"),
                "chapter": e.get("properties", {}).get("chapter"),
                "page": e.get("properties", {}).get("page"),
                "text": e.get("properties", {}).get("context"),
            })
    return rows