# phase_3_moe_raft/router.py

from phase_3_moe_raft.config import EXPERTS, DEFAULT_EXPERT

# Keyword sets (extend as needed)
ANCIENT_KEYWORDS   = ["vedic", "indus valley", "maurya", "gupta", "sangam", "ashoka", "kalinga"]
MEDIEVAL_KEYWORDS  = ["delhi sultanate", "mughal", "vijayanagara", "bhakti", "sufi", "chola"]
MODERN_KEYWORDS    = ["british", "colonial", "1857", "revolt", "independence", "partition of bengal",
                      "swadeshi", "non-cooperation", "quit india", "gandhi"]

MARXIST_KEYWORDS   = ["economic drain", "class struggle", "mode of production", "dialectical",
                      "materialist", "exploitation", "bourgeois", "proletariat"]
NATIONALIST_KEYWORDS = ["nationalism", "freedom struggle", "patriot", "swadeshi", "motherland",
                         "british yoke", "national movement"]

def classify_domain(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ANCIENT_KEYWORDS):
        return "Ancient"
    if any(k in q for k in MEDIEVAL_KEYWORDS):
        return "Medieval"
    if any(k in q for k in MODERN_KEYWORDS):
        return "Modern"
    # Default domain – you might change to "Modern" or "Ancient" based on your data distribution
    return "Modern"

def classify_perspective(query: str) -> str:
    q = query.lower()
    if any(k in q for k in MARXIST_KEYWORDS):
        return "Marxist"
    if any(k in q for k in NATIONALIST_KEYWORDS):
        return "Nationalist"
    return "Nationalist"   # default

def route_query(query: str):
    """
    Returns (expert_name, expert_dict) after classifying the query.
    """
    domain = classify_domain(query)
    perspective = classify_perspective(query)
    expert_name = f"{domain.lower()}_{perspective.lower()}"
    if expert_name not in EXPERTS:
        # Fallback: try with default perspective
        expert_name = f"{domain.lower()}_nationalist"
        if expert_name not in EXPERTS:
            expert_name = DEFAULT_EXPERT
    return expert_name, EXPERTS[expert_name]