"""Graph schema and reusable labels / relation types."""

from __future__ import annotations

NODE_LABELS = {
    # Modern history
    "king": "King",
    "battle": "Battle",
    "year": "Year",
    "historian": "Historian",
    "dynasty": "Dynasty",
    "event": "Event",
    "passage": "Passage",
    "source": "Source",
    "movement": "Movement",
    "organization": "Organization",
    "political_party": "PoliticalParty",
    "freedom_fighter": "FreedomFighter",
    "politician": "Politician",
    "treaty": "Treaty",
    "person": "Person",
    "place": "Place",
    "newspaper": "Newspaper",
    "legislation": "Legislation",

    # Ancient / medieval expansions
    "emperor": "Emperor",
    "queen": "Queen",
    "prince": "Prince",
    "minister": "Minister",
    "ruler": "Ruler",
    "kingdom": "Kingdom",
    "empire": "Empire",
    "province": "Province",
    "fort": "Fort",
    "temple": "Temple",
    "mosque": "Mosque",
    "inscription": "Inscription",
    "edict": "Edict",
    "text": "Text",
    "religious_figure": "ReligiousFigure",
    "saint": "Saint",
    "scholar": "Scholar",
    "poet": "Poet",
    "philosopher": "Philosopher",
    "tax": "Tax",
    "law": "Law",
    "decree": "Decree",
    "capital": "Capital",
    "region": "Region",
}

RELATION_TYPES = {
    "fought_against": "FOUGHT_AGAINST",
    "fought": "FOUGHT",
    "wrote_about": "WROTE_ABOUT",
    "occurred_in": "OCCURRED_IN",
    "happened_in": "HAPPENED_IN",
    "succeeded_by": "SUCCEEDED_BY",
    "ruled": "RULED",
    "allied_with": "ALLIED_WITH",
    "contemporary_of": "CONTEMPORARY_OF",
    "mentioned_in": "MENTIONED_IN",
    "related_to": "RELATED_TO",
    "associated_with": "ASSOCIATED_WITH",
    "founded": "FOUNDED",
    "led": "LED",
    "member_of": "MEMBER_OF",
    "joined": "JOINED",
    "signed": "SIGNED",
    "protested": "PROTESTED",
    "imprisoned": "IMPRISONED",
    "assassinated": "ASSASSINATED",
    "published_in": "PUBLISHED_IN",
    "demanded": "DEMANDED",
    "presided_over": "PRESIDED_OVER",
    "negotiated": "NEGOTIATED",
    "established": "ESTABLISHED",
    "influenced": "INFLUENCED",

    # Ancient / medieval expansions
    "conquered": "CONQUERED",
    "captured": "CAPTURED",
    "annexed": "ANNEXED",
    "invaded": "INVADED",
    "besieged": "BESIEGED",
    "raided": "RAIDED",
    "expanded_into": "EXPANDED_INTO",
    "controlled": "CONTROLLED",
    "governed": "GOVERNED",
    "administered": "ADMINISTERED",
    "patronized": "PATRONIZED",
    "built": "BUILT",
    "commissioned": "COMMISSIONED",
    "converted": "CONVERTED",
    "rebelled_against": "REBELLED_AGAINST",
    "tributary_to": "TRIBUTARY_TO",
    "vassal_of": "VASSAL_OF",
    "married_to": "MARRIED_TO",
    "opposed": "OPPOSED",
    "vanquished": "VANQUISHED",
    "defeated_by": "DEFEATED_BY",
}

SCHEMA_DESCRIPTION = """\
Nodes:
  - King(name, aliases, source_chunk_ids)
  - Emperor(name, aliases, source_chunk_ids)
  - Queen(name, aliases, source_chunk_ids)
  - Prince(name, aliases, source_chunk_ids)
  - Minister(name, aliases, source_chunk_ids)
  - Ruler(name, aliases, source_chunk_ids)
  - Source(chunk_id, content)
  - Battle(name, year, aliases, source_chunk_ids)
  - Year(value)
  - Historian(name, aliases)
  - Dynasty(name, aliases)
  - Event(name, aliases, year)
  - Passage(chunk_id, volume, chapter, page)
  - Movement (e.g., Quit India, Non-Cooperation)
  - Organization (e.g., Indian National Congress, HSRA)
  - PoliticalParty (e.g., INC,  Muslim League)
  - FreedomFighter, Politician
  - Treaty (e.g., Treaty of Paris)
  - Place (city, region, country)
  - Newspaper, Legislation
  - Kingdom, Empire, Province
  - Fort, Temple, Mosque
  - Inscription, Edict, Text
  - ReligiousFigure, Saint, Scholar
  - Poet, Philosopher
  - Tax, Law
  - Decree, Capital, Region

Edges:
  - (King|Emperor|Ruler)-[:FOUGHT_AGAINST]->(King|Emperor|Ruler)
  - (Battle)-[:OCCURRED_IN]->(Year)
  - (Historian)-[:WROTE_ABOUT]->(Event|King|Battle|Dynasty|Empire|Kingdom)
  - (King|Emperor|Ruler)-[:RULED]->(Dynasty|Kingdom|Empire|Province)
  - (King|Emperor|Ruler)-[:SUCCEEDED_BY]->(King|Emperor|Ruler)
  - (King|Emperor|Ruler)-[:ALLIED_WITH]->(King|Emperor|Ruler)
  - (Historian)-[:CONTEMPORARY_OF]->(Historian)
  - (Passage)-[:MENTIONED_IN]->(Event|King|Battle|Dynasty|Empire|Kingdom)
  - (Person)-[:LED]->(Movement|Organization|Revolt|Campaign)
  - (Person)-[:MEMBER_OF]->(Organization|PoliticalParty)
  - (Person)-[:FOUNDED]->  - (Person)-[:FOUNDED]->(Organization|Newspaper)
  - (Person)-[:SIGNED]->(Treaty)
  - (Person)-[:PROTESTED]->(Event|Legislation)
  - (Person)-[:IMPRISONED]->(Place)  or (Person)-[:IMPRISONED_BY]->(Government)
  - (Person)-[:ASSASSINATED]->(Person)
  - (Organization)-[:DEMANDED]->(Policy|Event)
  - (Organization)-[:PUBLISHED_IN]->(Newspaper)
  - (Person)-[:PRESIDED_OVER]->(Event)
  - (Person)-[:NEGOTIATED]->(Treaty)
  - (Person)-[:ESTABLISHED]->(Organization|Movement)
  - (Person)-[:INFLUENCED]->(Person|Movement|Organization)
  - (Ruler|King|Emperor)-[:CONQUERED|CAPTURED|ANNEXED|INVADED|BESIEGED|RAIDED]->(Kingdom|Empire|Fort|Region|City)
  - (Ruler|King|Emperor)-[:GOVERNED|ADMINISTERED|CONTROLLED]->(Kingdom|Empire|Province|Region|Capital)
  - (Ruler|King|Emperor|ReligiousFigure|Scholar|Person)-[:BUILT|COMMISSIONED|PATRONIZED]->(Temple|Mosque|Fort|Inscription|Text)
  - (Ruler|King|Emperor|Person)-[:CONVERTED]->(Religion|Person|Community)
  - (Ruler|King|Emperor|Person)-[:REBELLED_AGAINST|OPPOSED]->(Ruler|King|Emperor|Empire)
  - (Kingdom|Empire|Province)-[:TRIBUTARY_TO|VASSAL_OF]->(Kingdom|Empire|Ruler)
"""
