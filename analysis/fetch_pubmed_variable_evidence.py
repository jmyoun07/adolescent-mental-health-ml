from __future__ import annotations

import csv
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "reports" / "pubmed_variable_evidence_candidates.csv"


QUERIES = {
    "adolescent_suicide_ml": '((adolescent[Title/Abstract]) AND ("suicidal ideation" OR "suicide plan" OR "suicide attempt") AND ("machine learning" OR "prediction model"))',
    "yrbss_depression_suicide": '((Youth Risk Behavior Survey OR YRBS OR YRBSS) AND (suicide OR suicidal OR depression) AND adolescent)',
    "kyrbs_depression_suicide": '((Korea Youth Risk Behavior OR Korean Youth Risk Behavior OR KYRBS OR KYRBWS) AND (suicide OR suicidal ideation OR depression))',
    "risk_factor_domains": '((adolescent depression OR depressive symptoms) AND (suicidal ideation OR suicide attempt) AND (sleep OR bullying OR alcohol OR smoking OR smartphone OR social media))',
}


DOMAIN_PATTERNS = {
    "mental_health_overlap": r"depress|sad|hopeless|anxiety|lonel|stress|mental health|psychological",
    "suicidality": r"suicid",
    "sleep": r"sleep|insomnia",
    "digital_social_media": r"smartphone|screen time|social media|internet|digital",
    "violence_victimization": r"bully|bullying|victim|violence|fight|injur|abuse",
    "substance_alcohol_tobacco": r"alcohol|drink|smok|tobacco|cigarette|vaping|e-cigarette|substance|drug|marijuana|cannabis",
    "physical_activity": r"physical activity|exercise|sport|sedentary",
    "weight_body_image_bmi": r"body mass|BMI|weight|obes|body image|body shape",
    "academic_school": r"academic|school|grade|connectedness|achievement",
    "socioeconomic_family": r"socioeconomic|family|parent|household|income",
    "sex_age_demographics": r"sex|gender|female|male|age|grade|race|ethnic",
    "sexual_behavior_identity": r"sexual intercourse|sexual behavior|sexual minority|LGBT|orientation",
    "comorbidity_allergy_asthma": r"asthma|allerg|atopic|rhinitis|dermatitis",
}


def ncbi_get(endpoint: str, params: dict[str, str]) -> bytes:
    params = dict(params)
    params.setdefault("tool", "kyrbs_variable_review")
    params.setdefault("email", "not.provided@example.com")
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def esearch(query: str, retmax: int = 60) -> list[str]:
    xml = ncbi_get(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": str(retmax),
            "sort": "relevance",
        },
    )
    root = ET.fromstring(xml)
    return [elem.text for elem in root.findall(".//Id") if elem.text]


def join_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return " ".join(" ".join(elem.itertext()).split())


def article_year(article: ET.Element) -> str:
    for path in [
        ".//JournalIssue/PubDate/Year",
        ".//ArticleDate/Year",
        ".//PubMedPubDate[@PubStatus='pubmed']/Year",
        ".//PubMedPubDate[@PubStatus='entrez']/Year",
    ]:
        elem = article.find(path)
        if elem is not None and elem.text:
            return elem.text
    medline_date = article.findtext(".//JournalIssue/PubDate/MedlineDate", default="")
    match = re.search(r"(19|20)\d{2}", medline_date)
    return match.group(0) if match else ""


def efetch(pmids: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i in range(0, len(pmids), 80):
        batch = pmids[i : i + 80]
        xml = ncbi_get(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            },
        )
        root = ET.fromstring(xml)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", default="")
            title = join_text(article.find(".//ArticleTitle"))
            abstract_parts = [join_text(e) for e in article.findall(".//Abstract/AbstractText")]
            abstract = " ".join(part for part in abstract_parts if part)
            journal = join_text(article.find(".//Journal/Title"))
            year = article_year(article)
            mesh_terms = "; ".join(
                sorted(
                    {
                        join_text(desc)
                        for desc in article.findall(".//MeshHeading/DescriptorName")
                        if join_text(desc)
                    }
                )
            )
            text = f"{title} {abstract} {mesh_terms}"
            domains = [
                name
                for name, pattern in DOMAIN_PATTERNS.items()
                if re.search(pattern, text, flags=re.IGNORECASE)
            ]
            rows.append(
                {
                    "pmid": pmid,
                    "year": year,
                    "journal": journal,
                    "title": title,
                    "abstract": abstract,
                    "mesh_terms": mesh_terms,
                    "evidence_domains": ";".join(domains),
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                }
            )
        time.sleep(0.34)
    return rows


def main() -> None:
    query_hits: dict[str, list[str]] = {}
    pmid_order: list[str] = []
    for label, query in QUERIES.items():
        ids = esearch(query)
        query_hits[label] = ids
        for pmid in ids:
            if pmid not in pmid_order:
                pmid_order.append(pmid)
        time.sleep(0.34)

    rows = efetch(pmid_order)
    by_pmid = {row["pmid"]: row for row in rows}
    for label, ids in query_hits.items():
        for rank, pmid in enumerate(ids, start=1):
            if pmid in by_pmid:
                key = f"query_{label}_rank"
                by_pmid[pmid][key] = str(rank)

    all_keys = [
        "pmid",
        "year",
        "journal",
        "title",
        "abstract",
        "mesh_terms",
        "evidence_domains",
        "pubmed_url",
        *[f"query_{label}_rank" for label in QUERIES],
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for pmid in pmid_order:
            writer.writerow(by_pmid[pmid])

    print(f"Wrote {len(pmid_order)} PubMed candidate records to {OUT}")


if __name__ == "__main__":
    main()
