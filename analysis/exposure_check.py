"""
exposure_check.py: retrieval exposure check (§7.5).

How often does an ordinary retriever assemble the evidence that the unsafe
rungs construct? Each distinct corpus (one per seed; the conditions that share
a seed share the corpus) is indexed once over the sanitized text of its 25,000
fragments, and two query families are issued:

  claim    one query per cross-boundary probe: the sanitized values of its five
           source fragments, in role order (what a verification step or a
           follow-up turn would ask about)
  generic  40 hand-written threat-hunting queries, the same for every corpus

A top-k context is exposed when its fragments cover all five roles across two
or more tenants and no single event covers all five. For claim queries the
script also counts how many of the probe's five source fragments are retrieved.

Two retrievers are run: BM25 (rank-bm25, BM25Okapi over lower-cased tokens with
surrounding punctuation stripped; fragments with a score of zero or less are
not returned, so a query whose terms occur in fewer than k fragments returns
fewer than k, which the fewer_than_k column counts) and all-MiniLM-L6-v2
(sentence-transformers, cosine similarity of normalized embeddings rounded to
six decimals, computed on the CPU at a pinned model revision; every fragment
is ranked). Equal scores are ordered either by a hash of the provenance
handle or by position in fragments.store.jsonl; both orders are reported, per
corpus in the CSV and pooled in the log, which also lists the counts that
differ between the two orders.

Usage, from the repository root (needs rank-bm25 and sentence-transformers;
the MiniLM model is downloaded on first use), with the rerun in a temporary
folder:
  tmp=$(mktemp -d)
  python3 analysis/exposure_check.py --out "$tmp/exposure_check.csv" > "$tmp/exposure_check.log"
  python3 analysis/exposure_check.py --compare "$tmp/exposure_check.csv"

Writes the rows to --out, by default exposure_check.csv in a new temporary
folder: one row per corpus seed, retriever, tie-breaking order and query
family. The released rows are analysis/exposure_check.csv, and their log is
analysis/exposure_check.log: its first line names the CSV and its number of
rows, and its second line lists the library versions in use. MiniLM scores
depend on the versions of torch, transformers, sentence-transformers and
numpy, so --compare checks a rerun against the released rows: the BM25 rows
must be identical, and the MiniLM rows must give the counts that §7.5
reports.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import tempfile
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROLES = ("subject", "action", "object", "outcome", "time")
TOKEN = re.compile(r"[a-z0-9_:\-\.@+/]+")
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TIEBREAKS = ("hash", "file-order")

GENERIC_QUERIES = [
    "show me hosts that contacted suspicious external IPs",
    "find users who elevated privileges recently",
    "lateral movement from compromised credentials",
    "ransomware indicators across the environment",
    "data exfiltration to unknown destinations",
    "phishing email leading to credential theft",
    "domain admin account used outside normal hours",
    "powershell execution on workstation",
    "scheduled task creation by non-admin user",
    "DNS queries to known C2 infrastructure",
    "executable launched from temp directory",
    "service account password changed unexpectedly",
    "kerberoasting activity in the last week",
    "RDP connection from foreign IP",
    "file written to startup folder",
    "registry modification in run keys",
    "credential dumping from LSASS",
    "anomalous outbound traffic on port 443",
    "process injection into legitimate binary",
    "CVE-2024 exploitation attempts",
    "patch missing on critical server",
    "user account locked after multiple failures",
    "new service installed on domain controller",
    "WMI persistence mechanism",
    "browser extension installed silently",
    "USB device connected and files copied",
    "encrypted archive uploaded to cloud storage",
    "PowerShell encoded command execution",
    "scheduled task running suspicious payload",
    "antivirus disabled on endpoint",
    "log clearing event detected",
    "process accessing memory of LSASS",
    "outbound connection to TOR exit node",
    "DLL side-loading from non-standard path",
    "compromised host beaconing periodically",
    "credential rotation after suspected exposure",
    "host quarantined due to malware detection",
    "lateral SMB connection between workstations",
    "privilege escalation via token impersonation",
    "executable signed with revoked certificate",
]


def tokenize(s):
    return [t for t in (t.strip(".,;:") for t in TOKEN.findall(s.lower())) if t]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def distinct_corpora(data_dir):
    """Condition folders grouped by the content of their fragment store."""
    by_hash = {}
    for d in sorted(p for p in data_dir.iterdir() if (p / "fragments.store.jsonl").exists()):
        h = hashlib.sha256((d / "fragments.store.jsonl").read_bytes()).hexdigest()
        by_hash.setdefault(h, []).append(d)
    return list(by_hash.values())


class BM25:
    name = "BM25"
    drops_nonpositive_scores = True

    def __init__(self, texts):
        from rank_bm25 import BM25Okapi
        self.index = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, queries):
        return [[float(x) for x in self.index.get_scores(tokenize(q))] for q in queries]


class MiniLM:
    name = "MiniLM"
    drops_nonpositive_scores = False

    def __init__(self, texts):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MINILM, revision=MINILM_REVISION, device="cpu")
        self.vectors = np.asarray(self.model.encode(texts, batch_size=128, show_progress_bar=False,
                                                    normalize_embeddings=True))

    def scores(self, queries):
        import numpy as np
        q = self.model.encode(queries, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        sims = q @ self.vectors.T
        if not np.isfinite(sims).all():
            raise ValueError("non-finite similarity")
        # Rounded so that fragments with identical text tie exactly instead of by float noise.
        return [[round(x, 6) for x in row] for row in sims.tolist()]


def classify(fragments):
    """Exposure flags of one top-k context."""
    roles_by_tenant, roles_by_event = defaultdict(set), defaultdict(set)
    for f in fragments:
        roles_by_tenant[f["tenant_id"]].add(f["role"])
        roles_by_event[f["event_id"]].add(f["role"])
    roles = set().union(*roles_by_tenant.values()) if fragments else set()
    complete_event = any(set(ROLES) <= r for r in roles_by_event.values())
    role_complete_multi_tenant = len(roles_by_tenant) >= 2 and set(ROLES) <= roles
    return {
        "exposed": role_complete_multi_tenant and not complete_event,
        "role_complete": set(ROLES) <= roles,
        "role_complete_multi_tenant": role_complete_multi_tenant,
        "has_complete_event": complete_event,
        "multi_tenant": len(roles_by_tenant) >= 2,
    }


def run(data_dir, k, retriever_names):
    rows = []
    for dirs in distinct_corpora(data_dir):
        corpus = dirs[0]
        seed = json.loads((corpus / "corpus.json").read_text())["seed"]
        fragments = load_jsonl(corpus / "fragments.store.jsonl")
        by_handle = {f["public_provenance_id"]: f for f in fragments}
        handle_hash = [hashlib.sha1(f["public_provenance_id"].encode()).hexdigest() for f in fragments]
        queries = []
        for p in load_jsonl(corpus / "probes.jsonl"):
            if p["probe_type"] == "cross_boundary":
                by_role = {by_handle[h]["role"]: by_handle[h] for h in p["fragment_provenance_ids"]}
                queries.append(("claim", " ".join(by_role[c]["sanitized_value"] for c in ROLES),
                                set(p["fragment_provenance_ids"])))
        queries += [("generic", q, None) for q in GENERIC_QUERIES]
        texts = [f["sanitized_text"] for f in fragments]
        for name in retriever_names:
            retriever = {"BM25": BM25, "MiniLM": MiniLM}[name](texts)
            all_scores = retriever.scores([q for _, q, _ in queries])
            for tiebreak in TIEBREAKS:
                counts = defaultdict(Counter)
                for (family, _, sources), scores in zip(queries, all_scores):
                    candidates = [i for i, s in enumerate(scores) if s > 0 or not retriever.drops_nonpositive_scores]
                    tie_key = (lambda i: i) if tiebreak == "file-order" else (lambda i: handle_hash[i])
                    order = sorted(candidates, key=lambda i: (-scores[i], tie_key(i)))
                    top = [fragments[i] for i in order[:k]]
                    c = counts[family]
                    c["queries"] += 1
                    c["fewer_than_k"] += len(top) < k
                    for key, value in classify(top).items():
                        c[key] += value
                    if sources is not None:
                        hits = len({f["public_provenance_id"] for f in top} & sources)
                        c["source_fragments_retrieved"] += hits
                        c["source_fragments_total"] += len(sources)
                        c["all_five_sources_retrieved"] += hits == len(sources)
                for family, c in counts.items():
                    rows.append({"seed": seed, "conditions_sharing_corpus": len(dirs), "retriever": name,
                                 "tiebreak": tiebreak, "family": family, "k": k, **c})
    return rows


FIELDS = ["seed", "conditions_sharing_corpus", "retriever", "tiebreak", "family", "k", "queries", "exposed",
          "role_complete", "role_complete_multi_tenant", "has_complete_event", "multi_tenant", "fewer_than_k",
          "source_fragments_retrieved", "source_fragments_total", "all_five_sources_retrieved"]
# The counts §7.5 reports, pooled over the corpora for each retriever, tie-breaking order and query family.
REPORTED = ["queries", "exposed", "source_fragments_retrieved", "source_fragments_total", "all_five_sources_retrieved"]


def versions(retriever_names):
    packages = ["rank-bm25", "numpy"] + (["sentence-transformers", "transformers", "torch"]
                                         if "MiniLM" in retriever_names else [])
    v = [f"Python {platform.python_version()}"] + [f"{p} {metadata.version(p)}" for p in packages]
    if "MiniLM" in retriever_names:
        v.append(f"model {MINILM} at revision {MINILM_REVISION}, on CPU")
    return v


def compare(rerun_csv, released_csv):
    """0 if the rerun has the released BM25 rows and the released MiniLM counts of §7.5, else 1."""
    def load(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def pooled(rows, retriever):
        out = defaultdict(Counter)
        for r in rows:
            if r["retriever"] == retriever:
                out[r["tiebreak"], r["family"]].update({k: int(r[k]) for k in REPORTED})
        return out

    rerun, released = load(rerun_csv), load(released_csv)
    bm25 = [[r for r in rows if r["retriever"] == "BM25"] for rows in (rerun, released)]
    minilm = [pooled(rows, "MiniLM") for rows in (rerun, released)]
    print(f"BM25 rows: {'identical to' if bm25[0] == bm25[1] else 'DIFFERENT from'} the released rows")
    differ = [f"{t} {fam} {k} {minilm[1][t, fam][k]} -> {minilm[0][t, fam][k]}"
              for t, fam in sorted(minilm[1]) for k in REPORTED if minilm[0][t, fam][k] != minilm[1][t, fam][k]]
    if not minilm[1] or minilm[0].keys() != minilm[1].keys():
        differ.append("the MiniLM rows cover different tie-breaking orders or query families")
    print("MiniLM counts of §7.5 (queries, exposed contexts, source fragments retrieved, queries returning all "
          "five; both tie-breaking orders): " + ("identical to the released rows" if not differ else "DIFFERENT: " + "; ".join(differ)))
    return 0 if bm25[0] == bm25[1] and not differ else 1


def main():
    ap = argparse.ArgumentParser(description="Retrieval exposure check (§7.5).")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--retrievers", nargs="+", choices=["BM25", "MiniLM"], default=["BM25", "MiniLM"])
    ap.add_argument("--out", type=Path, help="CSV to write (default: exposure_check.csv in a new temporary folder)")
    ap.add_argument("--compare", type=Path, metavar="RERUN_CSV",
                    help="compare the rows of a rerun with analysis/exposure_check.csv and exit")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare, HERE / "exposure_check.csv")
    rows = run(args.data, args.k, args.retrievers)
    out = args.out or Path(tempfile.mkdtemp(prefix="exposure_check_")) / "exposure_check.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval=0)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.name} ({len(rows)} rows)")
    print(f"rows written to {out}", file=sys.stderr)
    print("versions: " + "; ".join(versions(args.retrievers)))
    print(f"\nTop-{args.k} contexts pooled over the {len({r['seed'] for r in rows})} distinct corpora")
    pooled = defaultdict(Counter)
    for r in rows:
        pooled[(r["retriever"], r["tiebreak"], r["family"])].update({k: r.get(k, 0) for k in FIELDS[6:]})
    for (retriever, tiebreak, family), c in pooled.items():
        n = c["queries"]
        line = (f"{retriever:6s} ties by {tiebreak:10s} {family:7s} exposed {c['exposed']}/{n} "
                f"({100 * c['exposed'] / n:.1f}%); covering all five roles {c['role_complete']}; "
                f"with a complete genuine event {c['has_complete_event']}; "
                f"returning fewer than {args.k} fragments {c['fewer_than_k']}")
        if c["source_fragments_total"]:
            line += (f"; source fragments retrieved {c['source_fragments_retrieved']}/{c['source_fragments_total']}"
                     f"; queries returning all five {c['all_five_sources_retrieved']}")
        print(line)
    keys = {"exposed claim contexts": ("claim", "exposed"), "exposed generic contexts": ("generic", "exposed"),
            "source fragments retrieved": ("claim", "source_fragments_retrieved"),
            "claim queries returning all five": ("claim", "all_five_sources_retrieved")}
    seeds = sorted({r["seed"] for r in rows})
    cell = {(r["retriever"], r["tiebreak"], r["family"], r["seed"]): r for r in rows}
    print("\nHash order against file order, per corpus seed and pooled")
    for retriever in args.retrievers:
        for name, (fam, key) in keys.items():
            pairs = {s: [cell[(retriever, t, fam, s)].get(key, 0) for t in TIEBREAKS] for s in seeds}
            differ = [f"seed {s}: {a} by hash, {b} by file order" for s, (a, b) in pairs.items() if a != b]
            a, b = (pooled[(retriever, t, fam)][key] for t in TIEBREAKS)
            if not differ:
                verdict = "equal for every seed"
            elif a == b:
                verdict = "; ".join(differ) + f"; equal when pooled ({a})"
            else:
                verdict = "; ".join(differ) + f"; pooled: {a} by hash, {b} by file order"
            print(f"  {retriever:6s} {name}: {verdict}")


if __name__ == "__main__":
    sys.exit(main())
