"""
Programming Question Retrieval - Phase 6 Terminal Interface
This interface reuses code directly from the Phase 4 and Phase 5 notebooks.
It does not reimplement the Boolean parser or TF-IDF/cosine logic.
Run from the project root or notebooks folder:
    python search_interface.py
"""
import os, sys, json, math, re
from collections import Counter
import pandas as pd

# ---------- Locate project files ----------
def locate(filename, candidates):
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"Cannot locate {filename}. Tried: {candidates}")

PROCESSED_PATH = locate("processed_documents.csv.gz", [
    "Data/artifacts/phase2/processed_documents.csv.gz",
    "../Data/artifacts/phase2/processed_documents.csv.gz",
])
INDEX_PATH = locate("inverted_index.json", [
    "Data/artifacts/phase3/inverted_index.json",
    "../Data/artifacts/phase3/inverted_index.json",
])
INFO_PATH = locate("doc_info.csv", [
    "Data/artifacts/phase3/doc_info.csv",
    "../Data/artifacts/phase3/doc_info.csv",
])
def find_notebook(patterns, description):
    import glob
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if candidates:
        # Prefer a notebook outside an "original" backup folder if present.
        candidates.sort(key=lambda p: ("original" in p.lower(), len(p)))
        return os.path.abspath(candidates[0])
    raise FileNotFoundError(f"Cannot locate {description}. Checked patterns: {patterns}")

PHASE4_NOTEBOOK = find_notebook([
    "notebooks/phase4-boolean-search*.ipynb",
    "notebooks/original/phase4-boolean-search*.ipynb",
    "phase4-boolean-search*.ipynb",
    "../notebooks/original/phase4-boolean-search*.ipynb",
], "Phase 4 Boolean Search notebook")
PHASE5_NOTEBOOK = find_notebook([
    "notebooks/phase5-TFIDF_Cosine_Similarity*.ipynb",
    "notebooks/original/phase5-TFIDF_Cosine_Similarity*.ipynb",
    "phase5-TFIDF_Cosine_Similarity*.ipynb",
    "../notebooks/original/phase5-TFIDF_Cosine_Similarity*.ipynb",
], "Phase 5 TF-IDF notebook")

# ---------- Load project artifacts ----------
processed_df = pd.read_csv(PROCESSED_PATH, compression="gzip", low_memory=False)
with open(INDEX_PATH, encoding="utf-8") as f:
    index_payload = json.load(f)
doc_info = pd.read_csv(INFO_PATH)

# Normalize fields for display
if "Id" in processed_df.columns:
    processed_df["Id"] = processed_df["Id"].astype(int)
if "Id" in doc_info.columns:
    doc_info["Id"] = doc_info["Id"].astype(int)
doc_info = doc_info.set_index("Id", drop=False)
doc_lookup = processed_df.set_index("Id", drop=False)

# ---------- Notebook code loader ----------
def notebook_cells(path):
    with open(path, encoding="utf-8") as f:
        notebook = json.load(f)
    # Keep the original notebook cell numbering (including Markdown cells).
    return notebook.get("cells", [])

def exec_cells(path, indices, namespace):
    cells = notebook_cells(path)
    for idx in indices:
        if idx >= len(cells) or cells[idx].get("cell_type") != "code":
            raise RuntimeError(f"Notebook cell index {idx} is not a code cell in {path}")
        source = "".join(cells[idx].get("source", []))
        exec(compile(source, f"{os.path.basename(path)} cell {idx}", "exec"),
             namespace, namespace)

# ---------- Reuse Phase 4 Boolean-search implementation ----------
boolean_ns = {
    "re": re, "json": json, "pd": pd, "Counter": Counter,
    "inverted_index": {
        term: set(int(posting[0]) for posting in postings)
        for term, postings in index_payload["index"].items()
    },
}
# Phase 4's own preprocessing and parser/operator implementation
exec_cells(PHASE4_NOTEBOOK, [6, 11, 13], boolean_ns)
boolean_search = boolean_ns["boolean_search"]
QueryError = boolean_ns["QueryError"]
boolean_preprocess = boolean_ns["preprocess"]
ALL_DOC_IDS = set(int(x) for x in processed_df["Id"].tolist())

# ---------- Reuse Phase 5 TF-IDF / cosine implementation ----------
ranked_ns = {
    "pd": pd, "Counter": Counter, "math": math, "re": re,
    "df": processed_df.copy(),
}
ranked_ns["N"] = len(ranked_ns["df"])
# Phase 5 computation cells and its original cosine/preprocessing functions
exec_cells(PHASE5_NOTEBOOK, [5, 7, 10, 12, 14, 16], ranked_ns)
idf_scores = ranked_ns["idf_scores"]
sparse_document_vectors = ranked_ns["sparse_document_vectors"]
document_norms = ranked_ns["document_norms"]
cosine_similarity = ranked_ns["cosine_similarity"]
preprocess_query = ranked_ns["preprocess_query"]

# ---------- Ranked query function using Phase 5 code ----------
def ranked_search(query, top_k=10):
    tokens = preprocess_query(query)
    query_tf = Counter(tokens)
    query_tfidf = {}
    for term, tf_value in query_tf.items():
        if term in idf_scores:
            query_tfidf[term] = tf_value * idf_scores[term]

    if not query_tfidf:
        return pd.DataFrame(columns=["Id", "Title", "Tags", "Similarity", "Body"])

    # Score only documents that contain at least one query term.
    candidate_ids = set()
    for term in query_tfidf:
        candidate_ids.update(boolean_ns["inverted_index"].get(term, set()))

    id_to_pos = {int(doc_id): pos for pos, doc_id in enumerate(processed_df["Id"])}
    scores = []
    for doc_id in candidate_ids:
        pos = id_to_pos.get(int(doc_id))
        if pos is None:
            continue
        score = cosine_similarity(query_tfidf, sparse_document_vectors[pos], document_norms[pos])
        scores.append((int(doc_id), score))

    scores.sort(key=lambda x: x[1], reverse=True)
    result_rows = []
    for doc_id, score in scores[:top_k]:
        row = doc_lookup.loc[doc_id]
        result_rows.append({
            "Id": doc_id,
            "Title": row.get("Title", ""),
            "Tags": row.get("Tags", ""),
            "Similarity": score,
            "Body": row.get("Body", row.get("Document", "")),
        })
    return pd.DataFrame(result_rows)

# ---------- Terminal UI ----------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt
    from rich.text import Text
    from rich import box
    console = Console()
    RICH = True
except ImportError:
    RICH = False

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def banner():
    if RICH:
        console.print(Panel.fit(
            "[bold cyan]PROGRAMMING QUESTION RETRIEVAL[/bold cyan]\n"
            "[white]Boolean Search  |  TF-IDF Ranked Search[/white]",
            border_style="bright_blue"
        ))
    else:
        print("=" * 64)
        print("PROGRAMMING QUESTION RETRIEVAL")
        print("Boolean Search | TF-IDF Ranked Search")
        print("=" * 64)

def choose_mode():
    if RICH:
        console.print("\n[bold]1[/bold] Boolean Search")
        console.print("[bold]2[/bold] Ranked Search")
        console.print("[bold]0[/bold] Exit")
        return Prompt.ask("Select mode", choices=["1", "2", "0"], default="0")
    print("\n[1] Boolean Search\n[2] Ranked Search\n[0] Exit")
    return input("Select mode: ").strip()

def print_results(results, mode, query):
    if results is None or len(results) == 0:
        msg = "No matching questions found."
        console.print(f"\n[yellow]{msg}[/yellow]") if RICH else print("\n" + msg)
        return
    if RICH:
        table = Table(title=f"Results — {mode}", box=box.ROUNDED, show_lines=True)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Question ID", style="magenta")
        table.add_column("Title", style="white", max_width=62)
        if mode == "Ranked":
            table.add_column("Similarity", style="green", justify="right")
        table.add_column("Tags", style="yellow", max_width=30)
        for i, (_, row) in enumerate(results.iterrows(), start=1):
            vals = [str(i), str(row["Id"]), str(row["Title"])]
            if mode == "Ranked":
                vals.append(f'{float(row["Similarity"]):.4f}')
            vals.append(str(row.get("Tags", "")))
            table.add_row(*vals)
        console.print(table)
        for i, (_, row) in enumerate(results.iterrows(), start=1):
            body = str(row.get("Body", "")).strip()
            if body:
                console.print(Panel(body[:1400], title=f"Question {i} — ID {row['Id']}", border_style="dim"))
    else:
        print("\n" + "-" * 80)
        for i, (_, row) in enumerate(results.iterrows(), start=1):
            print(f"{i}. ID: {row['Id']} | {row['Title']}")
            if mode == "Ranked":
                print(f"   Similarity: {row['Similarity']:.4f}")
            print(f"   Tags: {row.get('Tags','')}")
            body = str(row.get("Body", "")).strip()
            if body:
                print("   Body:", body[:1000])
            print("-" * 80)

def run_search(mode):
    query = Prompt.ask("\nEnter query") if RICH else input("\nEnter query: ")
    if not query.strip():
        console.print("[red]Query cannot be empty.[/red]") if RICH else print("Query cannot be empty.")
        return
    top_k = IntPrompt.ask("Number of results", default=10) if RICH else int(input("Number of results [10]: ") or "10")
    top_k = max(1, min(top_k, 50))
    try:
        if mode == "Boolean":
            ids, notes = boolean_search(query, index=boolean_ns["inverted_index"], all_ids=ALL_DOC_IDS)
            ids = sorted(ids)[:top_k]
            results = doc_lookup.loc[ids].reset_index(drop=True) if ids else pd.DataFrame()
            if not results.empty:
                results["Body"] = results.get("Body", results.get("Document", ""))
            for note in notes:
                console.print(f"[dim]Note: {note}[/dim]") if RICH else print("Note:", note)
        else:
            results = ranked_search(query, top_k)
        print_results(results, mode, query)
    except QueryError as e:
        console.print(f"[red]Query error: {e}[/red]") if RICH else print("Query error:", e)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]") if RICH else print("Error:", e)

def main():
    while True:
        clear()
        banner()
        mode = choose_mode()
        if mode == "0":
            break
        run_search("Boolean" if mode == "1" else "Ranked")
        if RICH:
            Prompt.ask("\nPress Enter to return to menu", default="")
        else:
            input("\nPress Enter to return to menu...")
    clear()
    if RICH:
        console.print("[bold cyan]Thank you for using the system![/bold cyan]")
    else:
        print("Thank you for using the system!")

if __name__ == "__main__":
    main()
