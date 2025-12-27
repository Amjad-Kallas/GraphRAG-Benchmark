import pickle
from collections import Counter

GRAPH_PATH = "graph.pickle"  # adjust if needed
GRAPH_PATH = "/home/kallas/project/GraphRAG-Benchmark/Examples/hipporag2_workspace/Medical/meta-llama_Llama-3.1-8B-Instruct_BAAI_bge-large-en-v1.5/graph.pickle"

def main():
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    print("=== GRAPH TYPE ===")
    print(type(G))
    print("Directed:", G.is_directed())

    print("\n=== SIZE ===")
    print("Number of vertices:", G.vcount())
    print("Number of edges:", G.ecount())

    print("\n=== VERTEX ATTRIBUTES ===")
    print(G.vertex_attributes())

    print("\n=== EDGE ATTRIBUTES ===")
    print(G.edge_attributes())

    print("\n=== NODE TYPE DISTRIBUTION ===")
    if "type" in G.vertex_attributes():
        types = Counter(G.vs["type"])
        for k, v in types.items():
            print(f"{k}: {v}")

    print("\n=== SAMPLE VERTEX ===")
    v = G.vs[0]
    for attr in G.vertex_attributes():
        val = v[attr]
        if isinstance(val, list) and len(val) > 10:
            print(f"{attr}: <list len={len(val)}>")
        else:
            print(f"{attr}: {val}")

    print("\n=== SAMPLE EDGE ===")
    e = G.es[0]
    print("Edge:", e.tuple)
    for attr in G.edge_attributes():
        print(f"{attr}: {e[attr]}")

    print("\n=== TRACE ONE FACT NODE (if exists) ===")
    if "type" in G.vertex_attributes():
        fact_ids = [i for i, t in enumerate(G.vs["type"]) if t == "fact"]
        if fact_ids:
            fid = fact_ids[0]
            print("Fact vertex id:", fid)
            print("Fact text:", G.vs[fid]["text"])

            neighbors = G.neighbors(fid)
            print("Connected vertices:")
            for n in neighbors:
                print(
                    f"  - id={n}, type={G.vs[n]['type']}, text={G.vs[n]['text'][:60]}"
                )

if __name__ == "__main__":
    main()
