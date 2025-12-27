import pickle

GRAPH_PATH = "graph.pickle"   # adjust if needed

def infer_type(hash_id: str):
    if hash_id.startswith("entity-"):
        return "entity"
    if hash_id.startswith("fact-"):
        return "fact"
    if hash_id.startswith("passage-"):
        return "passage"
    return "unknown"

def main():
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    print("=== GRAPH SUMMARY ===")
    print("Vertices:", G.vcount())
    print("Edges:", G.ecount())
    print("Directed:", G.is_directed())

    print("\n=== ALL NODES ===")
    for v in G.vs:
        hash_id = v["hash_id"]
        node_type = infer_type(hash_id)
        content = v["content"]

        print("\nNODE")
        print("  id      :", v.index)
        print("  hash_id :", hash_id)
        print("  type    :", node_type)
        print("  content :", content)

        # If embeddings exist
        if "embedding" in v.attributes():
            emb = v["embedding"]
            if emb is None:
                print("  embedding: None")
            else:
                print("  embedding: vector(len=%d)" % len(emb))

    print("\n=== ALL EDGES ===")
    for e in G.es:
        src, dst = e.tuple
        w = e["weight"] if "weight" in e.attributes() else None

        src_v = G.vs[src]
        dst_v = G.vs[dst]

        print("\nEDGE")
        print("  from:", src, infer_type(src_v["hash_id"]), src_v["content"])
        print("  to  :", dst, infer_type(dst_v["hash_id"]), dst_v["content"])
        print("  weight:", w)

if __name__ == "__main__":
    main()
