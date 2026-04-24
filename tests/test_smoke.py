from rfr.io import load_long_table
from rfr.core import evaluate_candidates


def test_smoke_evaluate_candidates(tmp_path):
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text(
        "actor_id,actor_type,doc_id,category_id,label\n"
        "h1,human,d1,c1,1\n"
        "h1,human,d1,c2,0\n"
        "h2,human,d1,c1,1\n"
        "h2,human,d1,c2,1\n"
        "h3,human,d1,c1,0\n"
        "h3,human,d1,c2,0\n"
        "ai1,ai,d1,c1,1\n"
        "ai1,ai,d1,c2,0\n"
    )
    ds = load_long_table(csv_path)
    rows = evaluate_candidates(ds, q=1.0, n_boot=50, seed=1)
    assert len(rows) == 1
