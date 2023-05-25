from typing import Optional

import networkx as nx
import pandas as pd
import polars as pl
from tqdm import tqdm


def map_to_chains_id(pairs: pl.DataFrame) -> pl.DataFrame:
    chains = _build_chains_graph(pairs)
    return _to_chain_id(chains, pairs)


def enrich_by_chains(
    pairs: pl.DataFrame,
    max_nodes: int = 120,
    cutoff: Optional[int] = None,
) -> pl.DataFrame:
    """Adds pairs derivable from existing ones, but not present in the data.

    Args:
        pairs (pl.DataFrame): Table with columns target, variantid1, variantid2
        max_nodes (int): Maximum number of nodes in a chain to create samples for it.
            Less `max_nodes` means less execution time and less additional samples. Defaults to 120.
        cutoff (int, optional): Maximum length of a path to stop search. None means any length is allowed.
            Less `cutoff` means less execution time and less additional samples. Defaults to None.

    Returns:
        pl.DataFrame: Table with columns from `pairs` and additional columns chain_id and enriched
    """

    pairs = map_to_chains_id(pairs)
    enriched = []
    for _, known_pairs in tqdm(
        pairs.groupby("chain_id"), total=pairs["chain_id"].n_unique()
    ):
        known_pairs = known_pairs.with_columns(pl.lit(False).alias("enriched"))
        if len(known_pairs) == 1:
            enriched.append(known_pairs)
            continue

        enriched_pairs = _enrich(known_pairs, max_nodes=max_nodes, cutoff=cutoff)
        if enriched_pairs is not None:
            enriched.append(pl.concat([known_pairs, enriched_pairs]))
        else:
            enriched.append(known_pairs)
    return pl.concat(enriched).filter(pl.col("enriched")).select(pl.exclude("enriched"))


def _build_chains_graph(pairs: pl.DataFrame) -> nx.Graph:
    chains = nx.Graph()
    for var1, var2, target in pairs.select(
        ["variantid1", "variantid2", "target"]
    ).iter_rows():
        chains.add_edge(var1, var2, target=target)
    return chains


def _to_chain_id(chains: nx.Graph, pairs: pl.DataFrame) -> pl.DataFrame:
    var_to_chain_id = {}
    for i, chain in enumerate(nx.connected_components(chains)):
        for var in chain:
            var_to_chain_id[var] = i
    return pairs.with_columns(
        pl.col("variantid1").map_dict(var_to_chain_id).cast(pl.UInt32).alias("chain_id")
    )


def _enrich(
    known_pairs: pl.DataFrame,
    max_nodes: int = 100,
    cutoff: Optional[int] = None,
) -> Optional[pl.DataFrame]:
    chains = _build_chains_graph(known_pairs)
    if len(chains.nodes) > max_nodes:
        return

    known_target = (
        known_pairs.to_pandas()
        .set_index(["variantid1", "variantid2"])["target"]
        .to_dict()
    )
    enriched_target = {}
    for var1, paths in nx.all_pairs_shortest_path(chains, cutoff=cutoff):
        for var2, path in paths.items():
            if var1 == var2:
                continue
            if (var1, var2) in known_target or (var2, var1) in known_target:
                continue
            if (var1, var2) in enriched_target or (var2, var1) in enriched_target:
                continue

            for path in nx.all_shortest_paths(chains, var1, var2):
                edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                targets = [chains.get_edge_data(*edge)["target"] for edge in edges]

                first = 0
                while first < len(targets) and targets[first] == 1:
                    first += 1
                if first == len(targets):
                    enriched_target[(var1, var2)] = 1
                    break

                last = len(targets) - 1
                while last > first and targets[last] == 1:
                    last -= 1
                if last == first:
                    enriched_target[(var1, var2)] = targets[last]
                    break

    if enriched_target:
        enriched = pd.Series(enriched_target).reset_index(drop=False)
        enriched.columns = ["variantid1", "variantid2", "target"]
        enriched["chain_id"] = known_pairs["chain_id"][0]
        enriched["enriched"] = True
        return (
            pl.from_pandas(enriched)
            .with_columns(
                pl.col(col).cast(dtype) for col, dtype in known_pairs.schema.items()
            )
            .select(known_pairs.columns)
        )


if __name__ == "__main__":
    from ozon_matching.andrey_solution.preprocessing import preprocess_pairs

    pairs = preprocess_pairs(pl.read_parquet("data/raw/train_pairs.parquet"))
    with_chains = enrich_by_chains(pairs, max_nodes=120, cutoff=None)
    with_chains.write_parquet("data/processed/enriched_train_pairs.pq")
