from typer import Option, Typer
from ozon_matching.kopatych_solution.utils import (
    extract_category_levels,
    get_and_create_dir,
    log_cli,
    read_json,
    read_model,
    read_parquet,
    write_json,
    write_model,
    write_parquet,
)
import os
import polars as pl


cli = Typer()


@cli.command()
def dummy():
    pass


@cli.command()
@log_cli
def prepare_data(data_dir:str = Option(...), n_folds: int = Option(...)):
    data = pl.concat(
        [
            read_parquet(os.path.join(data_dir, f"cv_{n_folds + 1}", "train", "data.parquet")),
            read_parquet(os.path.join(data_dir, f"cv_{n_folds + 1}", "test", "data.parquet"))
        ]
    )
    data = data.unique(subset=['variantid'])
    data = extract_category_levels(data, [3,4], category_col='categories')

    data = data.join(
        data.select(pl.col('category_level_3')).unique().with_row_count(name='category_level_3_id'),
        on=['category_level_3']
    )
    data = data.join(
        data.select(pl.col('category_level_4')).unique().with_row_count(name='category_level_4_id'),
        on=['category_level_4']
    )
    data = data.drop(['category_level_3', 'category_level_4'])

    write_parquet(
        data,
        os.path.join(os.path.join(data_dir, f"cv_{n_folds + 1}", "union_data.parquet"))
    )

if __name__ == "__main__":
    cli()
