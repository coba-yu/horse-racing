import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import polars as pl

from horse_racing.core.logging import logger


@dataclass
class Config:
    latest_train_start_date: str = "20220801"
    latest_train_end_date: str = "20240931"
    features: list[str] = field(
        default_factory=lambda: [
            "group",
            "number",
            "sex",
            "age",
            "weight",
            "weight_diff_dev",
            "odds",
            # race info
            "race_number",
            "start_hour",
            "distance",
            "rotate",
            "field_type",
            "weather",
            "field_condition",
            # target encoded
            "horse_id_target_encoded",
            "jockey_id_target_encoded",
            "trainer_id_target_encoded",
        ]
    )
    lgb_objective: str = "binary"


@dataclass
class RawData:
    result_df: pl.DataFrame
    horse_name_to_id: dict[str, str]
    horse_id_to_name: dict[str, str]
    jockey_name_to_id: dict[str, str]
    trainer_name_to_id: dict[str, str]


def load_data(data_dir: Path) -> RawData:
    result_df = pl.read_parquet(data_dir / "parquet" / "race_results")

    horse_name_to_id = {name: str(horse_id) for name, horse_id in result_df[["馬名", "horse_id"]].to_numpy()}
    horse_id_to_name = {horse_id: str(name) for name, horse_id in result_df[["馬名", "horse_id"]].to_numpy()}

    jockey_name_to_id = {name: str(jockey_id) for name, jockey_id in result_df[["騎手", "jockey_id"]].to_numpy()}
    trainer_name_to_id = {name: str(trainer_id) for name, trainer_id in result_df[["厩舎", "trainer_id"]].to_numpy()}

    return RawData(
        result_df=result_df,
        horse_name_to_id=horse_name_to_id,
        horse_id_to_name=horse_id_to_name,
        jockey_name_to_id=jockey_name_to_id,
        trainer_name_to_id=trainer_name_to_id,
    )


def _remove_debut_race(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(~pl.col("race_name").str.contains("新馬"))
    return df


def preprocess_train_data(data: RawData, config: Config) -> pl.DataFrame:
    select_cols = [
        pl.col("枠").cast(pl.Int32).alias("group"),
        pl.col("horse_number").cast(pl.Int32).alias("number"),
        pl.col("性齢")
        .str.extract(r"(牡|牝|セ)")
        .replace_strict({"牡": 0, "牝": 1, "セ": 2}, default=-1)
        .cast(pl.Int8)
        .alias("sex"),
        pl.col("性齢").str.extract(r"(\d+)").cast(pl.Int32).alias("age"),
        pl.col("斤量").cast(pl.Float32).alias("weight"),
        # race
        pl.col("race_number").str.extract(r"(\d+)").cast(pl.Int32).alias("race_number"),
        pl.col("start_at").str.extract(r"^(\d+)").cast(pl.Int32).alias("start_hour"),
        pl.col("distance").str.extract(r"(\d+)").cast(pl.Int32).alias("distance"),
        pl.col("distance")
        .str.extract(r"(左|右)")
        .replace_strict({"左": 0, "右": 1}, default=-1)
        .cast(pl.Int8)
        .alias("rotate"),
        pl.col("distance").str.extract(r"(芝|ダ)").replace_strict({"芝": 0, "ダ": 1}, default=-1).alias("field_type"),
        pl.col("weather")
        .str.extract(r"(晴|曇|雨)")
        .replace_strict({"晴": 0, "曇": 1, "雨": 2}, default=-1)
        .cast(pl.Int8),
        pl.col("field_condition").str.extract(r"(良|悪)").replace_strict({"良": 0, "悪": 1}, default=-1).cast(pl.Int8),
        # fresh
        pl.col("人 気").cast(pl.Int32).alias("popular"),
        pl.col("単勝 オッズ").cast(pl.Float32).alias("odds"),
        # target encoded
        pl.col("horse_id"),
        pl.col("jockey_id"),
        pl.col("trainer_id"),
        # ignore
        pl.col("着 順").cast(pl.Int8).alias("rank"),
        pl.col("race_id").cast(pl.Utf8),
        pl.col("race_name"),
        pl.col("race_date"),
    ]

    features = config.features
    if "weight_diff" in features or "weight_diff_dev" in features:
        select_cols.append(pl.col("馬体重 (増減)").str.extract(r"\(([-\+\d]+)\)").cast(pl.Int32).alias("weight_diff"))
    processed_df = data.result_df.filter(~pl.col("着 順").is_in({"中止", "除外", "取消"})).select(select_cols)

    # target encoding
    horse_target_encoded_df = processed_df.group_by("horse_id").agg(
        pl.col("rank").mean().alias("horse_id_target_encoded")
    )
    jockey_target_encoded_df = processed_df.group_by("jockey_id").agg(
        pl.col("rank").mean().alias("jockey_id_target_encoded")
    )
    trainer_target_encoded_df = processed_df.group_by("trainer_id").agg(
        pl.col("rank").mean().alias("trainer_id_target_encoded")
    )
    processed_df = processed_df.join(horse_target_encoded_df, on="horse_id", how="left")
    processed_df = processed_df.join(jockey_target_encoded_df, on="jockey_id", how="left")
    processed_df = processed_df.join(trainer_target_encoded_df, on="trainer_id", how="left")

    processed_df = _remove_debut_race(processed_df)
    processed_df = processed_df.drop("race_name")

    if "weight_diff_dev" not in features:
        return processed_df, None, horse_target_encoded_df, jockey_target_encoded_df, trainer_target_encoded_df

    weight_diff_avg_df = processed_df.group_by("horse_id").agg(pl.mean("weight_diff").alias("weight_diff_avg"))
    processed_df = processed_df.join(weight_diff_avg_df, on="horse_id", how="left")
    processed_df = processed_df.with_columns(
        (pl.col("weight_diff") - pl.col("weight_diff_avg")).alias("weight_diff_dev")
    )
    processed_df = processed_df.drop("weight_diff_avg")
    return (
        processed_df,
        weight_diff_avg_df,
        horse_target_encoded_df,
        jockey_target_encoded_df,
        trainer_target_encoded_df,
    )


def split_train_data(
    data_df: pl.DataFrame,
    config: Config,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # split train and valid
    train_df = data_df.filter(
        (pl.col("race_date") >= config.latest_train_start_date) & (pl.col("race_date") <= config.latest_train_end_date)
    )
    valid_df = data_df.filter(pl.col("race_date") > config.latest_train_end_date)

    return train_df, valid_df


def train_lgb(
    objective: str,
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    config: Config,
) -> lgb.Booster:
    # split features and target
    X_train_df = train_df.select(config.features)
    X_valid_df = valid_df.select(config.features)

    if objective == "binary":
        y_train = (train_df["rank"] == 1).cast(pl.Int8).to_numpy()
        y_valid = (valid_df["rank"] == 1).cast(pl.Int8).to_numpy()
    elif objective == "regression":
        y_train = train_df["rank"].cast(pl.Float32).to_numpy()
        y_valid = valid_df["rank"].cast(pl.Float32).to_numpy()
    else:
        raise ValueError(f"objective: {objective} is not supported")

    lgb_train = lgb.Dataset(X_train_df.to_pandas(), label=y_train)
    lgb_eval = lgb.Dataset(X_valid_df.to_pandas(), label=y_valid)

    params = {
        "objective": objective,
    }
    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_names=["train", "valid"],
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=10,
            ),
        ],
    )
    return lgb_model


def main() -> None:
    # setup
    config = Config()
    logger.info(f"config: {config}")

    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = Path(root_dir / "data" / "cache")
    model_dir = Path(root_dir / "model")

    # prepare train data
    raw_data = load_data(data_dir=data_dir)
    (
        train_data_df,
        weight_diff_avg_df,
        horse_target_encoded_df,
        jockey_target_encoded_df,
        trainer_target_encoded_df,
    ) = preprocess_train_data(data=raw_data, config=config)
    logger.info(f"features: {train_data_df.columns}")

    train_df, valid_df = split_train_data(
        data_df=train_data_df,
        config=config,
    )
    logger.info(f"train: {train_df.shape}")
    logger.info(f"valid: {valid_df.shape}")

    # training
    lgb_model = train_lgb(
        objective=config.lgb_objective,
        train_df=train_df,
        valid_df=valid_df,
        config=config,
    )

    # save
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    lgb_model_dir = model_dir / config.lgb_objective / now
    os.makedirs(lgb_model_dir, exist_ok=True)
    lgb_model.save_model(lgb_model_dir / "lgb_model.txt")

    out_data_dir = data_dir / now
    os.makedirs(out_data_dir, exist_ok=True)
    weight_diff_avg_df.write_parquet(out_data_dir / "weight_diff_avg.parquet")
    horse_target_encoded_df.write_parquet(out_data_dir / "horse_target_encoded.parquet")
    jockey_target_encoded_df.write_parquet(out_data_dir / "jockey_target_encoded.parquet")
    trainer_target_encoded_df.write_parquet(out_data_dir / "trainer_target_encoded.parquet")


if __name__ == "__main__":
    main()
