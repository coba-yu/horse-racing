import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import polars as pl

from horse_racing.core.logging import logger

HORSE_NAME_RAW_COLUMN = "馬名"
JOCKEY_NAME_RAW_COLUMN = "騎手"
TRAINER_NAME_RAW_COLUMN = "厩舎"


@dataclass
class Config:
    latest_train_start_date: str = "20220701"
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
    horse_id_df: pl.DataFrame
    jockey_id_df: pl.DataFrame
    trainer_id_df: pl.DataFrame


def _get_id_df(df: pl.DataFrame, raw_name_column: str, name_column: str, id_column: str) -> pl.DataFrame:
    return df.select(pl.col(raw_name_column).alias(name_column), pl.col(id_column)).unique()


def load_data(data_dir: Path) -> RawData:
    result_df = pl.read_parquet(data_dir / "parquet" / "race_results")
    return RawData(
        result_df=result_df,
        horse_id_df=_get_id_df(
            result_df,
            raw_name_column=HORSE_NAME_RAW_COLUMN,
            name_column="horse_name",
            id_column="horse_id",
        ),
        jockey_id_df=_get_id_df(
            result_df,
            raw_name_column=JOCKEY_NAME_RAW_COLUMN,
            name_column="jockey_name",
            id_column="jockey_id",
        ),
        trainer_id_df=_get_id_df(
            result_df,
            raw_name_column=TRAINER_NAME_RAW_COLUMN,
            name_column="trainer_name",
            id_column="trainer_id",
        ),
    )


def _label_encode(df: pl.DataFrame, column: str, label_dict: dict[str, int]) -> pl.DataFrame:
    return df.with_columns(
        pl.col(column)
        .str.extract(rf'({"|".join(list(label_dict))})')
        .replace_strict(label_dict, default=-1)
        .cast(pl.Int8)
    )


def encode_sex(df: pl.DataFrame) -> pl.DataFrame:
    return _label_encode(df, column="sex", label_dict={"牡": 0, "牝": 1, "セ": 2})


def encode_rotate(df: pl.DataFrame) -> pl.DataFrame:
    return _label_encode(df, column="rotate", label_dict={"左": 0, "右": 1})


def encode_field_type(df: pl.DataFrame) -> pl.DataFrame:
    return _label_encode(df, column="field_type", label_dict={"芝": 0, "ダ": 1})


def encode_weather(df: pl.DataFrame) -> pl.DataFrame:
    return _label_encode(df, column="weather", label_dict={"晴": 0, "曇": 1, "雨": 2})


def encode_field_condition(df: pl.DataFrame) -> pl.DataFrame:
    return _label_encode(df, column="field_condition", label_dict={"良": 0})


def _remove_debut_race(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(~pl.col("race_name").str.contains("新馬"))
    return df


def preprocess_train_data(
    data: RawData,
    config: Config,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame | None,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    select_cols = [
        pl.col("枠").cast(pl.Int32).alias("group"),
        pl.col("horse_number").cast(pl.Int32).alias("number"),
        pl.col("性齢").alias("sex"),
        pl.col("性齢").str.extract(r"(\d+)").cast(pl.Int32).alias("age"),
        pl.col("斤量").cast(pl.Float32).alias("weight"),
        # race
        pl.col("race_number").str.extract(r"(\d+)").cast(pl.Int32).alias("race_number"),
        pl.col("start_at").str.extract(r"^(\d+)").cast(pl.Int32).alias("start_hour"),
        pl.col("distance").str.extract(r"(\d+)").cast(pl.Int32).alias("distance"),
        pl.col("distance").alias("rotate"),
        pl.col("distance").alias("field_type"),
        pl.col("weather"),
        pl.col("field_condition"),
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

    # label encoding
    processed_df = encode_sex(processed_df)
    processed_df = encode_rotate(processed_df)
    processed_df = encode_field_type(processed_df)
    processed_df = encode_weather(processed_df)
    processed_df = encode_field_condition(processed_df)

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
    processed_data_dir = Path(root_dir / "data" / "processed")
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

    out_data_dir = processed_data_dir / now
    os.makedirs(out_data_dir, exist_ok=True)
    raw_data.horse_id_df.write_parquet(out_data_dir / "horse_id.parquet")
    raw_data.jockey_id_df.write_parquet(out_data_dir / "jockey_id.parquet")
    raw_data.trainer_id_df.write_parquet(out_data_dir / "trainer_id.parquet")
    if weight_diff_avg_df is not None:
        weight_diff_avg_df.write_parquet(out_data_dir / "weight_diff_avg.parquet")
    horse_target_encoded_df.write_parquet(out_data_dir / "horse_target_encoded.parquet")
    jockey_target_encoded_df.write_parquet(out_data_dir / "jockey_target_encoded.parquet")
    trainer_target_encoded_df.write_parquet(out_data_dir / "trainer_target_encoded.parquet")


if __name__ == "__main__":
    main()
