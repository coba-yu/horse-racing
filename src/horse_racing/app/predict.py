from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import polars as pl

from horse_racing.app.train import encode_sex, encode_rotate, encode_field_type, encode_weather, encode_field_condition
from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.logging import logger
from horse_racing.domain.race import RaceInfo
from horse_racing.usecase.race_card import RaceCardUsecase


@dataclass
class Args:
    dt: str = ""
    race_id: str = ""
    version: str = ""
    lgb_objective: str = ""


def preprocess(
    raw_df: pl.DataFrame,
    weight_diff_avg_df: pl.DataFrame,
    horse_id_df: pl.DataFrame,
    jockey_id_df: pl.DataFrame,
    trainer_id_df: pl.DataFrame,
    horse_target_encoded_df: pl.DataFrame,
    jockey_target_encoded_df: pl.DataFrame,
    trainer_target_encoded_df: pl.DataFrame,
    race_info: RaceInfo,
    features: list[str],
) -> pl.DataFrame:
    processed_df = raw_df.filter(~pl.col("人気").cast(pl.String).str.contains("-"))

    select_columns = [
        pl.col("枠").cast(pl.Int32).alias("group"),
        pl.col("馬 番").cast(pl.Int32).alias("number"),
        pl.col("性齢").alias("sex"),
        pl.col("性齢").str.extract(r"(\d+)").cast(pl.Int32).alias("age"),
        pl.col("斤量").cast(pl.Float32).alias("weight"),
        # race
        pl.lit(race_info.race_number).cast(pl.Int32).alias("race_number"),
        pl.lit(race_info.start_hour).cast(pl.Int32).alias("start_hour"),
        pl.lit(race_info.distance).str.extract(r"(\d+)").cast(pl.Int32).alias("distance"),
        pl.lit(race_info.distance).alias("rotate"),
        pl.lit(race_info.distance).alias("field_type"),
        pl.lit(race_info.weather).alias("weather"),
        pl.lit(race_info.field_condition).alias("field_condition"),
        # flesh
        pl.col("人気").cast(pl.Int32).alias("popular"),
        pl.col("オッズ 更新").cast(pl.Float32).alias("odds"),
        # target encoded
        pl.col("馬名").alias("horse_name"),
        pl.col("騎手").alias("jockey_name"),
        pl.col("厩舎").alias("trainer_name"),
    ]
    if "weight_diff" in features or "weight_diff_dev" in features:
        select_columns.append(
            pl.col("馬体重 (増減)").str.extract(r"\(([-\+\d]+)\)").cast(pl.Int32).alias("weight_diff")
        )
    processed_df = processed_df.select(select_columns)

    processed_df = encode_sex(processed_df)
    processed_df = encode_rotate(processed_df)
    processed_df = encode_field_type(processed_df)
    processed_df = encode_weather(processed_df)
    processed_df = encode_field_condition(processed_df)

    processed_df = processed_df.join(horse_id_df, on="horse_name", how="left")
    processed_df = processed_df.join(jockey_id_df, on="jockey_name", how="left")
    processed_df = processed_df.join(trainer_id_df, on="trainer_name", how="left")
    processed_df = processed_df.join(weight_diff_avg_df, on="horse_id", how="left")
    processed_df = processed_df.join(horse_target_encoded_df, on="horse_id", how="left")
    processed_df = processed_df.join(jockey_target_encoded_df, on="jockey_id", how="left")
    processed_df = processed_df.join(trainer_target_encoded_df, on="trainer_id", how="left")

    if "weight_diff_dev" in features:
        processed_df = processed_df.join(weight_diff_avg_df, on="horse_id", how="left")
        processed_df = processed_df.with_columns(
            (pl.col("weight_diff") - pl.col("weight_diff_avg")).alias("weight_diff_dev")
        )
        processed_df = processed_df.drop("weight_diff_avg")

    return processed_df.select([c for c in processed_df.columns if c in features])


def lgb_predict(model: lgb.Booster, data_df: pl.DataFrame) -> pl.DataFrame:
    predict_ignore_columns = [
        "horse_name",
    ]
    pred_feature_df = data_df.select([c for c in data_df.columns if c not in predict_ignore_columns])
    y_pred = model.predict(pred_feature_df.to_pandas())
    return data_df.with_columns(pl.Series("lgb_pred", y_pred))


def main() -> None:
    # setup
    parser = ArgumentParser()
    parser.add_argument("--dt", type=str, required=True)
    parser.add_argument("--race_id", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--lgb-objective", type=str, required=True)
    args = parser.parse_args(namespace=Args())

    root_dir = Path(__file__).parent.parent.parent.parent
    lgb_model_dir = root_dir / "model" / args.lgb_objective / args.version
    processed_data_dir = root_dir / "data" / "processed" / args.version
    processed_race_data_dir = (
        root_dir / "data" / "processed" / args.dt / args.race_id / args.lgb_objective / args.version
    )
    predict_dir = root_dir / "data" / "predicted" / args.dt / args.race_id / args.lgb_objective / args.version

    chrome_driver = ChromeDriver()

    # get race info
    race_card_usecase = RaceCardUsecase(driver=chrome_driver, cache_dir=processed_race_data_dir / "cache" / "race_card")
    race_info = race_card_usecase.get_race_info(race_id=args.race_id)
    logger.info(f"race_info: {race_info}")

    raw_race_df = race_card_usecase.get_race_card(race_id=args.race_id)
    logger.info(raw_race_df)

    # load model and past data
    lgb_model = lgb.Booster(model_file=lgb_model_dir / "lgb_model.txt")
    logger.info(lgb_model)

    horse_id_df = pl.read_parquet(processed_data_dir / "horse_id.parquet")
    jockey_id_df = pl.read_parquet(processed_data_dir / "jockey_id.parquet")
    trainer_id_df = pl.read_parquet(processed_data_dir / "trainer_id.parquet")
    weight_diff_avg_df = pl.read_parquet(processed_data_dir / "weight_diff_avg.parquet")
    horse_target_encoded_df = pl.read_parquet(processed_data_dir / "horse_target_encoded.parquet")
    jockey_target_encoded_df = pl.read_parquet(processed_data_dir / "jockey_target_encoded.parquet")
    trainer_target_encoded_df = pl.read_parquet(processed_data_dir / "trainer_target_encoded.parquet")

    # preprocess
    # TODO: load yaml
    features = [
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
    data_df = preprocess(
        raw_df=raw_race_df,
        weight_diff_avg_df=weight_diff_avg_df,
        horse_id_df=horse_id_df,
        jockey_id_df=jockey_id_df,
        trainer_id_df=trainer_id_df,
        horse_target_encoded_df=horse_target_encoded_df,
        jockey_target_encoded_df=jockey_target_encoded_df,
        trainer_target_encoded_df=trainer_target_encoded_df,
        race_info=race_info,
        features=features,
    )
    logger.info(data_df)

    # predict
    predict_df = lgb_predict(model=lgb_model, data_df=data_df)
    logger.info(predict_df)
    predict_dir.mkdir(parents=True, exist_ok=True)

    predict_df = predict_df.join(
        raw_race_df.select(pl.col("馬 番").alias("number"), pl.col("馬名").alias("horse_name")),
        on="number",
        how="left",
    )
    predict_df.write_csv(predict_dir / "lgb.tsv", separator="\t")


if __name__ == "__main__":
    main()
