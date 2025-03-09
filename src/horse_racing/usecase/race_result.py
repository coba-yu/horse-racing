from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository


class RaceResultUsecase:
    def __init__(
        self,
        race_result_repository: RaceResultNetkeibaRepository,
    ) -> None:
        self.race_result_repository = race_result_repository

    def get(self, race_date: str, race_id: str) -> str:
        # Try to use local cache
        # TODO

        # Download from GCS
        # TODO

        # Download from netkeiba
        html = self.race_result_repository.download(race_date=race_date, race_id=race_id)

        # Upload to GCS
        self.race_result_repository.upload_to_storage(race_date=race_date, race_id=race_id)

        return html
