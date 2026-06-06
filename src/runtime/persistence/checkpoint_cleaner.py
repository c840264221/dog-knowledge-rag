from datetime import datetime, timedelta


class CheckpointCleaner:

    def clean(self, store):

        now = datetime.now()

        for trace_id, record in list(
            store.items()
        ):

            age = now - record.metadata.updated_at

            if age > timedelta(hours=24):

                record.metadata.status = (
                    "expired"
                )

                del store[trace_id]