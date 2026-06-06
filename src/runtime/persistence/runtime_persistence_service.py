from src.runtime.persistence.runtime_serializer import (
    RuntimeSerializer
)
from src.runtime.persistence.runtime_snapshot import RuntimeSnapshot

from src.runtime.persistence.runtime_snapshot_store import (
    RuntimeSnapshotStore
)

from src.runtime.context.runtime_context import RuntimeContext

from datetime import datetime

from src.runtime.persistence.checkpoint_record import (
    CheckpointRecord
)

from src.runtime.persistence.checkpoint_metadata import (
    CheckpointMetadata
)

from src.runtime.persistence.checkpoint_status import (
    CheckpointStatus
)


class RuntimePersistenceService:

    def __init__(self):

        self.store = RuntimeSnapshotStore()

    def save(self, ctx):

        snapshot = RuntimeSerializer.to_snapshot(
            ctx
        )

        old_record = self.store.load(
            ctx.trace_id
        )

        if old_record:

            metadata = old_record.metadata

            metadata.updated_at = datetime.now()

            metadata.interrupt_count += 1

        else:

            metadata = CheckpointMetadata(

                trace_id=ctx.trace_id,

                status=CheckpointStatus.ACTIVE,

                created_at=datetime.now(),

                updated_at=datetime.now(),

                interrupt_count=0
            )

        record = CheckpointRecord(

            snapshot=snapshot,

            metadata=metadata
        )

        self.store.save(
            ctx.trace_id,
            record
        )

    def restore(self, trace_id):

        record = self.store.load(
            trace_id
        )

        if not record:
            return None

        record.metadata.status = (
            CheckpointStatus.RESUMED
        )

        record.metadata.updated_at = (
            datetime.now()
        )

        return RuntimeSerializer.from_snapshot(
            record.snapshot
        )

    def delete(self, trace_id):

        self.store.delete(
            trace_id
        )

    # 查看record的metadata
    def get_metadata(self, trace_id):

        record = self.store.load(
            trace_id
        )

        if not record:
            return None

        return record.metadata