from dataclasses import dataclass

from src.runtime.persistence.runtime_snapshot import (
    RuntimeSnapshot
)

from src.runtime.persistence.checkpoint_metadata import (
    CheckpointMetadata
)


@dataclass
class CheckpointRecord:

    snapshot: RuntimeSnapshot

    metadata: CheckpointMetadata