from src.runtime.persistence.checkpoint_record import CheckpointRecord


class RuntimeSnapshotStore:

    def __init__(self):

        self.snapshots = {}

    def save(self,trace_id, record: CheckpointRecord):

        # 加入checkpoint的生命周期管理 将原来的快照替换成CheckpointRecord类
        self.snapshots[trace_id] = record
        # self.snapshots[trace_id] = snapshot

    def load(self,trace_id) -> CheckpointRecord | None:
        return self.snapshots.get(trace_id)

    def delete(self,trace_id):

        self.snapshots.pop(trace_id,None)