class CheckpointManager:

    def __init__(self,persistence_service):

        self.persistence = (
            persistence_service
        )

    def save_checkpoint(self):

        from src.runtime.context import (
            runtime_ctx
        )

        ctx = runtime_ctx.get()

        if not ctx:
            return

        self.persistence.save(ctx)

    def restore_checkpoint(self,trace_id):

        return self.persistence.restore(
            trace_id
        )

    def clear_checkpoint(self,trace_id):

        self.persistence.delete(
            trace_id
        )

    # 获取检查点的metadata数据
    def get_checkpoint_metadata(self,trace_id):
        return self.persistence.get_metadata(
            trace_id
        )