from src.logger import logger


class ReportPrinter:

    @staticmethod
    def print(report):
        """
        打印 Runtime Report。

        功能：
            将 RuntimeReport 以控制台友好的摘要形式打印出来。

        参数：
            report:
                RuntimeReport 对象。

        返回值：
            None。
        """

        timeline_count = len(
            report.timeline
            or []
        )

        logger.info(
            "\n"
            "================ Runtime Report ================\n"
            f"Trace: {report.trace_id}\n"
            f"Agent: {report.current_agent}\n"
            f"Tools: {report.tool_count}\n"
            f"Errors: {report.error_count}\n"
            f"Tool Latency: {report.tool_latency:.3f}s\n"
            f"Timeline Events: {timeline_count}\n"
            "================================================"
        )