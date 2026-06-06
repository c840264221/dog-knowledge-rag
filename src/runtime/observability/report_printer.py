from src.logger import logger


class ReportPrinter:

    @staticmethod
    def print(report):

        logger.info(
            f"""
================ Runtime Report ================

Trace:
{report.trace_id}

Agent:
{report.current_agent}

Tools:
{report.tool_count}

Errors:
{report.error_count}

Tool Latency:
{report.tool_latency:.3f}s

Timeline:
{report.timeline}

================================================
"""
        )