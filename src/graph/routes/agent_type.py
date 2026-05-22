from enum import Enum


class AgentType(str, Enum):

    RECOMMENDATION = "recommendation_agent"

    EXACT_SEARCH = "exact_agent"

    GENERAL_QA = "general_agent"

    FINISH = "FINISH"