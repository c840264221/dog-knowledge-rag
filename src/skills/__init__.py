"""Dog Agent Framework 的 Skill（技能）基础公开接口。"""

from src.skills.default_catalog import (
    DOG_TRAINING_PLAN_SKILL,
    build_default_skill_input_extractor,
    build_default_skill_registry,
    build_default_skill_runtime,
)
from src.skills.input_checker import SkillInputChecker
from src.skills.input_extractor import (
    SkillExtractionRule,
    SkillInputExtractor,
)
from src.skills.loader import DisabledSkillError, SkillLoader
from src.skills.registry import (
    DuplicateSkillError,
    SkillNotFoundError,
    SkillRegistry,
)
from src.skills.runtime import SkillRuntime
from src.skills.schemas import (
    SkillCatalogItem,
    SkillDefinition,
    SkillInputCheckResult,
    SkillInputExtractionResult,
    SkillInputRequirement,
    SkillRuntimeResult,
    SkillSelectionResult,
)
from src.skills.selector import SkillSelector
from src.skills.state_adapter import build_skill_state_update


__all__ = [
    "DisabledSkillError",
    "DOG_TRAINING_PLAN_SKILL",
    "DuplicateSkillError",
    "SkillCatalogItem",
    "SkillDefinition",
    "SkillInputCheckResult",
    "SkillInputChecker",
    "SkillInputExtractionResult",
    "SkillInputExtractor",
    "SkillInputRequirement",
    "SkillLoader",
    "SkillNotFoundError",
    "SkillRegistry",
    "SkillRuntime",
    "SkillRuntimeResult",
    "SkillSelectionResult",
    "SkillSelector",
    "SkillExtractionRule",
    "build_default_skill_input_extractor",
    "build_default_skill_registry",
    "build_default_skill_runtime",
    "build_skill_state_update",
]
