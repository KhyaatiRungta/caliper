"""Grader package. Importing it registers every built-in grader."""

from caliper.graders.registry import (  # noqa: F401
    GraderError,
    grade,
    grader,
    get_grader,
    normalise_spec,
    registered,
)
from caliper.graders import final_answer as _final_answer  # noqa: F401
from caliper.graders import trajectory as _trajectory  # noqa: F401
from caliper.graders import llm_judge as _llm_judge  # noqa: F401

__all__ = ["grade", "grader", "get_grader", "registered", "normalise_spec", "GraderError"]
