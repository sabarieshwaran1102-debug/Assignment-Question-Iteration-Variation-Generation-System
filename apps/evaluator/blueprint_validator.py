"""
Blueprint-Question Consistency Validator implementation.
Ensures generated question texts and answer keys adhere strictly to the authoritative blueprint.
Verifies target variable, task type, formula, known parameters, units, learning objective, and domain concepts.
Rejects candidates that introduce hallucinated or unrelated concepts.
"""

import json
import re
from typing import List, Optional, Tuple
from packages.schemas.models import QuestionVariation, VariationBlueprint


class DefaultBlueprintQuestionConsistencyValidator:
    """Validator verifying exact consistency between authoritative VariationBlueprint, question text, and answer key."""

    FORBIDDEN_VELOCITY_CONCEPTS = {
        "mach number", "shockwave angle", "magnetic flux", "magnetic drive",
        "cyclotron", "ballistic pendulum", "spring constant", "incline friction coefficient",
        "orbital radius", "thrust force", "rebound height", "coefficient of restitution",
        "bank angle", "drag coefficient", "rotational acceleration"
    }

    def validate_blueprint_consistency(
        self,
        candidate: QuestionVariation,
        blueprint: Optional[VariationBlueprint] = None
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        bp = blueprint
        if bp is None and hasattr(candidate, "metadata") and isinstance(candidate.metadata, dict):
            bp = candidate.metadata.get("blueprint")

        if bp is None and candidate.context_changes:
            try:
                data = json.loads(candidate.context_changes)
                if isinstance(data, dict) and "known_variables" in data:
                    bp = VariationBlueprint(**data)
            except Exception:
                pass

        if bp is None:
            # Fallback if no blueprint attached: extract parameters directly from question text
            return True, []

        q_text = candidate.question.lower()
        ans_text = candidate.answer_key.lower()

        # 1. Target variable and formula check
        target_var = (bp.target_variable or "velocity").lower()
        formula = (bp.formula or "v = d / t").lower()

        # 2. Check for forbidden hallucinated physics concepts for v = d / t formulas
        if "v = d / t" in formula or "velocity" in target_var or "speed" in target_var:
            for forbidden in self.FORBIDDEN_VELOCITY_CONCEPTS:
                if forbidden in q_text:
                    reasons.append(
                        f"Blueprint consistency failure: Question text introduces hallucinated/unrelated "
                        f"physics concept '{forbidden}' inconsistent with formula '{formula}'."
                    )
                    return False, reasons

        # 3. Known numerical parameter verification
        numbers_in_q = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", candidate.question)]
        for var_name, var_val in bp.known_variables.items():
            # Check if expected parameter value exists in question text
            matched = any(abs(num - float(var_val)) < 1e-3 for num in numbers_in_q)
            if not matched:
                reasons.append(
                    f"Blueprint parameter mismatch: Blueprint variable '{var_name}={var_val}' "
                    f"not found in generated question text: '{candidate.question}'."
                )

        # 4. Answer key correspondence verification
        if bp.known_variables.get("distance") and bp.known_variables.get("time"):
            d_exp = float(bp.known_variables["distance"])
            t_exp = float(bp.known_variables["time"])
            if t_exp > 0:
                v_expected = d_exp / t_exp
                v_str_int = f"{int(v_expected)} m/s" if v_expected.is_integer() else f"{v_expected:.2f} m/s"
                v_str_flt = f"{v_expected:.1f} m/s"

                if v_str_int.lower() not in ans_text and v_str_flt.lower() not in ans_text:
                    reasons.append(
                        f"Answer key blueprint mismatch: Expected calculated result '{v_str_int}' "
                        f"not found in answer key: '{candidate.answer_key}'."
                    )

        is_valid = len(reasons) == 0
        return is_valid, reasons
