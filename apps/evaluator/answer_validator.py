"""
Default Answer Validator implementation.
Validates answer key formatting, completeness, and numerical consistency.
"""

import re
from typing import List, Optional
from apps.evaluator.interfaces import AnswerValidator


class DefaultAnswerValidator(AnswerValidator):
    """Validates answer keys for completeness, validity, and numerical correctness."""

    def validate_answer(self, question_text: str, answer_key: str) -> bool:
        if not answer_key or not answer_key.strip():
            return False
            
        if len(answer_key.strip()) < 5:
            return False

        q_lower = question_text.lower()
        ans_nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", answer_key)]

        # Check velocity calculation: v = distance / time with strict word boundaries
        has_velocity = bool(re.search(r"\b(velocity|speed)\b", q_lower))
        has_distance = bool(re.search(r"\b(m|meters|meter|km|kilometers|distance)\b", q_lower))
        has_time = bool(re.search(r"\b(s|sec|seconds|second|hrs|hours)\b", q_lower))

        if has_velocity and has_distance and has_time:
            q_nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", question_text)]
            if len(q_nums) >= 2 and q_nums[1] > 0:
                dist = q_nums[0]
                time_val = q_nums[1]
                expected_v = dist / time_val

                if ans_nums:
                    matches = [a for a in ans_nums if abs(a - expected_v) <= (0.05 * expected_v + 0.1)]
                    if not matches:
                        return False

        # Check force calculation: F = m * a with strict word boundaries
        has_force = bool(re.search(r"\b(force|net force)\b", q_lower))
        has_mass = bool(re.search(r"\b(mass|kg|kilograms)\b", q_lower))
        has_accel = bool(re.search(r"\b(acceleration|m/s2)\b", q_lower))

        if has_force and has_mass and has_accel:
            q_nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", question_text)]
            if len(q_nums) >= 2:
                m = q_nums[0]
                a = q_nums[1]
                expected_f = m * a
                if ans_nums:
                    matches = [ans_val for ans_val in ans_nums if abs(ans_val - expected_f) <= (0.05 * expected_f + 0.1)]
                    if not matches:
                        return False

        return True
