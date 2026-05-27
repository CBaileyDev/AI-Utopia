package dev.aiutopia.mod.bridge.skill;

/** Per-tick outcome of a skill execution attempt.
 *
 *  RUNNING        — skill still executing across more ticks; do not emit completion event yet
 *  COMPLETED      — skill finished successfully; emit SkillCompletionEvent
 *  FAILED_TIMEOUT — skill ran out of allotted ticks
 *  IMMEDIATE_FAILURE — preconditions not met or world rejected the action
 *                      (used by ExploitDetector's MEANINGLESS_TOOL_CALL rule)
 *  ABORTED        — skill was preempted by a new dispatch with same agentId
 */
public enum SkillResult {
    RUNNING,
    COMPLETED,
    FAILED_TIMEOUT,
    IMMEDIATE_FAILURE,
    ABORTED;
}
