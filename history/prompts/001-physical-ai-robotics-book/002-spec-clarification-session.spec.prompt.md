---
id: 002
title: Spec Clarification Robot and LLM Choices
stage: spec
date: 2025-12-10
surface: agent
model: claude-opus-4-5-20251101
feature: 001-physical-ai-robotics-book
branch: 001-physical-ai-robotics-book
user: user
command: /sp.clarify
labels: ["clarification", "robot-models", "llm-integration", "turtlebot3", "isaac-humanoid", "ollama"]
links:
  spec: specs/001-physical-ai-robotics-book/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-physical-ai-robotics-book/spec.md
tests:
  - Clarification integration validation PASS
---

## Prompt

/sp.clarify (no additional arguments)

Ran structured ambiguity scan on Physical AI & Humanoid Robotics Book specification to identify and resolve missing decision points before planning phase.

## Response snapshot

Performed coverage analysis across 10 taxonomy categories. Identified 3 material clarifications needed:

**Questions Asked & Answered:**

1. **Robot model for tutorials**
   - Q: Which robot model(s) should be used for tutorials?
   - A: TurtleBot3 for Weeks 1-6, transition to humanoid for Weeks 7+
   - Impact: Updated FR-004 to specify TurtleBot3 as reference platform

2. **Humanoid platform selection**
   - Q: Which humanoid platform for Weeks 7-13?
   - A: NVIDIA Humanoid assets from Isaac Sim
   - Impact: Updated FR-006 to specify NVIDIA Humanoid assets

3. **LLM integration approach**
   - Q: LLM integration approach for VLA chapters?
   - A: Local-first using Ollama with Llama 3/Mistral as primary; cloud APIs optional
   - Impact: Updated FR-007 and Assumptions section

**Sections Updated:**
- Added `## Clarifications` section with `### Session 2025-12-10`
- FR-004: Added TurtleBot3 reference
- FR-006: Added NVIDIA Humanoid assets reference
- FR-007: Added Ollama/Llama 3 as primary LLM approach
- Assumptions: Clarified LLM access approach

## Outcome

- ✅ Impact: 3 key architectural decisions resolved before planning phase
- 🧪 Tests: All clarifications integrated and validated
- 📁 Files: 1 file updated (spec.md)
- 🔁 Next prompts: `/sp.plan` to proceed with implementation planning
- 🧠 Reflection: Clarifications reduce downstream rework risk; robot model and LLM choices now explicit

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Clarification validation PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Validate choices against Isaac Sim documentation during planning
