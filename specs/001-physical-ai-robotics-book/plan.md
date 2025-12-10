# Implementation Plan: Physical AI & Humanoid Robotics Book

**Branch**: `001-physical-ai-robotics-book` | **Date**: 2025-12-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-physical-ai-robotics-book/spec.md`

## Summary

A 13-week hands-on book teaching Physical AI from ROS 2 fundamentals through VLA-powered humanoid robotics. Uses TurtleBot3 for foundational modules (Weeks 1-6) and NVIDIA Humanoid assets in Isaac Sim for advanced content (Weeks 7-13). Culminates in a capstone autonomous humanoid executing voice-to-action tasks. Deployed on Docusaurus/GitHub Pages with research-while-writing workflow ensuring all content is verified against primary documentation.

## Technical Context

**Language/Version**: Markdown/MDX for content; Python 3.10+ for code examples
**Primary Dependencies**: Docusaurus 3.x, ROS 2 Humble, Gazebo Harmonic, Isaac Sim 2023.1+, Ollama
**Storage**: Git-based (static site); no database required
**Testing**: Docusaurus build validation, code example execution, link checking, readability scoring
**Target Platform**: GitHub Pages (static site); Ubuntu 22.04 (learner environment)
**Project Type**: Documentation/Educational content (Docusaurus site)
**Performance Goals**: Page load <3s, build time <5 minutes, 100% link validity
**Constraints**: Flesch-Kincaid grade 9-11, APA citations, zero build warnings
**Scale/Scope**: 13 modules, ~50 chapters, 50+ code examples, 4 hours/module

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Technical Accuracy | ✅ PASS | Primary sources defined (ROS 2, Gazebo, Isaac docs); APA citation workflow established |
| II. Reproducibility | ✅ PASS | Version pinning (Humble, Harmonic, Isaac 2023.1+); tested environment defined |
| III. Clarity for Learners | ✅ PASS | FK 9-11 target; progressive module structure; prerequisites per module |
| IV. Source Rigor | ✅ PASS | APA format; bibliography per chapter; URL archiving planned |
| V. Practical Focus | ✅ PASS | Hands-on exercises every module; theory tied to implementation |
| VI. Vendor Neutrality | ✅ PASS | Open-source primary (ROS 2, Gazebo); Isaac explained as required tool |

**Gate Status**: ✅ PASS - All constitution principles satisfied

## Project Structure

### Documentation (this feature)

```text
specs/001-physical-ai-robotics-book/
├── plan.md              # This file
├── research.md          # Technology decisions and rationale
├── data-model.md        # Content entity definitions
├── quickstart.md        # Author setup guide
└── tasks.md             # Implementation tasks (created by /sp.tasks)
```

### Source Code (repository root)

```text
# Docusaurus Site Structure
docs/
├── intro.md                    # Book introduction
├── week-01/                    # Module 1: Physical AI Foundations
│   ├── index.md               # Module overview
│   ├── what-is-physical-ai.md
│   └── exercises/
├── week-02/                    # Module 2: ROS 2 Fundamentals I
│   ├── index.md
│   ├── ros2-nodes.md
│   ├── ros2-topics.md
│   └── exercises/
├── week-03/                    # Module 3: ROS 2 Fundamentals II
├── week-04/                    # Module 4: Gazebo Simulation I
├── week-05/                    # Module 5: Gazebo Simulation II
├── week-06/                    # Module 6: Navigation & SLAM
├── week-07/                    # Module 7: Isaac Sim Introduction
├── week-08/                    # Module 8: Synthetic Data Generation
├── week-09/                    # Module 9: Isaac ROS Perception
├── week-10/                    # Module 10: Conversational AI I
├── week-11/                    # Module 11: Conversational AI II
├── week-12/                    # Module 12: VLA Integration
├── week-13/                    # Module 13: Capstone
├── glossary.md
└── bibliography.md

src/
├── components/                 # Custom MDX components
│   ├── Terminal.js            # Styled terminal output
│   ├── ROSGraph.js            # ROS 2 node visualization
│   ├── HardwareReq.js         # Hardware requirement callout
│   └── Exercise.js            # Exercise block component
├── css/
│   └── custom.css
└── pages/
    └── index.js               # Landing page

static/
├── diagrams/                   # Diagram source files
│   ├── *.drawio
│   └── *.mermaid
├── code/                       # Companion code examples
│   ├── week-02/
│   ├── week-03/
│   ...
│   └── week-13/
└── img/

# Configuration
docusaurus.config.js
sidebars.js
package.json
```

**Structure Decision**: Docusaurus documentation site with 13 weekly module directories, custom MDX components for robotics content, and companion code repository in `static/code/`.

## High-Level Architecture

### Learning Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LEARNING PROGRESSION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FOUNDATION          SIMULATION           PERCEPTION          INTEGRATION   │
│  (Weeks 1-3)         (Weeks 4-6)          (Weeks 7-9)        (Weeks 10-13) │
│                                                                              │
│  ┌──────────┐       ┌──────────┐        ┌──────────┐        ┌──────────┐   │
│  │Physical  │──────▶│ Gazebo   │───────▶│ Isaac    │───────▶│   VLA    │   │
│  │AI + ROS 2│       │ + Nav2   │        │ Sim      │        │ Pipeline │   │
│  │Fundament.│       │          │        │          │        │          │   │
│  └──────────┘       └──────────┘        └──────────┘        └──────────┘   │
│       │                  │                   │                    │         │
│       ▼                  ▼                   ▼                    ▼         │
│  TurtleBot3         TurtleBot3          NVIDIA            Humanoid +       │
│  URDF               Simulation          Humanoid          Voice-to-Action  │
│                                                                              │
│                                                           ┌──────────┐      │
│                                                           │ CAPSTONE │      │
│                                                           │ Week 13  │      │
│                                                           └──────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MODULE TEMPLATE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │ Module Overview │  Prerequisites, Learning Objectives        │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Chapter 1     │  │   Chapter 2     │  │   Chapter N     │ │
│  │   (Concept)     │  │   (Deep Dive)   │  │   (Advanced)    │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                    │           │
│           └──────────┬─────────┴─────────┬──────────┘          │
│                      │                   │                      │
│                      ▼                   ▼                      │
│           ┌─────────────────┐  ┌─────────────────┐             │
│           │   Exercises     │  │   Assessment    │             │
│           │  (Hands-on)     │  │  (Optional)     │             │
│           └─────────────────┘  └─────────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### VLA Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VLA WORKFLOW (Weeks 10-13)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │  Voice   │───▶│ Whisper  │───▶│  Ollama  │───▶│     ROS 2 Actions    │  │
│  │  Input   │    │  (STT)   │    │ Llama 3  │    │                      │  │
│  │          │    │          │    │ /Mistral │    │  ┌────────────────┐  │  │
│  └──────────┘    └──────────┘    └──────────┘    │  │ Nav2 Navigate  │  │  │
│                                        │          │  ├────────────────┤  │  │
│                                        ▼          │  │ Perception     │  │  │
│                               ┌──────────────┐   │  │ (Detect Object)│  │  │
│                               │  Task Plan   │   │  ├────────────────┤  │  │
│                               │  JSON Output │   │  │ MoveIt2        │  │  │
│                               │              │   │  │ (Manipulate)   │  │  │
│                               │ {            │   │  └────────────────┘  │  │
│                               │  "actions":[ │───│                      │  │
│                               │   "navigate",│   └──────────────────────┘  │
│                               │   "detect",  │                              │
│                               │   "grasp"    │          ▲                   │
│                               │  ]           │          │ Feedback          │
│                               │ }            │          │                   │
│                               └──────────────┘    ┌─────┴─────┐            │
│                                                   │ Perception│            │
│                                                   │  Results  │            │
│                                                   └───────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Section Structure

### Phase 1: Foundation (Weeks 1-3)

| Week | Title | Key Topics | Robot | Exercises |
|------|-------|------------|-------|-----------|
| 1 | Physical AI Foundations | What is Physical AI, embodiment, course overview | N/A | Environment setup |
| 2 | ROS 2 Fundamentals I | Nodes, topics, messages, workspace | TurtleBot3 | Create publisher/subscriber |
| 3 | ROS 2 Fundamentals II | Services, actions, URDF/Xacro | TurtleBot3 | Build robot model |

### Phase 2: Simulation (Weeks 4-6)

| Week | Title | Key Topics | Robot | Exercises |
|------|-------|------------|-------|-----------|
| 4 | Gazebo Simulation I | Installation, world building, spawning | TurtleBot3 | Launch robot in Gazebo |
| 5 | Gazebo Simulation II | Sensors, physics, ROS 2 bridge | TurtleBot3 | Sensor data processing |
| 6 | Navigation & SLAM | Nav2, SLAM Toolbox, autonomous nav | TurtleBot3 | Map building, goal navigation |

### Phase 3: Advanced Simulation (Weeks 7-9)

| Week | Title | Key Topics | Robot | Exercises |
|------|-------|------------|-------|-----------|
| 7 | Isaac Sim Introduction | Installation, interface, ROS 2 bridge | NVIDIA Humanoid | Launch humanoid in Isaac |
| 8 | Synthetic Data Generation | Domain randomization, dataset export | NVIDIA Humanoid | Generate perception dataset |
| 9 | Isaac ROS Perception | VSLAM, depth estimation, object detection | NVIDIA Humanoid | Perception pipeline |

### Phase 4: Conversational AI (Weeks 10-12)

| Week | Title | Key Topics | Robot | Exercises |
|------|-------|------------|-------|-----------|
| 10 | Conversational AI I | Whisper STT, Ollama setup, prompting | NVIDIA Humanoid | Voice transcription |
| 11 | Conversational AI II | Task planning, ROS 2 action generation | NVIDIA Humanoid | LLM → action plans |
| 12 | VLA Integration | Full pipeline, error handling, re-planning | NVIDIA Humanoid | End-to-end VLA |

### Phase 5: Capstone (Week 13)

| Week | Title | Key Topics | Robot | Exercises |
|------|-------|------------|-------|-----------|
| 13 | Autonomous Humanoid Capstone | Integration, testing, documentation | NVIDIA Humanoid | Complete project |

## Research Approach

### Research-While-Writing Workflow

```
For each chapter:
  1. RESEARCH
     - Consult primary documentation (ROS 2, Gazebo, Isaac)
     - Identify key concepts and APIs
     - Note version-specific behaviors

  2. OUTLINE
     - Define learning objectives (3-5 per chapter)
     - Map prerequisite knowledge
     - Plan exercises

  3. DRAFT
     - Write content with inline APA citations
     - Include code examples with expected output
     - Add diagrams where needed

  4. VALIDATE
     - Execute all code on reference environment
     - Verify citations against sources
     - Check readability (FK 9-11)

  5. INTEGRATE
     - Add to Docusaurus
     - Validate build
     - Run link checker
```

### Primary Documentation Sources

| Domain | Primary Source | Citation Key |
|--------|---------------|--------------|
| ROS 2 | docs.ros.org/en/humble/ | openrobotics2024ros2 |
| Gazebo | gazebosim.org/docs/harmonic/ | openrobotics2024gazebo |
| Isaac Sim | docs.omniverse.nvidia.com/isaacsim/ | nvidia2024isaacsim |
| Isaac ROS | nvidia-isaac-ros.github.io/ | nvidia2024isaacros |
| TurtleBot3 | emanual.robotis.com/docs/en/platform/turtlebot3/ | robotis2024turtlebot3 |
| Whisper | github.com/openai/whisper | openai2024whisper |
| Ollama | ollama.ai/ | ollama2024 |
| Nav2 | docs.nav2.org/ | nav22024 |

## Quality Validation

### Validation Matrix

| Check | Tool | Frequency | Pass Criteria |
|-------|------|-----------|---------------|
| Code Reproducibility | Manual execution | Every example | Runs without modification |
| Readability | textstat (Python) | Every chapter | FK grade 9-11 |
| Link Validity | Docusaurus build | Pre-deploy | 0 broken links |
| Citation Format | Manual + regex | Every chapter | APA 7th edition |
| Build Success | `npm run build` | Every commit | 0 errors, 0 warnings |
| Search Indexing | Manual verification | Post-deploy | All content searchable |

### Reference Environment

```yaml
# .tested-environment.yml
os: Ubuntu 22.04.3 LTS
kernel: 6.5.0-generic
ros_distro: humble
ros_version: 2.0
python: 3.10.12
gazebo: harmonic
isaac_sim: 2023.1.1
nvidia_driver: 535.154.05
cuda: 12.2
ollama: 0.1.20
whisper: 20231117
node: 18.19.0
npm: 10.2.3
docusaurus: 3.1.0
last_verified: 2025-12-10
```

## Decisions Requiring ADRs

The following decisions are architecturally significant and should be documented:

1. **ADR-001: Simulation Stack Selection**
   - Decision: Layered Gazebo → Isaac Sim approach
   - Impact: Affects module structure, hardware requirements, learning curve
   - Suggest: `/sp.adr simulation-stack-selection`

2. **ADR-002: LLM Integration Strategy**
   - Decision: Local-first with Ollama
   - Impact: Reproducibility, cost, offline capability
   - Suggest: `/sp.adr llm-integration-strategy`

3. **ADR-003: Robot Platform Progression**
   - Decision: TurtleBot3 → NVIDIA Humanoid
   - Impact: URDF complexity, Isaac compatibility, manipulation capabilities
   - Suggest: `/sp.adr robot-platform-progression`

📋 **Architectural decisions detected.** Document reasoning and tradeoffs? Run `/sp.adr <decision-title>` for each.

## Implementation Phases

### Phase A: Infrastructure Setup
- [ ] Initialize Docusaurus project
- [ ] Configure custom MDX components
- [ ] Set up GitHub Pages deployment
- [ ] Create reference environment documentation

### Phase B: Foundation Content (Weeks 1-3)
- [ ] Write Week 1: Physical AI Foundations
- [ ] Write Week 2: ROS 2 Fundamentals I
- [ ] Write Week 3: ROS 2 Fundamentals II
- [ ] Validate all code examples

### Phase C: Simulation Content (Weeks 4-6)
- [ ] Write Week 4: Gazebo Simulation I
- [ ] Write Week 5: Gazebo Simulation II
- [ ] Write Week 6: Navigation & SLAM
- [ ] Validate all code examples

### Phase D: Isaac Content (Weeks 7-9)
- [ ] Write Week 7: Isaac Sim Introduction
- [ ] Write Week 8: Synthetic Data Generation
- [ ] Write Week 9: Isaac ROS Perception
- [ ] Validate all code examples

### Phase E: VLA Content (Weeks 10-12)
- [ ] Write Week 10: Conversational AI I
- [ ] Write Week 11: Conversational AI II
- [ ] Write Week 12: VLA Integration
- [ ] Validate full pipeline

### Phase F: Capstone & Polish (Week 13)
- [ ] Write Week 13: Capstone
- [ ] Create glossary and master bibliography
- [ ] Final validation pass
- [ ] Deploy to GitHub Pages

## Complexity Tracking

> **No constitution violations requiring justification.**

All design decisions align with constitution principles. The layered simulation approach (Gazebo → Isaac Sim) adds complexity but is necessary to balance accessibility (lower hardware requirements for beginners) with capability (industry-grade simulation for advanced content).

---

## Next Steps

1. Run `/sp.tasks` to generate actionable task list
2. Consider `/sp.adr` for architectural decisions identified above
3. Begin Phase A: Infrastructure Setup
