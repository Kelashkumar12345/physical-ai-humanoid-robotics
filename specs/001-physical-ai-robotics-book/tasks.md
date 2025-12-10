# Tasks: Physical AI & Humanoid Robotics Book

**Input**: Design documents from `/specs/001-physical-ai-robotics-book/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Docusaurus site**: `docs/`, `src/`, `static/` at repository root
- **Modules**: `docs/week-XX/` for each weekly module
- **Components**: `src/components/` for custom MDX components
- **Code examples**: `static/code/week-XX/` for companion code

---

## Phase 1: Setup (Shared Infrastructure) ✅ COMPLETE

**Purpose**: Initialize Docusaurus project and configure site infrastructure

- [x] T001 Initialize Docusaurus project with `npx create-docusaurus@latest` in repository root
- [x] T002 Configure docusaurus.config.ts with book metadata, theme, and plugins
- [x] T003 [P] Create sidebars.ts with 13-week module structure
- [x] T004 [P] Configure package.json scripts for build, deploy, and validation
- [x] T005 [P] Create src/css/custom.css with robotics-themed styling
- [x] T006 [P] Create .github/workflows/deploy.yml for GitHub Pages deployment
- [x] T007 Create .tested-environment.yml with reference environment specifications

---

## Phase 2: Foundational (Blocking Prerequisites) ✅ COMPLETE

**Purpose**: Core infrastructure that MUST be complete before ANY content can be written

**⚠️ CRITICAL**: No module content can begin until this phase is complete

- [x] T008 Create src/components/CodeBlock.tsx for styled code and terminal output blocks
- [x] T009 [P] Create src/components/ROSComponents.tsx for ROS 2 topic/service visualization
- [x] T010 [P] Create src/components/LearningObjectives.tsx for learning objectives and prerequisites
- [x] T011 [P] Create src/components/Exercise.tsx for hands-on activity blocks
- [x] T012 Create docs/index.md with book introduction and learning path overview
- [x] T013 [P] Create docs/appendices/glossary.md with robotics terms
- [x] T014 [P] Create docs/appendices/references.md with bibliography (APA format)
- [x] T015 Create static/diagrams/ directory structure for diagram source files
- [x] T016 [P] Create static/code/ directory structure for companion code
- [x] T017 Validate Docusaurus build completes with `npm run build` (zero errors/warnings)

**Checkpoint**: Infrastructure ready - module content creation can now begin

---

## Phase 3: User Story 1 - Complete ROS 2 Fundamentals Module (Priority: P1) 🎯 MVP - WEEK 1 COMPLETE

**Goal**: Learner completes Weeks 1-3 and can create a multi-node ROS 2 system with custom messages

**Independent Test**: Learner runs publisher/subscriber nodes, visualizes URDF in RViz2

### Week 1: ROS 2 Fundamentals I ✅ COMPLETE

- [x] T018 [US1] Create docs/module-1/week-1/introduction.md with ROS 2 architecture overview
- [x] T019 [P] [US1] Create docs/module-1/week-1/nodes-and-topics.md covering publishers and subscribers
- [x] T020 [P] [US1] Create docs/module-1/week-1/services-and-actions.md with service/action patterns
- [x] T021 [US1] Create docs/module-1/week-1/exercises.md with hands-on activities
- [x] T022 [US1] Create docs/getting-started/ with prerequisites, environment-setup, quick-start

### Week 2: ROS 2 Fundamentals I

- [ ] T023 [US1] Create docs/week-02/index.md with module overview referencing TurtleBot3
- [ ] T024 [P] [US1] Create docs/week-02/ros2-concepts.md covering computation graph and DDS
- [ ] T025 [P] [US1] Create docs/week-02/ros2-nodes.md with node creation tutorial
- [ ] T026 [P] [US1] Create docs/week-02/ros2-topics.md with publisher/subscriber tutorial
- [ ] T027 [US1] Create docs/week-02/ros2-messages.md with custom message types
- [ ] T028 [US1] Create static/code/week-02/minimal_publisher.py with tested code example
- [ ] T029 [P] [US1] Create static/code/week-02/minimal_subscriber.py with tested code example
- [ ] T030 [US1] Create docs/week-02/exercises/pubsub-exercise.md with hands-on activity
- [ ] T031 [US1] Add Week 2 citations to docs/bibliography.md

### Week 3: ROS 2 Fundamentals II

- [ ] T032 [US1] Create docs/week-03/index.md with module overview
- [ ] T033 [P] [US1] Create docs/week-03/ros2-services.md with service server/client tutorial
- [ ] T034 [P] [US1] Create docs/week-03/ros2-actions.md with action server/client tutorial
- [ ] T035 [P] [US1] Create docs/week-03/ros2-parameters.md with parameter handling
- [ ] T036 [US1] Create docs/week-03/ros2-launch.md with launch file creation
- [ ] T037 [US1] Create docs/week-03/urdf-basics.md with URDF/Xacro for TurtleBot3
- [ ] T038 [US1] Create static/code/week-03/turtlebot3_description/ with URDF files
- [ ] T039 [US1] Create docs/week-03/exercises/urdf-exercise.md with RViz2 visualization
- [ ] T040 [US1] Add Week 3 citations to docs/bibliography.md
- [ ] T041 [US1] Create static/diagrams/ros2-graph.mermaid with ROS 2 computation graph diagram

**Checkpoint**: User Story 1 complete - learner can create ROS 2 nodes, topics, services, actions, and URDF

---

## Phase 4: User Story 2 - Build and Run Simulations in Gazebo (Priority: P1)

**Goal**: Learner spawns TurtleBot3 in Gazebo, controls via ROS 2, runs Nav2 for autonomous navigation

**Independent Test**: Robot navigates autonomously to goals using SLAM-generated map

### Week 4: Gazebo Simulation I

- [ ] T042 [US2] Create docs/week-04/index.md with module overview and hardware requirements
- [ ] T043 [P] [US2] Create docs/week-04/gazebo-install.md with Gazebo Harmonic installation
- [ ] T044 [P] [US2] Create docs/week-04/gazebo-basics.md covering interface and world building
- [ ] T045 [US2] Create docs/week-04/robot-spawning.md with TurtleBot3 spawn tutorial
- [ ] T046 [US2] Create docs/week-04/ros2-gazebo-bridge.md with ros_gz_bridge configuration
- [ ] T047 [US2] Create static/code/week-04/turtlebot3_gazebo/ with launch files
- [ ] T048 [US2] Create docs/week-04/exercises/spawn-robot.md with hands-on spawning exercise
- [ ] T049 [US2] Add Week 4 citations to docs/bibliography.md

### Week 5: Gazebo Simulation II

- [ ] T050 [US2] Create docs/week-05/index.md with module overview
- [ ] T051 [P] [US2] Create docs/week-05/gazebo-sensors.md covering camera, LIDAR, IMU simulation
- [ ] T052 [P] [US2] Create docs/week-05/gazebo-physics.md with physics tuning and collision
- [ ] T053 [US2] Create docs/week-05/sensor-data.md with ROS 2 sensor topic processing
- [ ] T054 [US2] Create static/code/week-05/sensor_processor.py with camera/lidar processing
- [ ] T055 [US2] Create docs/week-05/exercises/sensor-exercise.md with sensor data visualization
- [ ] T056 [US2] Add Week 5 citations to docs/bibliography.md

### Week 6: Navigation & SLAM

- [ ] T057 [US2] Create docs/week-06/index.md with module overview
- [ ] T058 [P] [US2] Create docs/week-06/slam-toolbox.md with SLAM Toolbox configuration
- [ ] T059 [P] [US2] Create docs/week-06/nav2-intro.md with Nav2 stack overview
- [ ] T060 [US2] Create docs/week-06/nav2-setup.md with TurtleBot3 Nav2 configuration
- [ ] T061 [US2] Create docs/week-06/autonomous-nav.md with goal navigation tutorial
- [ ] T062 [US2] Create static/code/week-06/nav2_config/ with Nav2 parameter files
- [ ] T063 [US2] Create docs/week-06/exercises/mapping-exercise.md with map building
- [ ] T064 [P] [US2] Create docs/week-06/exercises/navigation-exercise.md with goal navigation
- [ ] T065 [US2] Add Week 6 citations to docs/bibliography.md
- [ ] T066 [US2] Create static/diagrams/nav2-stack.mermaid with Nav2 architecture diagram

**Checkpoint**: User Story 2 complete - learner can simulate TurtleBot3 with sensors and autonomous navigation

---

## Phase 5: User Story 3 - Master NVIDIA Isaac Sim and Synthetic Data (Priority: P2)

**Goal**: Learner launches Isaac Sim with ROS 2 bridge, generates synthetic dataset with domain randomization

**Independent Test**: Export 1000+ annotated images from Isaac Sim for object detection

### Week 7: Isaac Sim Introduction

- [ ] T067 [US3] Create docs/week-07/index.md with module overview and GPU requirements (RTX 3060+)
- [ ] T068 [P] [US3] Create docs/week-07/isaac-install.md with Isaac Sim 2023.1+ installation
- [ ] T069 [P] [US3] Create docs/week-07/isaac-interface.md covering Omniverse interface basics
- [ ] T070 [US3] Create docs/week-07/isaac-ros-bridge.md with ROS 2 bridge configuration
- [ ] T071 [US3] Create docs/week-07/humanoid-intro.md introducing NVIDIA Humanoid assets
- [ ] T072 [US3] Create docs/week-07/exercises/launch-humanoid.md with humanoid spawn exercise
- [ ] T073 [US3] Add Week 7 citations to docs/bibliography.md

### Week 8: Synthetic Data Generation

- [ ] T074 [US3] Create docs/week-08/index.md with module overview
- [ ] T075 [P] [US3] Create docs/week-08/replicator-intro.md with Omniverse Replicator basics
- [ ] T076 [P] [US3] Create docs/week-08/domain-randomization.md covering randomization techniques
- [ ] T077 [US3] Create docs/week-08/dataset-export.md with annotation format (COCO, KITTI)
- [ ] T078 [US3] Create static/code/week-08/replicator_script.py with dataset generation script
- [ ] T079 [US3] Create docs/week-08/exercises/synthetic-data-exercise.md with 1000 image export
- [ ] T080 [US3] Add Week 8 citations to docs/bibliography.md

### Week 9: Isaac ROS Perception

- [ ] T081 [US3] Create docs/week-09/index.md with module overview
- [ ] T082 [P] [US3] Create docs/week-09/isaac-ros-intro.md with Isaac ROS packages overview
- [ ] T083 [P] [US3] Create docs/week-09/vslam.md with visual SLAM using Isaac ROS
- [ ] T084 [US3] Create docs/week-09/depth-estimation.md with stereo depth perception
- [ ] T085 [US3] Create docs/week-09/object-detection.md with detection pipeline setup
- [ ] T086 [US3] Create static/code/week-09/perception_launch.py with perception pipeline launch
- [ ] T087 [US3] Create docs/week-09/exercises/perception-exercise.md with detection validation
- [ ] T088 [US3] Add Week 9 citations to docs/bibliography.md
- [ ] T089 [US3] Create static/diagrams/perception-pipeline.mermaid with perception architecture

**Checkpoint**: User Story 3 complete - learner can use Isaac Sim for synthetic data and perception

---

## Phase 6: User Story 4 - Implement VLA Workflow for Humanoid Control (Priority: P2)

**Goal**: Learner implements voice → LLM → ROS 2 action pipeline for humanoid control

**Independent Test**: Voice command "pick up the red cup" triggers navigation, detection, and grasp actions

### Week 10: Conversational AI I

- [ ] T090 [US4] Create docs/week-10/index.md with module overview and LLM requirements
- [ ] T091 [P] [US4] Create docs/week-10/whisper-setup.md with Whisper STT installation and usage
- [ ] T092 [P] [US4] Create docs/week-10/ollama-setup.md with Ollama + Llama 3/Mistral setup
- [ ] T093 [US4] Create docs/week-10/llm-prompting.md with prompting strategies for robotics
- [ ] T094 [US4] Create static/code/week-10/whisper_node.py with ROS 2 Whisper integration
- [ ] T095 [US4] Create docs/week-10/exercises/voice-transcription.md with STT exercise
- [ ] T096 [US4] Add Week 10 citations to docs/bibliography.md

### Week 11: Conversational AI II

- [ ] T097 [US4] Create docs/week-11/index.md with module overview
- [ ] T098 [P] [US4] Create docs/week-11/task-planning.md with LLM-based task decomposition
- [ ] T099 [P] [US4] Create docs/week-11/action-generation.md with ROS 2 action message generation
- [ ] T100 [US4] Create docs/week-11/llm-ros-bridge.md with LLM to ROS 2 integration
- [ ] T101 [US4] Create static/code/week-11/task_planner.py with LLM task planning node
- [ ] T102 [US4] Create static/code/week-11/action_executor.py with action execution node
- [ ] T103 [US4] Create docs/week-11/exercises/task-planning-exercise.md with planning demo
- [ ] T104 [US4] Add Week 11 citations to docs/bibliography.md

### Week 12: VLA Integration

- [ ] T105 [US4] Create docs/week-12/index.md with module overview
- [ ] T106 [P] [US4] Create docs/week-12/vla-pipeline.md with full pipeline architecture
- [ ] T107 [P] [US4] Create docs/week-12/error-handling.md with re-planning and failure recovery
- [ ] T108 [US4] Create docs/week-12/perception-feedback.md with closed-loop perception integration
- [ ] T109 [US4] Create static/code/week-12/vla_system/ with complete VLA ROS 2 package
- [ ] T110 [US4] Create docs/week-12/exercises/vla-exercise.md with end-to-end VLA demo
- [ ] T111 [US4] Add Week 12 citations to docs/bibliography.md
- [ ] T112 [US4] Create static/diagrams/vla-pipeline.mermaid with VLA architecture diagram

**Checkpoint**: User Story 4 complete - learner can run voice-to-action pipeline

---

## Phase 7: User Story 5 - Complete Capstone: Autonomous Humanoid (Priority: P1)

**Goal**: Learner integrates all modules into autonomous humanoid executing voice-to-action tasks

**Independent Test**: Voice command triggers full pipeline: navigate → detect → grasp → return

**Dependencies**: Requires US1-US4 completion for full integration

### Week 13: Autonomous Humanoid Capstone

- [ ] T113 [US5] Create docs/week-13/index.md with capstone overview and prerequisites
- [ ] T114 [P] [US5] Create docs/week-13/system-integration.md with component integration guide
- [ ] T115 [P] [US5] Create docs/week-13/testing-strategy.md with testing and validation approach
- [ ] T116 [US5] Create docs/week-13/troubleshooting.md with common issues and solutions
- [ ] T117 [US5] Create static/code/week-13/capstone_ws/ with complete capstone ROS 2 workspace
- [ ] T118 [US5] Create docs/week-13/exercises/capstone-project.md with step-by-step integration
- [ ] T119 [US5] Create docs/week-13/exercises/demo-scenarios.md with example voice commands
- [ ] T120 [US5] Add Week 13 citations to docs/bibliography.md
- [ ] T121 [US5] Create static/diagrams/capstone-architecture.mermaid with full system diagram

**Checkpoint**: User Story 5 complete - autonomous humanoid fully functional

---

## Phase 8: User Story 6 - Deploy Book on Docusaurus/GitHub Pages (Priority: P2)

**Goal**: Book deploys cleanly with functional navigation, search, and all links valid

**Independent Test**: `npm run build` succeeds with zero errors/warnings; deployed site accessible

- [ ] T122 [US6] Configure Algolia DocSearch or local search in docusaurus.config.js
- [ ] T123 [P] [US6] Create src/pages/index.js with landing page and course overview
- [ ] T124 [P] [US6] Optimize static/img/ with compressed images for fast loading
- [ ] T125 [US6] Validate all internal links with `npm run build`
- [ ] T126 [US6] Run link checker on external URLs (primary documentation sources)
- [ ] T127 [US6] Verify search indexes all 13 modules and returns relevant results
- [ ] T128 [US6] Test GitHub Pages deployment with `npm run deploy`
- [ ] T129 [US6] Validate page load time under 3 seconds on standard broadband
- [ ] T130 [US6] Create README.md with project overview and contribution guidelines

**Checkpoint**: User Story 6 complete - book deployed and accessible

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final quality checks and improvements across all modules

- [ ] T131 [P] Complete docs/glossary.md with all robotics terms from Weeks 1-13
- [ ] T132 [P] Review docs/bibliography.md for APA format compliance
- [ ] T133 Validate all code examples execute on reference environment (Ubuntu 22.04, ROS 2 Humble)
- [ ] T134 [P] Check Flesch-Kincaid readability (grade 9-11) across all chapters
- [ ] T135 [P] Verify all diagrams match actual system architectures
- [ ] T136 Review citation format in all chapters (APA 7th edition)
- [ ] T137 Final `npm run build` validation (zero errors, zero warnings)
- [ ] T138 Update .tested-environment.yml with final verification date
- [ ] T139 Create CHANGELOG.md documenting book version 1.0.0

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup
    ↓
Phase 2: Foundational (BLOCKS all user stories)
    ↓
┌───────────────────────────────────────────────────────┐
│  Phase 3: US1 (ROS 2) ───┐                           │
│  Phase 4: US2 (Gazebo) ──┼──→ Can run in parallel    │
│  Phase 5: US3 (Isaac) ───┘    after Foundational     │
│  Phase 6: US4 (VLA) ─────────────────────────────────│
└───────────────────────────────────────────────────────┘
    ↓ (US1-US4 complete)
Phase 7: US5 (Capstone) - requires prior stories
    ↓
Phase 8: US6 (Deploy) - can run after US1 content exists
    ↓
Phase 9: Polish - after all content complete
```

### User Story Dependencies

| Story | Priority | Depends On | Can Parallel With |
|-------|----------|------------|-------------------|
| US1 (ROS 2) | P1 | Foundational | US2, US3, US4, US6 |
| US2 (Gazebo) | P1 | US1 (conceptually) | US3, US4, US6 |
| US3 (Isaac) | P2 | US2 (simulation skills) | US4, US6 |
| US4 (VLA) | P2 | US1 (ROS 2 basics) | US3, US6 |
| US5 (Capstone) | P1 | US1, US2, US3, US4 | None |
| US6 (Deploy) | P2 | US1 (minimum content) | US2, US3, US4 |

### Within Each User Story

1. Module index.md first (sets context)
2. Concept chapters (can parallel)
3. Tutorial chapters (may depend on concepts)
4. Code examples (for tutorials)
5. Exercises (after tutorials)
6. Citations (ongoing)

### Parallel Opportunities by Phase

**Phase 1 (Setup)**: T003, T004, T005, T006 can run in parallel
**Phase 2 (Foundational)**: T009, T010, T011, T013, T014, T016 can run in parallel
**Phase 3-7 (User Stories)**: All [P] marked tasks within each week can run in parallel
**Phase 9 (Polish)**: T131, T132, T134, T135 can run in parallel

---

## Parallel Example: User Story 1, Week 2

```bash
# Launch parallel concept chapters:
Task: "Create docs/week-02/ros2-concepts.md"
Task: "Create docs/week-02/ros2-nodes.md"
Task: "Create docs/week-02/ros2-topics.md"

# Then (after concepts):
Task: "Create docs/week-02/ros2-messages.md"

# Launch parallel code examples:
Task: "Create static/code/week-02/minimal_publisher.py"
Task: "Create static/code/week-02/minimal_subscriber.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 + Setup Only)

1. Complete Phase 1: Setup (Docusaurus infrastructure)
2. Complete Phase 2: Foundational (components, templates)
3. Complete Phase 3: User Story 1 (Weeks 1-3 content)
4. **STOP and VALIDATE**: Test site builds, Week 1-3 content accessible
5. Deploy preview for early feedback

### Incremental Delivery

| Milestone | Content | Validation |
|-----------|---------|------------|
| MVP | Weeks 1-3 (ROS 2) | Build passes, exercises work |
| Alpha | + Weeks 4-6 (Gazebo) | Nav2 tutorial complete |
| Beta | + Weeks 7-9 (Isaac) | Synthetic data exports |
| RC | + Weeks 10-12 (VLA) | VLA pipeline works |
| 1.0 | + Week 13 (Capstone) | Full course complete |

### Parallel Team Strategy

With multiple authors:
1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Author A: Weeks 1-3 (ROS 2)
   - Author B: Weeks 4-6 (Gazebo)
   - Author C: Weeks 7-9 (Isaac)
   - Author D: Weeks 10-12 (VLA)
3. All authors collaborate on Week 13 (Capstone)
4. Final polish as team

---

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | 139 |
| Setup Tasks | 7 |
| Foundational Tasks | 10 |
| US1 Tasks (ROS 2) | 24 |
| US2 Tasks (Gazebo) | 25 |
| US3 Tasks (Isaac) | 23 |
| US4 Tasks (VLA) | 23 |
| US5 Tasks (Capstone) | 9 |
| US6 Tasks (Deploy) | 9 |
| Polish Tasks | 9 |
| Parallel Opportunities | 47 tasks marked [P] |

**MVP Scope**: Phase 1 + Phase 2 + Phase 3 (US1) = 41 tasks

---

## Notes

- [P] tasks = different files, no dependencies - can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story delivers an independently testable increment
- Commit after each task or logical group
- Run `npm run build` frequently to catch issues early
- All code examples must be tested on reference environment before inclusion
- Follow APA 7th edition for all citations
