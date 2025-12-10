# Data Model: Physical AI & Humanoid Robotics Book

**Feature**: 001-physical-ai-robotics-book
**Date**: 2025-12-10
**Status**: Complete

## Overview

This document defines the content entities, their relationships, and validation rules for the Physical AI & Humanoid Robotics Book. The model supports Docusaurus-based static site generation with structured learning progression.

---

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BOOK                                           │
│  (Physical AI & Humanoid Robotics)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ contains (1:13)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODULE                                          │
│  week_number, title, prerequisites[], learning_objectives[], duration       │
└─────────────────────────────────────────────────────────────────────────────┘
          │                         │                         │
          │ contains (1:N)          │ has (1:N)              │ references (0:N)
          ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    CHAPTER      │    │    EXERCISE     │    │   DEPENDENCY    │
│ title, content, │    │ title, steps[], │    │ module_ref,     │
│ diagrams[],     │    │ expected_output,│    │ dependency_type │
│ citations[]     │    │ troubleshooting │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                    │
          │ contains (0:N)     │ uses (0:N)
          ▼                    ▼
┌─────────────────┐    ┌─────────────────┐
│  CODE_EXAMPLE   │    │  CODE_EXAMPLE   │
│ language, code, │◄───│                 │
│ dependencies[], │    └─────────────────┘
│ tested_env,     │
│ expected_output │
└─────────────────┘
          │
          │ cites (0:N)
          ▼
┌─────────────────┐
│    CITATION     │
│ authors, title, │
│ publication,    │
│ year, url,      │
│ access_date     │
└─────────────────┘
```

---

## Entity Definitions

### 1. Book

Top-level container for the entire course.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Unique identifier: `physical-ai-humanoid-robotics` |
| `title` | string | Yes | "Physical AI & Humanoid Robotics" |
| `subtitle` | string | No | "A Hands-On Guide from ROS 2 to VLA-Powered Humanoids" |
| `version` | semver | Yes | Book version (e.g., "1.0.0") |
| `modules` | Module[] | Yes | Array of 13 modules |
| `bibliography` | Citation[] | Yes | Master bibliography |
| `glossary` | GlossaryTerm[] | Yes | Canonical terminology definitions |

**Validation Rules**:
- Must contain exactly 13 modules
- Version follows semantic versioning
- Bibliography must be non-empty

---

### 2. Module

A weekly unit of content (approximately 4 hours of study).

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `week_number` | integer | Yes | 1-13 |
| `id` | string | Yes | Slug: `week-01-physical-ai-foundations` |
| `title` | string | Yes | Human-readable title |
| `description` | string | Yes | 2-3 sentence summary |
| `prerequisites` | string[] | Yes | List of prior knowledge/modules required |
| `learning_objectives` | string[] | Yes | Measurable outcomes (3-5 per module) |
| `estimated_duration` | string | Yes | ISO 8601 duration (e.g., "PT4H") |
| `chapters` | Chapter[] | Yes | Ordered list of chapters |
| `exercises` | Exercise[] | Yes | Hands-on activities |
| `assessment` | Assessment | No | Optional module quiz/project |
| `dependencies` | Dependency[] | Yes | Module dependency graph edges |

**Validation Rules**:
- `week_number` must be unique and in range [1, 13]
- `prerequisites` for Week 1 must be empty or contain only "Python basics"
- `learning_objectives` must be measurable (contain action verbs)
- `estimated_duration` should not exceed "PT6H"

**State Transitions**:
```
draft → review → published → deprecated
```

---

### 3. Chapter

An individual lesson within a module.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Slug: `ros2-nodes-topics` |
| `title` | string | Yes | Chapter title |
| `content` | markdown | Yes | Main content body |
| `order` | integer | Yes | Position within module |
| `reading_time` | string | Yes | Estimated reading time (e.g., "PT20M") |
| `diagrams` | Diagram[] | No | Associated visual content |
| `code_examples` | CodeExample[] | No | Embedded code snippets |
| `citations` | Citation[] | Yes | In-text citations used |
| `key_terms` | string[] | No | Glossary terms introduced |

**Validation Rules**:
- `content` must pass Flesch-Kincaid grade 9-11 check
- All `citations` must exist in module or book bibliography
- `key_terms` must link to glossary definitions

---

### 4. Exercise

A hands-on activity with step-by-step instructions.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Slug: `exercise-create-ros2-node` |
| `title` | string | Yes | Exercise title |
| `difficulty` | enum | Yes | `beginner`, `intermediate`, `advanced` |
| `estimated_time` | string | Yes | ISO 8601 duration |
| `prerequisites` | string[] | No | Chapter/exercise dependencies |
| `objectives` | string[] | Yes | What learner will accomplish |
| `steps` | Step[] | Yes | Ordered list of steps |
| `code_examples` | CodeExample[] | No | Code to write/run |
| `expected_outcome` | string | Yes | What success looks like |
| `troubleshooting` | Troubleshooting[] | No | Common issues and solutions |
| `solution` | CodeExample | No | Complete solution (collapsible) |

**Validation Rules**:
- `steps` must be non-empty
- Each step must have clear action and expected result
- `expected_outcome` must be verifiable

---

### 5. CodeExample

Executable code snippet or project.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `language` | string | Yes | `python`, `bash`, `yaml`, `xml`, `cpp` |
| `code` | string | Yes | Source code content |
| `filename` | string | No | Suggested filename |
| `line_highlight` | integer[] | No | Lines to emphasize |
| `dependencies` | Dependency[] | Yes | Package/library requirements |
| `tested_environment` | Environment | Yes | Verified execution environment |
| `expected_output` | string | No | What running this produces |
| `runnable` | boolean | Yes | Whether code can be executed directly |

**Validation Rules**:
- `code` must be syntactically valid for `language`
- If `runnable` is true, must have `tested_environment`
- `dependencies` must include version pins

---

### 6. Citation

Reference to authoritative source (APA format).

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Unique key: `openrobotics2024ros2` |
| `authors` | string[] | Yes | Author names |
| `year` | integer | Yes | Publication year |
| `title` | string | Yes | Work title |
| `publication` | string | No | Journal/book/website name |
| `url` | url | No | Web address |
| `doi` | string | No | Digital Object Identifier |
| `access_date` | date | Conditional | Required if `url` present |
| `archived_url` | url | No | Wayback Machine archive |

**Validation Rules**:
- Must follow APA 7th edition format
- If `url` provided, `access_date` required
- `year` must be reasonable (1900-current)

---

### 7. Dependency

Module or package dependency declaration.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | enum | Yes | `module`, `package`, `system` |
| `name` | string | Yes | Dependency name |
| `version` | string | Conditional | Required for packages |
| `version_constraint` | string | No | `>=`, `==`, `~=` constraints |
| `optional` | boolean | No | Default false |
| `reason` | string | No | Why this dependency exists |

---

### 8. Environment

Tested execution environment specification.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `os` | string | Yes | `ubuntu-22.04`, `wsl2-ubuntu-22.04` |
| `ros_distro` | string | Yes | `humble` |
| `python_version` | string | Yes | `3.10+` |
| `gpu` | string | No | GPU requirement if applicable |
| `packages` | Dependency[] | Yes | Installed packages with versions |
| `tested_date` | date | Yes | Last verification date |

---

### 9. Diagram

Visual content with source files.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `title` | string | Yes | Diagram caption |
| `type` | enum | Yes | `mermaid`, `drawio`, `image` |
| `source_file` | path | Yes | Source file path |
| `export_file` | path | Conditional | PNG/SVG export (if not mermaid) |
| `alt_text` | string | Yes | Accessibility description |

**Validation Rules**:
- `alt_text` required for accessibility
- `source_file` must exist in repository

---

### 10. GlossaryTerm

Canonical terminology definition.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `term` | string | Yes | Canonical term |
| `definition` | string | Yes | Clear, concise definition |
| `aliases` | string[] | No | Alternative names |
| `first_use` | string | Yes | Chapter ID where introduced |
| `category` | enum | Yes | `ros2`, `simulation`, `perception`, `vla`, `general` |

---

## Module Dependency Graph

```
Week 1: Physical AI Foundations
    └── (no dependencies - entry point)

Week 2: ROS 2 Fundamentals I
    └── depends_on: Week 1

Week 3: ROS 2 Fundamentals II
    └── depends_on: Week 2

Week 4: Gazebo Simulation I
    └── depends_on: Week 3

Week 5: Gazebo Simulation II
    └── depends_on: Week 4

Week 6: Navigation & SLAM
    └── depends_on: Week 5

Week 7: Isaac Sim Introduction
    └── depends_on: Week 6

Week 8: Synthetic Data Generation
    └── depends_on: Week 7

Week 9: Isaac ROS Perception
    └── depends_on: Week 8

Week 10: Conversational AI I
    └── depends_on: Week 6 (can run parallel to Isaac track)

Week 11: Conversational AI II
    └── depends_on: Week 10

Week 12: VLA Integration
    └── depends_on: Week 9, Week 11

Week 13: Capstone
    └── depends_on: Week 12
```

---

## File Structure Mapping

```
docs/
├── intro.md                    # Book introduction
├── week-01/                    # Module 1
│   ├── index.md               # Module overview
│   ├── what-is-physical-ai.md # Chapter 1
│   ├── embodiment.md          # Chapter 2
│   └── exercises/
│       └── setup-environment.md
├── week-02/                    # Module 2
│   ├── index.md
│   ├── ros2-nodes.md
│   ├── ros2-topics.md
│   └── exercises/
│       ├── create-node.md
│       └── pubsub-example.md
...
├── week-13/                    # Capstone
│   ├── index.md
│   ├── integration.md
│   └── exercises/
│       └── capstone-project.md
├── glossary.md                 # Master glossary
└── bibliography.md             # Master bibliography

static/
├── diagrams/                   # Diagram source files
│   ├── ros2-graph.drawio
│   └── vla-pipeline.drawio
└── code/                       # Companion code repository
    ├── week-02/
    ├── week-03/
    ...
    └── week-13/
```
