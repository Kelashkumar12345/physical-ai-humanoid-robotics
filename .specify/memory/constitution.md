# Physical AI & Humanoid Robotics Constitution

AI/Spec-Driven Book for Docusaurus — 13-Week Course

## Core Principles

### I. Technical Accuracy

All content must be factually correct and verifiable against primary sources.
- Every factual claim must be traceable to authoritative sources
- Preferred sources: official documentation (ROS 2, Gazebo, Isaac), peer-reviewed robotics papers, conference proceedings
- No speculation or unverified claims; when uncertainty exists, acknowledge it explicitly
- Diagrams must accurately reflect real system architectures (ROS graph structures, VLA pipelines, sensor data flows)

### II. Reproducibility

All tutorials, code examples, and hands-on exercises must be reproducible.
- Every step (ROS 2, Gazebo, Isaac Sim, VLA models) must work end-to-end when followed exactly
- Code examples must be tested on the specified platform/version before inclusion
- Environment requirements (OS, dependencies, hardware) must be explicitly documented
- Version pinning required for all software dependencies

### III. Clarity for Learners

Content optimized for readers with basic AI background but new to robotics.
- Writing clarity: Flesch-Kincaid grade level 9–11
- Simple, instructional prose; avoid jargon without definition
- Concepts build progressively without gaps
- Each module provides prerequisite knowledge check and learning objectives

### IV. Source Rigor

Rely exclusively on authoritative, verifiable sources.
- Citation style: APA format (in-text references + bibliography per chapter)
- Primary sources preferred over secondary summaries
- Official documentation takes precedence over blog posts or tutorials
- Academic papers must be peer-reviewed or from reputable preprint servers with clear status noted

### V. Practical Focus

Prioritize hands-on learning over unnecessary theory.
- Theory included only when it directly supports practical robotics skills
- Every concept tied to a concrete implementation or exercise
- Avoid tangential topics that don't contribute to the Autonomous Humanoid capstone
- Balance depth vs. breadth toward actionable competence

### VI. Vendor Neutrality

Maintain objectivity; no product endorsements.
- Use vendor-neutral language except where specific tools are required (e.g., NVIDIA Isaac for simulation)
- When proprietary tools are necessary, explain why and note open alternatives if they exist
- Present tradeoffs objectively when comparing approaches

## Content Standards

### Writing Quality
- Flesch-Kincaid grade level: 9–11
- Active voice preferred
- Short paragraphs (3–5 sentences max)
- Code comments in plain English
- Glossary terms linked on first use per chapter

### Code Examples
- All code tested before inclusion
- Include expected output/behavior
- Error handling demonstrated where relevant
- Comments explain "why" not just "what"
- Repository structure follows ROS 2 conventions where applicable

### Diagrams & Visuals
- Must match actual system architectures
- Source files (draw.io, Mermaid, etc.) committed alongside exports
- Alt text required for accessibility
- Consistent visual style across chapters

### Citations
- APA format throughout
- In-text citations for all factual claims
- Chapter bibliographies + master bibliography
- URLs archived via Wayback Machine where possible
- Access dates included for online sources

## Structure Constraints

### Format
- Markdown optimized for Docusaurus
- MDX components permitted for interactive elements
- Admonitions for warnings, tips, notes
- Code blocks with language tags and line highlighting

### Course Structure
- 13-week module progression
- Each week: ~4 hours of content (reading + hands-on)
- Module dependencies explicitly mapped
- Week 13: Capstone integration (Autonomous Humanoid)

### Module Progression
1. Weeks 1–3: ROS 2 Fundamentals
2. Weeks 4–6: Gazebo Simulation
3. Weeks 7–9: Isaac Sim & Synthetic Data
4. Weeks 10–12: Vision-Language-Action (VLA) Models
5. Week 13: Autonomous Humanoid Capstone

## Success Criteria

### Reader Outcomes
- [ ] Readers can complete the Autonomous Humanoid capstone project by following the book
- [ ] All modules (ROS 2 → Gazebo → Isaac → VLA) function end-to-end
- [ ] A reader with basic Python/AI background can complete each module independently

### Content Quality
- [ ] Zero factual inaccuracies when verified against primary sources
- [ ] High coherence: no conceptual gaps between modules
- [ ] All code examples execute without modification on specified environment
- [ ] All diagrams verified against actual system behavior

### Technical Delivery
- [ ] Book deploys cleanly on GitHub Pages via Docusaurus
- [ ] Build passes with zero warnings
- [ ] All internal links valid
- [ ] Search functionality indexes all content

## Governance

### Amendment Process
- Constitution changes require explicit justification
- Changes must not break existing module dependencies
- All amendments documented with rationale and date

### Quality Gates
- Every chapter requires technical review against primary sources
- Code examples require execution verification
- Factual claims require citation verification
- Diagrams require architecture validation

### Compliance
- All contributors must read constitution before contributing
- PRs must demonstrate compliance with principles
- Technical accuracy disputes resolved by primary source verification

**Version**: 1.0.0 | **Ratified**: 2025-12-10 | **Last Amended**: 2025-12-10
