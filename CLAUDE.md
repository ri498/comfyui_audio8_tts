# comfyui_audio8_tts — Agent Instructions

## Project Guidelines

**You must always read `.agents/rules/project-guidelines.md` at the start of every task.**

This file contains the canonical reference for:
- Project architecture and file structure
- ComfyUI node conventions and AUDIO tensor format
- Model caching patterns and HuggingFace integration
- Branching and contribution strategy
- Coding standards

## Update Requirement

**You must keep `.agents/rules/project-guidelines.md` up to date.**

After completing any task that introduces changes to:
- Node definitions (new nodes, renamed inputs/outputs, changed types)
- Dependencies (`requirements.txt`)
- Model loading or caching logic
- Project structure (new files, renamed modules)
- ComfyUI registration mappings

...you **must update** `.agents/rules/project-guidelines.md` to reflect those changes before finishing.

Do not leave the guidelines file stale or out of sync with the actual codebase.
