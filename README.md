---
title: Autonomous PR Reviewer
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# Autonomous PR Reviewer Environment

## Description
A real-world simulation of a GitHub Pull Request review workflow. AI agents must inspect code diffs and identify security vulnerabilities like hardcoded secrets and SQL injections.

## Action Space
- `inspect_line`: Reveal code at a specific line.
- `request_changes`: Flag an issue type at a specific line.
- `approve_pr`: Approve the PR if no issues are found.

## Tasks
1. **Easy**: Detection of hardcoded API keys.
2. **Medium**: Identification of O(n²) inefficient loops.
3. **Hard**: Spotting SQL injection vulnerabilities.

## Setup
Built with OpenEnv. Run via `openenv build` or `docker build`.

## Author
Created by Satyam Singh Rawat & Bhumika Bahuguna for the OpenEnv Hackathon.