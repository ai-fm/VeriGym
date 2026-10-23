# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/) (while below 1.0, breaking changes bump the minor version).

Add your changes under `[Unreleased]`, one line per change, grouped as:

- `### Added` for new features
- `### Changed` for changes in existing behavior (mark breaking changes with **Breaking:**)
- `### Fixed` for bug fixes
- `### Removed` for removed features


The "Bump version" workflow moves the filled groups under the new version and resets `[Unreleased]` to empty groups. See the example `[0.0.0]` at the bottom.

## [Unreleased]

### Added
- First release on PyPI: `pip install verigym`

### Changed

### Fixed

### Removed

## [0.1.0] - 2026-10-01

### Added

- Base class `VeriGymEnv`
- Learning of MDPs and interval MDPs from trajectories
- AbstractionMapping: forward and backward mapping between original and abstract spaces
- Discretization of continuous spaces
- Import and export of explicit MDPs (from Storm, PRISM, UMB, ...)
- Exploration policies (Random, Entropy-based, Q-value-based)
- Abstraction Refinement
- Workflows providing example usage of verigym
