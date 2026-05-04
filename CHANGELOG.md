# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-05-04

### Added

- Added `onnx_shape` helpers to convert inferred dimensions into ONNX
  `TensorShapeProto` values.
- Dynamic dimensions now emit ONNX `dim_param` names when available so symbolic
  shape information survives export.

### Changed

- Exported the new `onnx_shape` module from the crate root for downstream
  converters.
- Added and unified local pre-commit checks around `cargo fmt --check` and
  `cargo clippy -- -D warnings`.

### Fixed

- Cleaned up clippy issues in the ONNX shape conversion path and related tests.

## [0.1.0]

### Added

- Initial release with shared WebNN and ONNX data type mappings, operation name
  mappings, attribute helpers, tensor data utilities, identifier sanitization,
  protobuf bindings, and shape inference primitives.
