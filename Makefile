SHELL := /bin/bash

CARGO ?= cargo
RUSTUP ?= rustup

# Optional: set CRATE=some-package on the command line.
CRATE ?=
PKGFLAG :=
ifneq ($(strip $(CRATE)),)
  PKGFLAG := -p $(CRATE)
endif

# Optional features list: make test FEATURES="foo,bar"
FEATURES ?=
FEATUREFLAG :=
ifneq ($(strip $(FEATURES)),)
  FEATUREFLAG := --features "$(FEATURES)"
endif

# Extra flags: e.g. CARGOFLAGS="--locked"
CARGOFLAGS ?=

# Default profile for build/run targets
PROFILE ?= debug

# More ergonomic output
CARGO_TERM_COLOR ?= always
export CARGO_TERM_COLOR

.PHONY: help
help:
	@echo "Targets:"
	@echo "  build                Build (debug by default)"
	@echo "  build-release        Build --release"
	@echo "  check                cargo check"
	@echo "  test                 cargo test"
	@echo "  test-release         cargo test --release"
	@echo "  test-all-features    cargo test --all-features"
	@echo "  test-no-default      cargo test --no-default-features"
	@echo "  bench                cargo bench"
	@echo "  run                  cargo run (pass ARGS='...')"
	@echo "  fmt                  cargo fmt"
	@echo "  fmt-check            cargo fmt --check"
	@echo "  clippy               cargo clippy (deny warnings)"
	@echo "  doc                  cargo doc"
	@echo "  doc-open             cargo doc --open"
	@echo "  clean                cargo clean"
	@echo "  update               cargo update"
	@echo "  tree                 cargo tree"
	@echo "  expand               cargo expand (needs cargo-expand)"
	@echo "  udeps                cargo udeps (nightly, needs cargo-udeps)"
	@echo "  audit                cargo audit (needs cargo-audit)"
	@echo "  deny                 cargo deny check (needs cargo-deny)"
	@echo "  msrv                 show rustc version; set MSRV in rust-toolchain.toml"
	@echo "  ci                   fmt-check + clippy + test"
	@echo ""
	@echo "Variables:"
	@echo "  CRATE=...            Limit to a workspace package (cargo -p)"
	@echo "  FEATURES=...         Enable features (comma-separated)"
	@echo "  CARGOFLAGS=...       Extra cargo flags (e.g. --locked)"
	@echo "  ARGS='...'           Arguments for run target"

.PHONY: build
build:
ifeq ($(PROFILE),release)
	$(CARGO) build $(PKGFLAG) $(FEATUREFLAG) --release $(CARGOFLAGS)
else
	$(CARGO) build $(PKGFLAG) $(FEATUREFLAG) $(CARGOFLAGS)
endif

.PHONY: build-release
build-release:
	$(CARGO) build $(PKGFLAG) $(FEATUREFLAG) --release $(CARGOFLAGS)

.PHONY: check
check:
	$(CARGO) check $(PKGFLAG) $(FEATUREFLAG) $(CARGOFLAGS)

.PHONY: test
test:
	$(CARGO) test $(PKGFLAG) $(FEATUREFLAG) $(CARGOFLAGS)

.PHONY: test-release
test-release:
	$(CARGO) test $(PKGFLAG) $(FEATUREFLAG) --release $(CARGOFLAGS)

.PHONY: test-all-features
test-all-features:
	$(CARGO) test $(PKGFLAG) --all-features $(CARGOFLAGS)

.PHONY: test-no-default
test-no-default:
	$(CARGO) test $(PKGFLAG) --no-default-features $(CARGOFLAGS)

.PHONY: bench
bench:
	$(CARGO) bench $(PKGFLAG) $(FEATUREFLAG) $(CARGOFLAGS)

.PHONY: run
run:
	$(CARGO) run $(PKGFLAG) $(FEATUREFLAG) $(CARGOFLAGS) -- $(ARGS)

.PHONY: fmt
fmt:
	$(CARGO) fmt

.PHONY: fmt-check
fmt-check:
	$(CARGO) fmt --check

.PHONY: clippy
clippy:
	$(CARGO) clippy $(PKGFLAG) $(FEATUREFLAG) $(CARGOFLAGS) -- -D warnings

.PHONY: doc
doc:
	$(CARGO) doc $(PKGFLAG) $(FEATUREFLAG) $(CARGOFLAGS)

.PHONY: doc-open
doc-open:
	$(CARGO) doc $(PKGFLAG) $(FEATUREFLAG) --open $(CARGOFLAGS)

.PHONY: clean
clean:
	$(CARGO) clean

.PHONY: update
update:
	$(CARGO) update

.PHONY: tree
tree:
	$(CARGO) tree $(PKGFLAG) $(FEATUREFLAG)

# Requires: cargo install cargo-expand
.PHONY: expand
expand:
	$(CARGO) expand $(PKGFLAG) $(FEATUREFLAG)

# Requires nightly + cargo install cargo-udeps
.PHONY: udeps
udeps:
	$(CARGO) +nightly udeps $(PKGFLAG) $(FEATUREFLAG)

# Requires: cargo install cargo-audit
.PHONY: audit
audit:
	$(CARGO) audit

# Requires: cargo install cargo-deny
.PHONY: deny
deny:
	$(CARGO) deny check

.PHONY: msrv
msrv:
	@$(RUSTUP) show active-toolchain
	@$(CARGO) --version
	@rustc --version

.PHONY: ci
ci: fmt-check clippy test

