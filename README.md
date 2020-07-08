# shallow-water

[![CI](https://github.com/rse-standrewscs/shallow-water/workflows/CI/badge.svg)](https://github.com/rse-standrewscs/shallow-water/actions)
[![codecov](https://codecov.io/gh/rse-standrewscs/shallow-water/branch/master/graph/badge.svg)](https://codecov.io/gh/rse-standrewscs/shallow-water)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=rse-standrewscs/shallow-water)](https://dependabot.com)
[![](https://tokei.rs/b1/github/rse-standrewscs/shallow-water)](https://github.com/XAMPPRocky/tokei)

[API documentation (master)](https://rse-standrewscs.github.io/shallow-water/shallow_water/index.html)

http://www-vortex.mcs.st-and.ac.uk/software.html

Fortran 3D shallow water code by David Dritschel

## Installation

Requires the latest stable Rust compiler (Minimum Supported Rust Version still to be specified), which can be installed with [rustup.rs](https://rustup.rs).

In the root directory of the project:

```
cargo install --path .
```

This will build and place a `shallow-water` binary in `~/.cargo/bin/` (which is added to `PATH` by default in the `rustup` installer).

### Usage

```
$ shallow-water --help
shallow_water 0.1.0

USAGE:
    shallow-water --parameters <PARAMETERS> [SUBCOMMAND]

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -p, --parameters <PARAMETERS>    Path to file containing simulation parameters.

SUBCOMMANDS:
    balinit    Re-initialises a flow with balanced fields obtained from the conditions delta_t=gamma_t=0 using data
               previously set up with a data generation routine.
    help       Prints this message or the help of the given subcommand(s)
    nhswps     The Horizontally Doubly-Periodic Three-Dimensional Non-Hydrostatic Shallow-Water Pseudo-Spectral
               Method
    swto3d     Converts 2D shallow-water fields to 3D fields needed by nhswps.
    vstrip     Initialises a PV strip with zero fields of divergence and acceleration divergence.
```

`cargo run` can be used to run from the source directory without installation.

For example:

```
cargo run --release -- -p parameters.yaml vstrip
cargo run --release -- -p parameters.yaml balinit
cargo run --release -- -p parameters.yaml swto3d
cargo run --release -- -p parameters.yaml nhswps
```

### Testing

To run all tests:

```
cargo test --release
```

If the `--release` flag is not set the tests may take a _very_ long time to complete.

### Benchmarking

[Criterion.rs](https://github.com/bheisler/criterion.rs) is the benchmarking framework used for this project. To run all benchmarks use:

```
cargo bench
```
