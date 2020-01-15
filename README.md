# shallow-water

[![Build Status](https://travis-ci.com/chocol4te/shallow-water.svg?token=Yy278F6KPxruhJLf8xog&branch=rust_port)](https://travis-ci.com/chocol4te/shallow-water)
[![codecov](https://codecov.io/gh/chocol4te/shallow-water/branch/master/graph/badge.svg?token=LLR3tmRGuE)](https://codecov.io/gh/chocol4te/shallow-water)

3D shallow water code by David Dritschel

http://www-vortex.mcs.st-and.ac.uk/software.html

## Usage

In the root directory of the project:

```
cargo run --release --bin vstrip
cargo run --release --bin balinit
cargo run --release --bin swto3d
cargo run --release --bin nhswps
```

This will execute all 4 binaries in the project, using parameters found in `parameters.toml`. Alternative files can be passed as an argument.

Final output is placed the `2d` and `3d` folders, as well as three `.asc` files.

### Testing

```
cargo test
```

### Benchmarking

```
cargo bench
```
