# shallow-water

[![Build Status](https://travis-ci.org/rse-standrewscs/shallow-water.svg)](https://travis-ci.org/rse-standrewscs/shallow-water)
[![codecov](https://codecov.io/gh/rse-standrewscs/shallow-water/branch/master/graph/badge.svg)](https://codecov.io/gh/rse-standrewscs/shallow-water)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=rse-standrewscs/shallow-water)](https://dependabot.com)
[![](https://tokei.rs/b1/github/rse-standrewscs/shallow-water)](https://github.com/XAMPPRocky/tokei)

3D shallow water code by David Dritschel

[API documentation (master)](https://rse-standrewscs.github.io/shallow-water/)

http://www-vortex.mcs.st-and.ac.uk/software.html

## Usage

In the root directory of the project:

```
cargo run --release -- vstrip
cargo run --release -- balinit
cargo run --release -- swto3d
cargo run --release -- nhswps
```

This will execute all 4 binaries in the project, using parameters found in `parameters.toml`. Alternative files can be passed as an argument.

Final output is placed in the `2d` and `3d` folders, as well as in three `.asc` files.

### Testing

```
cargo test
```

### Benchmarking

```
cargo bench
```
