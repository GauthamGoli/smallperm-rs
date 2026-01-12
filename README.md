# smallperm

High-performance pseudo-random permutations using Feistel networks.

[![Crates.io](https://img.shields.io/crates/v/smallperm.svg)](https://crates.io/crates/smallperm)
[![Documentation](https://docs.rs/smallperm/badge.svg)](https://docs.rs/smallperm)
[![License](https://img.shields.io/crates/l/smallperm.svg)](LICENSE)

## Overview

`smallperm` generates pseudo-random permutations with **O(1) memory** and **O(1) time** per element. Unlike Fisher-Yates shuffle which requires O(n) memory to store the entire permutation, this library computes each element on-demand using a Feistel network cipher.

This makes it ideal for:

- **Machine learning**: Shuffling billions of training samples without memory overhead
- **Distributed systems**: Each worker can independently compute any slice of the permutation
- **Streaming**: Get shuffled elements immediately without waiting for full materialization

## Features

- **O(1) memory**: No need to store the entire permutation
- **O(1) random access**: Get any element instantly via index
- **Invertible**: Bidirectional mapping between indices and values
- **Deterministic**: Same seed always produces the same permutation
- **Large scale**: Supports up to 2^128 elements

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
smallperm = "0.1"
```

## Quick Start

```rust
use smallperm::Permutation;

// Create a permutation of 1 million elements
let perm = Permutation::new(1_000_000, 42);

// Iterate through shuffled indices
for value in perm.iter().take(10) {
    println!("{}", value);
}

// O(1) random access - get the 500,000th element
let value = perm.get(500_000).unwrap();

// O(1) inverse lookup - find where a value appears
let index = perm.index_of(value).unwrap();
assert_eq!(index, 500_000);
```

## API

### `Permutation`

The main type for working with permutations.

```rust
use smallperm::Permutation;

// Create with a u64 seed
let perm = Permutation::new(1000, 42);

// Or with a 256-bit key for more control
let key: [u8; 32] = [0; 32];
let perm = Permutation::new_with_key(1000, key);

// Get length
assert_eq!(perm.len(), 1000);

// Random access (returns Option<u128>)
let value = perm.get(500).unwrap();

// Inverse lookup (returns Option<u128>)
let index = perm.index_of(value).unwrap();

// Iterate
for value in perm.iter() {
    // ...
}
```

### Helper Functions

Convenience functions for common operations:

```rust
use smallperm::{sample_indices, sample, shuffle, shuffle_in_place};

// Sample k unique random indices from [0, n)
let indices = sample_indices(100, 10, 42);
assert_eq!(indices.len(), 10);

// Sample k unique elements from a slice
let data = vec!["a", "b", "c", "d", "e"];
let sampled = sample(&data, 3, 42);

// Shuffle a slice (returns new Vec)
let shuffled = shuffle(&data, 42);

// Shuffle in place
let mut data = vec![1, 2, 3, 4, 5];
shuffle_in_place(&mut data, 42);
```

### Low-Level Access

For advanced use cases, the underlying types are also exposed:

```rust
use smallperm::{Permutor, FeistelNetwork};

// Permutor is the core iterator type
let permutor = Permutor::new_with_u64_key(1000, 42);
for value in permutor {
    // ...
}

// FeistelNetwork is the raw cipher
// (outputs may exceed max, use Permutor for bounded results)
```

## Performance

The library uses a Feistel network with FxHash (a fast, non-cryptographic hash function) as the round function. Each lookup requires 8-32 hash operations depending on the permutation size.

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Create permutation | O(1) | O(1) |
| Get element by index | O(1)* | O(1) |
| Inverse lookup | O(1)* | O(1) |
| Iterate n elements | O(n) | O(1) |

\* Amortized O(1) due to rejection sampling when the Feistel domain exceeds the permutation size.

### Comparison with Fisher-Yates

| Approach | Memory | Time to first element | Random access |
|----------|--------|----------------------|---------------|
| Fisher-Yates | O(n) | O(n) | O(1) after shuffle |
| smallperm | O(1) | O(1) | O(1) |

For a dataset of 1 billion elements:
- Fisher-Yates: ~8 GB memory, must shuffle everything first
- smallperm: ~100 bytes, instant access to any element

## Use Cases

### ML Data Loading

```rust
use smallperm::Permutation;

fn load_epoch(dataset_size: u128, epoch: u64) -> impl Iterator<Item = u128> {
    // Different seed per epoch = different shuffle
    Permutation::new(dataset_size, epoch).into_iter()
}

// Stream shuffled indices without loading everything into memory
for sample_idx in load_epoch(1_000_000_000, 0).take(1000) {
    // Load and process sample at index sample_idx
}
```

### Distributed Training

```rust
use smallperm::Permutation;

fn get_worker_batch(
    dataset_size: u128,
    seed: u64,
    worker_id: usize,
    num_workers: usize,
    batch_size: usize,
) -> Vec<u128> {
    let perm = Permutation::new(dataset_size, seed);
    let start = worker_id * batch_size;

    (start..start + batch_size)
        .map(|i| perm.get(i as u128).unwrap())
        .collect()
}
```

### Reproducible Sampling

```rust
use smallperm::sample;

let data: Vec<i32> = (0..10000).collect();

// Same seed = same sample, every time
let sample1 = sample(&data, 100, 42);
let sample2 = sample(&data, 100, 42);
assert_eq!(sample1, sample2);
```

## Algorithm

The library implements a [Feistel network](https://en.wikipedia.org/wiki/Feistel_cipher), a symmetric cipher structure that creates a bijection (one-to-one mapping) over the input domain. Key properties:

1. **Bijective**: Every input maps to exactly one output, and vice versa
2. **Invertible**: Given the output, we can compute the original input
3. **Pseudo-random**: Output appears random but is deterministic given the key

For permutations of size n, the Feistel network operates on a domain of size 2^k where k is the smallest even integer such that 2^k >= n. Values outside [0, n) are handled via rejection sampling (recursively applying the permutation until the result is in range).

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Based on [permutation-iterator](https://github.com/asimihsan/permutation-iterator) by Asim Ihsan.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
