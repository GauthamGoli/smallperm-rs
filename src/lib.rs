//! # smallperm
//!
//! High-performance pseudo-random permutations using Feistel networks.
//!
//! This library provides O(1) memory and O(1) expected time per element for generating
//! pseudo-random permutations, making it ideal for shuffling large datasets in machine
//! learning contexts where time-to-first-sample and memory overhead are critical.
//!
//! ## Features
//!
//! - **O(1) memory**: No need to store the entire permutation
//! - **O(1) random access**: Get any element of the permutation instantly
//! - **Invertible**: Map both forward and backward between indices and values
//! - **Deterministic**: Same seed produces same permutation across runs
//! - **Supports up to 2^128 elements**
//!
//! ## Example
//!
//! ```rust
//! use smallperm::Permutation;
//!
//! // Create a permutation of 1000 elements with seed 42
//! let perm = Permutation::new(1000, 42);
//!
//! // Iterate through the permutation
//! for (i, value) in perm.iter().enumerate().take(10) {
//!     println!("perm[{}] = {}", i, value);
//! }
//!
//! // Random access
//! let value = perm.get(500).unwrap();
//! println!("perm[500] = {}", value);
//!
//! // Inverse lookup
//! let index = perm.index_of(value).unwrap();
//! assert_eq!(index, 500);
//! ```

mod feistel;

pub use feistel::{FeistelNetwork, Permutor};

/// A pseudo-random permutation with O(1) memory and O(1) access time.
///
/// This struct wraps `Permutor` and provides a cleaner API for common use cases.
#[derive(Debug, Clone)]
pub struct Permutation {
    permutor: Permutor,
}

impl Permutation {
    /// Create a new permutation of `len` elements using a u64 seed.
    ///
    /// # Panics
    ///
    /// Panics if `len` is 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use smallperm::Permutation;
    ///
    /// let perm = Permutation::new(100, 42);
    /// assert_eq!(perm.len(), 100);
    /// ```
    pub fn new(len: u128, seed: u64) -> Self {
        assert!(len > 0, "Permutation length must be greater than 0");
        Self {
            permutor: Permutor::new_with_u64_key(len, seed),
        }
    }

    /// Create a new permutation using a 256-bit key.
    ///
    /// # Panics
    ///
    /// Panics if `len` is 0.
    pub fn new_with_key(len: u128, key: [u8; 32]) -> Self {
        assert!(len > 0, "Permutation length must be greater than 0");
        Self {
            permutor: Permutor::new_with_slice_key(len, key),
        }
    }

    /// Returns the length of the permutation.
    #[inline]
    pub fn len(&self) -> u128 {
        self.permutor.max
    }

    /// Returns true if the permutation is empty (always false since len > 0 is required).
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Get the value at index `i` in the permutation.
    ///
    /// Returns `None` if `i >= len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use smallperm::Permutation;
    ///
    /// let perm = Permutation::new(100, 42);
    /// let value = perm.get(50).unwrap();
    /// assert!(value < 100);
    /// ```
    #[inline]
    pub fn get(&self, i: u128) -> Option<u128> {
        if i >= self.permutor.max {
            None
        } else {
            Some(self.permutor.forward(i))
        }
    }

    /// Get the value at index `i` without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `i < len`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: u128) -> u128 {
        self.permutor.forward(i)
    }

    /// Find the index of `value` in the permutation (inverse lookup).
    ///
    /// Returns `None` if `value >= len`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use smallperm::Permutation;
    ///
    /// let perm = Permutation::new(100, 42);
    /// let value = perm.get(50).unwrap();
    /// let index = perm.index_of(value).unwrap();
    /// assert_eq!(index, 50);
    /// ```
    #[inline]
    pub fn index_of(&self, value: u128) -> Option<u128> {
        if value >= self.permutor.max {
            None
        } else {
            Some(self.permutor.backward(value))
        }
    }

    /// Returns an iterator over the permutation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use smallperm::Permutation;
    ///
    /// let perm = Permutation::new(10, 42);
    /// let values: Vec<u128> = perm.iter().collect();
    /// assert_eq!(values.len(), 10);
    /// ```
    pub fn iter(&self) -> PermutationIter {
        PermutationIter {
            permutor: self.permutor.clone(),
        }
    }
}

impl IntoIterator for Permutation {
    type Item = u128;
    type IntoIter = PermutationIter;

    fn into_iter(self) -> Self::IntoIter {
        PermutationIter {
            permutor: self.permutor,
        }
    }
}

impl<'a> IntoIterator for &'a Permutation {
    type Item = u128;
    type IntoIter = PermutationIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over a permutation.
#[derive(Debug, Clone)]
pub struct PermutationIter {
    permutor: Permutor,
}

impl Iterator for PermutationIter {
    type Item = u128;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.permutor.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.permutor.max - self.permutor.values_returned) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PermutationIter {}

/// Sample `k` unique random indices from `[0, n)`.
///
/// This is equivalent to taking the first `k` elements of a permutation of `n` elements.
///
/// # Panics
///
/// Panics if `n == 0` or `k > n`.
///
/// # Example
///
/// ```rust
/// use smallperm::sample_indices;
///
/// let indices = sample_indices(100, 10, 42);
/// assert_eq!(indices.len(), 10);
/// for &idx in &indices {
///     assert!(idx < 100);
/// }
/// ```
pub fn sample_indices(n: u128, k: usize, seed: u64) -> Vec<u128> {
    assert!(n > 0, "n must be greater than 0");
    assert!(k as u128 <= n, "k must be <= n");

    let perm = Permutation::new(n, seed);
    perm.iter().take(k).collect()
}

/// Sample `k` unique random elements from a slice.
///
/// # Panics
///
/// Panics if `slice.is_empty()` or `k > slice.len()`.
///
/// # Example
///
/// ```rust
/// use smallperm::sample;
///
/// let data = vec![10, 20, 30, 40, 50];
/// let sampled = sample(&data, 3, 42);
/// assert_eq!(sampled.len(), 3);
/// ```
pub fn sample<T: Clone>(slice: &[T], k: usize, seed: u64) -> Vec<T> {
    assert!(!slice.is_empty(), "slice must not be empty");
    assert!(k <= slice.len(), "k must be <= slice length");

    let indices = sample_indices(slice.len() as u128, k, seed);
    indices.iter().map(|&i| slice[i as usize].clone()).collect()
}

/// Shuffle a slice and return a new Vec with the shuffled elements.
///
/// # Panics
///
/// Panics if `slice.is_empty()`.
///
/// # Example
///
/// ```rust
/// use smallperm::shuffle;
///
/// let data = vec![1, 2, 3, 4, 5];
/// let shuffled = shuffle(&data, 42);
/// assert_eq!(shuffled.len(), 5);
/// ```
pub fn shuffle<T: Clone>(slice: &[T], seed: u64) -> Vec<T> {
    assert!(!slice.is_empty(), "slice must not be empty");

    let perm = Permutation::new(slice.len() as u128, seed);
    (0..slice.len())
        .map(|i| {
            let perm_idx = perm.get(i as u128).unwrap() as usize;
            slice[perm_idx].clone()
        })
        .collect()
}

/// Shuffle a mutable slice in place.
///
/// Note: This requires O(n) additional memory for the temporary buffer.
/// For true O(1) memory shuffling, iterate using `Permutation` directly.
///
/// # Panics
///
/// Panics if `slice.is_empty()`.
///
/// # Example
///
/// ```rust
/// use smallperm::shuffle_in_place;
///
/// let mut data = vec![1, 2, 3, 4, 5];
/// shuffle_in_place(&mut data, 42);
/// ```
pub fn shuffle_in_place<T: Clone>(slice: &mut [T], seed: u64) {
    if slice.is_empty() {
        return;
    }

    let shuffled = shuffle(slice, seed);
    slice.clone_from_slice(&shuffled);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_basic() {
        let perm = Permutation::new(100, 42);
        assert_eq!(perm.len(), 100);

        let value = perm.get(50).unwrap();
        assert!(value < 100);

        let index = perm.index_of(value).unwrap();
        assert_eq!(index, 50);
    }

    #[test]
    fn test_permutation_iterator() {
        let perm = Permutation::new(10, 42);
        let values: Vec<u128> = perm.iter().collect();
        assert_eq!(values.len(), 10);

        // Check all values are unique and in range
        let mut sorted = values.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
        assert!(sorted.iter().all(|&v| v < 10));
    }

    #[test]
    fn test_sample_indices() {
        let indices = sample_indices(100, 10, 42);
        assert_eq!(indices.len(), 10);

        // Check all indices are unique and in range
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
        assert!(sorted.iter().all(|&i| i < 100));
    }

    #[test]
    fn test_sample() {
        let data: Vec<i32> = (0..100).collect();
        let sampled = sample(&data, 10, 42);
        assert_eq!(sampled.len(), 10);
    }

    #[test]
    fn test_shuffle() {
        let data: Vec<i32> = (0..10).collect();
        let shuffled = shuffle(&data, 42);
        assert_eq!(shuffled.len(), 10);

        // Check all elements are present
        let mut sorted = shuffled.clone();
        sorted.sort();
        assert_eq!(sorted, data);
    }

    #[test]
    fn test_shuffle_in_place() {
        let mut data: Vec<i32> = (0..10).collect();
        let original = data.clone();
        shuffle_in_place(&mut data, 42);

        // Check all elements are present
        let mut sorted = data.clone();
        sorted.sort();
        assert_eq!(sorted, original);
    }

    #[test]
    fn test_deterministic() {
        let perm1 = Permutation::new(100, 42);
        let perm2 = Permutation::new(100, 42);

        for i in 0..100 {
            assert_eq!(perm1.get(i), perm2.get(i));
        }
    }

    #[test]
    fn test_different_seeds() {
        let perm1 = Permutation::new(100, 42);
        let perm2 = Permutation::new(100, 43);

        // Very unlikely to be the same
        let same_count = (0..100)
            .filter(|&i| perm1.get(i) == perm2.get(i))
            .count();
        assert!(same_count < 10);
    }

    #[test]
    #[should_panic(expected = "Permutation length must be greater than 0")]
    fn test_zero_length_panics() {
        Permutation::new(0, 42);
    }

    #[test]
    fn test_out_of_bounds() {
        let perm = Permutation::new(10, 42);
        assert!(perm.get(10).is_none());
        assert!(perm.get(100).is_none());
        assert!(perm.index_of(10).is_none());
    }
}
