//! Parallelized operations over slices of numerical data.

use crate::util::*;
use rayon::prelude::*;

/// Sets each element to the same value.
/// `chunk_size` is break the work into tasks for multi-threading.
pub fn set_value<NumType: NumTrait>(
    a_slice: &mut [NumType],
    value: NumType,
    chunk_size: usize,
) {
    a_slice
        .par_chunks_mut(chunk_size)
        .for_each(|a_chunk: &mut [NumType]| {
            for a in a_chunk {
                *a = value;
            }
        });
}

pub fn square<NumType: NumTrait>(a_slice: &mut [NumType], chunk_size: usize) {
    a_slice
        .par_chunks_mut(chunk_size)
        .for_each(|a_chunk: &mut [NumType]| {
            for a in a_chunk {
                *a = *a * *a;
            }
        });
}

/// Implements a = a * b over slice elements.
pub fn multiply_by<NumType: NumTrait>(
    a_slice: &mut [NumType],
    b_slice: &[NumType],
    chunk_size: usize,
) {
    a_slice
        .par_chunks_mut(chunk_size)
        .zip(b_slice.par_chunks(chunk_size))
        .for_each(|(a_chunk, b_chunk)| {
            for (a, b) in a_chunk.iter_mut().zip(b_chunk.iter()) {
                *a = *a * *b;
            }
        });
}

/// Implements a = a / c over slice elements.
pub fn div<NumType: NumTrait>(
    a_slice: &mut [NumType],
    c: NumType,
    chunk_size: usize,
) {
    a_slice.par_chunks_mut(chunk_size).for_each(|a_chunk| {
        for a in a_chunk.iter_mut() {
            *a = *a / c;
        }
    });
}

/// Implements a = b over slice elements.
pub fn copy<NumType: NumTrait>(
    a_slice: &mut [NumType],
    b_slice: &[NumType],
    chunk_size: usize,
) {
    a_slice
        .par_chunks_mut(chunk_size)
        .zip(b_slice.par_chunks(chunk_size))
        .for_each(|(a_chunk, b_chunk)| {
            for (a, b) in a_chunk.iter_mut().zip(b_chunk.iter()) {
                *a = *b;
            }
        });
}

/// Returns the number of bits needed to represent i.
pub fn n_binary_digits(i: usize) -> usize {
    (usize::BITS - i.leading_zeros()) as usize
}

/// Implements power operation with repeated square algorithm.
pub fn power<NumType: Num + Copy + Send + Sync + std::marker::Sized>(
    n: usize,
    x_buffer: &mut [NumType],
    result_buffer: &mut [NumType],
    chunk_size: usize,
) {
    debug_assert!(x_buffer.len() == result_buffer.len());
    let mut exp = n;

    // Result buffer starts as 1
    set_value(result_buffer, NumType::one(), chunk_size);

    let k = n_binary_digits(exp);
    for _ in 0..k {
        // Is the Least significant bit 1?
        if exp & 1 == 1 {
            multiply_by(result_buffer, x_buffer, chunk_size);
        }
        exp >>= 1;
        square(x_buffer, chunk_size);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn set_values_test() {
        {
            let mut a = vec![0, 1, 2, 3, 4, 5];
            set_value(&mut a, 7, 6);
            for v in a {
                assert_eq!(v, 7);
            }
        }

        {
            let n = 1000;
            let chunk_size = 10;
            let value = n + 1;
            let mut a = Vec::with_capacity(n);
            for i in 0..n {
                a.push(i);
            }

            set_value(&mut a, value, chunk_size);

            for v in a {
                assert_eq!(v, value);
            }
        }
    }

    #[test]
    fn n_binary_digits_test() {
        assert_eq!(n_binary_digits(1), 1);
        assert_eq!(n_binary_digits(2), 2);
        assert_eq!(n_binary_digits(3), 2);
        assert_eq!(n_binary_digits(4), 3);
        assert_eq!(n_binary_digits(5), 3);
        assert_eq!(n_binary_digits(6), 3);
        assert_eq!(n_binary_digits(7), 3);
        assert_eq!(n_binary_digits(8), 4);
        assert_eq!(n_binary_digits(9), 4);
    }

    #[test]
    fn square_test() {
        let mut data = vec![1, 2, 3, 4, 5];
        square(&mut data, 1);
        for (i, x) in data.iter().enumerate() {
            assert_eq!(*x, (i + 1).pow(2));
        }
    }

    #[test]
    fn multiply_by_test() {
        let mut a = vec![1, 2, 3, 4, 5];
        let b = vec![6, 7, 8, 9, 10];
        multiply_by(&mut a, &b, 1);
        for (i, x) in a.iter().enumerate() {
            assert_eq!(*x, (i + 1) * (i + 6));
        }
    }

    #[test]
    fn power_test() {
        {
            let mut data = vec![1, 2, 3, 4, 5];
            let mut buffer = vec![0; 5];
            power(5, &mut data, &mut buffer, 1);
            for (i, x) in buffer.iter().enumerate() {
                assert_eq!(*x, (i + 1).pow(5));
            }
        }
        {
            let mut data = vec![1, 2, 3, 4, 5];
            let mut buffer = vec![0; 5];
            power(5, &mut data, &mut buffer, 100);
            for (i, x) in buffer.iter().enumerate() {
                assert_eq!(*x, (i + 1).pow(5));
            }
        }
    }
}
