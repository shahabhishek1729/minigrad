//!
#![warn(missing_debug_implementations, missing_docs)]
#![allow(dead_code)]

/// A trait that applies to the following numeric types: unsigned and signed integers (i8 -> i32,
/// u8 -> u32, isize and usize) and floating point numbers (f32). 64-bit types are not yet
/// supported (u64, i64, f64).
pub trait Numeric {
    /// Every time that implements `Numeric` must be castable to a f32, so that it can be used to
    /// store a Scalar's `data` and `grad` values.
    fn to_f32(self) -> f32;
}

// For eacch of the following implementations, the `to_f32` method is simply a cast from the value
// specified to an `f32`.
impl Numeric for i8 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for i16 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for i32 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for i64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for isize {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for u8 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for u16 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for u32 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for u64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for usize {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Numeric for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}
