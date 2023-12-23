//! An automatic gradient calculator, implemented in pure Rust for operations on individual
//! scalars. Will update the crate with support for vectors, matrices and tensors in a future
//! release
#![warn(missing_debug_implementations, missing_docs, rust_2018_idioms)]
#![allow(dead_code)]

mod numeric;

use derivative::{self, Derivative}; // Allows for ignoring a label field when comparing Scalars
use std::cell::{Cell, RefCell}; // Allows for interior mutability of a Scalar's gradient
use std::fmt::{Debug, Display};
use std::ops;

use crate::numeric::Numeric;
use float_cmp::approx_eq;

// Currently the four basic operations are supported (excluding the base operator, which is a base
// operator for leaf nodes with no children). All operations must be performed with either one or
// two children. To add an operator, the following must be implemented:
//  1. The operator must be added to the enum below.
//  2. The formatting of the operator must be defined in `Operation`'s `Display` impl.
//  3. The actual functionality of the operator must be defined (either by overriding a default
//     operator or creating a new one)
//  4. The derivative for the operator must be specified. That is, for some one-child operation
//     z(x), ∂z/∂x must be defined, and for a two-child operation z(x, y), ∂z/∂x and ∂z/∂y must be
//     defined.
//  5. (Optional, but recommended) add tests for both the operator's functionality and derivative.
#[derive(Debug, PartialEq, Eq, Ord, PartialOrd, Clone, Copy)]
enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Base,
}

/// Override the way operators are formatted
impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let print: &str = match self {
            Operation::Add => "+",
            Operation::Sub => "-",
            Operation::Mul => "*",
            Operation::Div => "/",
            Operation::Base => "BASE",
        };

        write!(f, "{}", print)
    }
}

trait Derivable {
    fn derive(&mut self);
    fn backward(&mut self);
}
// Default derivation of `Clone`, while the `Derivative` crate allows for more advanced derivations
// of `PartialEq` and `Eq` (in this case, allows us to ignore the `_label` field when comparing two
// Scalars)
#[derive(Derivative, Clone)]
// Allows for more advanced derivations
#[derivative(PartialEq, Eq, PartialOrd, Ord)]
struct Scalar<'a> {
    data_sign: i8,
    data_int: u32,
    data_frac: u32,
    data_digits: u32,
    _children: Vec<&'a Self>,
    _grad_sign: Cell<i8>,
    _grad_int: Cell<u32>,
    _grad_frac: Cell<u32>,
    _grad_digits: Cell<u32>,
    _op: Operation,
    #[derivative(PartialEq = "ignore")]
    _label: &'static str,
}

impl Debug for Scalar<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut child_str = String::new();
        for c in &self._children[..] {
            child_str.push_str(&format!(
                "Scalar(label = {}, data = {}, grad = {}), ",
                c._label,
                c.join_data(),
                c.join_grad()
            ));
        }

        write!(
            f,
            "Scalar(label = {}, data = {}, grad = {}, children = [{}], operation = {})",
            self._label,
            self.join_data(),
            self.join_grad(),
            child_str,
            self._op
        )
    }
}

impl Scalar<'_> {
    fn split_f32(x: f32) -> (i8, u32, u32, u32) {
        let sign = if x < 0. { -1i8 } else { 1i8 };

        if x == x.floor() {
            return (sign, x.abs() as u32, 0, 1);
        }

        let s = x.to_string();
        let mut spl = s.split('.');

        let mut int: &str = "";
        let mut frac: &str = "";

        while let Some(s) = spl.next() {
            if int.is_empty() {
                int = s;
            } else {
                frac = s;
            }
        }

        // Ignore a preceding negative sign (if it exists). The following is used instead of a
        // `int.replace("-", "")` because a negative sign should only be at the beginning of the
        // number. Anywhere else and we should panic.
        if &int[0..1] == "-" {
            int = &int[1..];
        }

        (
            sign,
            int.parse::<u32>()
                .expect("The integral (whole) number before the decimal point should be valid!"),
            frac.parse::<u32>()
                .expect("The fractional part after the decimal point should be valid!"),
            frac.len() as u32,
        )
    }

    fn join_f32(sign: i8, int: u32, frac: u32, n_digits: u32) -> f32 {
        (sign as f32) * (int as f32 + frac as f32 / 10f32.powi(n_digits as i32))
    }

    fn join_data(&self) -> f32 {
        Self::join_f32(
            self.data_sign,
            self.data_int,
            self.data_frac,
            self.data_digits,
        )
    }

    fn join_grad(&self) -> f32 {
        Self::join_f32(
            self._grad_sign.get(),
            self._grad_int.get(),
            self._grad_frac.get(),
            self._grad_digits.get(),
        )
    }

    fn new(data: impl Numeric, label: &'static str) -> Self {
        let (data_sign, data_int, data_frac, data_digits) = Self::split_f32(data.to_f32());

        Scalar {
            data_sign,
            data_int,
            data_frac,
            data_digits,
            _children: vec![],
            _grad_sign: Cell::new(1i8),
            _grad_int: Cell::new(0u32),
            _grad_frac: Cell::new(0u32),
            _grad_digits: Cell::new(0u32),
            _op: Operation::Base,
            _label: label,
        }
    }

    fn new_full<'a>(
        data: impl Numeric,
        _ch: Vec<&'a Self>,
        _grad: f32,
        _op: Operation,
        label: &'static str,
    ) -> Scalar<'a> {
        let (data_sign, data_int, data_frac, data_digits) = Self::split_f32(data.to_f32());
        let (_grad_sign, _grad_int, _grad_frac, _grad_digits) = Scalar::split_f32(_grad);

        Scalar {
            data_sign,
            data_int,
            data_frac,
            data_digits,
            _children: _ch,
            _grad_sign: Cell::new(_grad_sign),
            _grad_int: Cell::new(_grad_int),
            _grad_frac: Cell::new(_grad_frac),
            _grad_digits: Cell::new(_grad_digits),
            _op,
            _label: label,
        }
    }

    fn update_grad(&self, new_grad: f32) {
        let (new_grad_sign, new_grad_int, new_grad_frac, new_grad_digits): (i8, u32, u32, u32) =
            Self::split_f32(new_grad);

        (*self)._grad_sign.set(new_grad_sign);
        (*self)._grad_int.set(new_grad_int);
        (*self)._grad_frac.set(new_grad_frac);
        (*self)._grad_digits.set(new_grad_digits);
    }
}

impl Derivable for Scalar<'_> {
    fn derive(&mut self) {
        let _parent_grad = self.join_grad();

        if self._children.len() == 0 {
            return;
        }

        let orig_grad0 = self._children[0].join_grad();
        let orig_grad1 = self._children[1].join_grad();

        match self._op {
            // For each of the following operations, let z be the final output value produced by
            // the overall computation, a (and b, if applicable) be the current child nodes being
            // processed, and y be the output of applying the found operation to those nodes.
            Operation::Add => {
                // Here, we have y = a + b. The following holds:
                //  1. ∂y/∂a = 1.0, and therefore, ∂z/∂a = ∂z/∂y
                self._children[0].update_grad(orig_grad0 + _parent_grad);
                //  2. ∂y/∂b = 1.0, and therefore, ∂z/∂b = ∂z/∂y
                self._children[1].update_grad(orig_grad1 + _parent_grad);
            }
            Operation::Sub => {
                // Here, we have y = a - b. The following holds:
                //  1. ∂y/∂a = 1.0, and therefore, ∂z/∂a = ∂z/∂y
                self._children[0].update_grad(orig_grad0 + _parent_grad);
                //  2. ∂y/∂b = -1.0, and therefore, ∂z/∂b = -1.0 * ∂z/∂y
                self._children[1].update_grad(orig_grad1 + _parent_grad * -1.);
            }
            Operation::Mul => {
                // We will need access to `a` and `b` to calculate the derivatives, unlike the
                // previous operations.
                let orig_data0 = self._children[0].join_data(); // Represents `a` here
                let orig_data1 = self._children[1].join_data(); // Represents `b` here

                // Here, we have y = ab. The following holds:
                //  1. ∂y/∂a = b, and therefore, ∂z/∂a = ∂z/∂y * b
                self._children[0].update_grad(orig_grad0 + _parent_grad * orig_data1);
                //  2. ∂y/∂b = a, and therefore, ∂z/∂b = ∂z/∂y * a
                self._children[1].update_grad(orig_grad1 + _parent_grad * orig_data0);
            }
            Operation::Div => {
                // We will need access to `a` and `b` here as well.
                let orig_data0 = self._children[0].join_data();
                let orig_data1 = self._children[1].join_data();

                // Here, we have y = a ÷ b, or y = 1/b * a. The following holds:
                //  1. ∂y/∂a = 1/b, and therefore, ∂z/∂a = ∂z/∂y * 1/b
                (*self._children[0]).update_grad(orig_grad0 + _parent_grad * 1. / orig_data1);
                //  2. ∂y/∂b = -a * b^-2, and therefore, ∂z/∂b = -∂z/∂y * (a/b^2)
                self._children[1]
                    .update_grad(orig_grad1 - _parent_grad * orig_data0 / orig_data1.powi(2));
            }
            // TODO: Implement more operations here
            _ => (), // The only other case here is the Base operation, which is just the default
                     // for leaf nodes, so no need to handle those (leaf nodes have no children).
        }
    }

    /// Given a Scalar, takes its derivative and the derivative of all its children (direct or
    /// indirect) in a recursive fashion, until every node in the Scalar's
    fn backward(&mut self) {
        self.update_grad(1.0);

        //let mut topology: Vec<Scalar<'_>> = Vec::new();
        //let mut visited: Vec<Scalar<'_>> = Vec::new();

        //let mut topology = build_topo(&mut topology, &mut visited, self.clone());
        let topology = parse_topology(self);

        dbg!(&topology);

        for s in topology.iter() {
            //&mut (*s).derive();
            let node = s.borrow_mut();
            node.clone().derive();
            dbg!(&node);
        }
    }
}

fn parse_topology<'a>(node: &'a Scalar<'a>) -> Vec<RefCell<&'a Scalar<'a>>> {
    let mut topology: Vec<RefCell<&Scalar<'_>>> = vec![RefCell::new(node)];
    let mut curr_level: Vec<&Scalar<'_>> = vec![node];
    let mut visited: Vec<&Scalar<'_>> = vec![];

    while curr_level.len() > 0 {
        for n in curr_level[0]._children.iter() {
            if !visited.contains(&n) {
                visited.push(n);
                curr_level.push(n);
                topology.push(RefCell::new(n));
            }
        }
        curr_level.remove(0);
    }

    topology
}

fn build_topo<'a>(
    topo: &mut Vec<Scalar<'a>>,
    visited: &mut Vec<Scalar<'a>>,
    v: Scalar<'a>,
) -> Vec<Scalar<'a>> {
    assert_eq!(
        &Scalar::new_full(3.0, vec![], 3.2, Operation::Add, "d"),
        &Scalar::new_full(3.0, vec![], 3.2, Operation::Add, "e")
    );

    //if !visited.contains(v) {
    if !visited.contains(&v) {
        dbg!(&v);
        visited.push(v.clone());
        topo.push(v.clone());
        for child in v._children.iter() {
            let c_ = child.clone().clone();
            build_topo(topo, visited, c_);
        }
    }

    topo.clone()
}

impl<'a> ops::Add for &'a Scalar<'a> {
    type Output = Scalar<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        Scalar::<'a>::new_full(
            self.join_data() + rhs.join_data(),
            vec![self, rhs],
            0.0,
            Operation::Add,
            "",
        )
    }
}
//
// impl<'a> ops::Add<&dyn Numeric> for &'a Scalar<'a> {
//     type Output = Scalar<'a>;
//     fn add(self, rhs: &dyn Numeric) -> Self::Output {
//         static rhs = Scalar::new(rhs.to_f32())
//         Scalar::<'a>::new_full(
//             self.join_data() + rhs.to_f32(),
//             vec![self, &Scalar::new(rhs.to_f32(), "tmp")],
//             0.0,
//             Operation::Add,
//             "",
//         )
//     }
// }
//
impl<'a> ops::Sub for &'a Scalar<'a> {
    type Output = Scalar<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        Scalar::<'a>::new_full(
            self.join_data() - rhs.join_data(),
            vec![self, rhs],
            0.0,
            Operation::Sub,
            "",
        )
    }
}

impl<'a> ops::Mul for &'a Scalar<'a> {
    type Output = Scalar<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        Scalar::<'a>::new_full(
            self.join_data() * rhs.join_data(),
            vec![self, rhs],
            0.0,
            Operation::Mul,
            "",
        )
    }
}

impl<'a> ops::Div for &'a Scalar<'a> {
    type Output = Scalar<'a>;
    fn div(self, rhs: Self) -> Self::Output {
        Scalar::<'a>::new_full(
            self.join_data() / rhs.join_data(),
            vec![self, rhs],
            0.0,
            Operation::Div,
            "",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod basic_ops {
        use super::*;
        #[test]
        fn test_add() {
            let s1 = &Scalar::new(3.2f32, "s1");
            let s2 = &Scalar::new(4.7, "s2");

            let result = Scalar::new_full(3.2 + 4.7, vec![s1, s2], 0.0, Operation::Add, "result");

            assert_eq!(s1 + s2, result);
        }

        #[test]
        fn test_sub() {
            let s1 = &Scalar::new(3.2, "s1");
            let s2 = &Scalar::new(4.7, "s2");

            let result = Scalar::new_full(3.2 - 4.7, vec![s1, s2], 0.0, Operation::Sub, "result");

            assert_eq!(s1 - s2, result);
        }

        #[test]
        fn test_mul() {
            let s1 = &Scalar::new(3.2, "s1");
            let s2 = &Scalar::new(4.7, "s2");

            let result = Scalar::new_full(3.2 * 4.7, vec![s1, s2], 0.0, Operation::Mul, "result");

            assert_eq!(s1 * s2, result);
        }

        #[test]
        fn test_div() {
            let s1 = &Scalar::new(3.2, "s1");
            let s2 = &Scalar::new(4.7, "s2");

            let result = Scalar::new_full(3.2 / 4.7, vec![s1, s2], 0.0, Operation::Div, "result");

            assert_eq!(s1 / s2, result);
        }

        #[test]
        fn test_joins() {
            let s1 = Scalar::join_f32(1, 23, 3, 2);
            assert_eq!(s1, 23.03);

            let s2 = Scalar::join_f32(-1, 23, 3, 2);
            assert_eq!(s2, -23.03);
        }

        #[test]
        fn test_partial_eq() {
            let s1 = Scalar::new(3.2, "s1");
            let s2 = Scalar::new(3.2, "s2");
            let s3 = Scalar::new(3.2, "s3");
            let s4 = Scalar::new(3.2, "s4");
            let v = vec![s1.clone(), s2.clone(), s3.clone()];

            assert!(v.contains(&s4));
        }
    }

    mod backward {
        use float_cmp::approx_eq;

        use super::*;

        #[test]
        fn test_add_backward() {
            let a = &Scalar::new(3.1, "a");
            let b = &Scalar::new(4.2, "b");
            let mut c = a + b;

            c.backward();

            assert_eq!(c.join_grad(), 1.0);
            assert_eq!(b.join_grad(), 1.0);
            assert_eq!(a.join_grad(), 1.0);
        }

        #[test]
        fn test_sub_backward() {
            let a = &Scalar::new(3.1, "a");
            let b = &Scalar::new(4.2, "b");
            let mut c = a - b;

            c.backward();

            assert_float_eq(c.join_data(), -1.1);
            assert_eq!(c.join_grad(), 1.0);
            assert_eq!(a.join_grad(), 1.0);
            assert_eq!(b.join_grad(), -1.0);
        }

        #[test]
        fn test_mul_backward() {
            let a = &Scalar::new(3.1, "a");
            let b = &Scalar::new(4.2, "b");
            let mut c = a * b;

            c.backward();

            assert_eq!(c.join_grad(), 1.0);
            assert_eq!(b.join_grad(), 3.1);
            assert_eq!(a.join_grad(), 4.2);
        }

        #[test]
        fn test_div_backward() {
            let a = &Scalar::new(3.1, "a");
            let b = &Scalar::new(4.2, "b");
            let mut c = a / b;

            c.backward();

            assert_eq!(c.join_grad(), 1.);
            assert!(approx_eq!(f32, a.join_grad(), 1. / 4.2, ulps = 4));
            assert_eq!(b.join_grad(), -3.1 * 4.2f32.powi(-2));
        }

        #[test]
        fn test_compound_fn() {
            let a = &Scalar::new(-4.0, "a");
            let b = &Scalar::new(2.0, "b");
            let mut c = a + b; // -2.0
            c._label = "c";
            let mut d = a * b; // -8.0
            d._label = "d";
            let mut e = &d / &c; // 4.0
            e._label = "e";
            let f = &Scalar::new(10.0, "f");
            let mut g = f / &e;
            g._label = "g";

            g.backward();

            // g(a, b) = 10 / ab/(a+b)
            //         = 10/b + 10/a
            //         = 5 - 2.5 = 2.5.
            //
            // g(e, f) = f/e
            // ∂g/∂e = -f/e^2, ∂g/∂f = 1/e
            //
            // g(c, d) = d/c
            // ∂g/∂e = -d/c^2, ∂g/∂d = 1/c
            //
            // ∂g/∂a = -10/a^2, ∂g/∂b = -10/b^2
            // ∂g/da = -0.625,    ∂g/∂b = -2.5
            assert_float_eq(g.join_data(), 2.5);
            assert_float_eq(g.join_grad(), 1.0);
            assert_float_eq(f.join_grad(), 0.25);
            assert_float_eq(e.join_grad(), -0.625);

            assert_float_eq(c.join_grad(), -1.25);
            assert_float_eq(d.join_grad(), 0.3125);
            assert_float_eq(b.join_grad(), -2.5);
            assert_float_eq(a.join_grad(), -0.625);
        }
    }
}

/// Given two floats `a` and `b`, asserts that the two floats are equal to each other, taking
/// into account precision and rounding errors that might change some of the later decimal
/// points of either float.
fn assert_float_eq(a: f32, b: f32) {
    assert!(approx_eq!(f32, a, b, ulps = 4));
}
