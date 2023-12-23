# MiniGrad: A Rust crate to backpropagate on Scalar operations.

MiniGrad allows you to create `Scalar`s (wrappers around numeric data values), and perform basic operations on them; MiniGrad internally stores a computation graph whenever an operation is performed, so that when `.backward()` is called on a specific Scalar, gradient values for each node in the graph can be computed. 

The design of the API was inspired by the current state-of-the-art deep learning frameworks (e.g., [PyTorch]([url](https://github.com/pytorch/pytorch)), [TensorFlow]([url](https://github.com/tensorflow/tensorflow))), albeit slightly altered to better fit in with idiomatic Rust. 

### Details
Internally, due to the way Rust stores integers and floating point values, all numeric data and gradients are stored in two parts, with a whole number part and a fractional part. For instance, the value would be stored with a `data_int` value of $3$, and a `data_frac` value of $2$. Therefore, to get the data stored in this `Scalar`, a convenience method `join_data()` is provided on all `Scalar` objects. The same holds true of gradients, which can be retrieved with `join_grad()`.

### Examples
```rust
// Creates a Scalar with label "a" and value 3.1
let a = &Scalar::new(3.1, "a");

// Creates a Scalar with label "c" and value 4.2
let b = &Scalar::new(4.2, "b");

// `c` should have value 3.1 + 4.2 = 7.3
let mut c = a + b;
c._label = "c";

let mut d = a * b;
d._label = "d";

let mut e = &d / &c;
e._label = "e";

let f = &Scalar::new(10.0, "f");

let mut g = f / &e;
g._label = "g";

// Backpropagate on the internal computation graph and set gradients of each Scalar involved.
g.backward();

// Check the result of the computation itself
dbg!(g.join_data()); // 2.5

// Check gradients of each node involved in computing `g`
dbg!(g.join_grad()); // 1.0
dbg!(f.join_grad()); // 0.25
dbg!(e.join_grad()); // -0.625
dbg!(c.join_grad()); // -1.25
dbg!(d.join_grad()); // 0.3125
dbg!(b.join_grad()); // -2.5
dbg!(a.join_grad()); // -0.625
```

### Credits
The inspiration and framework for this project was derived from Andrej Karpathy's [lecture]([url](https://www.youtube.com/watch?v=VMj-3S1tku0&pp=ygUJbWljcm9ncmFk)) and [library]([url](https://github.com/karpathy/micrograd)https://github.com/karpathy/micrograd) on backpropagation and deep learning.
