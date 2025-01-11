# MiniGrad: A Rust crate to backpropagate on Scalar operations.

MiniGrad allows you to create `Scalar`s (wrappers around numeric data values), and perform basic operations on them; MiniGrad internally stores a computation graph whenever an operation is performed, so that when `.backward()` is called on a specific Scalar, gradient values for each node in the graph can be computed. 

The design of the API was inspired by the current state-of-the-art deep learning frameworks (e.g., [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow)), albeit slightly altered to better fit in with idiomatic Rust. 

### Details
##### Handling of Numeric Values
Internally, due to the way Rust stores integers and floating point values, all numeric data and gradients are stored in two parts, with a whole number part and a fractional part. For instance, the value would be stored with a `data_int` value of $3$, and a `data_frac` value of $2$. Therefore, to get the data stored in this `Scalar`, a convenience method `join_data()` is provided on all `Scalar` objects. The same holds true of gradients, which can be retrieved with `join_grad()`.

##### Borrowing and Referencing Scalars
Due to another implementation detail in Rust, you can only operate on references to `Scalar`s, rather than `Scalar`s themselves. That is, if you have `let a = Scalar::new(3.1, "a")`, and `let b = Scalar::new(3.1, "b")`, in order to add these `Scalar`s, you will first need a reference to both. 
```rust
let a = Scalar::new(3.1, "a");
let b = Scalar::new(3.1, "b");

let c = a + b; // NOT VALID
let c = &a + &b; // VALID
```

### Examples
The following code segment uses the following series of computations to generate the final output:<br>
$a = 3.1$<br>
$b = 4.2$<br>
$c = a + b$<br>
$d = ab$<br>
$e = \frac{d}{c}$<br>
$f = 10$<br>
$g = \frac{f}{e}$<br>

Or, in a single expression, $g = 10 / \frac{(3.1)(4.2)}{3.1 + 4.2}$.

Calling `g.backward()` then computes the following gradients: $\frac{\partial g}{\partial f}$ (read "derivative of g with respect of f"), $\frac{\partial g}{\partial e}$, $\frac{\partial g}{\partial d}$, $\frac{\partial g}{\partial c}$, $\frac{\partial g}{\partial b}$, $\frac{\partial g}{\partial a}$ 
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

let ref mut e = &d / &c;
e._label = "e";

let f = &Scalar::new(10.0, "f");

let mut g = f / e;
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
The inspiration and framework for this project was derived from Andrej Karpathy's [lecture](https://www.youtube.com/watch?v=VMj-3S1tku0&pp=ygUJbWljcm9ncmFk) and [library](https://github.com/karpathy/micrograd) on backpropagation and deep learning.
