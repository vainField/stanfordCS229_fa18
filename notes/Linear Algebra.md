- 还没有完全清楚的地方
  - 正定
  - 特征值和特征向量，diagonalizable

# **<u>Linear Algebra Review and Reference</u>**

## **1 Basic Concepts and Notation**

- Linear algebra provides <u>*a way of compactly representing and operating on sets of linear equations*</u>.

- For example, consider the following system of equations:
  $$
  \begin{aligned}
  4 x_{1}-5 x_{2} &=-13 \\
  -2 x_{1}+3 x_{2} &=9 .
  \end{aligned}
  $$

  - In matrix notation, we can write the system more compactly
    $$
    A x=b
    $$
    with
    $$
    A=\left[\begin{array}{cc}
    4 & -5 \\
    -2 & 3
    \end{array}\right], \quad b=\left[\begin{array}{c}
    -13 \\
    9
    \end{array}\right] .
    $$
    

### 1.1 Basic Notation

- By $A \in \mathbb{R}^{m \times n}$ we denote a **matrix** with $m$ rows and $n$ columns, where the entries of $A$ are real numbers.
- By $x \in \mathbb{R}^{n}$, we denote a vector with $n$ entries. By convention, an $n$-dimensional vector is often thought of as a matrix with $n$ rows and 1 column, known as a **column vector**. If we want to explicitly represent a **row vector** - a matrix with 1 row and $n$ columns - we typically write $x^{T}$.
- We denote the $j$ th column of $A$ by $a_{j}$ or $A_{:, j}$ 
- We denote the $i$ th row of $A$ by $a_{i}^{T}$ or $A_{i,:}$ 

## **2 Matrix Multiplication**

-  The product of two matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$ is the matrix
   $$
   C=A B \in \mathbb{R}^{m \times p}
   $$
   where
   $$
   C_{i j}=\sum_{k=1}^{n} A_{i k} B_{k j}
   $$

   - Note that in order for the matrix product to exist, *<u>the number of columns in $A$ must equal the number of rows in $B$.</u>*

### 2.1 Vector-Vector Products

- Given two vectors $x, y \in \mathbb{R}^{n}$, the quantity $x^{T} y$, sometimes called the **inner product** or **dot product** of the vectors, is a real number given by
  $$
  x^{T} y \in \mathbb{R}=\left[\begin{array}{llll}
  x_{1} & x_{2} & \cdots & x_{n}
  \end{array}\right]\left[\begin{array}{c}
  y_{1} \\
  y_{2} \\
  \vdots \\
  y_{n}
  \end{array}\right]=\sum_{i=1}^{n} x_{i} y_{i}
  $$

  - Observe that inner products are really just special case of matrix multiplication. 
  - Note that <u>*it is always the case that $x^{T} y=y^{T} x$.*</u>

- Given vectors $x \in \mathbb{R}^{m}, y \in \mathbb{R}^{n}$ (not necessarily of the same size), $x y^{T} \in \mathbb{R}^{m \times n}$ is called the **outer product** of the vectors. It is a matrix whose entries are given by $\left(x y^{T}\right)_{i j}=x_{i} y_{j}$, i.e.,
  $$
  x y^{T} \in \mathbb{R}^{m \times n}=\left[\begin{array}{c}
  x_{1} \\
  x_{2} \\
  \vdots \\
  x_{m}
  \end{array}\right]\left[\begin{array}{llll}
  y_{1} & y_{2} & \cdots & y_{n}
  \end{array}\right]=\left[\begin{array}{cccc}
  x_{1} y_{1} & x_{1} y_{2} & \cdots & x_{1} y_{n} \\
  x_{2} y_{1} & x_{2} y_{2} & \cdots & x_{2} y_{n} \\
  \vdots & \vdots & \ddots & \vdots \\
  x_{m} y_{1} & x_{m} y_{2} & \cdots & x_{m} y_{n}
  \end{array}\right]
  $$

  - As an example of <u>*how the outer product can be useful*</u>, let $\mathbf{1} \in \mathbb{R}^{n}$ denote an $n$-dimensional vector whose entries are all equal to 1 . Furthermore, consider the matrix $A \in \mathbb{R}^{m \times n}$ whose columns are all equal to some vector $x \in \mathbb{R}^{m}$. Using outer products, we can represent $A$ compactly as,
    $$
    A=\left[\begin{array}{cccc}
    \mid & \mid & & \mid \\
    x & x & \cdots & x \\
    \mid & \mid & & \mid
    \end{array}\right]=\left[\begin{array}{cccc}
    x_{1} & x_{1} & \cdots & x_{1} \\
    x_{2} & x_{2} & \cdots & x_{2} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{m} & x_{m} & \cdots & x_{m}
    \end{array}\right]=\left[\begin{array}{c}
    x_{1} \\
    x_{2} \\
    \vdots \\
    x_{m}
    \end{array}\right]\left[\begin{array}{llll}
    1 & 1 & \cdots & 1
    \end{array}\right]=x \mathbf{1}^{T}
    $$

### 2.2 Matrix-Vector Products

- Given a matrix $A \in \mathbb{R}^{m \times n}$ and a vector $x \in \mathbb{R}^{n}$, their product is a vector $y=A x \in \mathbb{R}^{m}$. There are a couple ways of looking at matrix-vector multiplication, and we will look at each of them in turn.

  - If we write $A$ by rows, then we can express $A x$ as,
    $$
    y=A x=\left[\begin{array}{ccc}
    - & a_{1}^{T} & - \\
    - & a_{2}^{T} & - \\
    \vdots \\
    - & a_{m}^{T} & -
    \end{array}\right] x=\left[\begin{array}{c}
    a_{1}^{T} x \\
    a_{2}^{T} x \\
    \vdots \\
    a_{m}^{T} x
    \end{array}\right]
    $$

    - In other words, <u>*the $i$ th entry of $y$ is equal to the inner product of the $i$ th row of $A$*</u> and $x$, $y_{i}=a_{i}^{T} x$.

  - Alternatively, let's write $A$ in column form. In this case we see that,
    $$
    y=A x=\left[\begin{array}{cccc}
    \mid & \mid & & \mid \\
    a_{1} & a_{2} & \cdots & a_{n} \\
    \mid & \mid & & \mid
    \end{array}\right]\left[\begin{array}{c}
    x_{1} \\
    x_{2} \\
    \vdots \\
    x_{n}
    \end{array}\right]=\left[a_{1}\right] x_{1}+\left[a_{2}\right] x_{2}+\ldots+\left[a_{n}\right] x_{n} .
    $$

    - In other words, <u>*$y$ is a linear combination of the columns of $A$*</u>, where the coefficients of the linear combination are given by the entries of $x$.

- So far we have been multiplying on the right by a column vector, but it is also possible to multiply on the left by a row vector. This is written, $y^{T}=x^{T} A$ for $A \in \mathbb{R}^{m \times n}, x \in \mathbb{R}^{m}$, and $y \in \mathbb{R}^{n}$. As before, we can express $y^{T}$ in two obvious ways, depending on whether we express $A$ in terms on its rows or columns. 

  - In the first case we express $A$ in terms of its columns, which gives

    $$
    y^{T}=x^{T} A=x^{T}\left[\begin{array}{cccc}
    \mid & \mid & & \mid \\
    a_{1} & a_{2} & \cdots & a_{n} \\
    \mid & \mid & & \mid
    \end{array}\right]=\left[\begin{array}{llll}
    x^{T} a_{1} & x^{T} a_{2} & \cdots & x^{T} a_{n}
    \end{array}\right]
    $$

    - which demonstrates that *<u>the $i$ th entry of $y^{T}$ is equal to the inner product of $x$ and the $i$ th column of $A$.</u>*

  - Finally, expressing $A$ in terms of rows we get the final representation of the vector-matrix product,
    $$
    \begin{aligned}
    y^{T} &=x^{T} A \\
    &=\left[\begin{array}{llll}
    x_{1} & x_{2} & \cdots & x_{n}
    \end{array}\right]\left[\begin{array}{ccc}
    - & a_{1}^{T} & - \\
    - & a_{2}^{T} & - \\
    & \vdots \\
    - & a_{m}^{T} & -
    \end{array}\right] \\
    &=x_{1}\left[\begin{array}{lll}
    - & a_{1}^{T} & -
    \end{array}\right]+x_{2}\left[\begin{array}{lll}
    - & a_{2}^{T} & -
    \end{array}\right]+\ldots+x_{n}\left[\begin{array}{lll}
    - & a_{n}^{T} & -
    \end{array}\right]
    \end{aligned}
    $$

    - so we see that <u>*$y^{T}$ is a linear combination of the rows of $A$*</u>, where the coefficients for the linear combination are given by the entries of $x$.

### 2.3 Matrix-Matrix Products

- Armed with this knowledge, we can now look at <u>*four different (but, of course, equivalent) ways of viewing the matrix-matrix multiplication*</u> $C=A B$ as defined at the beginning of this section.

  - <u>First</u>, we can view matrix-matrix multiplication as **a set of vector-vector products**. The most obvious viewpoint, which follows immediately from the definition, is that the $(i, j)$th entry of $C$ is equal to the inner product of the $i$ th row of $A$ and the $j$ th column of $B$. Symbolically, this looks like the following,
    $$
    C=A B=\left[\begin{array}{ccc}
    - & a_{1}^{T} & - \\
    - & a_{2}^{T} & - \\
    & \vdots \\
    - & a_{m}^{T} & -
    \end{array}\right]\left[\begin{array}{cccc}
    \mid & \mid & & \mid \\
    b_{1} & b_{2} & \cdots & b_{p} \\
    \mid & \mid & & \mid
    \end{array}\right]=\left[\begin{array}{cccc}
    a_{1}^{T} b_{1} & a_{1}^{T} b_{2} & \cdots & a_{1}^{T} b_{p} \\
    a_{2}^{T} b_{1} & a_{2}^{T} b_{2} & \cdots & a_{2}^{T} b_{p} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m}^{T} b_{1} & a_{m}^{T} b_{2} & \cdots & a_{m}^{T} b_{p}
    \end{array}\right]
    $$

    - Remember that since $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, $a_{i} \in \mathbb{R}^{n}$ and $b_{j} \in \mathbb{R}^{n}$, so these inner products all make sense. <u>*This is the most "natural" representation*</u> when we represent $A$ by rows and $B$ by columns.
    - <u>Alternatively</u>, we can represent $A$ by columns, and $B$ by rows. This representation leads to a much trickier interpretation of $A B$ as <u>*a sum of outer products*</u>. Symbolically,
      $$
      C=A B=\left[\begin{array}{cccc}
      \mid & \mid & & \mid \\
      a_{1} & a_{2} & \cdots & a_{n} \\
      \mid & \mid & & \mid
      \end{array}\right]\left[\begin{array}{ccc}
      - & b_{1}^{T} & - \\
      - & b_{2}^{T} & - \\
      \vdots \\
      - & b_{n}^{T} & -
      \end{array}\right]=\sum_{i=1}^{n} a_{i} b_{i}^{T}
      $$

      - Put another way, $A B$ is equal to the sum, over all $i$, of the outer product of the $i$ th column of $A$ and the $i$ th row of $B$. Since, in this case, $a_{i} \in \mathbb{R}^{m}$ and $b_{i} \in \mathbb{R}^{p}$, the dimension of the outer product $a_{i} b_{i}^{T}$ is $m \times p$, which coincides with the dimension of $C$.

  - <u>Second</u>, we can also view matrix-matrix multiplication as **a set of matrix-vector products**. Specifically, if we represent $B$ by columns, we can view the columns of $C$ as matrix-vector products between $A$ and the columns of $B$. Symbolically,
    $$
    C=A B=A\left[\begin{array}{cccc}
    \mid & \mid & & \mid \\
    b_{1} & b_{2} & \cdots & b_{p} \\
    \mid & \mid & & \mid
    \end{array}\right]=\left[\begin{array}{cccc}
    \mid & \mid & & \mid \\
    A b_{1} & A b_{2} & \cdots & A b_{p} \\
    \mid & \mid & & \mid
    \end{array}\right]
    $$

    - Here the $i$ th column of $C$ is given by the matrix-vector product with the vector on the right, $c_{i}=A b_{i}$. These matrix-vector products can in turn be interpreted using both viewpoints given in the previous subsection.

    - <u>Finally</u>, we have the analogous viewpoint, where we represent $A$ by rows, and view <u>*the rows of $C$ as the matrix-vector product between the rows of $A$ and $B$.*</u> Symbolically,
      $$
      C = A B = \left[\begin{array}{ccc}
      - & a_{1}^{T} & - \\
      - & a_{2}^{T} & - \\
      \vdots \\
      - & a_{m}^{T} & -
      \end{array}\right] B=\left[\begin{array}{ccc}
      - & a_{1}^{T} B & - \\
      - & a_{2}^{T} B & - \\
      \vdots \\
      - & a_{m}^{T} B & -
      \end{array}\right]
      $$

      - Here the $i$ th row of $C$ is given by the matrix-vector product with the vector on the left, $c_i^T = a_i^TB$

- In addition to this, it is useful to know a few basic properties of matrix multiplication at a higher level:

  - Matrix multiplication is **associative**: $(A B) C=A(B C)$.
  - Matrix multiplication is **distributive**: $A(B+C)=A B+A C$.
  - Matrix multiplication is, in general, **not commutative**; that is, it can be the case that $A B \neq B A$.

- For example, to check the associativity of matrix multiplication, we can verify this directly using the definition of matrix multiplication:
  $$
  \begin{aligned}
  ((A B) C)_{i j} &=\sum_{k=1}^{p}(A B)_{i k} C_{k j}=\sum_{k=1}^{p}\left(\sum_{l=1}^{n} A_{i l} B_{l k}\right) C_{k j} \\
  &=\sum_{k=1}^{p}\left(\sum_{l=1}^{n} A_{i l} B_{l k} C_{k j}\right)=\sum_{l=1}^{n}\left(\sum_{k=1}^{p} A_{i l} B_{l k} C_{k j}\right) \\
  &=\sum_{l=1}^{n} A_{i l}\left(\sum_{k=1}^{p} B_{l k} C_{k j}\right)=\sum_{l=1}^{n} A_{i l}(B C)_{l j}=(A(B C))_{i j}
  \end{aligned}
  $$
  

  - Here, the first and last two equalities simply use the definition of matrix multiplication, 
  - the third and fifth equalities use <u>*the distributive property for scalar multiplication over addition*</u>, 
  - and the fourth equality uses <u>*the commutative and associativity of scalar addition*</u>. 
  - This technique for **proving matrix properties by reduction to simple scalar properties** will come up often, so make sure you're familiar with it.


## **3 Operations and Properties**

### 3.1 The Identity Matrix and Diagonal Matrices

- The **identity matrix**, denoted $I \in \mathbb{R}^{n \times n}$, is a square matrix with ones on the diagonal and zeros everywhere else. That is,
  $$
  I_{i j}= \begin{cases}1 & i=j \\ 0 & i \neq j\end{cases}
  $$

  - It has the property that for all $A \in \mathbb{R}^{m \times n}$,

    $$
    A I=A=I A .
    $$

- A **diagonal matrix** is a matrix where all non-diagonal elements are 0 . This is typically denoted $D=\operatorname{diag}\left(d_{1}, d_{2}, \ldots, d_{n}\right)$, with
  $$
  D_{i j}= \begin{cases}d_{i} & i=j \\ 0 & i \neq j\end{cases}
  $$

- Clearly, $I=\operatorname{diag}(1,1, \ldots, 1)$.

### 3.2 The Transpose

- The **transpose** of a matrix results from "flipping" the rows and columns. Given a matrix $A \in \mathbb{R}^{m \times n}$, its transpose, written $A^{T} \in \mathbb{R}^{n \times m}$, is the $n \times m$ matrix whose entries are given by
  $$
  \left(A^{T}\right)_{i j}=A_{j i} .
  $$

- The following properties of transposes are easily verified:

  - $\left(A^{T}\right)^{T}=A$
  - $(A B)^{T}=B^{T} A^{T}$
  - $(A+B)^{T}=A^{T}+B^{T}$

### 3.3 Symmetric Matrices

- A square matrix $A \in \mathbb{R}^{n \times n}$ is **symmetric** if $A=A^{T}$. It is **anti-symmetric** if $A=-A^{T}$. 

  - It is easy to show that for any matrix $A \in \mathbb{R}^{n \times n}$, the matrix $A+A^{T}$ is symmetric and the matrix $A-A^{T}$ is anti-symmetric.

  - From this it follows that any square matrix $A \in \mathbb{R}^{n \times n}$ can be represented as a sum of a symmetric matrix and an anti-symmetric matrix, since
    $$
    A=\frac{1}{2}\left(A+A^{T}\right)+\frac{1}{2}\left(A-A^{T}\right)
    $$

- <u>*Symmetric matrices occur a great deal in practice, and they have many nice properties.*</u>

- It is common to denote the set of all symmetric matrices of size $n$ as $\mathbb{S}^{n}$, so that $A \in \mathbb{S}^{n}$ means that $A$ is a symmetric $n \times n$ matrix.

### 3.4 The Trace

- The **trace** of a <u>*square matrix*</u> $A \in \mathbb{R}^{n \times n}$, denoted $\operatorname{tr}(A)$ (or just $\operatorname{tr} A$ if the parentheses are obviously implied), is the sum of diagonal elements in the matrix:
  $$
  \operatorname{tr} A=\sum_{i=1}^{n} A_{i i} .
  $$

- The trace has the following properties:

  - For $A \in \mathbb{R}^{n \times n}, \operatorname{tr} A=\operatorname{tr} A^{T}$.
  - For $A, B \in \mathbb{R}^{n \times n}, \operatorname{tr}(A+B)=\operatorname{tr} A+\operatorname{tr} B$.
  - For $A \in \mathbb{R}^{n \times n}, t \in \mathbb{R}, \operatorname{tr}(t A)=t \operatorname{tr} A$.
  - For $A, B$ such that $A B$ is square, $\operatorname{tr} A B=\operatorname{tr} B A$.
  - For $A, B, C$ such that $A B C$ is square, $\operatorname{tr} A B C=\operatorname{tr} B C A=\operatorname{tr} C A B$, and so on for the product of more matrices.

- We'll prove the fourth property given above. Suppose that $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times m}$
  $$
  \begin{aligned}
  \operatorname{tr} A B &=\sum_{i=1}^{m}(A B)_{i i}=\sum_{i=1}^{m}\left(\sum_{j=1}^{n} A_{i j} B_{j i}\right) \\
  &=\sum_{i=1}^{m} \sum_{j=1}^{n} A_{i j} B_{j i}=\sum_{j=1}^{n} \sum_{i=1}^{m} B_{j i} A_{i j} \\
  &=\sum_{j=1}^{n}\left(\sum_{i=1}^{m} B_{j i} A_{i j}\right)=\sum_{j=1}^{n}(B A)_{j j}=\operatorname{tr} B A
  \end{aligned}
  $$

### 3.5 Norms

- A **norm** of a vector $\|x\|$ is informally a measure of the "length" of the vector. For example, we have the commonly-used Euclidean or $\ell_{2}$ norm,
  $$
  \|x\|_{2}=\sqrt{\sum_{i=1}^{n} x_{i}^{2}}
  $$

  - Note that $\|x\|_{2}^{2}=x^{T} x$.

  - Other examples of norms are the $\ell_{1}$ norm,
    $$
    \|x\|_{1}=\sum_{i=1}^{n}\left|x_{i}\right|
    $$

  - and the $\ell_{\infty}$ norm,
    $$
    \|x\|_{\infty}=\max _{i}\left|x_{i}\right| .
    $$
    

- More formally, a norm is any function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ that satisfies 4 properties:

  1. For all $x \in \mathbb{R}^{n}, f(x) \geq 0$ (non-negativity).
  2. $f(x)=0$ if and only if $x=0$ (definiteness).
  3. For all $x \in \mathbb{R}^{n}, t \in \mathbb{R}, f(t x)=|t| f(x)$ (homogeneity).
  4. For all $x, y \in \mathbb{R}^{n}, f(x+y) \leq f(x)+f(y)$ (triangle inequality).

- In fact, all three norms presented so far are examples of the family of $\ell_{p}$ norms, which are parameterized by a real number $p \geq 1$, and defined as
  $$
  \|x\|_{p}=\left(\sum_{i=1}^{n}\left|x_{i}\right|^{p}\right)^{1 / p}
  $$

- Norms can also be defined for matrices, such as the Frobenius norm,
  $$
  \|A\|_{F}=\sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} A_{i j}^{2}}=\sqrt{\operatorname{tr}\left(A^{T} A\right)}
  $$

### 3.6 Linear Independence and Rank

- A set of vectors $\left\{x_{1}, x_{2}, \ldots x_{n}\right\} \subset \mathbb{R}^{m}$ is said to be **(linearly) independent** if no vector can be represented as a linear combination of the remaining vectors.
- The **column rank** of a matrix $A \in \mathbb{R}^{m \times n}$ is the size of the largest subset of columns of $A$ that constitute a linearly independent set. In the same way, the **row rank** is the largest number of rows of $A$ that constitute a linearly independent set.
- For any matrix $A \in \mathbb{R}^{m \times n}$, it turns out that <u>*the column rank of $A$ is equal to the row rank of $A$*</u> (though we will not prove this), and so both quantities are referred to collectively as the **rank** of $A$, denoted as $\operatorname{rank}(A)$.
- The following are some basic properties of the rank:
  - For $A \in \mathbb{R}^{m \times n}, \operatorname{rank}(A) \leq \min (m, n)$. If $\operatorname{rank}(A)=\min (m, n)$, then $A$ is said to be **full rank**.
  - For $A \in \mathbb{R}^{m \times n}, \operatorname{rank}(A)=\operatorname{rank}\left(A^{T}\right)$.
  - For $A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}, \operatorname{rank}(A B) \leq \min (\operatorname{rank}(A), \operatorname{rank}(B))$.
  - For $A, B \in \mathbb{R}^{m \times n}, \operatorname{rank}(A+B) \leq \operatorname{rank}(A)+\operatorname{rank}(B)$.

### 3.7 The Inverse

- The **inverse** of a square matrix $A \in \mathbb{R}^{n \times n}$ is denoted $A^{-1}$, and is the unique matrix such that
  $$
  A^{-1} A=I=A A^{-1} .
  $$

  - Note that not all matrices have inverses. Non-square matrices, for example, do not have inverses by definition.
  - In particular, we say that $A$ is **invertible** or **non-singular** if $A^{-1}$ exists and **non-invertible** or **singular** otherwise. 

- In order for a square matrix $A$ to have an inverse $A^{-1}$, then $A$ must be *<u>full rank</u>*.

- The following are properties of the inverse; all assume that $A, B \in \mathbb{R}^{n \times n}$ are non-singular:

  - $\left(A^{-1}\right)^{-1}=A$
  - $(A B)^{-1}=B^{-1} A^{-1}$
  - $\left(A^{-1}\right)^{T}=\left(A^{T}\right)^{-1}$. For this reason this matrix is often denoted $A^{-T}$.

### 3.8 Orthogonal Matrices

- Two vectors $x, y \in \mathbb{R}^{n}$ are **orthogonal** if $x^{T} y=0$. A vector $x \in \mathbb{R}^{n}$ is **normalized** if $\|x\|_{2}=1$. A square matrix $U \in \mathbb{R}^{n \times n}$ is **orthogonal** (note the different meanings when talking about vectors versus matrices) if all its columns are orthogonal to each other and are normalized (the columns are then referred to as being **orthonormal**).

- It follows immediately from the definition of orthogonality and normality that
  $$
  U^{T} U=I=U U^{T} .
  $$

  - In other words, the inverse of an orthogonal matrix is its transpose.
  - Note that if $U$ is not square - i.e., $U \in \mathbb{R}^{m \times n}, \quad n<m$ - but its columns are still orthonormal, then $U^{T} U=I$, but $U U^{T} \neq I$. 
    - We generally only use the term orthogonal to describe the previous case, where $U$ is square.

- Another nice property of orthogonal matrices is that operating on a vector with an orthogonal matrix will not change its Euclidean norm, i.e.,
  $$
  \|U x\|_{2}=\|x\|_{2}
  $$
  for any $x \in \mathbb{R}^{n}, U \in \mathbb{R}^{n \times n}$ orthogonal.

### 3.9 Range and Nullspace of a Matrix

- The **span** of a set of vectors $\left\{x_{1}, x_{2}, \ldots x_{n}\right\}$ is the set of all vectors that can be expressed as a linear combination of $\left\{x_{1}, \ldots, x_{n}\right\}$. That is,
  $$
  \operatorname{span}\left(\left\{x_{1}, \ldots x_{n}\right\}\right)=\left\{v: v=\sum_{i=1}^{n} \alpha_{i} x_{i}, \quad \alpha_{i} \in \mathbb{R}\right\}
  $$

  - It can be shown that if $\left\{x_{1}, \ldots, x_{n}\right\}$ is a set of $n$ linearly independent vectors, where each $x_{i} \in \mathbb{R}^{n}$, then $\operatorname{span}\left(\left\{x_{1}, \ldots x_{n}\right\}\right)=\mathbb{R}^{n}$.

- The **projection** of a vector $y \in \mathbb{R}^{m}$ onto the span of $\left\{x_{1}, \ldots, x_{n}\right\}$ (here we assume $\left.x_{i} \in \mathbb{R}^{m}\right)$ is the vector $v \in \operatorname{span}\left(\left\{x_{1}, \ldots x_{n}\right\}\right.$ ), such that $v$ is as close as possible to $y$, as measured by the Euclidean norm $\|v-y\|_{2}$. We denote the projection as $\operatorname{Proj}\left(y ;\left\{x_{1}, \ldots, x_{n}\right\}\right)$ and can define it formally as,
  $$
  \operatorname{Proj}\left(y ;\left\{x_{1}, \ldots x_{n}\right\}\right)=\operatorname{argmin}_{v \in \operatorname{span}\left(\left\{x_{1}, \ldots, x_{n}\right\}\right)}\|y-v\|_{2}
  $$

- The **range** (sometimes also called the columnspace) of a matrix $A \in \mathbb{R}^{m \times n}$, denoted $\mathcal{R}(A)$, is the the span of the columns of $A$. In other words,
  $$
  \mathcal{R}(A)=\left\{v \in \mathbb{R}^{m}: v=A x, x \in \mathbb{R}^{n}\right\}
  $$

  - Making a few technical assumptions (namely that $A$ is full rank and that $n<m$ ), the projection of a vector $y \in \mathbb{R}^{m}$ onto the range of $A$ is given by,
    $$
    \operatorname{Proj}(y ; A)=\operatorname{argmin}_{v \in \mathcal{R}(A)}\|v-y\|_{2}=A\left(A^{T} A\right)^{-1} A^{T} y
    $$

  - When $A$ contains only a single column, $a \in \mathbb{R}^{m}$, this gives the special case for a projection of a vector on to a line:
    $$
    \operatorname{Proj}(y ; a)=\frac{a a^{T}}{a^{T} a} y .
    $$

- The **nullspace** of a matrix $A \in \mathbb{R}^{m \times n}$, denoted $\mathcal{N}(A)$ is the set of all vectors that equal $0$ when multiplied by $A$, i.e.,
  $$
  \mathcal{N}(A)=\left\{x \in \mathbb{R}^{n}: A x=0\right\}
  $$
  
- Note that vectors in $\mathcal{R}(A)$ are of size $m$, while vectors in the $\mathcal{N}(A)$ are of size $n$, so vectors in $\mathcal{R}\left(A^{T}\right)$ and $\mathcal{N}(A)$ are both in $\mathbb{R}^{n}$. It turns out that
  $$
  \left\{w: w=u+v, u \in \mathcal{R}\left(A^{T}\right), v \in \mathcal{N}(A)\right\}=\mathbb{R}^{n} \text { and } \mathcal{R}\left(A^{T}\right) \cap \mathcal{N}(A)=\{\mathbf{0}\}
  $$

  - In other words, $\mathcal{R}\left(A^{T}\right)$ and $\mathcal{N}(A)$ are <u>*disjoint subsets*</u> that together span the entire space of $\mathbb{R}^{n}$. Sets of this type are called **orthogonal complements**, and we denote this $\mathcal{R}\left(A^{T}\right)=$ $\mathcal{N}(A)^{\perp}$.

### 3.10 The Determinant

- The **determinant** of a square matrix $A \in \mathbb{R}^{n \times n}$, is a function $\operatorname{det}: \mathbb{R}^{n \times n} \rightarrow \mathbb{R}$, and is denoted $|A|$ or $\operatorname{det} A$.

- Given a matrix
  $$
  \left[\begin{array}{ccc}
  - & a_{1}^{T} & - \\
  - & a_{2}^{T} & - \\
  & \vdots & \\
  - & a_{n}^{T} & -
  \end{array}\right]
  $$

  - consider the set of points $S \subset \mathbb{R}^{n}$ formed by taking all possible linear combinations of the row vectors $a_{1}, \ldots, a_{n} \in \mathbb{R}^{n}$ of $A$, where the coefficients of the linear combination are all between 0 and 1. Formally,

    $$
    S=\left\{v \in \mathbb{R}^{n}: v=\sum_{i=1}^{n} \alpha_{i} a_{i} \text { where } 0 \leq \alpha_{i} \leq 1, i=1, \ldots, n\right\}
    $$

- The absolute value of the determinant of $A$, it turns out, is a measure of the "volume" of the set $S$.

  - e.g.<img src="image.assets/Screen Shot 2022-04-02 at 12.09.11.png" alt="Screen Shot 2022-04-02 at 12.09.11" style="zoom:25%;" />
  - For two-dimensional matrices, $S$ generally has the shape of a *<u>parallelogram</u>*. In three dimensions, the set $S$ corresponds to an object known as a *<u>parallelepiped</u>*. In even higher dimensions, the set $S$ is an object known as an $n$-dimensional *<u>parallelotope</u>*.

- Algebraically, the determinant satisfies the following three properties (from which all other properties follow, including the general formula):

  1. The determinant of the identity is $1,|I|=1$. (Geometrically, the volume of a unit hypercube is 1 ).

  2. Given a matrix $A \in \mathbb{R}^{n \times n}$, if we multiply a single row in $A$ by a scalar $t \in \mathbb{R}$, then the determinant of the new matrix is $t|A|$,
     (Geometrically, multiplying one of the sides of the set $S$ by a factor $t$ causes the volume to increase by a factor $t$.)

  3. If we exchange any two rows $a_{i}^{T}$ and $a_{j}^{T}$ of $A$, then the determinant of the new matrix is $-|A|$, for example
     $$
     \left|\left[\begin{array}{ccc}
     - & a_{2}^{T} & - \\
     - & a_{1}^{T} & - \\
     & \vdots & \\
     - & a_{m}^{T} & -
     \end{array}\right]\right|=-|A|
     $$

- Several properties that follow from the three properties above include:

  - For $A \in \mathbb{R}^{n \times n},|A|=\left|A^{T}\right|$.
  - For $A, B \in \mathbb{R}^{n \times n},|A B|=|A||B|$.
  - For $A \in \mathbb{R}^{n \times n},|A|=0$ if and only if $A$ is singular (i.e., non-invertible). (If $A$ is singular then it does not have full rank, and hence its columns are linearly dependent. In this case, the set $S$ corresponds to a "flat sheet" within the $n$-dimensional space and hence has zero volume.)
  - For $A \in \mathbb{R}^{n \times n}$ and $A$ non-singular, $\left|A^{-1}\right|=1 /|A|$.

- Before giving the general definition for the determinant, we define, for $A \in \mathbb{R}^{n \times n}, A_{\backslash i, \backslash j} \in$ $\mathbb{R}^{(n-1) \times(n-1)}$ to be the matrix that results from deleting the $i$ th row and $j$ th column from $A$. The general (recursive) formula for the determinant is
  $$
  \begin{aligned}
  |A| &=\sum_{i=1}^{n}(-1)^{i+j} a_{i j}\left|A_{\backslash i, \backslash j}\right| \quad(\text { for any } j \in 1, \ldots, n) \\
  &=\sum_{j=1}^{n}(-1)^{i+j} a_{i j}\left|A_{\backslash i, \backslash j}\right| \quad(\text { for any } i \in 1, \ldots, n)
  \end{aligned}
  $$
  with the initial case that $|A|=a_{11}$ for $A \in \mathbb{R}^{1 \times 1}$.

- we hardly ever explicitly write the complete equation of the determinant for matrices bigger than $3 \times 3$. However, the equations for determinants of matrices up to size $3 \times 3$ are fairly common, and it is good to know them:
  $$
  \left|\left[\begin{array}{lll}a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33}\end{array}\right]\right|=\begin{array}{r}a_{11} a_{22} a_{33}+a_{12} a_{23} a_{31}+a_{13} a_{21} a_{32} \\ -a_{11} a_{23} a_{32}-a_{12} a_{21} a_{33}-a_{13} a_{22} a_{31}\end{array}
  $$

- The **classical adjoint** (often just called the adjoint) of a matrix $A \in \mathbb{R}^{n \times n}$, is denoted $\operatorname{adj}(A)$, and defined as
  $$
  \operatorname{adj}(A) \in \mathbb{R}^{n \times n}, \quad(\operatorname{adj}(A))_{i j}=(-1)^{i+j}\left|A_{\backslash j, \backslash i}\right|
  $$

  - (note the switch in the indices $A_{\backslash j, \backslash i}$ ). It can be shown that for any nonsingular $A \in \mathbb{R}^{n \times n}$,

  $$
  A^{-1}=\frac{1}{|A|} \operatorname{adj}(A)
  $$

### 3.11 Quadratic Forms and Positive Semidefinite Matrices

- Given a square matrix $A \in \mathbb{R}^{n \times n}$ and a vector $x \in \mathbb{R}^{n}$, the scalar value $x^{T} A x$ is called a **quadratic form**. Written explicitly, we see that
  $$
  x^{T} A x=\sum_{i=1}^{n} x_{i}(A x)_{i}=\sum_{i=1}^{n} x_{i}\left(\sum_{j=1}^{n} A_{i j} x_{j}\right)=\sum_{i=1}^{n} \sum_{j=1}^{n} A_{i j} x_{i} x_{j}
  $$
  Note that,
  $$
  x^{T} A x=\left(x^{T} A x\right)^{T}=x^{T} A^{T} x=x^{T}\left(\frac{1}{2} A+\frac{1}{2} A^{T}\right) x
  $$

  - From this, we can conclude that <u>*only the symmetric part of $A$ contributes to the quadratic form*</u>. 
  - For this reason, *<u>we often implicitly assume that the matrices appearing in a quadratic form are symmetric</u>*.

- We give the following definitions:

  - A symmetric matrix $A \in \mathbb{S}^{n}$ is **positive definite** (PD) if for all non-zero vectors $x \in \mathbb{R}^{n}, x^{T} A x>0$. This is usually denoted $A \succ 0$ (or just $A>0$ ), and often times the set of all positive definite matrices is denoted $\mathbb{S}_{++}^{n}$.
  - A symmetric matrix $A \in \mathbb{S}^{n}$ is **positive semidefinite** (PSD) if for all vectors $x^{T} A x \geq$ 0 . This is written $A \succeq 0$ (or just $A \geq 0$ ), and the set of all positive semidefinite matrices is often denoted $\mathbb{S}_{+}^{n}$.
  - Likewise, a symmetric matrix $A \in \mathbb{S}^{n}$ is **negative definite** (ND), denoted $A \prec 0$ (or just $A<0$ ) if for all non-zero $x \in \mathbb{R}^{n}, x^{T} A x<0$.
  - Similarly, a symmetric matrix $A \in \mathbb{S}^{n}$ is **negative semidefinite** (NSD), denoted $A \preceq 0$ (or just $A \leq 0$ ) if for all $x \in \mathbb{R}^{n}, x^{T} A x \leq 0$.
  - Finally, a symmetric matrix $A \in \mathbb{S}^{n}$ is **indefinite**, if it is neither positive semidefinite nor negative semidefinite - i.e., if there exists $x_{1}, x_{2} \in \mathbb{R}^{n}$ such that $x_{1}^{T} A x_{1}>0$ and $x_{2}^{T} A x_{2}<0$.

- <u>*One important property of positive definite and negative definite matrices is that they are always full rank, and hence, invertible.*</u> To see why this is the case, suppose that some matrix $A \in \mathbb{R}^{n \times n}$ is not full rank. Then, suppose that the $j$ th column of $A$ is expressible as a linear combination of other $n-1$ columns:
  $$
  a_{j}=\sum_{i \neq j} x_{i} a_{i}
  $$

  - for some $x_{1}, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{n} \in \mathbb{R}$. Setting $x_{j}=-1$, we have
    $$
    A x=\sum_{i=1}^{n} x_{i} a_{i}=0
    $$
    But this implies $x^{T} A x=0$ for some non-zero vector $x$, so $A$ must be neither positive definite nor negative definite.

- Given any matrix $A \in \mathbb{R}^{m \times n}$ (not necessarily symmetric or even square), <u>*the matrix $G=A^{T} A$ (sometimes called a **Gram matrix**) is always positive semidefinite. Further, if $m \geq n$ (and we assume for convenience that $A$ is full rank), then $G=A^{T} A$ is positive definite*</u>.

### 3.12 Eigenvalues and Eigenvectors

- Given a square matrix $A \in \mathbb{R}^{n \times n}$, we say that $\lambda \in \mathbb{C}$ is an **eigenvalue** of $A$ and $x \in \mathbb{C}^{n}$ is the corresponding eigenvector ${ }^{3}$ if
  $$
  A x=\lambda x, \quad x \neq 0 .
  $$

  - Intuitively, this definition means that multiplying $A$ by the vector $x$ results in a new vector that points in the same direction as $x$, but scaled by a factor $\lambda$.
  - When we talk about "the" eigenvector associated with $\lambda$, we usually assume that the eigenvector is normalized to have length 1 (this still creates some ambiguity, since $x$ and $-x$ will both be eigenvectors, but we will have to live with this).

- We can rewrite the equation above to state that $(\lambda, x)$ is an eigenvalue-eigenvector pair of $A$ if,
  $$
  (\lambda I-A) x=0, \quad x \neq 0 .
  $$

  - But $(\lambda I-A) x=0$ has a non-zero solution to $x$ if and only if $(\lambda I-A)$ has <u>*a non-empty nullspace*</u>, which is only the case if $(\lambda I-A)$ is singular, i.e.,

    $$
    |(\lambda I-A)|=0 .
    $$

    - We can now use the previous definition of the determinant to expand this expression into a (very large) polynomial in $\lambda$, where $\lambda$ will have maximum degree $n$. 
    - We then find the $n$ (possibly complex) roots of this polynomial to find the $n$ eigenvalues $\lambda_{1}, \ldots, \lambda_{n}$. 
    - To find the eigenvector corresponding to the eigenvalue $\lambda_{i}$, we simply solve the linear equation $\left(\lambda_{i} I-A\right) x=0$. 
      - (It should be noted that this is not the method which is actually used in practice to numerically compute the eigenvalues and eigenvectors (remember that the complete expansion of the determinant has $n!$ terms); it is rather a mathematical argument.)

- The following are properties of eigenvalues and eigenvectors (in all cases assume $A \in \mathbb{R}^{n \times n}$ has eigenvalues $\lambda_{i}, \ldots, \lambda_{n}$ and associated eigenvectors $\left.x_{1}, \ldots x_{n}\right)$ :

  - The trace of a $A$ is equal to the sum of its eigenvalues,

    $$
    \operatorname{tr} A=\sum_{i=1}^{n} \lambda_{i}
    $$

  - The determinant of $A$ is equal to the product of its eigenvalues,

    $$
    |A|=\prod_{i=1}^{n} \lambda_{i}
    $$

  - The rank of $A$ is equal to the number of non-zero eigenvalues of $A$.

  - If $A$ is non-singular then $1 / \lambda_{i}$ is an eigenvalue of $A^{-1}$ with associated eigenvector $x_{i}$, i.e., $A^{-1} x_{i}=\left(1 / \lambda_{i}\right) x_{i}$. 
    (To prove this, take the eigenvector equation, $A x_{i}=\lambda_{i} x_{i}$ and left-multiply each side by $A^{-1}$.)

  - The eigenvalues of a diagonal matrix $D=\operatorname{diag}\left(d_{1}, \ldots d_{n}\right)$ are just the diagonal entries $d_{1}, \ldots d_{n}$.

- We can write all the eigenvector equations simultaneously as
  $$
  A X=X \Lambda
  $$
  where the columns of $X \in \mathbb{R}^{n \times n}$ are the eigenvectors of $A$ and $\Lambda$ is a diagonal matrix whose entries are the eigenvalues of $A$, i.e.,
  $$
  X \in \mathbb{R}^{n \times n}=\left[\begin{array}{cccc}
  \mid & \mid & & \mid \\
  x_{1} & x_{2} & \cdots & x_{n} \\
  \mid & \mid & & \mid
  \end{array}\right], \Lambda=\operatorname{diag}\left(\lambda_{1}, \ldots, \lambda_{n}\right) .
  $$
  - If the eigenvectors of $A$ are linearly independent, then the matrix $X$ will be invertible, so $A=X \Lambda X^{-1}$. A matrix that can be written in this form is called **diagonalizable**.

### 3.13 Eigenvalues and Eigenvectors of Symmetric Matrices

- Two remarkable properties come about when we look at the eigenvalues and eigenvectors of a symmetric matrix $A \in \mathbb{S}^{n}$. 

  - First, it can be shown that all the eigenvalues of $A$ are real. 
  - Secondly, the eigenvectors of $A$ are orthonormal, i.e., the matrix $X$ defined above is an orthogonal matrix (for this reason, we denote the matrix of eigenvectors as $U$ in this case).

- We can therefore represent $A$ as $A=U \Lambda U^{T}$, remembering from above that the inverse of an orthogonal matrix is just its transpose.

  - Using this, we can show that the definiteness of a matrix depends entirely on the sign of its eigenvalues. Suppose $A \in \mathbb{S}^{n}=U \Lambda U^{T}$. Then
    $$
    x^{T} A x=x^{T} U \Lambda U^{T} x=y^{T} \Lambda y=\sum_{i=1}^{n} \lambda_{i} y_{i}^{2}
    $$

    - where $y=U^{T} x$ (and since $U$ is full rank, any vector $y \in \mathbb{R}^{n}$ can be represented in this form). 
    - Because $y_{i}^{2}$ is always positive, the sign of this expression depends entirely on the $\lambda_{i}$ 's. If all $\lambda_{i}>0$, then the matrix is positive definite; if all $\lambda_{i} \geq 0$, it is positive semidefinite.

- An application where eigenvalues and eigenvectors come up frequently is in maximizing some function of a matrix. In particular, for a matrix $A \in \mathbb{S}^{n}$, consider the following maximization problem,
  $$
  \max _{x \in \mathbb{R}^{n}} x^{T} A x \quad \text { subject to }\|x\|_{2}^{2}=1
  $$
  i.e., we want to find the vector (of norm 1) which maximizes the quadratic form. 

  - Assuming the eigenvalues are ordered as $\lambda_{1} \geq \lambda_{2} \geq \ldots \geq \lambda_{n}$, the optimal $x$ for this optimization problem is $x_{1}$, the eigenvector corresponding to $\lambda_{1}$. In this case the maximal value of the quadratic form is $\lambda_{1}$.

## **4 Matrix Calculus**

### 4.1 The Gradient

- Suppose that $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$ is a function that takes as input a matrix $A$ of size $m \times n$ and returns a real value. Then the **gradient** of $f$ (with respect to $A \in \mathbb{R}^{m \times n}$​ ) is the matrix of partial derivatives, defined as:
  $$
  \nabla_{A} f(A) \in \mathbb{R}^{m \times n}=\left[\begin{array}{cccc}
  \frac{\partial f(A)}{\partial A_{11}} & \frac{\partial f(A)}{\partial A_{12}} & \cdots & \frac{\partial f(A)}{\partial A_{1 n}} \\
  \frac{\partial f(A)}{\partial A_{21}} & \frac{\partial f(A)}{\partial A_{22}} & \cdots & \frac{\partial f(A)}{\partial A_{2 n}} \\
  \vdots & \vdots & \ddots & \vdots \\
  \frac{\partial f(A)}{\partial A_{m 1}} & \frac{\partial f(A)}{\partial A_{m 2}} & \cdots & \frac{\partial f(A)}{\partial A_{m n}}
  \end{array}\right]
  $$
  i.e., $\operatorname{an} m \times n$ matrix with
  $$
  \left(\nabla_{A} f(A)\right)_{i j}=\frac{\partial f(A)}{\partial A_{i j}} .
  $$

  - Note that the size of $\nabla_{A} f(A)$ is always the same as the size of $A$.
  - It is very important to remember that <u>*the gradient of a function is only defined if the function is real-valued, that is, if it returns a scalar value*</u>. We can not, for example, take the gradient of $A x, A \in \mathbb{R}^{n \times n}$ with respect to $x$, since this quantity is vector-valued.

- Working with gradients can sometimes be tricky for notational reasons. For example, suppose that $A \in \mathbb{R}^{m \times n}$ is a matrix of fixed coefficients and suppose that $b \in \mathbb{R}^{m}$ is a vector of fixed coefficients. Let $f: \mathbb{R}^{m} \rightarrow \mathbb{R}$ be the function defined by $f(z)=z^{T} z$, such that $\nabla_{z} f(z)=2 z$. But now, consider the expression,
  $$
  \nabla f(A x) .
  $$
  - How should this expression be interpreted? There are at least two possibilities:
    1. In the first interpretation, recall that $\nabla_{z} f(z)=2 z$. Here, we interpret $\nabla f(A x)$ as evaluating the gradient at the point $A x$, hence,
       $$
       \nabla f(A x)=2(A x)=2 A x \in \mathbb{R}^{m}
       $$
  
    2. In the second interpretation, we consider the quantity $f(A x)$ as a function of the input variables $x$. More formally, let $g(x)=f(A x)$. Then in this interpretation,
       $$
       \nabla f(A x)=\nabla_{x} g(x) \in \mathbb{R}^{n}
       $$
  
  - We denote the first case as $\nabla_{z} f(A x)$ and the second case as $\nabla_{x} f(A x)$.  <u>*Keeping the notation clear is extremely important.*</u>
  

### 4.2 The Hessian

- Suppose that $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ is a function that takes a vector in $\mathbb{R}^{n}$ and returns a real number. Then the **Hessian** matrix with respect to $x$, written $\nabla_{x}^{2} f(x)$ or simply as $H$ is the $n \times n$ matrix of partial derivatives,
  $$
  \nabla_{x}^{2} f(x) \in \mathbb{R}^{n \times n}=\left[\begin{array}{cccc}
  \frac{\partial^{2} f(x)}{\partial x_{1}^{2}} & \frac{\partial^{2} f(x)}{\partial x_{1} \partial x_{2}} & \cdots & \frac{\partial^{2} f(x)}{\partial x_{1} \partial x_{n}} \\
  \frac{\partial^{2} f(x)}{\partial x_{2} \partial x_{1}} & \frac{\partial^{2} f(x)}{\partial x_{2}^{2}} & \cdots & \frac{\partial^{2} f(x)}{\partial x_{2} \partial x_{n}} \\
  \vdots & \vdots & \ddots & \vdots \\
  \frac{\partial^{2} f(x)}{\partial x_{n} \partial x_{1}} & \frac{\partial^{2} f(x)}{\partial x_{n} \partial x_{2}} & \cdots & \frac{\partial^{2} f(x)}{\partial x_{n}^{2}}
  \end{array}\right]
  $$

  - In other words, $\nabla_{x}^{2} f(x) \in \mathbb{R}^{n \times n}$, with
    $$
    \left(\nabla_{x}^{2} f(x)\right)_{i j}=\frac{\partial^{2} f(x)}{\partial x_{i} \partial x_{j}}
    $$

  - Note that the Hessian is always symmetric, since
    $$
    \frac{\partial^{2} f(x)}{\partial x_{i} \partial x_{j}}=\frac{\partial^{2} f(x)}{\partial x_{j} \partial x_{i}}
    $$

  - Similar to the gradient, the Hessian is defined only when $f(x)$ is real-valued.

- It is natural to think of the gradient as the analogue of the first derivative for functions of vectors, and the Hessian as the analogue of the second derivative. This intuition is generally correct, but there a few caveats to keep in mind.

  - For functions of a vector, *<u>the gradient of the function is a vector, and we cannot take the gradient of a vector, and $\nabla_{x}\nabla_x f(x)$ is not defined.</u>* 

  - However, this is almost true, in the following sense: If we look at the $i$ th entry of the gradient $\left(\nabla_{x} f(x)\right)_{i}=\partial f(x) / \partial x_{i}$, and take the gradient with respect to $x$ we get
    $$
    \nabla_{x} \frac{\partial f(x)}{\partial x_{i}}=\left[\begin{array}{c}
    \frac{\partial^{2} f(x)}{\partial x_{i} \partial x_{1}} \\
    \frac{\partial^{2} f(x)}{\partial x_{i} \partial x_{2}} \\
    \vdots \\
    \frac{\partial f(x)}{\partial x_{i} \partial x_{n}}
    \end{array}\right]
    $$
    which is the $i$ th column (or row) of the Hessian. 

    - Therefore,
      $$
      \nabla_{x}^{2} f(x)=\left[\begin{array}{llll}
      \nabla_{x}\left(\nabla_{x} f(x)\right)_{1} & \nabla_{x}\left(\nabla_{x} f(x)\right)_{2} & \cdots & \nabla_{x}\left(\nabla_{x} f(x)\right)_{n}
      \end{array}\right] .
      $$

  - If we don't mind being a little bit sloppy <u>*we can say that (essentially) $\nabla_{x}^{2} f(x)=\nabla_{x}\left(\nabla_{x} f(x)\right)^{T}$*</u>, so long as we understand that this really means taking the gradient of each entry of $\left(\nabla_{x} f(x)\right)^{T}$, not the gradient of the whole vector.

### 4.3 Gradients and Hessians of Quadratic and Linear Functions

- For $x \in \mathbb{R}^{n}$, let $f(x)=b^{T} x$ for some known vector $b \in \mathbb{R}^{n}$. Then
  $$
  f(x)=\sum_{i=1}^{n} b_{i} x_{i}
  $$
  so
  $$
  \frac{\partial f(x)}{\partial x_{k}}=\frac{\partial}{\partial x_{k}} \sum_{i=1}^{n} b_{i} x_{i}=b_{k} .
  $$

  - From this we can easily see that $\nabla_{x} b^{T} x=b$.
  - Single-variable calculus analogy: ${\partial \over\partial x} a x=a$.
  
- Now consider the quadratic function $f(x)=x^{T} A x$ for $A \in \mathbb{S}^{n}$. Remember that
  $$
  f(x)=\sum_{i=1}^{n} \sum_{j=1}^{n} A_{i j} x_{i} x_{j} .
  $$

  - To take the partial derivative, we'll consider the terms including $x_{k}$ and $x_{k}^{2}$ factors separately:
    $$
    \begin{aligned}
    \frac{\partial f(x)}{\partial x_{k}} &=\frac{\partial}{\partial x_{k}} \sum_{i=1}^{n} \sum_{j=1}^{n} A_{i j} x_{i} x_{j} \\
    &=\frac{\partial}{\partial x_{k}}\left[\sum_{i \neq k} \sum_{j \neq k} A_{i j} x_{i} x_{j}+\sum_{i \neq k} A_{i k} x_{i} x_{k}+\sum_{j \neq k} A_{k j} x_{k} x_{j}+A_{k k} x_{k}^{2}\right] \\
    &=\sum_{i \neq k} A_{i k} x_{i}+\sum_{j \neq k} A_{k j} x_{j}+2 A_{k k} x_{k} \\
    &=\sum_{i=1}^{n} A_{i k} x_{i}+\sum_{j=1}^{n} A_{k j} x_{j}=2 \sum_{i=1}^{n} A_{k i} x_{i}
    \end{aligned}
    $$
    where the last equality follows since $A$ is symmetric (which we can safely assume, since it is appearing in a quadratic form).

    - Note that the $k$ th entry of $\nabla_{x} f(x)$ is just the inner product of the $k$ th row of $A$ and $x$. 
    - Therefore, $\nabla_{x} x^{T} A x=2 A x$.
    - Single-variable calculus analogy: ${\partial \over \partial x} a x^{2}=2 a x$.
  
- Finally, let's look at the Hessian of the quadratic function $f(x)=x^{T} A x$ (it should be obvious that the Hessian of a linear function $b^{T} x$ is zero). In this case,
  $$
  \frac{\partial^{2} f(x)}{\partial x_{k} \partial x_{\ell}}=\frac{\partial}{\partial x_{k}}\left[\frac{\partial f(x)}{\partial x_{\ell}}\right]=\frac{\partial}{\partial x_{k}}\left[2 \sum_{i=1}^{n} A_{\ell i} x_{i}\right]=2 A_{\ell k}=2 A_{k \ell} .
  $$

  - Therefore, it should be clear that $\nabla_{x}^{2} x^{T} A x=2 A$, which should be entirely expected.
  - Single-variable calculus analogy: ${\partial^{2} \over \partial x^{2}} a x^{2}=2 a$
  
- To recap,

  - $\nabla_{x} b^{T} x=b$
  - $\nabla_{x} x^{T} A x=2 A x$ (if $A$ symmetric)
  - $\nabla_{x}^{2} x^{T} A x=2 A$ (if $A$ symmetric)

### 4.4 Least Squares

- Suppose we are given matrices $A \in \mathbb{R}^{m \times n}$ (for simplicity we assume $A$ is full rank) and a vector $b \in \mathbb{R}^{m}$ such that $b \notin \mathcal{R}(A)$. 

  - In this situation we will not be able to find a vector $x \in \mathbb{R}^{n}$, such that $A x=b$, 
  - so instead we want to find a vector $x$ such that $A x$ is as close as possible to $b$, as measured by the square of the Euclidean norm $\|A x-b\|_{2}^{2}$.

- Using the fact that $\|x\|_{2}^{2}=x^{T} x$, we have
  $$
  \begin{aligned}
  \|A x-b\|_{2}^{2} &=(A x-b)^{T}(A x-b) \\
  &=x^{T} A^{T} A x-2 b^{T} A x+b^{T} b
  \end{aligned}
  $$

  - Taking the gradient with respect to $x$ we have, and using the properties we derived in the previous section
    $$
    \begin{aligned}
    \nabla_{x}\left(x^{T} A^{T} A x-2 b^{T} A x+b^{T} b\right) &=\nabla_{x} x^{T} A^{T} A x-\nabla_{x} 2 b^{T} A x+\nabla_{x} b^{T} b \\
    &=2 A^{T} A x-2 A^{T} b
    \end{aligned}
    $$

  - Setting this last expression equal to zero and solving for $x$ gives the normal equations
    $$
    x=\left(A^{T} A\right)^{-1} A^{T} b
    $$

### 4.5 Gradients of the Determinant

- For $A \in \mathbb{R}^{n \times n}$, we want to find $\nabla_{A}|A|$. Recall from our discussion of determinants that
  $$
  |A|=\sum_{i=1}^{n}(-1)^{i+j} A_{i j}\left|A_{\backslash i, \backslash j}\right| \quad(\text { for any } j \in 1, \ldots, n)
  $$
  so
  $$
  \frac{\partial}{\partial A_{k \ell}}|A|=\frac{\partial}{\partial A_{k \ell}} \sum_{i=1}^{n}(-1)^{i+j} A_{i j}\left|A_{\backslash i, \backslash j}\right|=(-1)^{k+\ell}\left|A_{\backslash k, \backslash \ell}\right|=(\operatorname{adj}(A))_{\ell k}
  $$

  - From this it immediately follows from the properties of the adjoint that
    $$
    \nabla_{A}|A|=(\operatorname{adj}(A))^{T}=|A| A^{-T}
    $$

- Now let's consider the function $f: \mathbb{S}_{++}^{n} \rightarrow \mathbb{R}, f(A)=\log |A|$. 
  (Note that we have to restrict the domain of $f$ to be the positive definite matrices, since this ensures that $|A|>0$, so that the $\log$ of $|A|$ is a real number.)

  - In this case we can use the chain rule to see that
    $$
    \frac{\partial \log |A|}{\partial A_{i j}}=\frac{\partial \log |A|}{\partial|A|} \frac{\partial|A|}{\partial A_{i j}}=\frac{1}{|A|} \frac{\partial|A|}{\partial A_{i j}}
    $$

  - From this it should be obvious that
    $$
    \nabla_{A} \log |A|=\frac{1}{|A|} \nabla_{A}|A|=A^{-1},
    $$

    - where we can drop the transpose in the last expression because $A$ is symmetric. 
    - Single-variable calculus analogy: ${\partial \over\partial x} \log x=1 / x$.

### 4.6 Eigenvalues as Optimization

- Consider the following, equality constrained optimization problem:
  $$
  \max _{x \in \mathbb{R}^{n}} x^{T} A x \quad \text { subject to }\|x\|_{2}^{2}=1
  $$
  for a symmetric matrix $A \in \mathbb{S}^{n}$. 

  - A standard way of solving optimization problems with equality constraints is by forming the **Lagrangian**, an objective function that includes the equality constraints. The Lagrangian in this case can be given by
    $$
    \mathcal{L}(x, \lambda)=x^{T} A x-\lambda x^{T} x
    $$

    - where $\lambda$ is called the Lagrange multiplier associated with the equality constraint. 

  - It can be established that for $x^{*}$ to be a optimal point to the problem, the gradient of the Lagrangian has to be zero at $x^{*}$ (this is not the only condition, but it is required). That is,
    $$
    \nabla_{x} \mathcal{L}(x, \lambda)=\nabla_{x}\left(x^{T} A x-\lambda x^{T} x\right)=2 A^{T} x-2 \lambda x=0
    $$

    - Notice that this is just the linear equation $A x=\lambda x$. 

  - This shows that the only points which can possibly maximize (or minimize) $x^{T} A x$ assuming $x^{T} x=1$ are the eigenvectors of $A$.



