package org.emergentorder.onnx


  trait Dim
  sealed trait Axes

  sealed case class Scalar()                             extends Axes
  sealed case class Vec[I <: XInt, T <: Dim](i: I, t: T) extends Axes
  sealed case class Mat[I <: XInt, T <: Dim, J <: XInt, U <: Dim](i: I, t: T, j: J, u: U)
      extends Axes
      //TODO: rename
  sealed case class Tuple3OfDim[I <: XInt, T <: Dim, J <: XInt, U <: Dim, K <: XInt, V <: Dim](
      i: I,
      t: T,
      j: J,
      u: U,
      k: K,
      v: V
  ) extends Axes
//TODO: 4+ dimensional

  object AxesFactory {
    def getAxes[T](shape: Array[XInt], dims: Array[Dim]): Axes = {
      if (shape.length == 3) {
        val t0 = shape(0)
        val d0 = dims(0)
        val t1 = shape(1)
        val d1 = dims(1)
        val t2 = shape(2)
        val d2 = dims(2)
        new Tuple3OfDim[t0.type, d0.type, t1.type, d1.type, t2.type, d2.type](
          t0,
          d0,
          t1,
          d1,
          t2,
          d2
        )
      } else if (shape.length == 1) {
        val t0 = shape(0)
        val d0 = dims(0)
        new Vec[t0.type, d0.type](t0, d0)
      } else if (shape.length == 0) (new Scalar)
      else {
        val t0 = shape(0)
        val d0 = dims(0)
        val t1 = shape(1)
        val d1 = dims(1)
        new Mat[t0.type, d0.type, t1.type, d1.type](t0, d0, t1, d1)
      }

    }
  }

  type TypesafeTensor[T, A <: Axes] = Tuple3[Array[T], Array[Int], A]

  type Tensor[T]       = TypesafeTensor[T, Axes]
  type SparseTensor[T] = Tensor[T]

  type XInt = Int with Singleton

  object TensorFactory {

    def getTensor[T](data: Array[T], t: Array[Int]): Tensor[T] = {
      val shape: Array[XInt] = t.map(z => z: XInt)
      require(data.size == shape.foldLeft(1)(_ * _))
      (data, t, AxesFactory.getAxes(shape, Array.fill(shape.size) { new Dim {} }))
    }
    def getTypesafeTensor[T, A <: Axes](data: Array[T], axes: A): TypesafeTensor[T, A] = {

//      val axes: A = AxesFactory.getAxes(shape, dims)
      val t: Array[Int] = axes match {
        case Scalar()                      => Array()
        case Vec(i, _)                     => Array(i)
        case Mat(i, _, j, _)               => Array(i, j)
        case Tuple3OfDim(i, _, j, _, k, _) => Array(i, j, k)
      }

      val shape: Array[XInt] = t.map(z => z: XInt)

      require(data.size == shape.foldLeft(1)(_ * _))
      (data, t, axes)
    }
  }

