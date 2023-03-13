# Paddle PHI C++ API 组合运算示例

## 1. 背景

Paddle 自 2.3 版本起，通过 PHI 算子库开放了部分 C++ API，支持通过 C++ API 复用 PHI 算子库内实现的算子，当前官方主要推荐在 [自定义算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/custom_op/new_cpp_op_cn.html) 开发时使用，降低在框架外部组合复杂算子的开发成本。

在 2.4 版本，Paddle 开放的 C++ 算子API 个数达 350+ ，基本将框架内仍在使用的运算类算子均通过 C++ API 开放给外部用户调用，需要时只需要 include 一个汇总的扩展头文件即可。

```
#include <paddle/extension.h>
```

具体的 API 列表可以在 Paddle 安装目录下的相应头文件中的查看，如 `${your python path}/lib/python3.7/site-packages/paddle/include/paddle/phi/api/include/api.h`。示例如下：

```c++
PADDLE_API Tensor atan2(const Tensor& x, const Tensor& y);

PADDLE_API Tensor bernoulli(const Tensor& x);

PADDLE_API Tensor cholesky(const Tensor& x, bool upper = false);

PADDLE_API Tensor cholesky_solve(const Tensor& x, const Tensor& y, bool upper = false);

PADDLE_API Tensor cross(const Tensor& x, const Tensor& y, int axis = 9);

PADDLE_API Tensor diag(const Tensor& x, int offset = 0, float padding_value = 0.0);

PADDLE_API Tensor diagonal(const Tensor& x, int offset = 0, int axis1 = 0, int axis2 = 1);

PADDLE_API Tensor digamma(const Tensor& x);

PADDLE_API Tensor dist(const Tensor& x, const Tensor& y, float p = 2.0);

PADDLE_API Tensor dot(const Tensor& x, const Tensor& y);

PADDLE_API Tensor erf(const Tensor& x);

PADDLE_API Tensor erfinv(const Tensor& x);

PADDLE_API Tensor& erfinv_(Tensor& x);

...
```

> 注1：由于一些历史原因，Paddle内诸多算子的参数与 Python API 参数并不一致，导致将其开放为 C++ API 也会存在与 Python API 参数不一致的情况，这是一种不规范的现象，因此这些 API 暂时放在 paddle::experimental 命名空间下，部分一致的 API 手动放在 paddle 命名空间下。

> 注2：Paddle 目前仅推进对外使用 phi/api 目录下以 Tensor 为中心的 API，在 phi/core 中的以 DenseTensor，SparseXXXTensor为中心的是底层使用的 API ，复杂度比较高，暂时不推荐外部用户使用

除自定义算子场景外，开发者也可以使用此类 C++ API 在外部实现一段 C++ 计算程序，但由于 Paddle 开放的算子只有计算功能，并不支持 autograd，因此并不能通过这样的方式进行 C++ 训练。

通过这种方式使用时，需要链接 Paddle 安装目录下的 C++ 动态库 `libpaddle.so` 。

## 2. 示例

本 Demo 提供了一个使用 Paddle PHI C++ API 的简单示例，调用 PHI 的 matmul 算子在框架外实现矩阵乘法运算，代码示例见 `main.cc` 。

下面介绍如何编译并执行本示例：

1. 安装 Paddle

建议直接安装 Paddle 的 develop (Nightly build) 版本，按照官方文档确保 Paddle 安装无误。（[安装链接](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)）

2. 编译

首先创建并进入 build 目录

```shell
$ mkdir build
$ cd build
```

执行cmake

This works for me

```shell
cmake .. -DPADDLE_LIBRARY=$(python -c "import paddle, os, inspect; print(os.path.dirname(inspect.getsourcefile(paddle)))") -DCMAKE_PREFIX_PATH=$(python -c "import pybind11, os, inspect; print(os.path.dirname(inspect.getsourcefile(pybind11)))")/..
```

This is being used by Weihang

```shell
$ cmake .. -DPYTHON_LIBRARY=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

-- The CXX compiler identification is GNU 8.2.0
-- Check for working CXX compiler: /usr/local/gcc-8.2/bin/c++
-- Check for working CXX compiler: /usr/local/gcc-8.2/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- PYTHON_LIBRARY = /work/dev1/paddle/build_venv/lib/python3.7/site-packages
-- Configuring done
-- Generating done
-- Build files have been written to: /work/toolset/paddle-cpp-demo/build
```

执行make

```shell
$ make

[ 50%] Building CXX object CMakeFiles/main.dir/main.cc.o
[100%] Linking CXX executable main
[100%] Built target main
```

3. 执行

```shell
$ ./main

Demo execute start
Tensor(Place(cpu), float32)[24 24 24 24 24 24 24 24 24 24 24 24 24 24 24]
Demo execute start
```

## 3. 小结

理论上，Paddle 可以正常运行的话，不需要再依赖其他的第三方库，确保相关路径配置正确即可编译执行。

如果本机有多个 Python 环境，注意指定正确的 Python Lib 路径，也可以不使用 `get_python_lib` 直接指定准确的路径。

本实验仅在 Linux(Ubuntu) 上进行了验证，Windows 和 Mac 并未验证，但理论上也可以使用。
