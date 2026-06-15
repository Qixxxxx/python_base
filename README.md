# python_base - Python 与 NumPy 学习笔记

这是一个面向初学者的 Python 学习笔记仓库，主要使用 Jupyter Notebook 记录 Python 基础语法、常用标准库、面向对象、异常处理、迭代器/生成器/装饰器、多线程与多进程，以及 NumPy 数值计算基础。

当前进度：

- Python 基础：已完成 1-17 章
- NumPy 基础：已完成 1-8 章

笔记整体风格保持一致：概念说明、常用方法表、示例代码、详细中文注释、常见错误总结，并在适合练习的章节加入练习题和参考代码。

## 目录结构

```text
python_base/
|-- README.md
|-- python_learn/
|   |-- 1.数据类型.ipynb
|   |-- 2.数据类型转换.ipynb
|   |-- 3.注释使用.ipynb
|   |-- 4.运算符.ipynb
|   |-- 5.字符串.ipynb
|   |-- 6.列表.ipynb
|   |-- 7.元组.ipynb
|   |-- 8.字典.ipynb
|   |-- 9.集合.ipynb
|   |-- 10.条件控制语句.ipynb
|   |-- 11.循环结构.ipynb
|   |-- 12.函数.ipynb
|   |-- 13.文件读取写入和OS.ipynb
|   |-- 14.面向对象编程.ipynb
|   |-- 15.错误与异常.ipynb
|   |-- 16.迭代器生成器和装饰器.ipynb
|   |-- 17.Python多线程和多进程.ipynb
|   `-- res/
|-- numpy_learn/
|   |-- 1.NumPy入门与数组创建.ipynb
|   |-- 2.数组属性索引切片与形状操作.ipynb
|   |-- 3.数据类型复制视图与缺失值.ipynb
|   |-- 4.数学运算通用函数和统计.ipynb
|   |-- 5.广播机制向量化与矩阵运算.ipynb
|   |-- 6.条件筛选排序和集合操作.ipynb
|   |-- 7.随机数采样和模拟.ipynb
|   |-- 8.文件读写常用方法速查与综合案例.ipynb
|   `-- res/
`-- .gitignore
```

## Python 基础章节

| 章节 | 文件 | 主题 |
| --- | --- | --- |
| 1 | `1.数据类型.ipynb` | 常见数据类型、变量基础 |
| 2 | `2.数据类型转换.ipynb` | 类型转换、输入输出中的类型处理 |
| 3 | `3.注释使用.ipynb` | 单行注释、多行注释、注释规范 |
| 4 | `4.运算符.ipynb` | 算术、比较、逻辑、赋值、成员运算符 |
| 5 | `5.字符串.ipynb` | 字符串创建、索引切片、常用方法、格式化 |
| 6 | `6.列表.ipynb` | 列表增删改查、排序、遍历、列表推导式 |
| 7 | `7.元组.ipynb` | 元组创建、不可变特性、拆包 |
| 8 | `8.字典.ipynb` | 字典增删改查、遍历、嵌套结构 |
| 9 | `9.集合.ipynb` | 集合去重、交并差、成员判断 |
| 10 | `10.条件控制语句.ipynb` | `if`、`if...else`、`if...elif...else`、`match...case` |
| 11 | `11.循环结构.ipynb` | `for`、`while`、`range`、`break`、`continue`、循环嵌套 |
| 12 | `12.函数.ipynb` | 函数定义、参数、返回值、作用域、`lambda`、闭包、递归 |
| 13 | `13.文件读取写入和OS.ipynb` | 文件读写、CSV、JSON、`os`、`pathlib`、目录操作 |
| 14 | `14.面向对象编程.ipynb` | 类和对象、属性、方法、封装、继承、多态、特殊方法 |
| 15 | `15.错误与异常.ipynb` | 常见异常、`try...except`、`raise`、自定义异常 |
| 16 | `16.迭代器生成器和装饰器.ipynb` | 迭代器、生成器、`yield`、装饰器 |
| 17 | `17.Python多线程和多进程.ipynb` | 线程、线程池、锁、队列、进程池、多线程/多进程选择 |

## NumPy 学习章节

| 章节 | 文件 | 主题 |
| --- | --- | --- |
| 1 | `1.NumPy入门与数组创建.ipynb` | NumPy 入门、数组创建、`array`、`arange`、`linspace`、`zeros`、`ones`、`full`、`eye` |
| 2 | `2.数组属性索引切片与形状操作.ipynb` | 数组属性、索引切片、元素修改、`reshape`、`ravel`、`flatten`、转置、拼接、拆分 |
| 3 | `3.数据类型复制视图与缺失值.ipynb` | `dtype`、`astype`、视图与拷贝、`NaN`、无穷值、缺失值处理 |
| 4 | `4.数学运算通用函数和统计.ipynb` | 数学运算、通用函数、统计函数、`axis`、累计计算 |
| 5 | `5.广播机制向量化与矩阵运算.ipynb` | 广播机制、向量化、`np.where`、矩阵乘法、线性代数基础 |
| 6 | `6.条件筛选排序和集合操作.ipynb` | 布尔索引、多条件筛选、排序、`argsort`、去重和集合操作 |
| 7 | `7.随机数采样和模拟.ipynb` | 随机数生成器、随机整数、随机小数、正态分布、抽样、模拟 |
| 8 | `8.文件读写常用方法速查与综合案例.ipynb` | NumPy 文件读写、`loadtxt`、`genfromtxt`、`save`、`load`、`savez`、综合案例 |

## 配套资源

### Python 资源

`python_learn/res/` 中保存了 Python 文件读写、多进程等章节需要用到的示例文件：

| 文件/目录 | 用途 |
| --- | --- |
| `chapter13_text.txt` | 第 13 章文本读取示例 |
| `chapter13_students.csv` | 第 13 章 CSV 读取和写入示例 |
| `chapter13_config.json` | 第 13 章 JSON 读取和写入示例 |
| `chapter13_workspace/` | 第 13 章代码运行时生成的练习输出目录 |
| `chapter17_process_demo.py` | 第 17 章多进程示例脚本 |

### NumPy 资源

`numpy_learn/res/` 中保存了 NumPy 文件读写和综合案例需要用到的示例文件：

| 文件/目录 | 用途 |
| --- | --- |
| `matrix.txt` | 第 8 章 `np.loadtxt` 读取纯数字矩阵 |
| `scores.csv` | 第 8 章 `np.genfromtxt` 读取成绩数据 |
| `numpy_workspace/` | 第 8 章运行保存文件示例时生成的输出目录 |

## 环境准备

建议使用 Python 3 和 Jupyter Notebook。

```bash
pip install notebook numpy
```

启动 Jupyter：

```bash
jupyter notebook
```

然后在浏览器中打开对应的 `.ipynb` 文件，按单元格顺序运行学习。

## 学习建议

1. 先按 `python_learn` 的 1-17 章学习 Python 基础，再进入 `numpy_learn`。
2. 学 NumPy 时建议重点理解 `shape`、`axis`、广播机制、布尔索引，这些是后续 Pandas 和数据分析的基础。
3. 每个 Notebook 都建议亲自运行一遍，并修改数组形状、数值或条件，观察输出变化。
4. 第 13 章和 NumPy 第 8 章会写入文件，建议只在配套的 `res` 工作目录中练习。
5. 第 17 章多进程示例在 Windows/Jupyter 环境中可能不适合直接在 Notebook 内运行，推荐运行 `python_learn/res/chapter17_process_demo.py`。

## 当前进度

| 日期 | 内容 |
| --- | --- |
| 2026-06-12 | 完成 Python 基础 1-17 章学习笔记 |
| 2026-06-12 | 补充 Python 第 13 章文件读写、OS 常用方法表和配套资源 |
| 2026-06-13 | 重整并完成 NumPy 基础 1-8 章学习笔记 |

## License

本仓库为个人学习笔记，内容仅供学习、复习和交流参考。
