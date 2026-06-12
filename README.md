# python_base - Python 基础学习笔记

这是一个面向 Python 初学者的个人学习笔记仓库，主要使用 Jupyter Notebook 记录 Python 基础语法、常用标准库、面向对象、异常处理、迭代器/生成器/装饰器、多线程与多进程，以及 NumPy 基础内容。

当前 Python 基础部分已经整理完成，共 17 章。笔记以“概念说明 + 示例代码 + 详细中文注释 + 常见错误总结”为主，适合按章节顺序学习，也适合之后复习查阅。

## 目录结构

```text
python_base/
├── README.md
├── python_learn/
│   ├── 1.数据类型.ipynb
│   ├── 2.数据类型转换.ipynb
│   ├── 3.注释使用.ipynb
│   ├── 4.运算符.ipynb
│   ├── 5.字符串.ipynb
│   ├── 6.列表.ipynb
│   ├── 7.元组.ipynb
│   ├── 8.字典.ipynb
│   ├── 9.集合.ipynb
│   ├── 10.条件控制语句.ipynb
│   ├── 11.循环结构.ipynb
│   ├── 12.函数.ipynb
│   ├── 13.文件读取写入和OS.ipynb
│   ├── 14.面向对象编程.ipynb
│   ├── 15.错误与异常.ipynb
│   ├── 16.迭代器生成器和装饰器.ipynb
│   ├── 17.Python多线程和多进程.ipynb
│   └── res/
│       ├── chapter13_text.txt
│       ├── chapter13_students.csv
│       ├── chapter13_config.json
│       ├── chapter17_process_demo.py
│       └── chapter13_workspace/
└── numpy_learn/
    ├── numpy01.ipynb
    ├── numpy02.ipynb
    └── numpy03.ipynb
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

## 配套资源

`python_learn/res/` 中保存了部分章节需要用到的示例文件：

| 文件/目录 | 用途 |
| --- | --- |
| `chapter13_text.txt` | 第 13 章文本读取示例 |
| `chapter13_students.csv` | 第 13 章 CSV 读取和写入示例 |
| `chapter13_config.json` | 第 13 章 JSON 读取和写入示例 |
| `chapter13_workspace/` | 第 13 章代码运行时生成的练习输出目录 |
| `chapter17_process_demo.py` | 第 17 章多进程示例脚本，适合在终端中运行 |

## NumPy 笔记

| 文件 | 主题 |
| --- | --- |
| `numpy01.ipynb` | NumPy 数组创建、属性查看、索引切片、形状调整 |
| `numpy02.ipynb` | axis 概念、聚合统计、矩阵运算、广播、布尔索引、排序去重 |
| `numpy03.ipynb` | 随机数生成基础 |

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

1. 按 `1` 到 `17` 的顺序学习 Python 基础，先打牢语法，再进入 NumPy。
2. 每个 Notebook 建议亲自运行一遍，并尝试修改变量值观察输出变化。
3. 第 13 章涉及文件写入、重命名、删除等操作，建议只在配套的 `res/chapter13_workspace/` 中练习。
4. 第 17 章多进程示例在 Windows/Jupyter 环境中可能不适合直接在 Notebook 内运行，推荐运行 `res/chapter17_process_demo.py`。
5. 学习时可以在原 Notebook 下方新增自己的实验单元，保留错误和修正过程，这会比只看最终答案更有帮助。

## 当前进度

| 日期 | 内容 |
| --- | --- |
| 2026-06-12 | 完成 Python 基础 1-17 章学习笔记 |
| 2026-06-12 | 补充第 13 章文件读写、OS 常用方法表和配套资源 |
| 2026-06-12 | 保留 NumPy 基础 3 个 Notebook |

## License

本仓库为个人学习笔记，内容仅供学习、复习和交流参考。
