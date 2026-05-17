# python_base — Python 基础学习笔记

> 📚 个人 Python 与数据科学基础学习仓库，从语法入门到 NumPy 数组操作，逐步夯实基础。

---

## 🎯 项目简介

本项目是一个**系统化的 Python 基础学习笔记库**，以 Jupyter Notebook 为主要载体，记录 Python 核心语法、常用标准库以及 NumPy 数值计算的学习过程。所有内容面向初学者，代码与注释均使用中文，力求做到“每一行都有解释”。

无论你是刚开始学 Python，还是想复习基础、查漏补缺，这里都希望能帮到你。

---

## 📂 目录结构

### 已完成 ✅

```
python_base/
├── python_learn/          # Python 基础语法
│   ├── python01.ipynb     # 数据类型、变量、格式化输出、列表、元组、字典、集合、字符串方法
│   ├── python02.ipynb     # 流程控制（if / 循环）、函数、类与面向对象、异常处理
│   ├── os_learn.ipynb     # 文件读写与 os / os.path 模块使用
│   ├── test_1.txt         # 文件读写练习测试文件
│   ├── test_2.txt         # 文件读写练习测试文件
│   └── mkdir_test/
│       └── create_file.txt# 目录操作练习测试文件
│
└── numpy_learn/           # NumPy 数组操作
    ├── numpy01.ipynb      # 数组创建、属性查看、修改元素、切片与翻转
    ├── numpy02.ipynb      # axis 概念、聚合统计、矩阵运算、广播机制、布尔索引、排序与去重
    └── numpy03.ipynb      # 随机数生成
```

### 规划中 🚧

| 模块 | 主题 | 计划内容 |
|:---|:---|:---|
| `python_learn/` | `python03.ipynb` | 迭代器、生成器、装饰器、上下文管理器 |
| `python_learn/` | `python04.ipynb` | 正则表达式（re 模块） |
| `python_learn/` | `python05.ipynb` | 常用标准库速查：datetime、json、collections、itertools |
| `pandas_learn/` | `pandas01.ipynb` | Series 与 DataFrame 基础创建与索引 |
| `pandas_learn/` | `pandas02.ipynb` | 数据清洗：缺失值处理、去重、类型转换 |
| `pandas_learn/` | `pandas03.ipynb` | 数据筛选、分组聚合（groupby）、合并（merge / concat） |
| `pandas_learn/` | `pandas04.ipynb` | 数据透视表、时间序列基础 |
| `matplotlib_learn/` | `matplotlib01.ipynb` | 折线图、散点图、柱状图、直方图基础绘制 |
| `matplotlib_learn/` | `matplotlib02.ipynb` | 图表美化：标题、标签、图例、子图布局 |
| `advanced_python/` | `python_advance01.ipynb` | 面向对象进阶：魔术方法、属性描述符、元类简介 |
| `advanced_python/` | `python_advance02.ipynb` | 并发基础：多线程、多进程、异步 IO（asyncio）入门 |

---

## 🛠️ 环境准备

本项目基于 **Python 3** 与 **Jupyter Notebook**。

### 1. 安装依赖

```bash
# 建议先创建虚拟环境（可选）
# python -m venv venv
# source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

pip install notebook numpy
# 未来若增加 pandas、matplotlib 等内容：
# pip install pandas matplotlib
```

### 2. 启动 Jupyter

```bash
jupyter notebook
```

然后在浏览器中打开对应的 `.ipynb` 文件，即可逐单元运行学习。

---

## 💡 学习建议

1. **顺序学习**：`python01` → `python02` → `os_learn`，打牢语法基础后再进入 NumPy。
2. **边学边练**：每个 Notebook 都建议亲自运行一遍，修改参数观察结果变化。
3. **做笔记**：可以在已有单元格下方新增自己的理解和实验，形成属于你的笔记。
4. **善用输出**：Notebook 中保留了部分执行输出，可先观察输出，再对照代码理解逻辑。

---

## 📌 更新日志

| 日期 | 内容 |
|:---|:---|
| 2024-XX | 创建仓库，完成 `python01` ~ `python02` |
| 2024-XX | 新增 `os_learn`，补充文件与目录操作 |
| 2024-XX | 新增 `numpy01` ~ `numpy03`，完成 NumPy 基础篇 |

---

## 📄 License

本仓库为个人学习笔记，内容仅供参考交流，欢迎 Star & Fork。

---

> 🚀 **学无止境，日拱一卒。** 如果你也在学习 Python，欢迎一起交流进步！
