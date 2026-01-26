# self-llm-model

## 项目简介

这是一个从头实现大语言模型的项目，旨在帮助开发者理解大语言模型的核心原理和实现方法。

## 功能特点

- 从零开始构建大语言模型
- 支持 CPU 和 GPU (CUDA) 运行环境
- 使用 PyTorch 作为深度学习框架
- 集成 tiktoken 用于 token 处理

## 安装说明

### 系统要求

- Python 3.12 或更高版本
- （可选）支持 CUDA 12.1 的 NVIDIA GPU

### 安装方法

#### 1. 克隆项目

```bash
git clone <repository-url>
cd self-llm-model
```

#### 2. 安装依赖

根据您的硬件环境选择合适的安装命令：

##### CPU 版本（无 GPU）

```bash
uv sync --extra cpu
```

##### GPU 版本（支持 CUDA 12.1）

```bash
uv sync --extra cu121
```

## 使用方法

### 数据集
从 [下载链接](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 下载需要的数据文件（创建./dataset目录）并放到./dataset下

### 基本使用

```python
```

## 项目结构

```
self-llm-model/
├── pyproject.toml          # 项目配置文件
├── README.md               # 项目说明文档
└── self_llm_model/         # 主源码目录
    ├── __init__.py
    ├── model.py            # 模型实现
    └── tokenizer.py        # Tokenizer 实现
```

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件
