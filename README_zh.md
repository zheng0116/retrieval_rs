# 基于 CLIP 的图像检索
## 概述
一个使用 CLIP 模型计算文本和图像相似度的快速高效图像搜索工具，使用 Rust 开发。通过计算 CLIP 嵌入向量的余弦相似度，实现使用自然语言搜索图像的功能。

## 特性

- 使用自然语言进行文本到图像的搜索
- 批处理实现快速图像处理
- 特征缓存以提升重复搜索性能
- 支持多种图像格式（JPG、JPEG、PNG、GIF、BMP）
## 待办事项
- 支持网页访问前端界面
- 支持grpc和http接口
- 支持多模态搜索
- 支持向量数据库

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/zheng0116/retrieval_rs.git
cd retrieval_rs
```

2. 构建项目：
```bash
sh run.sh build
```

## 使用方法

基本用法：
```bash
sh run.sh search
```

可用选项：
- `--image-dir`：要搜索的图像目录
- `--query`：自然语言搜索查询
- `--model`：自定义 CLIP 模型路径（可选）
- `--tokenizer`：自定义分词器路径（可选）
- `--cpu`：强制使用 CPU 执行
- `--similarity-threshold`：最小相似度阈值（默认：0.2）
- `--top-k`：显示结果数量（默认：5）
- `--cache-file`：缓存文件路径
- `--save-cache`：启用缓存保存
## 许可证

本项目采用 [AGPL-3.0](LICENSE) 许可证开源。
## 致谢
 - [candle](https://github.com/huggingface/candle)