# Image retrieval with CLIP
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-AGPL3.0-blue)
![Rust Version](https://img.shields.io/badge/rust-1.80.1-blue)
 
 <strong>[中文](./README_zh.md) |
    English</strong>
## 🌟 Introduction
A fast and efficient image retrieval tool using CLIP model to compute similarity between text and images, implemented in Rust. This tool allows you to search images using natural language by calculating cosine similarity between CLIP embeddings.

## 🚀 Features

- Text-to-image search using natural language queries
- Fast image processing with batch operations
- Feature caching to improve performance on repeated searches
- Support for various image formats (JPG, JPEG, PNG, GIF, BMP)
- GPU acceleration support
## TODO
 - Add web-based user interface
 - Implement gRPC and HTTP API endpoints
 - Support multi-modal search capabilities
 - Integrate with vector databases
## Installation

1. Clone the repository:
```bash
git clone https://github.com/zheng0116/retrieval_rs.git
cd retrieval_rs
```

2. Build the project:
```bash
sh run.sh build
```

## Usage

Basic usage:
```bash
sh run.sh search
```

Available options:
- `--image-dir`: Directory containing images to search
- `--query`: Search query in natural language
- `--model`: Path to custom CLIP model (optional)
- `--tokenizer`: Path to custom tokenizer (optional)
- `--cpu`: Force CPU execution
- `--similarity-threshold`: Minimum similarity score (default: 0.5)
- `--top-k`: Number of results to show (default: 5)
- `--cache-file`: Path to cache file
- `--save-cache`: Enable cache saving



## License

This project is open-sourced under the [AGPL-3.0](LICENSE) license.

## Acknowledgement
   - [candle](https://github.com/huggingface/candle)

