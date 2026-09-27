# RAGImage

📚 **拍照 → 知识库 → 出题** 的最小可行原型

把书拍成照片，本地 OCR 提取文字，建立知识库，调用 LLM 生成结构化选择题。

## 架构

```
[Web 前端]
    ↓ 上传照片
[FastAPI 后端]
    ↓
[RapidOCR 本地 OCR]  ← CPU 推理，~2 秒/页
    ↓
[知识库 TF-IDF 检索]  ← 切块+向量索引
    ↓
[AGNES LLM 出题]     ← Qwen3.8-Flash-Next，~77 秒/次
    ↓
[JSON 持久化]         ← data/questions_*.json
```

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url> ragimage
cd ragimage
```

### 2. 安装依赖

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 3. 配置 API Key

```bash
cp backend/.env.example backend/.env
# 编辑 backend/.env，填入你的 VLM_API_KEY
```

### 4. 启动服务器

```bash
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8765 --reload
```

浏览器打开 http://127.0.0.1:8765

### 5. 试试

1. 拖拽上传几张书页照片（参考 `examples/test_page1.jpg`）
2. 等 2 秒 OCR 完成
3. 点「❓ 题库」tab → 「🎲 出题」
4. 等 60-90 秒，查看生成的题目

## 项目结构

```
RAGImage/
├── backend/                    # Python FastAPI 后端
│   ├── app.py                  # 路由 + 请求处理
│   ├── config.py               # 环境配置加载
│   ├── ocr_service.py          # RapidOCR 本地 OCR
│   ├── chunker.py              # 文本切块（1图=1页）
│   ├── knowledge_base.py       # TF-IDF 检索 + JSON 持久化
│   ├── question_generator.py   # LLM 出题 + JSON 修复
│   ├── requirements.txt        # 依赖清单
│   ├── .env.example            # 配置模板（提交）
│   └── .env                    # 真实配置（不提交）
├── frontend/
│   └── index.html              # 单文件前端（原生 JS）
├── examples/                   # 示例图片
│   ├── test_page1.jpg
│   └── test_page2.jpg
├── uploads/                    # 运行时：用户上传（gitignore）
├── data/                       # 运行时：知识库数据（gitignore）
├── .gitignore
└── README.md
```

## API

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/api/health` | 健康检查 + 知识库统计 |
| GET | `/api/books` | 列出所有书籍 |
| GET | `/api/chunks?book_id=X` | 查看书籍的所有 chunk |
| POST | `/api/books` | 上传书页（multipart） |
| GET | `/api/books/{id}/text` | 查看书籍全文 |
| POST | `/api/books/{id}/questions` | 生成题目 |
| GET | `/api/books/{id}/questions` | 获取已生成题目 |
| POST | `/api/books/{id}/clear` | 删除书籍 |
| POST | `/api/clear-all` | 清空所有数据 |

## 技术栈

- **OCR**：[RapidOCR](https://github.com/RapidAI/RapidOCR)（本地 ONNX，CPU 友好，中文好）
- **检索**：sklearn TF-IDF（char_wb 2-3 gram，中文友好）
- **LLM**：AGNES Qwen3.8-Flash-Next（OpenAI 兼容接口）
- **后端**：FastAPI + Uvicorn
- **前端**：原生 HTML + CSS + JS（单文件，零构建）

## 已知限制

1. **出题慢**：Qwen3.8 单次 60-90 秒，前端没有进度条反馈
2. **答案全 A**：LLM 偷懒模式，未强制随机答案位置
3. **页码顺序**：按上传顺序，乱序上传需手动调整
4. **答案全是 A**：未在 prompt 里强制答案位置随机

## 后续优化方向

- 加 VLM（多模态大模型）直接看图，跳过 OCR
- 题目答案位置随机化
- 大上下文分块出题
- 数据库持久化（SQLite/PostgreSQL）
- Docker 化部署
- 图片质量预检（模糊/倾斜检测）
