
# 🎨 SRT to Stick Figure Image Generator

将 SRT 字幕文件自动转换为火柴人风格插画图片，基于 ModelScope 文生图 API 和 DeepSeek 大语言模型。

## ✨ 功能特点

- 📂 上传 SRT 字幕文件，自动保留原始编号
- 🧠 调用 DeepSeek 进行语义分组（每组 1-5 句，自动判断连贯性）
- ✍️ 生成火柴人风格正向提示词（动态描述人物、动作、道具、思考气泡等）
- 🎨 调用 ModelScope 文生图 API，支持多个备选模型自动切换（额度耗尽自动换下一个）
- 🖼️ 纯白色背景输出，便于后期抠图
- 🌐 Web 界面操作，实时显示进度
- 💾 图片命名直观：`img_001_起始句-结束句.png`

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- pip

### 2. 安装依赖

```bash
pip install flask pillow requests urllib3
```

可选：如需额外抠图功能，安装 `rembg`：

```bash
pip install "rembg[cpu,cli]"
```

### 3. 配置 API Key

复制配置示例文件并填入您的密钥：

```bash
cp config/API_key.example.json config/API_key.json
cp config/PicModel.example.json config/PicModel.json
```

编辑 `config/API_key.json`，填入：

- `modelscope.api_key`：从 [ModelScope](https://modelscope.cn) 个人中心获取（需开通文生图服务）
- `deepseek.api_key`：从 [DeepSeek Platform](https://platform.deepseek.com) 获取

### 4. 运行 Web 服务

```bash
python app.py
```

打开浏览器访问 `http://127.0.0.1:5000` 即可。

## 📁 项目结构

```
项目根目录/
├── app.py                      # Flask Web 主程序
├── srt_pre_prompt.py           # SRT 解析 + 语义分组
├── llm_prompt_generator.py     # 调用 DeepSeek 生成正向提示词
├── generate_image.py           # 调用 ModelScope 批量生图
├── remove_background.py        # 背景抠图（可选）
├── test_models.py              # 测试 ModelScope 模型可用性
├── config/
│   ├── API_key.json            # 实际密钥
│   └── PicModel.json           # 实际模型列表
├── templates/
│   └── index.html              # Web 前端页面
├── uploads/                    # 用户上传的 SRT 文件（自动生成）
├── tmp/                        # 中间文件（提示词预览、LLM 结果）
├── output/                     # 最终生成的图片
└── README.md
```

## 🖥️ 使用流程

1. **上传 SRT**：选择字幕文件，点击上传。
2. **语义分组**：系统调用 DeepSeek 将句子按语义分成 1-5 句一组（保留原始序号）。
3. **生成提示词**：为每组文本生成火柴人风格的正向提示词（包含人物、动作、道具、思考气泡等）。
4. **生成图片**：遍历所有提示词，依次调用 ModelScope 文生图 API，若当前模型额度用完则自动切换至下一个备选模型。
5. **结果展示**：所有图片保存在 `output/<字幕名>/` 下，Web 页面列出所有图片并提供下载。

## ⚙️ 自定义模型列表

编辑 `config/PicModel.json`，按顺序排列您希望尝试的模型，例如：

```json
[
    {
        "display_name": "Qwen Image",
        "model_id": "Qwen/Qwen-Image"
    },
    {
        "display_name": "Z-Image-Turbo",
        "model_id": "Tongyi-MAI/Z-Image-Turbo"
    }
]
```

脚本会按此顺序尝试，直到某模型成功生成图片。

## 📌 注意事项

- **API 额度**：ModelScope 和 DeepSeek 均有免费额度，请合理使用，额度不足时可切换模型或充值。
- **生成时间**：每张图片约需 20-60 秒，大量图片需要耐心等待。
- **Windows 编码**：若控制台打印表情符号出错（如 `✅` 显示乱码），脚本已内置 UTF-8 重定向，通常无需额外操作。如仍有问题，请设置系统区域为 UTF-8 支持。
- **空格处理**：文件名中的空格会被自动替换为下划线，避免路径错误。
- **隐私安全**：上传的 SRT 文件仅保存在 `uploads/` 目录，不会上传到任何外部服务器。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

## 📄 许可证

MIT License
