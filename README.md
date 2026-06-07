# AI-Sentinel

**AI-Sentinel** 是一个基于人工智能的实时视频监控与分析系统。系统通过接入 RTSP 摄像头或本地视频文件，利用视觉语言模型 (VLM) 对监控画面进行场景理解与异常检测，并将分析结果与元数据持久化到向量数据库 (ChromaDB) 以支持基于自然语言的语义检索与回溯。系统采用了前后端解耦的设计，保证了视频流的低延迟播放与后台 AI 分析的稳定运行。

## 🌟 核心功能

- **多源视频接入**：支持无缝拉取 RTSP 网络摄像头流（基于稳定的 FFmpeg 管道）以及本地视频文件（MP4/AVI 等）循环播放。
- **AI 实时分析**：支持集成多款 VLM 模型（支持本地部署如 Qwen2-VL，或兼容 OpenAI 格式的远程 API 和 Ollama 接口），通过自定义提示词生成准确的场景描述并判定异常（如人员跌倒、火灾、入侵等）。
- **智能时间同步 (SmartTimeSync)**：采用极低负载的 OCR 和插值技术提取画面水印时间，使 CPU 开销大幅降低 95%+。
- **多维度检索与历史回放**：依托 ChromaDB 的文本与图像双模态向量化引擎，支持按时间范围与语义进行自然语言检索（如：“红衣服的人”、“晚间的异常”）。
- **全功能 Web 大屏展示**：基于 Streamlit 打造的可视化交互界面，实时显示画面预览、最新警报信息，以及包含时间轴归档的数据库回溯功能。
- **动态存储优化**：支持配置最大保存帧数，自动进行滚动覆盖，避免硬盘空间被打满。

## ⚙️ 系统架构

本项目采用“前端播放与后台分析完全解耦”的并发多线程架构：
- **视频采集模块 (`src/ingestion`)**：负责视频源读取、抽帧、运动检测以及 OCR 时间戳提取，利用循环队列将画面传输给分析端。
- **AI 分析模块 (`src/analysis`)**：通过 Latest-Frame-Only 策略读取最新帧，避免 AI 慢导致的时延堆积；随后调用 VLM 生成文本特征并写入 ChromaDB。
- **WebUI 交互 (`src/webui`)**：使用优化的快照模式（0.3秒高频刷新）或通过 go2rtc 提供 HLS/WebRTC 超低延迟视频流。

## 🚀 快速开始

### 1. 环境准备

推荐使用 Conda 创建独立的 Python 虚拟环境，支持 Python 3.10+。系统部分特性需安装 `FFmpeg` 并加入系统 `PATH`。

```bash
# 克隆仓库
git clone https://github.com/alccs/AI-Sentinel.git
cd AI-Sentinel

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置说明

系统配置采用 JSON 格式管理，默认保存在 `config/settings.json` 中。
初次使用，请拷贝模板文件：

```bash
# 复制配置文件模板
cp config/settings.json.template config/settings.json
```

编辑 `config/settings.json` 设置你的视频源路径与模型 API 密钥：
- **视频源**：`video_source.type` ("file" 或 "rtsp")，以及对应的 `video_source.path`。
- **VLM API 设置**：`vlm.api_url`、`vlm.api_key` 和 `vlm.model_name`。
- **提取与向量库**：推荐开启本地特征提取（需安装 torch 和 transformers），或配置 `embedding` 相关的 API 密钥。

### 3. 运行系统

在 Windows 环境下可直接双击运行：
```bat
start.bat
```

或者在命令行终端中启动：
```bash
streamlit run src/webui/app.py
```

### 4. 界面使用说明

1. **主控制台**：运行后可通过浏览器访问 `http://localhost:8501` 查看监控画面大屏。
2. **报警滚动**：侧边栏会实时滚动展示最新分析的报警卡片与异常截图。
3. **设置与调整**：点击“⚙️ 高级设置”修改 AI 分析间隔（默认为 2 秒一次）、开启/关闭运动检测、设置存储上线，或使用鼠标直接框选 OCR 识别时间戳的区域（ROI）。
4. **历史检索**：切换至“历史检索与搜索”页面，可以按具体日期以及描述文本查询归档视频片段和报警情况。

## 📊 视频流加速优化指南

AI-Sentinel 支持多种流媒体加速协议，如果您觉得原生刷新模式卡顿，可以在系统中选用进阶模式：
- **原生快照模式（默认）**：配置简单，约 300 毫秒延迟，通过后端直传 NumPy 数据降低性能损耗。
- **go2rtc 代理直连（推荐）**：支持使用 go2rtc 服务器转发您的 RTSP 摄像头流并在浏览器端通过 HLS（~500ms延迟）或 WebRTC（<200ms延迟）播放，完全不占用 Python 后端转码开销。只需在主页面切换对应流地址即可。

## 🤝 贡献与支持

如果有任何问题或优化建议，欢迎提交 Issue。
