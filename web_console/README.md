# Text2SQL Web Console

这个目录是一个独立的本地网页控制台，用来包装你现有的 Text2SQL 评测流程。

它不会改动 `src/` 里的主逻辑，只是通过页面提供这些能力：

- 一键发起单组评测
- 一键发起 six-way 六组对比
- 在页面里切换 `deepseek / aliyun / lab` API 模式
- 查看当前运行状态和日志
- 查看最近的历史结果
- 用柱状图快速对比最近结果
- 直接在页面里查看 `summary.json` 和 `result.json`

## 启动方式

推荐直接执行：

```bash
./web_console/run.sh
```

默认地址：

```text
http://127.0.0.1:8765
```

如果你想改端口：

```bash
./web_console/run.sh --port 8788
```

如果你更习惯自己指定 Python，也可以用项目虚拟环境直接启动：

```bash
.venv/bin/python -m web_console.app
```

## 依赖说明

这个控制台本身没有新增第三方依赖，直接复用你当前项目的 Python 环境即可。

如果评测本身需要模型 API 环境变量，仍然沿用你原来的配置方式，例如：

- `API_MODE`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`
- `KB_NAME`

页面里的 API 模式切换会优先读取：

```text
scripts/api_mode_exports.private.sh
```

也就是说，如果你已经把各模式的 key 配在这份私有脚本里，页面切换模式后就可以直接生效。

## 页面功能

页面分成四块：

- 运行面板：发起单组或六组评测
- 状态面板：查看任务运行日志
- 对比面板：显示最近结果的柱状图
- 历史面板：浏览最近结果并打开摘要/明细文件

## 说明

- 单组测试默认写入 `data/evaluation_runs/ui_runs`
- 六组对比仍沿用项目原来的输出方式
- 页面只允许读取仓库内部的 `json/jsonl/txt/log` 文件
