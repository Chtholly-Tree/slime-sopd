# Think-with-Image

VLM 多轮工具调用强化学习训练框架。通过让模型在回答视觉问题前主动使用放大工具观察细节，实现"边思考边观察"的推理范式。

## 整体流程

```
用户问题 + 图片
       ↓
┌──────────────────────────────┐
│     Rollout (rollout.py)      │
│  多轮对话循环，最多 max_turns 轮 │
└──────────────────────────────┘
       ↓ 每一轮：
   模型生成 (loss_mask=1)
       ↓
┌──────────────────────────────┐
│    Env (env.py) 解析响应       │
│                              │
│  1. 检测 <answer> → 结束回合   │
│  2. 检测 <tool_call> → 执行工具│
│  3. 都无 → 提示模型继续        │
└──────────────────────────────┘
       ↓
   工具执行 / 答案提取
       ↓
┌──────────────────────────────┐
│  观测编码 (loss_mask=0)        │
│  新图片加入图像列表            │
└──────────────────────────────┘
       ↓
   循环直到结束

       ↓
┌──────────────────────────────┐
│  LLM Judge (llm_judge.py)    │
│  调用 LLM 判断答案是否正确     │
│  reward = 1.0 或 0.0          │
└──────────────────────────────┘
```

## 核心组件

### 1. Rollout (`rollout.py`)

多轮生成循环，管理整个 episode 的 token 序列和图像累积。

**关键逻辑：**
- **loss_mask=1**：模型的 assistant 输出参与训练
- **loss_mask=0**：工具返回的观测结果不参与训练（防止模型记忆工具输出）
- **图像累积**：每轮工具生成的新图片加入 `current_images`，供后续轮次使用

### 2. Env (`env.py`)

交互环境，解析模型输出并执行工具。

**标签解析：**
- `<tool_call>{...}</tool_call>` → 提取工具调用，执行工具
- `<answer>...</answer>` → 提取最终答案，结束回合

**工具调用示例：**
```json
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [100, 200, 400, 600], "label": "文字区域", "img_idx": 0}}
```

### 3. Tools (`tools/`)

| 工具 | 功能 |
|------|------|
| `image_zoom_in_tool` | 裁剪并放大图像的指定区域，便于观察细节 |

工具 schema 用于 `apply_chat_template` 格式化工具调用。

### 4. LLM Judge (`rewards/llm_judge.py`)

调用外部 LLM API 判断模型答案是否正确。

**评判 Prompt：**
```
你是一个题目评判专家。请根据【题目】和【标准答案】，判断【模型答案】是否正确。
请只输出 yes 或 no。
```

**返回：**
- `reward=1.0`：正确
- `reward=0.0`：错误或 API 调用失败

## 训练配置

关键参数在 `config.yaml` 中设置：

| 参数 | 说明 |
|------|------|
| `max_turns` | 每轮对话的最大轮数 |
| `rollout_interaction_env_path` | 环境模块路径 |
| `vlm_rollout_work_dir` | 放大图片的存储目录 |

## 启动训练

```bash
bash run_think_with_image.sh
```

## 数据格式

输入 JSONL 格式：
```json
{"prompt": "图中显示的是什么？", "label": "A", "multi_modal_inputs": {"images": ["path/to/image.jpg"]}}
```

- `prompt`：用户问题
- `label`：标准答案
- `multi_modal_inputs.images`：图片路径列表
