我已经完成了现有项目的最小化提取工作，仅保留了 GSM8K 数据集实验所需的代码，并按照 `run_gsm8k.py` 中描述的 Transformer 结构进行了精简。

### 已完成的修改：

1.  **保留并清理核心文件**：
    *   **`GDesigner/graph/graph.py`**:
        *   移除了所有 GCN (Graph Convolutional Network) and MLP (Multi-Layer Perceptron) 相关的代码、引用和初始化逻辑。
        *   移除了不被 Transformer 模式使用的旧版连接构建逻辑（如 `construct_spatial_connection`, `construct_features` 等）。
        *   仅保留了 `run_gsm8k.py` 实际使用的 `routing_transformer` 相关逻辑和 `run_next_agent_prediction` 等核心方法。
    *   **`GDesigner/tools/reader/readers.py`**:
        *   移除了所有未使用的文件读取器（如 PDF, Docx, Image 等）及其相关依赖库引用。
        *   仅保留了 `run_gsm8k.py` 必须的 `JSONLReader`。
    *   **`GDesigner/agents/__init__.py` & `GDesigner/prompt/__init__.py`**:
        *   更新了导出列表，仅保留 GSM8K 相关的 Agent 和 PromptSet。

2.  **删除无关文件和目录**：
    *   **实验脚本 (`experiments/`)**: 删除了 `run_mmlu.py`, `run_aqua.py` 等所有非 GSM8K 的实验脚本。
    *   **数据集处理 (`datasets_my/`)**: 删除了除 `gsm8k_dataset.py` 和 `gsm8k/` 目录外的所有其他数据集代码。
    *   **Agent 实现 (`GDesigner/agents/`)**: 删除了 `code_writing.py`, `adversarial_agent.py` 等未被 GSM8K 实验使用的 Agent。
    *   **Prompt 集合 (`GDesigner/prompt/`)**: 删除了 `aqua_prompt_set.py` 等无关的提示词文件。
    *   **其他模块**:
        *   删除了 `GDesigner/gnn/` 目录（GCN 相关）。
        *   删除了 `GDesigner/tools/` 下的 `search/`, `web/`, `vgen/` 等未使用的工具目录。

3.  **保持不变的部分**：
    *   **`experiments/run_gsm8k.py`**: 保持原样，未修改任何逻辑，确保训练、测试和输出流程与原项目完全一致。
    *   **`GDesigner/transformer/`**: 保留了 Transformer 模型定义。
    *   **`GDesigner/agents/math_solver.py` & `final_decision.py`**: 保留了实验所需的 Agent 类。

现在的项目结构已经是一个仅针对 GSM8K 任务、基于 Transformer 路由结构的最小可运行子集。
