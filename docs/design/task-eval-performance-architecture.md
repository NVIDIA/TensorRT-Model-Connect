# Task Eval 重构与 Performance Evaluation 架构设计

- **状态：** Implemented for the ETTh1 vertical slice
- **日期：** 2026-07-17
- **范围：** 现有 Task Eval 的渐进式重构，以及在同一验证工作流中加入可重复、可比较的 Performance Evaluation
- **实现原则：** 先固化契约和兼容层，再迁移 task；不做一次性重写，不降低任何现有验收门槛

## 1. 决策摘要

在 Task Eval 中加入 Perf 之前，应先做一次**有边界的结构重构**。原因不是现有功能不能继续增加，而是当前 task、backend、计时、scoring、gate 和 artifact 逻辑耦合在一个大型脚本中；不同 task 的 `wall_ms` 又代表不同计时范围。直接把这些值聚合成 p50/p95，会得到形式统一但语义不可比的结果。

本设计采用三层组合：

1. 外部使用最小、plan-driven 的接口：`compile_plan()` 和 `run()`。
2. 内部使用强类型的 `TaskAdapter`、`BackendAdapter` 和 `BackendSession`，统一各 task 的扩展方式。
3. 迁移期间保留 `tools/task_eval.py`、现有 YAML、命令行行为和结果格式，通过 compatibility facade 兼容旧调用方。

命名上建议：

- 对外总称使用 **Model Validation**。
- 内部明确区分三个 assessment lane：**Task Evaluation**、**Fidelity Validation** 和 **Performance Evaluation**。
- 迁移期继续保留 `task_eval` 文件名、命令和配置键；完成主要 suite 迁移后，再单独决定是否弃用旧名字。

Performance Evaluation 初期只产出观察结果，不立即成为 blocking gate。只有测量语义、环境身份、baseline 和稳定性均验证完成后，指定 suite 才可逐个升级为 blocking。

## 2. 背景与问题

### 2.1 当前实现的主要结构问题

当前 Task Eval 已支持 text、encoder、vision、VLM、ASR、TTS、diffusion、time series 等多类任务，但 task 分派分散在以下多个阶段：

- dataset prepare；
- HF/reference 执行；
- TRT/TRTFB 执行；
- prompt、cache 和生成参数处理；
- scoring 与 HF/TRT agreement；
- result 汇总与 artifact 发布。

新增一个 task 往往需要同时修改这些区域。task 的概念没有集中到一个局部 seam，导致：

- 扩展成本随 task 数量增长；
- task 之间容易出现隐式差异；
- 单元测试需要穿过大量全局分支；
- Perf 很难复用统一的生命周期和计时规则。

### 2.2 现有 `wall_ms` 不能直接作为统一 Perf 指标

当前记录的时间有多种语义。例如：

- 有的 text 路径只测量已加载 session 的 generate；
- 有的 VLM/ASR 路径按 sample 启动子进程，时间包含启动甚至模型加载；
- 有的 HF 路径只包围 inference 调用；
- 有的 media 路径直接采用 runner 返回的 `timing_s`；
- 现有 GPU memory before/after 更适合清理或泄漏检查，不等于运行期间 peak memory。

因此旧 `wall_ms` 可以继续作为诊断字段，但不得作为新 Performance Evaluation 的 baseline 或 gate 数据源。

### 2.3 现有 Perf 工具的关系

仓库已经存在 `tools/perf_compare.py` 和 `tools/perfdb.py`：

- `perf_compare.py` 已提供 text generation 场景的 warmup、iterations、prefill/decode、throughput 和 peak memory 等能力；
- `perfdb.py` 已提供环境 fingerprint、历史记录、baseline 和 regression compare。

新设计不立即替换它们。第一阶段复用其成熟概念；后续通过 reporter/publisher adapter 接入统一结果。Task Eval 的核心结果 schema 不直接依赖 SQLite，也不把当前 text 专用的 PerfDB 行结构强行推广到所有 task。

## 3. 目标与非目标

### 3.1 目标

- 将一个 task 的 prepare、输入输出契约、quality/fidelity reducer 集中到一个 `TaskAdapter`。
- 将 backend 生命周期集中到 `BackendAdapter` 和 `BackendSession`。
- 让同一份确定性 workload 同时支持 task quality、backend fidelity 和 performance 测量。
- 明确区分 workload、execution、measurement 三种身份，避免不同输入或不同测量方法之间错误比较。
- 统一 warmup、iterations、同步点、冷启动/热启动范围、统计量和 peak memory 语义。
- 保持旧 CLI、suite 配置、结果字段、退出码和已有 gate 行为兼容。
- 允许 suite 逐个迁移，并为迁移前后的结果建立 golden equivalence。
- 让未迁移 task 在请求 Perf 时 fail closed，而不是静默使用不可靠的旧计时值。

### 3.2 非目标

- 不一次性重写所有 task。
- 不为通过 CI 而放宽、删除或重解释现有 quality/fidelity gate。
- 不把所有 profiling 工具合并进 Task Eval。
- 不在第一阶段定义跨硬件可比较的绝对性能分数。
- 不用 microbenchmark 替代 E2E/task-level performance；两者可以并存。
- 不把报告发布到 GitHub Pages。
- 不在这次重构中扩展无关的验证系统。

## 4. 领域模型与术语

### 4.1 Assessment lanes

**Task Evaluation**

评价模型对用户任务本身做得是否正确。例如 accuracy、F1、WER、MAE、MSE、CLIP score。

**Fidelity Validation**

评价 TRT 输出相对 HF/reference 是否一致。例如 token-ID parity、relative L2、max absolute error、mask mIoU 或 correctness agreement。

**Performance Evaluation**

评价一个明确 backend 在明确 workload 和明确 measurement scenario 下的执行表现。例如 warm latency p50/p95、throughput、TTFT、TPOT、peak memory。

三者独立产生状态，不能用一种结果替代另一种结果。性能更快不能抵消 task quality 或 fidelity 失败。

### 4.2 三层身份与 comparison key

```text
workload_digest
  = suite contract
  + dataset revision
  + ordered sample IDs
  + prepared inputs
  + generation/inference config
  + seed
  + task adapter version

execution_digest
  = workload_digest
  + model and revision
  + backend and backend adapter version
  + bundle/engine identity
  + precision
  + runtime configuration
  + environment fingerprint

measurement_digest
  = execution_digest
  + measurement scope
  + warmup count
  + measured iteration count
  + process repetition count
  + concurrency
  + synchronization policy

comparison_key
  = workload comparison identity
  + logical model ID
  + backend kind
  + precision
  + performance profile ID and version
  + environment compatibility class
```

三个 digest 用于完整 provenance。当前版本与 baseline 的 source commit、bundle 或 engine identity 正常情况下会不同，因此不能要求两个运行的 `measurement_digest` 完全相等。

`comparison_key` 专门表达 regression comparison 的等价类。它排除 timestamp、source commit、bundle digest 等被比较变量，但保留 workload 语义、backend 类型、precision、measurement profile 和环境兼容等级。任何影响测量解释的变更都必须改变 comparison key 或 profile version。

比较规则：

- Quality/Fidelity 结果至少要求 `workload_digest` 一致。
- Backend performance regression 要求 `comparison_key` 一致；两个运行各自的完整 `measurement_digest` 继续保留用于追溯。
- comparison key 不匹配时返回 `BLOCKED` 或 `NOT_COMPARABLE`，不得继续生成“性能提升/回退”结论。

### 4.3 核心对象

```python
@dataclass(frozen=True)
class ValidationRequest:
    suite_id: str
    model_selectors: tuple[str, ...]
    assessments: frozenset[Assessment]
    dataset_override: Path | None = None
    limit: int | None = None
    seed: int | None = None
    performance_profile_id: str | None = None


@dataclass(frozen=True)
class ValidationPlan:
    schema_version: str
    suite: SuiteContract
    cases: tuple[CasePlan, ...]
    ordered_samples: tuple[SampleRef, ...]
    assessment_specs: tuple[AssessmentSpec, ...]
    artifact_policy: ArtifactPolicy


@dataclass(frozen=True)
class MeasurementScenario:
    scope: MeasurementScope
    warmup_iterations: int
    measured_iterations: int
    process_repetitions: int
    concurrency: int
    synchronization: SynchronizationPolicy
    metrics: tuple[MetricSpec, ...]


@dataclass(frozen=True)
class ValidationReport:
    schema_version: str
    plan_digest: str
    cases: tuple[CaseResult, ...]
    artifacts: tuple[ArtifactRef, ...]
    overall_status: OverallStatus
```

完整实现还应包含：

- `PreparedWorkload`：已验证、已规范化的有序输入；
- `WorkloadIdentity`、`ExecutionIdentity`、`MeasurementIdentity`；
- `ComparisonKey`：定义 baseline 与当前运行的可比较维度；
- `MetricValue` 和带单位、方向、聚合方式的 `MetricSpec`；
- `GateSnapshot`：记录使用的阈值、baseline、比较方向和判断结果；
- `EnvironmentFingerprint`：GPU、driver、CUDA、TensorRT、runtime、host class 等；
- `ArtifactRef`：逻辑名称、schema、相对路径、digest 和敏感性分类。

所有进入 plan 或 report 的对象都必须可序列化、可版本化。核心 plan 使用 immutable dataclass；`run()` 不允许在执行时覆盖影响语义的参数。

## 5. 目标架构

```text
Legacy CLI / Existing YAML                  New CLI
             |                                |
             +-------- Compatibility Facade --+
                              |
                        compile_plan()
                              |
                    Immutable ValidationPlan
                              |
                            run()
                              |
                  +-----------+-----------+
                  |                       |
            Task Adapter Registry    Backend Registry
                  |                       |
            PreparedWorkload        BackendSession(s)
                  +-----------+-----------+
                              |
              +---------------+----------------+
              |               |                |
        Task Evaluation  Fidelity Validation  Measurement Engine
              |               |                |
              +---------------+----------------+
                              |
                          Gate Engine
                              |
                       ValidationReport
                              |
              +---------------+----------------+
              |               |                |
       Versioned artifacts  Legacy projection  Optional publishers
```

建议的代码 locality：

```text
tools/model_validation/
  __init__.py
  contracts.py
  planner.py
  engine.py
  registry.py
  measurement.py
  gates.py
  artifacts.py
  compatibility.py
  adapters/
    tasks/
    backends/

tools/task_eval.py                         # 迁移期兼容入口
tests/task_eval/validation_suites.yaml     # 现有 task contract
tests/task_eval/performance_profiles.yaml  # 新增 measurement contract
```

`tools/model_validation/` 是测试与验证工具，不进入 runtime 产品包，避免给 `python/tensorrt_model_connect/` 增加不必要依赖。

## 6. 外部接口

对新调用方只暴露两个主要操作：

```python
def compile_plan(
    request: ValidationRequest,
    *,
    catalog: ValidationCatalog,
) -> ValidationPlan:
    ...


def run(
    plan: ValidationPlan,
    *,
    artifact_root: Path,
) -> ValidationReport:
    ...
```

约束：

- `compile_plan()` 解析 suite、model、dataset revision、sample 顺序、adapter 版本和 performance profile。
- `run()` 只接受 plan 与 artifact 输出位置；不得再接收 `limit`、seed、threshold、warmup 等覆盖参数。
- 所有有效配置必须写入 `plan.json`，避免命令行显示一种配置、实际运行另一种配置。
- plan 编译失败属于配置错误；资源不可用属于 execution `BLOCKED`；模型运行异常属于 execution `ERROR`。

兼容 CLI 可以继续支持现有命令，但内部先翻译成 `ValidationRequest`。未来可增加：

```text
python tools/model_validation.py plan ...
python tools/model_validation.py run --plan <plan.json> ...
```

新 CLI 不是第一阶段的前置条件。

## 7. 内部 interfaces

### 7.1 TaskAdapter

```python
class TaskAdapter(Protocol[InputT, OutputT]):
    kind: str
    version: str

    def prepare(self, context: PrepareContext) -> PreparedWorkload[InputT]: ...

    def quality_reducer(
        self,
        workload: PreparedWorkload[InputT],
        outputs: Sequence[SampleOutput[OutputT]],
    ) -> MetricSet: ...

    def fidelity_reducer(
        self,
        workload: PreparedWorkload[InputT],
        candidate: Sequence[SampleOutput[OutputT]],
        reference: Sequence[SampleOutput[OutputT]],
    ) -> MetricSet: ...

    def measurement_units(self) -> tuple[MeasurementUnit, ...]: ...
```

要求：

- adapter 只处理 task 语义，不自行决定 CI lane 或 regression baseline。
- `prepare()` 必须产生稳定 sample ID、确定顺序，并验证输入资产。
- reducer 必须拒绝 count、ID、shape 和 non-finite mismatch；不得静默跳过。
- task 特有的 prompt、normalization、token parity、media metadata 都局部放在相应 adapter。

### 7.2 BackendAdapter 与 BackendSession

```python
class BackendAdapter(Protocol[InputT, OutputT]):
    kind: str
    version: str

    def preflight(self, case: CasePlan) -> PreflightResult: ...

    def open(self, context: BackendContext) -> BackendSession[InputT, OutputT]: ...


class BackendSession(Protocol[InputT, OutputT]):
    def infer(self, sample: InputT) -> OutputT: ...
    def synchronize(self) -> None: ...
    def reset_measurement_state(self) -> None: ...
    def runtime_stats(self) -> Mapping[str, MetricValue]: ...
    def close(self) -> None: ...
```

要求：

- session 明确表示“模型/backend 已加载”的生命周期。
- warm latency 只能在同一个已打开 session 内测量。
- cold/load 场景通过重新 `open()` 或重新启动受控 process 测量，不能和 warm latency 混在同一数组中。
- backend 输出必须先以 task adapter 规定的 raw contract 保存，再进入 postprocess/scoring。

### 7.3 MeasurementEngine

Measurement Engine 负责执行 `MeasurementScenario`，不包含 task scorer 或 gate policy。它输出原始 observation 和统计量：

```python
class MeasurementEngine:
    def measure(
        self,
        scenario: MeasurementScenario,
        session_factory: BackendSessionFactory,
        workload: PreparedWorkload[Any],
    ) -> MeasurementResult:
        ...
```

Measurement Engine 必须：

- warmup 不进入统计样本；
- 每个 observation 记录 sample ID、iteration、process repetition 和单位；
- 在 profile 要求的边界执行 backend/CUDA synchronize；
- 保存原始 observation，聚合值可以重新计算；
- 在任何 iteration 失败时显式记录，不用剩余成功值掩盖失败；
- 对不能提供的 metric 返回 `UNAVAILABLE` 及原因。

### 7.4 GateEngine

Gate Engine 只消费 metrics、baseline 和 policy：

- quality gate 只判断 task metrics；
- fidelity gate 只判断 candidate/reference agreement；
- performance gate 只判断 comparable measurement；
- overall status 根据明确 policy 聚合，不能把 lane 状态压缩成一个含义不清的 `passed`。

### 7.5 Compatibility Facade

Compatibility Facade 负责：

- 解析现有 `task_eval.py` 参数和 `validation_suites.yaml`；
- 对已迁移 suite 使用 native adapter；
- 对未迁移 suite 使用 `LegacySuiteAdapter`；
- 通过 `LegacyResultProjector` 生成旧路径、字段、summary 和退出码；
- 阻止 legacy suite 请求 Performance Evaluation。

明确规则：

```text
legacy suite + Task/Fidelity       -> 继续运行，行为兼容
native suite + Task/Fidelity       -> 新 engine，输出新旧两套 artifact
native suite + Performance         -> 新 Measurement Engine
legacy suite + Performance         -> BLOCKED，说明该 task 尚未迁移
```

## 8. 执行时序

一次完整运行按以下顺序进行：

1. 解析请求并编译 immutable plan。
2. 解析 dataset revision，固定 ordered sample IDs。
3. Task adapter prepare 并计算 `workload_digest`。
4. Backend adapter preflight，检查模型、bundle、precision、资源和能力。
5. 打开 reference/candidate session，执行需要的 Task/Fidelity workload。
6. 先保存 raw output，再做 task-specific reducer。
7. 若启用 Perf，按 performance profile 重新建立所需 session 并测量。
8. 计算统计量，解析 baseline，判断 comparable 性。
9. Gate Engine 分别产生 quality、fidelity、performance 状态。
10. 原子写入 versioned artifacts，再生成 legacy projection 和可选 publisher 输出。

若 quality 或 fidelity 失败：

- Performance measurement 可以继续收集诊断数据，前提是执行仍安全且结果不会误导。
- Performance assessment 状态必须为 `BLOCKED`，原因标记为 correctness prerequisite failed。
- 该次数据不得更新 baseline，也不得宣称性能通过。

## 9. Performance profile

Perf 的测量方法与硬件策略独立存放：

```yaml
schema_version: 1

profiles:
  gb300_single_stream_v1:
    description: Stable single-stream warm inference on a dedicated GB300 GPU
    environment:
      gpu_class: gb300
      exclusive_gpu: true
      max_background_gpu_utilization_pct: 5
    scenario:
      scope: warm_session
      concurrency: 1
      warmup_iterations: 5
      measured_iterations: 30
      process_repetitions: 3
      sample_order: fixed
      synchronization: backend_and_cuda
    metrics:
      required:
        - request_latency_ms.p50
        - request_latency_ms.p95
        - throughput_units_per_second
        - peak_device_memory_mb
      optional:
        - load_latency_ms
        - time_to_first_output_ms.p50
        - time_per_output_unit_ms.p50
    comparison:
      baseline_policy: explicit_approved
      missing_baseline: observed
      max_regression_fraction:
        request_latency_ms.p50: 0.05
        request_latency_ms.p95: 0.10
        throughput_units_per_second: 0.05
```

这只是 schema 形状示例，不代表已经批准具体数值。所有实际 threshold 需要基于目标机器的重复测量单独 review。

suite 选择 profile 时只引用 ID，不复制测量细节。例如：

```yaml
ci:
  lane: nightly
  performance_profile: gb300_single_stream_v1
  performance_mode: observation
```

`performance_mode` 分三阶段：

- `disabled`：不运行；
- `observation`：产出结果，但不因 regression 使 job 失败；
- `blocking`：满足 baseline 和稳定性前提后，regression 可使 job 失败。

## 10. Perf 指标语义

### 10.1 通用必选指标

- `request_latency_ms.p50`：请求延迟中位数。50% 的有效 observation 小于或等于该值。p50 不是平均值。
- `request_latency_ms.p95`：95% observation 小于或等于该值，用于观察尾延迟。
- `throughput_units_per_second`：profile 定义的 output unit 每秒处理量。
- `peak_device_memory_mb`：指定测量窗口内的设备内存峰值；必须先 reset peak tracker。
- `error_rate`：失败 observation 占比；blocking profile 默认要求为 0。

同时建议记录 `mean`、`min`、`max`、standard deviation 和 MAD，但回归门禁优先使用 p50/p95 和明确方向的 throughput。

百分位算法必须在 schema 中固定。建议使用 nearest-rank，或采用标准库 quantile 方法并写入 `quantile_method`；不可在不同运行之间隐式改变算法。

### 10.2 Task-family 可选指标

- 自回归 text/VLM：TTFT、TPOT、tokens/s、generated token count、engine launch count。
- Encoder/vision/time series：batch latency、samples/s。
- ASR：real-time factor、audio seconds/s。
- TTS：real-time factor、generated audio seconds/s。
- Image/video generation：seconds/image、seconds/frame、frames/s。

所有归一化指标必须同时保存实际 input/output 数量。例如 tokens/s 必须保存真实生成 token 数，不能只保存 `max_new_tokens`。

### 10.3 Cold、load 与 warm 分离

以下值不得聚合到同一个 percentile 数组：

- `load_latency_ms`：读取 bundle、创建 runtime/session 的耗时；
- `cold_request_latency_ms`：session 创建后的首次请求；
- `warm_request_latency_ms`：完成 warmup 后的请求。

如果 task 当前只能通过“每 sample 启动 CLI”运行，则只能声明 `process_e2e` scope。它不能冒充 `warm_session`，也不能与 in-process warm latency baseline 比较。

### 10.4 环境与稳定性要求

blocking Perf 至少要求：

- 固定 GPU class、precision、model revision、bundle identity 和 runtime config；
- dedicated/exclusive GPU，记录运行前后的 utilization、temperature、power state；
- 固定 sample、seed、输入长度/尺寸和输出预算；
- 至少 3 个 process repetitions；
- 报告每轮结果和跨轮离散程度；
- 明确的 baseline 选择与人工批准记录。

环境不满足时状态为 `BLOCKED`，而不是 `FAILED`。性能值仍可作为诊断 observation 保存，但不能用于 regression gate。

## 11. 状态模型

```python
class ExecutionStatus(Enum):
    COMPLETED = "completed"
    ERROR = "error"
    BLOCKED = "blocked"


class AssessmentStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    OBSERVED = "observed"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"
```

每个 case 都至少输出：

```json
{
  "execution_status": "completed",
  "assessments": {
    "task": {"status": "passed"},
    "fidelity": {"status": "passed"},
    "performance": {"status": "observed"}
  }
}
```

overall policy：

- execution `ERROR` 或 required assessment `FAILED` => overall `FAILED`；
- required assessment `BLOCKED` => overall `BLOCKED`；
- observation-only Perf 的 regression 只在报告中标记，不改变 overall pass/fail；
- optional assessment `NOT_RUN` 不影响 required lanes。

## 12. Artifact 与 schema

建议每次运行输出：

```text
<artifact_root>/
  plan.json
  environment.json
  workloads/
    <workload_digest>/manifest.json
  executions/
    <case_id>/<backend>/raw_outputs.jsonl
  measurements.jsonl
  results.json
  summary.md
  junit.xml
  legacy/
    ... existing Task Eval layout ...
```

要求：

- 所有 JSON 顶层包含 `schema_version`。
- 路径在报告中使用相对路径，artifact 内容保存 SHA-256。
- `measurements.jsonl` 保存原始 observation；`results.json` 保存聚合指标和 gate snapshot。
- baseline 只引用不可变的 prior result identity，不复制一个无法追溯的裸数值。
- 公开 artifact 继续执行现有路径和敏感信息清理规则。
- 写入采用 staging directory + atomic rename，避免部分成功结果被误认为完整报告。

PerfDB 接入采用可选 publisher：

```python
class ResultPublisher(Protocol):
    def publish(self, report: ValidationReport) -> PublishResult: ...
```

第一阶段可以不写 PerfDB。若接入，应先扩展为 generic metric rows 或建立版本化映射，不能丢失 task、workload 和 measurement digest。

## 13. 渐进式迁移方案

### Phase 0：锁定兼容契约

- 为现有 CLI、退出码、summary 和关键 JSON 字段补 golden fixtures。
- 建立现有 suite 的结果样例，明确哪些字段属于 public compatibility contract。
- 禁止新功能继续扩大 `task_eval.py` 的跨阶段分支。

完成标准：在不运行 GPU 的情况下，可以验证 legacy projection 是否兼容。

### Phase 1：引入 contracts、planner 和 facade

- 新增 immutable contracts、schema validator、registry 和 plan serialization。
- `tools/task_eval.py` 仍调用 legacy path，但先经过 Compatibility Facade。
- 不改变任何现有 suite 的执行和 gate 结果。

完成标准：现有 `tests/tools/test_task_eval.py` 全部通过，旧 CLI smoke 输出兼容。

### Phase 2：迁移 ETTh1 time-series suite

选择 ETTh1 作为首个 native adapter，因为它具备：

- 确定性 sample/window；
- 明确的 HF/TRT 输出张量；
- 已审查的 task 与 fidelity gate；
- 已进入 nightly 验证。

工作项：

- 实现 time-series `TaskAdapter`；
- 实现 HF 和 TRTFB backend session；
- 新旧路径在相同 sample 上执行 golden equivalence；
- 新结果通过 LegacyResultProjector 生成原有格式。

完成标准：sample IDs、raw output、task metrics、fidelity metrics、pass/fail 与旧路径一致；允许的浮点序列化差异必须显式记录。

### Phase 3：ETTh1 Perf observation

- 增加一个非 blocking performance profile。
- 验证 profile-defined warmup/measurement、3 次 process repetition、p50/p95 和 peak memory。
- 连续收集足够 nightly 数据，评估 noise 和 runner health。
- correctness prerequisite 失败时 Perf 为 `BLOCKED`。

完成标准：相同 measurement identity 可重现；原始 observation 可重新聚合出报告值；旧 `wall_ms` 未参与 gate。

### Phase 4：扩展 task adapters

建议迁移顺序：

1. encoder 与单输入 vision；
2. text generation 与 ASR；
3. VLM、TTS、diffusion/video 等复合 media task。

每迁移一类 task 都必须先完成：输入输出 contract、golden equivalence、backend session 生命周期和 task-specific measurement unit。

### Phase 5：逐 suite 启用 blocking Perf

只有同时满足以下条件才允许从 `observation` 升为 `blocking`：

- correctness/fidelity gate 稳定；
- dedicated runner 环境可识别且稳定；
- baseline 经显式批准；
- 误报率达到团队接受标准；
- regression threshold 来自重复数据，而非为某次 CI 结果临时调整。

### Phase 6：命名与弃用

- 新报告和文档使用 Model Validation 总称。
- 评估新增 `tools/model_validation.py` 入口的必要性。
- 只有在主要 suite 已迁移、调用方已盘点后，才为 `tools/task_eval.py` 发布 deprecation timeline。
- 旧入口在至少一个稳定发布周期内保持 wrapper 行为。

## 14. 测试策略

### 14.1 GPU-free tests

- contract/schema 序列化与版本拒绝；
- plan immutability 与 deterministic digest；
- ordered sample identity；
- TaskAdapter contract tests；
- fake BackendSession lifecycle；
- warmup 不进入 observation；
- p50/p95、MAD、吞吐和错误率计算；
- cold/warm scope 不可混合；
- comparison key mismatch 返回 not comparable；
- quality/fidelity/performance gate 分离；
- legacy result projection golden tests；
- legacy suite 请求 Perf 时 fail closed。

### 14.2 GPU integration tests

- session 加载一次并进行多次 warm inference；
- backend synchronize 边界正确；
- peak memory reset 和读取窗口正确；
- 固定 input/seed/output budget 可重现；
- process repetition 隔离资源；
- 环境不满足 blocking profile 时返回 `BLOCKED`。

### 14.3 迁移 equivalence tests

每个迁移 suite 在同一输入上比较旧路径与新路径：

- sample 数量、ID、顺序；
- backend raw output；
- task metric 和 fidelity metric；
- gate snapshot；
- exit code 与 legacy summary。

equivalence 失败时修复新架构或明确评审旧行为，不能通过调整 gate 使测试通过。

## 15. 验收标准

本次架构重构完成的最低验收标准：

1. 现有未迁移 suite 的 CLI、结果和 gate 行为不变。
2. ETTh1 native adapter 与 legacy path 达到 golden equivalence。
3. Perf 只从 Measurement Engine 产生，不读取旧 `wall_ms` 作为 baseline/gate。
4. Perf 报告包含 measurement identity、环境、原始 observation、p50、p95、throughput 和 peak memory。
5. unsupported legacy task 请求 Perf 时显式 `BLOCKED`。
6. Task、Fidelity、Performance 三个 assessment 状态独立可见。
7. correctness/fidelity 失败的运行不能更新性能 baseline。
8. 所有 threshold 和 baseline 都可追溯，且未因迁移而降低现有通过标准。
9. 新增一个 task adapter 不需要修改 planner、engine、measurement 或其他 task adapter。
10. GPU-free source/unit suite通过；目标 GPU 上的 ETTh1 smoke 和 observation run 通过。

## 16. 实施前需要单独确认的决策

以下问题不阻塞 contracts/facade 的第一阶段，但必须在启用 blocking Perf 前确认：

- Performance profile 的正式 owner，以及 profile/threshold 的 review 流程。
- baseline 使用显式批准的固定 run，还是经批准的 rolling window；本设计默认前者。
- 通用 PerfDB schema 是扩展现有 `perf_runs`，还是新增 versioned generic measurement tables。
- 各 backend 的可靠 peak device memory 实现方式。
- runner health 指标和 blocking profile 的 noise 上限。
- observation 数据保留周期与 artifact 大小预算。

## 17. 当前实现状态

本次实现已经完成 Phase 0 至 Phase 3 的首个纵向切片：

- 保留原有 CLI、suite、结果字段和 gate 行为，并新增可校验的 immutable validation plan；
- 将 ETTh1 的 workload identity 和 fidelity reducer 迁入 native task adapter；
- 新增独立的 performance profile、Measurement Engine、环境兼容检查、comparison key 和显式批准 baseline；
- 使用外层单调时钟测量 HF/TRTFB process E2E，不读取旧 wall_ms；
- 保存 warmup 与 measured 原始 observation，并生成 p50、p95、mean、standard deviation、MAD、throughput、error rate、peak memory（可用时）和跨 process repetition 离散度；
- 每次 Perf observation 的输出必须匹配已经通过 correctness/fidelity 阶段的对应输出，否则 Perf 为 BLOCKED 且 baseline candidate 不可批准；
- observation 不改变原有结果；blocking 仅在环境匹配且 baseline 同时满足可批准、人工批准和 comparison-key 一致时生效；
- 未迁移 task 请求 Perf 时 fail closed。

其他 task family 按 Phase 4 的顺序逐类迁移。它们没有统一、可信的 backend 生命周期之前，不会复用旧 wall_ms 或伪装成 native Perf。

## 17. 推荐的下一步

先实施 Phase 0 和 Phase 1，只引入 contracts、planner、compatibility facade 及测试，不迁移 runtime 行为。随后用 ETTh1 完成第一个纵向切片：prepare -> HF/TRT sessions -> task/fidelity -> Perf observation -> artifacts -> legacy projection。

这个顺序既能尽早验证新 interface 是否足够，又把首轮风险限制在一个已有稳定契约的 suite 内。
