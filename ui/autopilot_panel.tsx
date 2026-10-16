import {
  Alert,
  Button,
  Card,
  Inline,
  Page,
  Stack,
  StatusBadge,
  Text,
  useToast,
} from "@neko/plugin-ui"
import type { HostedAction, PluginSurfaceProps } from "@neko/plugin-ui"

// 与插件侧 autopilot_bridge.py 的类型保持一致。
// 字段一律可选：主控台没跑时 context 会全部返回"空"形态，
// 前端不该因为缺字段就崩在一半。
type BridgeStats = {
  tick_hz?: number | null
  tick_p95_ms?: number | null
  scene?: string | null
  intent_count?: number | null
  idle_streak?: number | null
}

type AutopilotState = {
  console_alive?: boolean
  console_version?: string | null
  // Python 侧 autopilot_bridge.plugin_version() 透出。
  // 与 console_version 并排显示，用户一眼判断"这套代码是不是新的"。
  plugin_version?: string | null
  engine_state?: string
  overlay_state?: string
  stats?: BridgeStats | null
  // ↓ 诊断字段。**存在的意义就是这个"未启动"页面** ——
  //   以前它只有一句"请手动启动"，用户根本不知道是没启动还是路径不对。
  bridge_path?: string
  bridge_exists?: boolean
  context_calls?: number
}

/**
 * 两个分支**共用**的页头版本行。
 *
 * ★ 为什么要抽出来、且两个分支都渲染
 *   这行是"我这套代码到底跑起来了吗"的自证：它同时显示插件版本和
 *   主控台版本，版本对不上就立刻看出来。如果只有 `alive` 分支才有，
 *   一旦判定走错分支，反而看不到唯一能暴露问题的那行。
 *
 * ★ 用 `||` 而不是 `??`
 *   能正常工作的 settings 面板里 `??` 一次都没出现过。全站统一用
 *   `||` / `?.`，不在这个 TSX 里引入没有先例的运算符。
 */
function VersionHeader(props: { plugin?: string | null; console?: string | null }) {
  return (
    <div style={rowStyle}>
      <Text tone="info">
        插件 v{props.plugin || "?"} · 主控台 v{props.console || "?"}
      </Text>
    </div>
  )
}

type Cmd = "START_ENGINE" | "STOP_ENGINE" | "START_OVERLAY" | "STOP_OVERLAY" | "QUIT_CONSOLE"

type Tone = "info" | "success" | "warning" | "danger"

function actionById(actions: HostedAction[], id: string): HostedAction | undefined {
  return actions.find((action) => action.id === id || action.entry_id === id)
}

function engineLabel(state?: string): string {
  switch (state) {
    case "running":
      return "运行中"
    case "stopped":
      return "未启动"
    case "crashed":
      return "已崩溃"
    default:
      return "未知"
  }
}

function engineTone(state?: string): Tone {
  if (state === "running") return "success"
  if (state === "crashed") return "danger"
  return "info"
}

function overlayLabel(state?: string): string {
  return state === "running" ? "已显示" : "已隐藏"
}

function overlayTone(state?: string): Tone {
  return state === "running" ? "success" : "info"
}

function fmt(value: number | null | undefined, digits = 1, unit = ""): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—"
  return `${value.toFixed(digits)}${unit}`
}

const rowStyle: Record<string, string | number> = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  flexWrap: "wrap",
}

const btnRowStyle: Record<string, string | number> = {
  display: "flex",
  gap: 8,
  flexWrap: "wrap",
}

export default function AutopilotPanel(props: PluginSurfaceProps<AutopilotState>) {
  const s = props.state || {}
  const toast = useToast()
  const cmdAction = actionById(props.actions || [], "autopilot_cmd")

  const alive = s.console_alive === true
  const engineRunning = s.engine_state === "running"
  const overlayRunning = s.overlay_state === "running"
  const stats = s.stats || {}

  async function send(cmd: Cmd, label: string) {
    if (!cmdAction) {
      toast.error("控制动作不可用，请确认插件已启动。")
      return
    }
    try {
      await props.api.call("autopilot_cmd", { payload: { cmd } })
      toast.success(`${label}：指令已发送`)
      await props.api.refresh()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error))
    }
  }

  // 主控台没跑：整张卡片只讲一件事 —— 怎么让它跑起来。
  // 不显示按钮，因为在这个状态下点了必然失败，给一个注定失败的按钮
  // 比不给更糟（用户会以为是 bug）。
  //
  // ★ 这里必须把**诊断信息也显示出来**
  //   以前这一屏只有一句"请手动启动"，于是"主控台确实没开"和
  //   "主控台开着但我们找错文件了"在用户眼里完全一样 —— 后者会让
  //   人反复去点启动图标，永远解决不了。现在直接告诉他：找的是哪个
  //   文件、文件在不在、这个页面被打开过几次。
  if (!alive) {
    const found = s.bridge_exists === true
    const calls = s.context_calls || 0
    return (
      <Page>
        <Card title="Autopilot 控制台">
          <Stack gap={12}>
            <VersionHeader plugin={s.plugin_version} console={s.console_version} />
            <StatusBadge tone="danger" label="主控台未启动" />
            <Text>
              未检测到 NEKO-Console 主控台。请手动启动它（双击
              NEKO-Console 目录下的 Console.exe，或运行
              python -m neko_console），然后点右上角刷新。
            </Text>
            {/* Alert 只用 tone —— settings 面板里 <Alert> 也只传 tone，
                多传 title 是全站没有先例的用法，不赌运行时是否容错。 */}
            <Alert tone={found ? "warning" : "info"}>
              {found
                ? "桥接文件找到了，但内容判定主控台已失联（心跳超过 3 秒）—— 主控台可能崩溃了。"
                : "桥接文件不存在：主控台还没写过状态，或它跑在另一个目录。"}
            </Alert>
            <Text>查找路径：{s.bridge_path || "（未知）"}</Text>
            <Text>
              本页已请求数据 {calls} 次
              {calls === 0
                ? "（0 次说明 Python context 根本没被调用，问题不在数据而在面板装配）"
                : ""}
            </Text>
            <Text tone="info">
              提示：点窗口的 X 只是缩到托盘，进程仍在运行；托盘图标右键「完全退出」才会结束进程。
            </Text>
          </Stack>
        </Card>
      </Page>
    )
  }

  return (
    <Page>
      <Card title="Autopilot 控制台">
        <Stack gap={12}>
          <VersionHeader plugin={s.plugin_version} console={s.console_version} />
          <div style={rowStyle}>
            <Text>主控台：</Text>
            <StatusBadge tone="success" label={`v${s.console_version || "?"} 运行中`} />
          </div>

          <div style={rowStyle}>
            <Text>引擎：</Text>
            <StatusBadge tone={engineTone(s.engine_state)} label={engineLabel(s.engine_state)} />
          </div>

          <div style={rowStyle}>
            <Text>卡片：</Text>
            <StatusBadge tone={overlayTone(s.overlay_state)} label={overlayLabel(s.overlay_state)} />
          </div>

          <div style={rowStyle}>
            <Text>
              帧率：{fmt(stats.tick_hz, 1, " Hz")} / {fmt(stats.tick_p95_ms, 0, " ms P95")}
            </Text>
          </div>

          <div style={rowStyle}>
            <Text>
              场景：{stats.scene || "—"}　意图：{fmt(stats.intent_count, 0)}
              （空转 {fmt(stats.idle_streak, 0)}）
            </Text>
          </div>

          <Inline>
            <div style={btnRowStyle}>
              <Button
                tone="success"
                disabled={engineRunning}
                onClick={() => send("START_ENGINE", "启动引擎")}
              >
                启动引擎
              </Button>
              <Button
                tone="warning"
                disabled={!engineRunning}
                onClick={() => send("STOP_ENGINE", "停止引擎")}
              >
                停止引擎
              </Button>
            </div>
            <div style={btnRowStyle}>
              <Button
                tone="success"
                disabled={overlayRunning}
                onClick={() => send("START_OVERLAY", "启动卡片")}
              >
                启动卡片
              </Button>
              <Button
                tone="warning"
                disabled={!overlayRunning}
                onClick={() => send("STOP_OVERLAY", "停止卡片")}
              >
                停止卡片
              </Button>
            </div>
          </Inline>

          <Alert tone="warning" title="完全退出主控台">
            会立即结束主控台进程（引擎与卡片一并停止）。面板随即显示「主控台未启动」。
          </Alert>
          <div style={btnRowStyle}>
            <Button tone="danger" onClick={() => send("QUIT_CONSOLE", "完全退出主控台")}>
              完全退出主控台
            </Button>
          </div>
        </Stack>
      </Card>
    </Page>
  )
}
