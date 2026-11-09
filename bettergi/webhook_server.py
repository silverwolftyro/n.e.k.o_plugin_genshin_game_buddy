"""genshin_game_buddy · BetterGI 桥接 —— Webhook HTTP 服务器。

职责：监听 BetterGI 推送的 Webhook 事件，解析后交给回调函数。

★ 2026-11-07：从独立插件 ``bettergi_companion`` **物理并入** ``genshin_game_buddy``。
  代码与行为逐字保留（只改了日志前缀与本段文档），单插件架构下
  「Bridge 事件接收」与「YOLO/OCR 事件触发」共用一个插件进程。

设计约束
--------
* **零第三方依赖** —— 只用 Python 标准库 ``http.server`` / ``json`` / ``threading``。
* **不 import NEKO 插件 SDK** —— 这样本模块可以脱离宿主单独跑单元测试；
  与 NEKO 的耦合全部落在同级的 ``bridge.py``（回调函数那一层）。
* 正向事件只接受 ``POST <path>``，其余方法/路径一律拒绝。
* 给了 ``control``（指令队列）时，另开两条反向控制端点：
  ``GET <path>/command``（长轮询取指令）与 ``POST <path>/ack``（回执）。
* 读请求体有超时（默认 5s）与体积上限，避免 BetterGI 侧被拖住。
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional

__all__ = ["WebhookServer", "WebhookConfig", "DEFAULT_TIMEOUT_S", "DEFAULT_MAX_BODY_BYTES"]

#: 读请求体的超时（秒）—— 防止 BetterGI 侧连接被长期挂住
DEFAULT_TIMEOUT_S = 5.0

#: 请求体体积上限（字节）。
#: BetterGI 的 WebHook 载荷固定 7 个字段，其中 ``screenshot``（base64 JPEG）
#: 可能有几 MB —— 上限按同机 ``_bettergi_bridge`` 的 32 MiB 经验值取。
#: 解析后会主动丢弃 ``screenshot`` 字段，避免大 base64 常驻内存。
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024  # 32 MiB

#: 时间戳允许的最大偏差（秒）—— 超过视为重放
DEFAULT_REPLAY_WINDOW_S = 300.0

#: 反向控制通道：取指令时的长轮询保持时长（秒）。
#: BetterGI 的 JS 脚本每轮 ``GET /bettergi/command``，没有指令时最多挂这么久。
#: 取 5s 是折中：太长会让指令下发延迟变大，太短会让 BetterGI 侧频繁空转。
DEFAULT_CONTROL_HOLD_S = 5.0

#: 事件回调签名：``(event, message, data, result) -> bool``
#: 返回 ``True`` 表示已处理，``False`` 表示忽略。
EventCallback = Callable[[str, str, Any, Any], bool]


class WebhookConfig:
    """WebHook 端点配置。"""

    __slots__ = (
        "host",
        "port",
        "path",
        "timeout_s",
        "max_body_bytes",
        "replay_window_s",
        "control_hold_s",
    )

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 17890,
        path: str = "/bettergi",
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        replay_window_s: float = DEFAULT_REPLAY_WINDOW_S,
        control_hold_s: float = DEFAULT_CONTROL_HOLD_S,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.path = path if path.startswith("/") else "/" + path
        self.timeout_s = float(timeout_s)
        self.max_body_bytes = int(max_body_bytes)
        self.replay_window_s = float(replay_window_s)
        self.control_hold_s = float(control_hold_s)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return "WebhookConfig(host=%r, port=%d, path=%r)" % (self.host, self.port, self.path)


def _parse_timestamp(value: Any) -> Optional[float]:
    """把 ``timestamp`` 字段解析成 epoch 秒。

    支持三种形态：数字（epoch 秒 / 毫秒）、ISO-8601 字符串、``datetime``。
    解析不出来返回 ``None``（表示不校验）。

    ⚠ **不带时区的 ISO 字符串按「本地时间」解释**。
    BetterGI 的 WebHook 通道发的是 ISO-8601 但**不带时区偏移**（本地时间），
    若按 UTC 解释会凭空差一个时区偏移，导致每次请求都被判为重放。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        # 大于 1e11 的当毫秒处理
        return num / 1000.0 if num > 1e11 else num
    if isinstance(value, datetime):
        # naive datetime：.timestamp() 按本地时区解释，与 BetterGI 一致
        return value.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return _parse_timestamp(int(text))
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        # naive datetime 按本地时区解释（BetterGI 发的是本地时间）
        return dt.timestamp()
    return None


class _WebhookRequestHandler(BaseHTTPRequestHandler):
    """处理 ``POST <path>`` 请求。"""

    server_version = "BetterGICompanion/0.1"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # ── 工具 ──────────────────────────────────────────────────────────
    @property
    def _hook(self) -> "WebhookServer":
        return self.server.hook_server  # type: ignore[attr-defined]

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            # 客户端提前断开（BrokenPipe / ConnectionReset / ConnectionAborted）
            # —— 长轮询被客户端超时掐断时很常见，不该在服务端刷栈。
            pass

    def _log(self, level: str, message: str) -> None:
        logger = self._hook.logger
        if logger is None:
            return
        fn = getattr(logger, level, None)
        if fn is None:
            return
        try:
            fn(message)
        except Exception:  # pragma: no cover - 日志失败不能影响服务
            pass

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """接管默认的 stderr 访问日志。"""
        self._log("debug", "bettergi http: " + (fmt % args))

    # ── 方法分发 ──────────────────────────────────────────────────────
    def do_POST(self) -> None:  # noqa: N802
        hook = self._hook
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        want = hook.config.path.rstrip("/") or "/"

        # 反向控制通道：BetterGI 回执
        if hook.control is not None and path == hook.ack_path:
            self._handle_control_ack()
            return

        if path != want:
            self._send_json(404, {"ok": False, "reason": "not_found", "path": self.path})
            self._log("debug", f"bettergi: 路径不匹配 {self.path}")
            return

        # 读请求体（带体积上限）
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(400, {"ok": False, "reason": "empty_body"})
            return
        if length > hook.config.max_body_bytes:
            self._send_json(413, {"ok": False, "reason": "body_too_large"})
            return

        try:
            raw = self.rfile.read(length)
        except (TimeoutError, OSError) as exc:
            self._send_json(408, {"ok": False, "reason": "read_timeout"})
            self._log("warning", f"bettergi: 读取请求体失败 {exc!r}")
            return

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "reason": "invalid_json", "detail": str(exc)})
            self._log("warning", f"bettergi: JSON 解析失败 {exc!r}")
            return

        if not isinstance(payload, dict):
            self._send_json(400, {"ok": False, "reason": "invalid_payload"})
            return

        # BetterGI 的「附带截图」会产生几 MB 的 base64 JPEG，本插件用不上，
        # 解析完立刻丢掉，避免大字符串常驻内存。
        if "screenshot" in payload:
            payload.pop("screenshot", None)

        # 防重放：只有 payload 里带 timestamp 才校验；replay_window_s = 0 表示关闭。
        stamp = _parse_timestamp(payload.get("timestamp"))
        if stamp is not None and hook.config.replay_window_s > 0:
            drift = abs(hook.clock() - stamp)
            if drift > hook.config.replay_window_s:
                self._send_json(409, {"ok": False, "reason": "stale_timestamp", "drift_s": round(drift, 3)})
                self._log("warning", f"bettergi: 时间戳超出窗口 drift={drift:.1f}s")
                return

        event = payload.get("event")
        if not isinstance(event, str) or not event.strip():
            self._send_json(400, {"ok": False, "reason": "missing_event"})
            return
        event = event.strip()

        message = payload.get("message")
        message = "" if message is None else str(message)
        data = payload.get("data")
        result = payload.get("result")

        try:
            handled = hook.on_event(event, message, data, result)
        except Exception as exc:  # noqa: BLE001 - 回调异常不能拖垮服务器
            self._log("error", f"bettergi: 事件处理异常 {exc!r}")
            self._send_json(500, {"ok": False, "reason": "handler_error", "detail": str(exc)})
            return

        self._send_json(200, {"ok": True, "handled": bool(handled), "event": event})
        self._log("info", f"bettergi: 收到事件 {event} handled={bool(handled)}")

    def do_GET(self) -> None:  # noqa: N802
        hook = self._hook
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        # 反向控制通道：BetterGI 的 JS 脚本长轮询取指令
        if hook.control is not None and path == hook.command_path:
            self._serve_next_command()
            return

        self.send_response(405)
        self.send_header("Allow", "POST")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve_next_command(self) -> None:
        """长轮询取下一条待执行指令。

        没有指令时最多阻塞 ``control_hold_s`` 秒，超时返回 **204**
        （BetterGI 侧据此继续下一轮轮询）。
        """
        hook = self._hook
        # ★ 记一笔「BetterGI 来取过指令」—— 启动自检靠它判断通道通没通
        hook.note_control_poll()
        try:
            cmd = hook.control.take_next(hold_s=hook.config.control_hold_s)
        except Exception as exc:  # noqa: BLE001 - 队列异常不能拖垮服务器
            self._log("error", f"bettergi: 取指令异常 {exc!r}")
            self._send_json(500, {"ok": False, "reason": "control_error"})
            return

        if cmd is None:
            self._send_json(204, {})
            return

        self._send_json(
            200,
            {
                "ok": True,
                "command_id": cmd.command_id,
                "action": cmd.action,
                "args": cmd.args,
                "created_at": cmd.created_at,
            },
        )
        self._log("info", f"bettergi: 指令已下发 {cmd.action} ({cmd.command_id[:8]})")

    def _handle_control_ack(self) -> None:
        """处理 BetterGI 回执（``POST <path>/ack``）。"""
        hook = self._hook
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(400, {"ok": False, "reason": "empty_body"})
            return
        if length > hook.config.max_body_bytes:
            self._send_json(413, {"ok": False, "reason": "body_too_large"})
            return

        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "reason": "invalid_json", "detail": str(exc)})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"ok": False, "reason": "invalid_payload"})
            return

        command_id = payload.get("command_id")
        if not isinstance(command_id, str) or not command_id.strip():
            self._send_json(400, {"ok": False, "reason": "missing_command_id"})
            return

        ok = bool(payload.get("ok", False))
        message = payload.get("message")
        message = "" if message is None else str(message)
        detail = payload.get("detail")
        detail = detail if isinstance(detail, dict) else {}

        try:
            hit = hook.control.ack(command_id.strip(), ok, message, detail)
        except Exception as exc:  # noqa: BLE001
            self._log("error", f"bettergi: 回执处理异常 {exc!r}")
            self._send_json(500, {"ok": False, "reason": "control_error"})
            return

        self._send_json(200, {"ok": True, "matched": hit})

    def do_PUT(self) -> None:  # noqa: N802
        self.do_GET()

    def do_DELETE(self) -> None:  # noqa: N802
        self.do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()


class WebhookServer:
    """带后台线程的 Webhook HTTP 服务器。

    用法::

        server = WebhookServer("127.0.0.1", 17890, "/bettergi", on_event)
        port = server.start()
        ...
        server.stop()
    """

    def __init__(
        self,
        host: str,
        port: int,
        path: str,
        on_event: EventCallback,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        replay_window_s: float = DEFAULT_REPLAY_WINDOW_S,
        control_hold_s: float = DEFAULT_CONTROL_HOLD_S,
        logger: Any = None,
        clock: Callable[[], float] = time.time,
        control: Any = None,
    ) -> None:
        """
        :param host: 监听地址，建议只用 ``127.0.0.1``。
        :param port: 监听端口，``0`` 表示由系统分配（测试用）。
        :param path: 只接受这个路径的 POST。
        :param on_event: 事件回调 ``(event, message, data, result) -> bool``。
        :param timeout_s: 读请求体超时（秒）。
        :param max_body_bytes: 请求体上限。
        :param replay_window_s: 时间戳允许的偏差（秒）。
        :param control_hold_s: 取指令时的长轮询保持时长（秒）。
        :param logger: 可选 logger（需有 debug/info/warning/error 方法）。
        :param clock: 时钟函数，测试时可注入假时钟。
        :param control: 可选指令队列（``bettergi_control.CommandQueue``）。
            给了才会开反向控制端点；不给则只有正向事件通道。
        """
        self.config = WebhookConfig(
            host=host,
            port=port,
            path=path,
            timeout_s=timeout_s,
            max_body_bytes=max_body_bytes,
            replay_window_s=replay_window_s,
            control_hold_s=control_hold_s,
        )
        self.on_event = on_event
        self.logger = logger
        self.clock = clock
        self.control = control
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # ── 反向通道连通性探针 ──────────────────────────────────────
        # BetterGI 的 JS 代理脚本一上线就会不停 `GET <path>/command`。
        # 「有没有人来取过指令」就是反向通道**有没有接上**的最好证据 ——
        # 比「BetterGI 进程在不在」准得多（装了不等于配好了）。
        self._control_poll_count = 0
        self._last_control_poll_at = 0.0

    # ── 反向控制端点 ──────────────────────────────────────────────────
    @property
    def command_path(self) -> str:
        """取指令的路径（``<path>/command``）。"""
        base = self.config.path.rstrip("/")
        return f"{base}/command"

    @property
    def ack_path(self) -> str:
        """回执的路径（``<path>/ack``）。"""
        base = self.config.path.rstrip("/")
        return f"{base}/ack"

    # ── 反向通道连通性 ────────────────────────────────────────────────
    def note_control_poll(self) -> int:
        """记一次「BetterGI 的 JS 代理来取指令」。

        :returns: 累计取指令次数。
        """
        with self._lock:
            self._control_poll_count += 1
            self._last_control_poll_at = self.clock()
            return self._control_poll_count

    @property
    def control_poll_count(self) -> int:
        """BetterGI 取过几次指令（0 = 反向通道还没接上）。"""
        with self._lock:
            return self._control_poll_count

    @property
    def last_control_poll_at(self) -> float:
        """最后一次取指令的时间（epoch 秒）；没取过是 ``0.0``。"""
        with self._lock:
            return self._last_control_poll_at

    # ── 生命周期 ──────────────────────────────────────────────────────
    def start(self) -> int:
        """绑定端口并启动后台线程。

        :returns: 实际绑定的端口（请求 ``port=0`` 时是系统分配的）。
        :raises OSError: 端口被占用等绑定失败。
        """
        with self._lock:
            if self._httpd is not None:
                return self._httpd.server_address[1]

            httpd = ThreadingHTTPServer((self.config.host, self.config.port), _WebhookRequestHandler)
            httpd.daemon_threads = True
            httpd.hook_server = self  # type: ignore[attr-defined]

            handler = _WebhookRequestHandler
            handler.timeout = self.config.timeout_s
            httpd.timeout = self.config.timeout_s

            self._httpd = httpd
            port = httpd.server_address[1]

            def run() -> None:
                try:
                    httpd.serve_forever(poll_interval=0.2)
                except Exception as exc:  # pragma: no cover - 服务线程异常
                    self._log("error", f"bettergi: HTTP 服务异常 {exc!r}")

            self._thread = threading.Thread(target=run, daemon=True, name="bettergi-companion-webhook")
            self._thread.start()
            self._log("info", f"bettergi: Webhook 监听 {self.config.host}:{port}{self.config.path}")
            return port

    def stop(self) -> None:
        """停止服务器并回收线程（可重复调用）。"""
        with self._lock:
            httpd, thread = self._httpd, self._thread
            self._httpd = None
            self._thread = None
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:  # pragma: no cover
                pass
            try:
                httpd.server_close()
            except Exception:  # pragma: no cover
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._log("info", "bettergi: Webhook 已停止")

    # ── 状态 ──────────────────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        """服务器是否在跑。"""
        with self._lock:
            return self._httpd is not None

    @property
    def port(self) -> int:
        """当前实际绑定的端口。"""
        with self._lock:
            if self._httpd is None:
                return self.config.port
            return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        """完整 Webhook URL。"""
        return f"http://{self.config.host}:{self.port}{self.config.path}"

    def _log(self, level: str, message: str) -> None:
        if self.logger is None:
            return
        fn = getattr(self.logger, level, None)
        if fn is None:
            return
        try:
            fn(message)
        except Exception:  # pragma: no cover
            pass
