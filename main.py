from __future__ import annotations

import asyncio
import inspect
import re
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:
    from astrbot.api.message_components import At, Plain
except ImportError:
    try:
        from astrbot.core.message.components import At, Plain
    except ImportError:
        At, Plain = None, None


@register(
    "astrbot_plugin_qq_like",
    "Codex",
    "LLM点赞：为用户送上每日上限名片赞，支持当前人设与语音插件(genie)个性化反应",
    "1.1.0",
)
class QQLikePlugin(Star):
    """QQ profile like plugin with LLM persona reaction and genie voice adaptation."""

    def __init__(self, context: Context, config: dict[str, Any] | Any = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}
        self._user_last_triggered: dict[str, float] = {}
        self._in_flight_users: set[str] = set()

    def _cfg(self, key: str, default: Any) -> Any:
        """Read configuration value safely."""
        if hasattr(self.config, "get"):
            val = self.config.get(key)
            if val is not None:
                return val
        return default

    # ─── Target Resolution ──────────────────────────────────────────────
    def _resolve_target(self, event: AstrMessageEvent) -> tuple[int, str, bool]:
        """Resolve target QQ number, nickname, and whether target is the sender.

        Returns:
            (target_id, target_name, is_self)
        """
        sender_id_str = str(event.get_sender_id() or "").strip()
        sender_name = str(event.get_sender_name() or "").strip() or "你"

        # 1. Check for At components in message_obj
        msg_obj = getattr(event, "message_obj", None)
        components = getattr(msg_obj, "message", None) or []
        self_id_str = str(getattr(msg_obj, "self_id", "") or "").strip()

        for comp in components:
            if hasattr(comp, "qq"):
                target_qq = str(getattr(comp, "qq", "")).strip()
                if target_qq and target_qq.isdigit() and target_qq != "all" and target_qq != self_id_str:
                    target_id = int(target_qq)
                    is_self = (target_qq == sender_id_str)
                    return target_id, ("你" if is_self else f"QQ({target_id})"), is_self

        # 2. Check for numeric argument in message text (e.g. /赞我 123456789)
        text = str(event.get_message_str() or "").strip()
        match = re.search(r"\b(\d{5,11})\b", text)
        if match:
            target_qq = match.group(1)
            if target_qq != self_id_str:
                target_id = int(target_qq)
                is_self = (target_qq == sender_id_str)
                return target_id, ("你" if is_self else f"QQ({target_id})"), is_self

        # 3. Default to sender
        sender_id = int(sender_id_str) if sender_id_str.isdigit() else 0
        return sender_id, sender_name, True

    # ─── OneBot Action Call ─────────────────────────────────────────────
    async def _call_action(
        self,
        event: AstrMessageEvent,
        action: str,
        **params: Any,
    ) -> Any:
        """Call OneBot action with event bot or fallback to registered platform."""
        bot = getattr(event, "bot", None)

        call_action_fn = getattr(bot, "call_action", None) if bot else None
        if not callable(call_action_fn):
            api = getattr(bot, "api", None) if bot else None
            call_action_fn = getattr(api, "call_action", None) if api else None

        if not callable(call_action_fn):
            platform_manager = getattr(self.context, "platform_manager", None)
            if platform_manager is not None:
                try:
                    for inst in platform_manager.get_insts():
                        meta = getattr(inst, "meta", None)
                        if meta and getattr(meta(), "name", "") == "aiocqhttp":
                            client = getattr(inst, "get_client", None) or getattr(inst, "bot", None)
                            b = client() if callable(client) else client
                            if b:
                                call_action_fn = getattr(b, "call_action", None) or getattr(getattr(b, "api", None), "call_action", None)
                                if callable(call_action_fn):
                                    break
                except Exception as exc:
                    logger.debug(f"[qq_like] Failed to resolve bot from platform_manager: {exc}")

        if not callable(call_action_fn):
            raise RuntimeError("未检测到可用的 aiocqhttp (OneBot v11) 平台客户端，点赞功能仅支持 QQ 协议端。")

        self_id = str(getattr(getattr(event, "message_obj", None), "self_id", "") or "").strip()
        if self_id and self_id.isdigit():
            params.setdefault("self_id", int(self_id))

        return await call_action_fn(action=action, **params)

    # ─── Like Execution ────────────────────────────────────────────────
    async def _execute_send_like(
        self,
        event: AstrMessageEvent,
        target_id: int,
    ) -> tuple[int, int, str, str]:
        """Send maximum likes in batches.

        Returns:
            (actual_likes, max_likes_attempted, status, error_detail)
            status in: "full_success", "partial_success", "already_maxed", "failed"
        """
        max_likes = max(1, int(self._cfg("max_likes", 20)))
        chunk_size = max(1, min(int(self._cfg("chunk_size", 10)), 10))
        actual_likes = 0
        limit_keywords = ("上限", "限制", "最多", "100", "20003", "已达", "己达", "每天", "达到", "超限", "超过")

        while actual_likes < max_likes:
            batch_times = min(max_likes - actual_likes, chunk_size)
            try:
                res = await self._call_action(
                    event,
                    "send_like",
                    user_id=target_id,
                    times=batch_times,
                )
                if isinstance(res, dict) and res.get("status") == "failed":
                    msg = str(res.get("wording") or res.get("message") or "")
                    if any(k in msg for k in limit_keywords):
                        if actual_likes > 0:
                            return actual_likes, max_likes, "partial_success", "已达今日上限"
                        return 0, max_likes, "already_maxed", "今日点赞已达上限"
                    return actual_likes, max_likes, ("partial_success" if actual_likes > 0 else "failed"), msg

                actual_likes += batch_times
                if actual_likes < max_likes:
                    await asyncio.sleep(0.35)
            except Exception as exc:
                err_text = str(exc)
                logger.warning(f"[qq_like] send_like exception: {err_text}")
                is_limit_error = any(k in err_text for k in limit_keywords)
                if is_limit_error:
                    if actual_likes > 0:
                        return actual_likes, max_likes, "partial_success", "已达今日上限"
                    return 0, max_likes, "already_maxed", "今日点赞已达上限"
                if actual_likes > 0:
                    return actual_likes, max_likes, "partial_success", err_text
                return 0, max_likes, "failed", err_text

        return actual_likes, max_likes, "full_success", ""

    # ─── Persona Prompt Helpers ────────────────────────────────────────
    @staticmethod
    async def _maybe_await(val: Any) -> Any:
        if inspect.isawaitable(val):
            return await val
        return val

    @staticmethod
    def _extract_persona_prompt(obj: Any) -> str:
        if obj is None:
            return ""
        try:
            if hasattr(obj, "get"):
                for key in ("prompt", "system_prompt"):
                    try:
                        v = obj.get(key)
                    except Exception:
                        v = None
                    if v:
                        return str(v)
        except Exception:
            pass
        for attr in ("system_prompt", "prompt"):
            try:
                v = getattr(obj, attr, None)
            except Exception:
                v = None
            if v:
                return str(v)
        return ""

    @staticmethod
    def _persona_name_of(obj: Any) -> str:
        if obj is None:
            return ""
        try:
            if hasattr(obj, "get"):
                for key in ("name", "id", "persona_id"):
                    v = obj.get(key)
                    if v:
                        return str(v)
        except Exception:
            pass
        for attr in ("name", "id", "persona_id"):
            v = getattr(obj, attr, None)
            if v:
                return str(v)
        return ""

    async def _resolve_persona_prompt(self, pm: Any, persona_id: str | None, umo: str) -> str:
        """Resolve persona prompt across different AstrBot versions."""
        if persona_id and persona_id not in ("default", "[%None]"):
            for method_name in ("get_persona_v3_by_id", "get_persona_by_id", "get_persona"):
                fn = getattr(pm, method_name, None)
                if fn is None:
                    continue
                try:
                    persona = await self._maybe_await(fn(persona_id))
                    prompt = self._extract_persona_prompt(persona)
                    if prompt:
                        return prompt
                except Exception:
                    continue
            for attr_name in ("personas_v3", "personas"):
                for persona in getattr(pm, attr_name, None) or []:
                    if self._persona_name_of(persona) == persona_id:
                        prompt = self._extract_persona_prompt(persona)
                        if prompt:
                            return prompt
            get_all = getattr(pm, "get_all_personas", None)
            if get_all is not None:
                try:
                    for persona in await self._maybe_await(get_all()):
                        if self._persona_name_of(persona) == persona_id:
                            prompt = self._extract_persona_prompt(persona)
                            if prompt:
                                return prompt
                except Exception:
                    pass

        # Fallback to default persona
        for method_name in ("get_default_persona_v3", "get_default_persona", "get_selected_default_persona"):
            fn = getattr(pm, method_name, None)
            if fn is None:
                continue
            try:
                persona = await self._maybe_await(fn(umo))
                prompt = self._extract_persona_prompt(persona)
                if prompt:
                    return prompt
            except Exception:
                try:
                    persona = await self._maybe_await(fn())
                    prompt = self._extract_persona_prompt(persona)
                    if prompt:
                        return prompt
                except Exception:
                    continue
        return ""

    async def _get_persona_prompt(self, event: AstrMessageEvent) -> str:
        """Retrieve current conversation active persona system prompt."""
        umo = getattr(event, "unified_msg_origin", None)
        persona_id = None
        try:
            cm = getattr(self.context, "conversation_manager", None)
            if cm is not None and umo:
                cid = await cm.get_curr_conversation_id(umo)
                if cid:
                    conv = await cm.get_conversation(umo, cid)
                    persona_id = getattr(conv, "persona_id", None)
        except Exception as exc:
            logger.debug(f"[qq_like] Failed to read conversation persona: {exc}")

        if not persona_id or persona_id == "[%None]":
            persona_id = None
            try:
                cfg = self.context.persona_manager.acm.get_conf(umo)
                persona_id = cfg.get("provider_settings", {}).get(
                    "default_personality", "default"
                )
            except Exception:
                persona_id = "default"

        pm = getattr(self.context, "persona_manager", None)
        if pm is not None:
            try:
                return await self._resolve_persona_prompt(pm, persona_id, umo)
            except Exception as exc:
                logger.debug(f"[qq_like] Failed to resolve persona prompt: {exc}")

        return ""

    # ─── GENIE Voice Plugin Integration ────────────────────────────────
    def _find_genie(self) -> Any | None:
        """Find loaded and active genie voice plugin instance."""
        if not self._cfg("enable_voice_plugin", True):
            return None
        try:
            get_all = getattr(self.context, "get_all_stars", None)
            if get_all is None:
                return None
            for md in get_all():
                if not getattr(md, "activated", True):
                    continue
                if not getattr(md, "star_cls", None):
                    continue
                name = (getattr(md, "name", "") or "").lower()
                root = (getattr(md, "root_dir_name", "") or "").lower()
                if name in ("genie", "astrbot_plugin_genie") or root in (
                    "genie",
                    "astrbot_plugin_genie",
                ):
                    return md.star_cls
        except Exception as exc:
            logger.warning(f"[qq_like] 查找 genie 语音插件失败: {exc}")
        return None

    @staticmethod
    def _clean_display_text(text: str) -> str:
        """Clean residual tags if voice plugin is not active."""
        zh_match = re.search(r"<zh>(.*?)</zh>", text, re.DOTALL | re.IGNORECASE)
        if zh_match:
            text = zh_match.group(1).strip()
        else:
            text = re.sub(r"</?(?:zh|ja|thought|think)>", "", text, flags=re.IGNORECASE).strip()
        return text

    # ─── LLM Reaction ──────────────────────────────────────────────────
    async def _generate_reaction(
        self,
        event: AstrMessageEvent,
        target_name: str,
        is_self: bool,
        actual_likes: int,
        max_likes: int,
        status: str,
        error_detail: str,
    ) -> str:
        """Call LLM to produce persona-based reaction and integrate genie voice if available."""
        target_desc = "用户本人" if is_self else f"用户指定的群友「{target_name}」"

        # Build fallback plain text
        if status == "full_success":
            fallback = f"好啦，已为{target_name}送上 {actual_likes} 个赞啦！记得也回赞我哦~"
        elif status == "partial_success":
            fallback = f"已为{target_name}送上 {actual_likes} 个赞，再点就达到今日上限啦！"
        elif status == "already_maxed":
            fallback = f"今天已经为{target_name}点过赞（或名片赞已达今日上限）啦，明天再来找我吧！"
        else:
            fallback = f"点赞出了点小状况：{error_detail or '未知错误'}"

        if not self._cfg("enable_llm_reaction", True):
            return fallback

        if status == "full_success":
            status_desc = f"你刚刚成功为{target_desc}送出了 {actual_likes} 个名片赞（已达到本次点赞上限）。"
        elif status == "partial_success":
            status_desc = f"你为{target_desc}送出了 {actual_likes} 个赞，随后因达到今日点赞上限停止。"
        elif status == "already_maxed":
            status_desc = f"你尝试为{target_desc}点赞，但系统提示对方今天已经达到点赞上限或今日已被点满（送出 0 个赞）。"
        else:
            status_desc = f"你尝试为{target_desc}点赞时遭遇失败，原因：{error_detail}。"

        guide = str(self._cfg("custom_prompt_guide", "") or "").strip()
        guide_note = f"\n额外指引要求：{guide}" if guide else ""

        sender_name = str(event.get_sender_name() or "").strip() or "用户"
        prompt = (
            f"【系统事件】：用户「{sender_name}」发起了点赞互动。\n"
            f"事件结果：{status_desc}\n"
            f"请根据你当前的人设、性格与说话风格，直接对用户做出自然的口吻反应（可傲娇、邀功、卖萌、求回赞、吐槽、暖心鼓励等）。"
            f"{guide_note}\n"
            f"重要要求：直接输出要说的话，不要带任何多余前缀、旁白、括号注释或解释说明。"
        )

        try:
            umo = getattr(event, "unified_msg_origin", None)
            provider = self.context.get_using_provider(umo)
            if provider is not None:
                persona_prompt = await self._get_persona_prompt(event)
                genie = self._find_genie()

                # If genie is loaded, inject bilingual prompt instruction
                if genie is not None:
                    try:
                        hint = genie.build_prompt_injection_hint() or ""
                        if hint:
                            persona_prompt = f"{persona_prompt or ''}{hint}"
                    except Exception as exc:
                        logger.warning(f"[qq_like] 读取 genie 双语注入提示失败: {exc}")

                resp = await provider.text_chat(
                    prompt=prompt,
                    session_id=None,
                    contexts=[],
                    image_urls=[],
                    system_prompt=persona_prompt or None,
                )
                text = (resp.completion_text or "").strip()
                if text:
                    # Clean surrounding quotes if LLM added them
                    if text.startswith(('"', "“")) and text.endswith(('"', "”")):
                        text = text[1:-1].strip()

                    # Route through genie voice plugin for TTS enqueue and display format
                    if genie is not None:
                        try:
                            voice_plugin_reply = getattr(genie, "voice_plugin_reply", None)
                            if callable(voice_plugin_reply):
                                display = await voice_plugin_reply(event, text)
                                if display:
                                    return display
                        except Exception as exc:
                            logger.warning(f"[qq_like] genie 语音接入失败，按原文展示: {exc}")

                    return self._clean_display_text(text)
        except Exception as exc:
            logger.warning(f"[qq_like] LLM 反应生成失败，将使用兜底回复: {exc}")

        return fallback

    # ─── Event Deduplication ───────────────────────────────────────────
    @staticmethod
    def _claim_event(event: AstrMessageEvent) -> bool:
        """Prevent duplicate execution across command and plain message handlers."""
        if getattr(event, "_qq_like_claimed", False):
            return True
        event._qq_like_claimed = True
        return False

    # ─── Unified Execution Pipeline ────────────────────────────────────
    async def _handle_like_flow(self, event: AstrMessageEvent):
        """Unified like execution with cooldown and message yielding."""
        sender_id_str = str(event.get_sender_id() or "").strip()
        if not sender_id_str:
            yield event.plain_result("无法识别当前发送者 QQ。")
            return

        # Cooldown & concurrent prevention
        now = time.time()
        cd = max(1, int(self._cfg("cooldown_seconds", 10)))
        last_time = self._user_last_triggered.get(sender_id_str, 0.0)

        if sender_id_str in self._in_flight_users:
            yield event.plain_result("正在为你处理点赞请求，请稍候~")
            return

        if now - last_time < cd:
            remain = int(cd - (now - last_time))
            yield event.plain_result(f"点赞过于频繁啦，请稍候 {remain} 秒后再试哦。")
            return

        self._in_flight_users.add(sender_id_str)
        self._user_last_triggered[sender_id_str] = now

        try:
            target_id, target_name, is_self = self._resolve_target(event)
            if not target_id:
                yield event.plain_result("未找到有效的点赞目标 QQ 号。")
                return

            actual_likes, max_likes, status, err_detail = await self._execute_send_like(
                event,
                target_id,
            )

            reply_text = await self._generate_reaction(
                event,
                target_name=target_name,
                is_self=is_self,
                actual_likes=actual_likes,
                max_likes=max_likes,
                status=status,
                error_detail=err_detail,
            )

            # Check if group and whether to At sender
            group_id = event.get_group_id()
            at_sender = bool(self._cfg("at_sender", False))

            if group_id and at_sender and At is not None and Plain is not None:
                chain = [At(qq=int(sender_id_str)), Plain(f" {reply_text}")]
                yield event.chain_result(chain)
            else:
                yield event.plain_result(reply_text)
        except Exception as exc:
            logger.exception(f"[qq_like] 处理点赞流程发生异常: {exc}")
            yield event.plain_result(f"点赞时出现异常：{exc}")
        finally:
            self._in_flight_users.discard(sender_id_str)

    # ─── Commands (with wake prefix / @) ───────────────────────────────
    @filter.command("赞我", alias={"like", "点赞", "赞", "点名片赞", "名片赞", "给我点赞"})
    async def like_cmd(self, event: AstrMessageEvent):
        """【指令触发】给用户点满名片赞并调用当前人设做出反应。"""
        if self._claim_event(event):
            return
        async for res in self._handle_like_flow(event):
            yield res
        event.stop_event()

    # ─── Plain Message Trigger (no prefix required) ────────────────────
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_plain_message(self, event: AstrMessageEvent):
        """【普通消息触发】监听无需前缀的点赞请求，并阻断后续未处理拦截。"""
        if not self._cfg("enable_plain_trigger", True):
            return
        if self._claim_event(event):
            return

        text = str(event.get_message_str() or "").strip()
        match = re.match(
            r"^[/~#.]?\s*(赞我|求赞|点赞|点个赞|帮我点赞|赞我一下|给我点赞|名片赞|like)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            async for res in self._handle_like_flow(event):
                yield res
            event.stop_event()
