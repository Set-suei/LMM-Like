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
    from astrbot.api.message_components import At, Plain, Reply
except ImportError:
    try:
        from astrbot.core.message.components import At, Plain, Reply
    except ImportError:
        At, Plain, Reply = None, None, None


@register(
    "astrbot_plugin_LLM_like",
    "Codex",
    "LLM赞我：为用户送上每日上限名片赞，支持LLM考虑给不给赞、全方式要赞识别、函数调用(Tool Calling)、当前人设与语音插件(genie)个性化反应",
    "1.3.0",
)
class QQLikePlugin(Star):
    """QQ profile like plugin with LLM consideration, omni-channel like request recognition, Tool Calling, persona reaction and genie voice adaptation."""

    RE_INVISIBLE_CHARS = re.compile(
        r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u1160\u3164]+"
    )

    RE_LEADING_MENTION = re.compile(
        r"^(?:\s*@[^\s@()（）]{1,32}(?:\(\d{5,12}\)|（\d{5,12}）)?[\s,，。！!？?~～:：]*)+"
    )

    RE_LIKE_COMMAND = re.compile(
        r"^[/~#.]?\s*("
        r"赞我|求赞|点赞|点个赞|点点赞|点一下赞|点下赞|帮我点赞|帮我点个赞|赞我一下|赞下我|"
        r"给我点赞|给我点个赞|给我点点赞|给俺点赞|给咱点赞|给个赞|给我赞|要赞|来个赞|赏个赞|"
        r"刷赞|名片赞|点名片赞|主页赞|主页点赞|资料卡赞|资料赞|互赞|回赞|求名片赞|求主页赞|"
        r"like|like\s+me"
        r")(?:[\s\d@()（）！!？?~～。.…呀啊吧哦呢嘛呗啦捏呜♡❤❤️⭐✨]*)$",
        re.IGNORECASE,
    )

    RE_LIKE_NATURAL = re.compile(
        r"(?:给我|给俺|给咱|帮我|为我|替我|去我|来我|到我|上我).*?(?:主页|名片|资料卡|资料|空间|卡片)?.*?(?:点个赞|点点赞|点一下赞|点下赞|点赞|点名片赞|刷赞|刷个赞|给个赞|送个赞|来个赞|赏个赞|回个赞|回赞|互赞|按个赞|点个小手|踩踩)|"
        r"(?:主页|名片|资料卡|资料).*?(?:点个赞|点点赞|点一下赞|点下赞|点赞|刷赞|给个赞|送个赞|来个赞|赞一下|赞下)|"
        r"(?:会|能|可以|想|要|愿不愿意|愿意|敢不敢|能不能|可不可以|行不行|好不好|肯不肯|舍得|要不要|求|麻烦|拜托|快|赶紧|顺便).*?(?:点个赞|点点赞|点一下赞|点下赞|点赞|赞我|名片赞|主页赞|资料卡赞|资料赞|给个赞|给我赞|送个赞|赏个赞|来个赞|回赞|互赞)|"
        r"(?:为什么|为啥|怎么|咋|难道|居然|竟然).*?不.*?(?:点赞|点个赞|赞我|给我赞|给俺赞|名片赞|主页赞)|"
        r"(?:(?:求|讨|要|给|送|赏|来|点|刷|回|互|骗|混)(?:我|俺|咱)?(?:个|一下|下|点|些)?(?:名片|主页|资料卡|资料)?赞)|(?:赞我|赞下我|赞一下我|给我点赞|给俺点赞|给咱点赞|帮我点赞|想要赞|想要你的赞|要你的赞|点我赞|还没有赞我|还没赞我|不赞我)",
        re.IGNORECASE,
    )

    RE_META_EXCLUDE = re.compile(
        r"^(?:点赞失败\s*[:：]|.*?(?:发赞我|要发赞我|发送赞我|说赞我|打赞我|发个赞我|判断赞我|优化.*?赞我|改.*?赞我|测试.*?赞我))",
        re.IGNORECASE,
    )

    RE_SELF_TARGET = re.compile(
        r"(?:赞我|给我|给俺|给咱|帮我|为我|替我|我主页|我的主页|我名片|我的名片|我资料|我的资料|求赞|要赞|回赞|互赞|like\s*me)",
        re.IGNORECASE,
    )

    RE_EXPLICIT_QQ_ARG = re.compile(
        r"^[/~#.]?\s*(?:赞我|赞他|赞她|赞|点赞|点名片赞|名片赞|给我点赞|帮他点赞|帮她点赞|给ta点赞|like)\s+(\d{5,11})\b",
        re.IGNORECASE,
    )

    RE_ACTION_TAG = re.compile(r"\[ACTION\s*:\s*(LIKE|REJECT)\]", re.IGNORECASE)

    def __init__(self, context: Context, config: dict[str, Any] | Any = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}
        self._user_last_triggered: dict[str, float] = {}
        self._in_flight_users: set[str] = set()
        self._pending_like_challenges: dict[tuple[str, str], dict[str, Any]] = {}

    def _cfg(self, key: str, default: Any) -> Any:
        """Read configuration value safely."""
        if hasattr(self.config, "get"):
            val = self.config.get(key)
            if val is not None:
                return val
        return default

    def _should_llm_decide(self) -> bool:
        """Return whether LLM should consider whether to give a like."""
        return bool(self._cfg("enable_llm_decision", True)) and bool(
            self._cfg("enable_autonomous_decision", True)
        )

    @classmethod
    def _normalize_message_text(cls, text: str) -> str:
        """Clean invisible characters and strip leading @mentions for reliable intent matching."""
        if not text:
            return ""
        cleaned = cls.RE_INVISIBLE_CHARS.sub("", text).strip()
        stripped = cls.RE_LEADING_MENTION.sub("", cleaned).strip()
        return stripped or cleaned

    def _is_directed_at_other_human(self, event: AstrMessageEvent, text: str) -> bool:
        """Check if a group message explicitly @mentions another human (not the bot) without @mentioning the bot."""
        msg_obj = getattr(event, "message_obj", None)
        components = getattr(msg_obj, "message", None) or []
        self_id_str = str(getattr(msg_obj, "self_id", "") or "").strip()
        if not self_id_str or not components:
            return False

        at_qqs: list[str] = []
        has_reply = False
        for comp in components:
            comp_cls_name = comp.__class__.__name__.lower()
            comp_type = str(getattr(comp, "type", "") or "").lower()
            if (Reply is not None and isinstance(comp, Reply)) or comp_cls_name == "reply" or comp_type == "reply":
                has_reply = True
                continue
            if (At is not None and isinstance(comp, At)) or comp_cls_name == "at" or comp_type == "at":
                qq_val = str(getattr(comp, "qq", "") or "").strip()
                if qq_val and qq_val != "all":
                    at_qqs.append(qq_val)

        if not at_qqs:
            return False

        if self_id_str in at_qqs:
            return False

        norm_text = self._normalize_message_text(text)
        if has_reply and self.RE_LIKE_COMMAND.search(norm_text) and self.RE_SELF_TARGET.search(norm_text):
            return False

        return True

    def _resolve_target(self, event: AstrMessageEvent) -> tuple[int, str, bool]:
        """Resolve target QQ number, nickname, and whether target is the sender."""
        sender_id_str = str(event.get_sender_id() or "").strip()
        sender_name = str(event.get_sender_name() or "").strip() or "你"
        raw_text = str(event.get_message_str() or "").strip()
        norm_text = self._normalize_message_text(raw_text)

        msg_obj = getattr(event, "message_obj", None)
        components = getattr(msg_obj, "message", None) or []
        self_id_str = str(getattr(msg_obj, "self_id", "") or "").strip()

        explicit_qq_match = self.RE_EXPLICIT_QQ_ARG.search(norm_text)
        if explicit_qq_match:
            target_qq = explicit_qq_match.group(1)
            if target_qq != self_id_str:
                target_id = int(target_qq)
                is_self = target_qq == sender_id_str
                return target_id, (sender_name if is_self else f"QQ({target_id})"), is_self

        is_explicit_self = bool(self.RE_SELF_TARGET.search(norm_text))
        if not is_explicit_self:
            for comp in components:
                comp_cls_name = comp.__class__.__name__.lower()
                comp_type = str(getattr(comp, "type", "") or "").lower()
                if (At is not None and isinstance(comp, At)) or comp_cls_name == "at" or comp_type == "at":
                    target_qq = str(getattr(comp, "qq", "") or "").strip()
                    if target_qq and target_qq.isdigit() and target_qq != "all" and target_qq != self_id_str:
                        target_id = int(target_qq)
                        is_self = target_qq == sender_id_str
                        target_display = (
                            sender_name
                            if is_self
                            else str(getattr(comp, "name", "") or "").strip() or f"QQ({target_id})"
                        )
                        return target_id, target_display, is_self

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
                                call_action_fn = getattr(b, "call_action", None) or getattr(
                                    getattr(b, "api", None), "call_action", None
                                )
                                if callable(call_action_fn):
                                    break
                except Exception as exc:
                    logger.debug(f"[LLM_like] Failed to resolve bot from platform_manager: {exc}")

        if not callable(call_action_fn):
            raise RuntimeError("未找到可用的 OneBot (aiocqhttp) 客户端连接，无法调用点赞接口。")

        res = call_action_fn(action, **params)
        if inspect.isawaitable(res):
            res = await res
        return res

    # ─── Like Execution ────────────────────────────────────────────────
    async def _execute_send_like(
        self,
        event: AstrMessageEvent,
        target_id: int,
    ) -> tuple[int, int, str, str]:
        """Execute send_like in batches until max_likes or limit reached.

        Returns:
            (actual_likes, max_likes, status, error_detail)
            status: 'full_success' | 'partial_success' | 'already_maxed' | 'failed'
        """
        max_likes = max(1, int(self._cfg("max_likes", 50)))
        chunk_size = max(1, int(self._cfg("chunk_size", 10)))

        actual_likes = 0
        limit_keywords = ("上限", "超过", "频繁", "已达", "每天最多", "满", "limit", "max", "reach")

        while actual_likes < max_likes:
            batch_times = min(chunk_size, max_likes - actual_likes)
            try:
                ret = await self._call_action(
                    event,
                    "send_like",
                    user_id=target_id,
                    times=batch_times,
                )

                ret_code = 0
                msg = ""
                if isinstance(ret, dict):
                    ret_code = ret.get("retcode", 0)
                    msg = str(ret.get("msg") or ret.get("wording") or "")

                if ret_code != 0:
                    logger.info(f"[LLM_like] send_like retcode={ret_code}, msg={msg}")
                    event._qq_like_executed = True
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
                logger.warning(f"[LLM_like] send_like exception: {err_text}")
                is_limit_error = any(k in err_text for k in limit_keywords)
                event._qq_like_executed = True
                if is_limit_error:
                    if actual_likes > 0:
                        return actual_likes, max_likes, "partial_success", "已达今日上限"
                    return 0, max_likes, "already_maxed", "今日点赞已达上限"
                if actual_likes > 0:
                    return actual_likes, max_likes, "partial_success", err_text
                return 0, max_likes, "failed", err_text

        event._qq_like_executed = True
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

        for method_name in ("get_default_persona_v3", "get_default_persona"):
            fn = getattr(pm, method_name, None)
            if fn is None:
                continue
            try:
                for arg in (umo, None):
                    try:
                        persona = await self._maybe_await(fn(arg) if arg else fn())
                        prompt = self._extract_persona_prompt(persona)
                        if prompt:
                            return prompt
                    except TypeError:
                        continue
            except Exception:
                continue

        for attr_name in ("default_persona_v3", "default_persona"):
            val = getattr(pm, attr_name, None)
            prompt = self._extract_persona_prompt(val)
            if prompt:
                return prompt

        return ""

    async def _get_persona_prompt(self, event: AstrMessageEvent) -> str:
        """Get the currently effective persona system prompt for this event/session."""
        umo = getattr(event, "unified_msg_origin", None)
        persona_id = None

        conversation = getattr(event, "conversation", None)
        if conversation is not None:
            persona_id = getattr(conversation, "persona_id", None)

        if not persona_id:
            msg_obj = getattr(event, "message_obj", None)
            if msg_obj is not None:
                session = getattr(msg_obj, "session", None)
                if session is not None:
                    persona_id = getattr(session, "persona_id", None)

        pm = getattr(self.context, "persona_manager", None)
        if pm is not None:
            try:
                return await self._resolve_persona_prompt(pm, persona_id, umo)
            except Exception as exc:
                logger.debug(f"[LLM_like] Failed to resolve persona prompt: {exc}")

        return ""

    # ─── GENIE Voice Plugin Integration ────────────────────────────────
    def _find_genie(self) -> Any | None:
        """Find loaded and active genie voice plugin instance."""
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
            logger.warning(f"[LLM_like] 查找 genie 语音插件失败: {exc}")
        return None

    def _is_genie_active(self, event: AstrMessageEvent) -> bool:
        """检查当前会话与环境下是否启用了语音插件。"""
        genie = self._find_genie()
        if genie is None:
            return False
        try:
            sid = str(getattr(event, "unified_msg_origin", None) or "default")
            if hasattr(genie, "data_manager") and hasattr(genie.data_manager, "is_tts_disabled"):
                if genie.data_manager.is_tts_disabled(sid):
                    return False
            if hasattr(genie, "_should_auto_tts"):
                if not genie._should_auto_tts(event, sid):
                    return False
            if hasattr(genie, "prompt_injection_enabled") and not genie.prompt_injection_enabled:
                return False
            return True
        except Exception:
            return False

    @classmethod
    def _clean_display_text(cls, text: str) -> str:
        """未开启语音插件时，清洗提取纯中文单语言，彻底剥除动作标签、日文标签与假名行。"""
        if not text:
            return ""
        text = cls.RE_ACTION_TAG.sub("", text).strip()
        zh_matches = re.findall(r"<zh>(.*?)</zh>", text, re.DOTALL | re.IGNORECASE)
        if zh_matches:
            return "\n".join(m.strip() for m in zh_matches if m.strip())
        cleaned = re.sub(r"<ja>[\s\S]*?</ja>", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<thought>[\s\S]*?</thought>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?[A-Za-z_][^<>]*>", "", cleaned).strip()
        lines = []
        kana_re = re.compile(r"[぀-ゟ゠-ヿ]")
        cjk_re = re.compile(r"[一-鿿]")
        for line in cleaned.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            if kana_re.search(line_s) and not cjk_re.search(line_s):
                continue
            lines.append(line_s)
        res = "\n".join(lines).strip()
        return res or cleaned

    async def _finalize_reply_text(self, event: AstrMessageEvent, raw_text: str) -> str:
        """Pass reply text through genie TTS if active, otherwise clean to Chinese text."""
        cleaned_raw = self.RE_ACTION_TAG.sub("", raw_text or "").strip()
        if cleaned_raw.startswith(('"', "“")) and cleaned_raw.endswith(('"', "”")):
            cleaned_raw = cleaned_raw[1:-1].strip()

        if self._is_genie_active(event):
            genie = self._find_genie()
            if genie is not None:
                try:
                    voice_plugin_reply = getattr(genie, "voice_plugin_reply", None)
                    if callable(voice_plugin_reply):
                        display = await voice_plugin_reply(event, cleaned_raw)
                        if display:
                            return display
                except Exception as exc:
                    logger.warning(f"[LLM_like] genie 语音处理失败，退回单语言展示: {exc}")

        return self._clean_display_text(cleaned_raw)

    # ─── LLM Reaction (After Like Execution) ───────────────────────────
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
        """调用 LLM 生成角色反应。开启语音插件时走双语+日文TTS，未开启时严格输出单语言中文。"""
        target_desc = "用户本人" if is_self else f"用户指定的群友「{target_name}」"

        if status == "full_success":
            fallback = f"刚才成功为{target_name}点了 {actual_likes} 个赞，记得也给我点点哦~"
        elif status == "partial_success":
            fallback = f"已为{target_name}点了 {actual_likes} 个赞，再点就达到系统限制啦。"
        elif status == "already_maxed":
            fallback = f"今天已经为{target_name}点过赞，名片赞已达上限，明天再来找我点吧~"
        else:
            fallback = f"点赞出了点小状况：{error_detail or '未知错误'}"

        if not self._cfg("enable_llm_reaction", True):
            return fallback

        if status == "full_success":
            status_desc = f"你刚刚成功为{target_desc}送出了 {actual_likes} 个名片赞，已达到本次点赞上限。"
        elif status == "partial_success":
            status_desc = f"你为{target_desc}送出了 {actual_likes} 个赞，因达到今日点赞上限而停止。"
        elif status == "already_maxed":
            status_desc = f"你刚刚答应为{target_desc}点赞，但调用接口时系统提示今日为TA点的赞已经达到上限（已点满，本次送出 0 个赞），只能明天再点。"
        else:
            status_desc = f"你尝试为{target_desc}点赞时操作失败，原因：{error_detail}。"

        guide = str(self._cfg("custom_prompt_guide", "") or "").strip()
        guide_note = f"\n额外指令要求：{guide}" if guide else ""

        sender_name = str(event.get_sender_name() or "").strip() or "用户"
        prompt = (
            f"【系统事件】：用户「{sender_name}」向你请求点赞。\n"
            f"事件结果：{status_desc}\n"
            f"请根据你当前的人设、性格与说话风格，直接对用户做出一句自然的口语反应（可包含撒娇、感谢、调侃、炫耀、吐槽、暖心提醒明天再来等）。\n"
            f"{guide_note}\n"
            f"重要要求：直接输出要说的话，不要带任何动作前缀、旁白、内心注释或多余说明。"
        )

        try:
            umo = getattr(event, "unified_msg_origin", None)
            provider = self.context.get_using_provider(umo)
            if provider is not None:
                persona_prompt = await self._get_persona_prompt(event)
                is_voice_active = self._is_genie_active(event)
                genie = self._find_genie() if is_voice_active else None

                if is_voice_active and genie is not None:
                    try:
                        hint = genie.build_prompt_injection_hint() or ""
                        if hint:
                            persona_prompt = f"{persona_prompt or ''}{hint}"
                        req_prompt = (
                            f"{prompt}\n"
                            "【特别输出规范】：必须同时严格输出 <zh>中文回复</zh> 与 <ja>日本語の返信</ja> 两种语言标签格式，"
                            "<ja> 标签内必须为地道日语台词（严禁任何英文或动作括号，涉及中文人名/昵称必须转为日文假名读法，如若曦->アガサシ）。"
                        )
                    except Exception as exc:
                        logger.warning(f"[LLM_like] 读取 genie 双语注入提示失败: {exc}")
                        req_prompt = prompt
                else:
                    req_prompt = (
                        f"{prompt}\n"
                        "【语言输出要求】：当前未启用语音功能，请直接且仅使用自然流畅的纯中文单语言回复，"
                        "严禁输出任何日文翻译、日文假名或多语言结构，直接输出纯中文内容以节省token。"
                    )

                resp = await provider.text_chat(
                    prompt=req_prompt,
                    session_id=None,
                    contexts=[],
                    image_urls=[],
                    system_prompt=persona_prompt or None,
                )
                text = (resp.completion_text or "").strip()
                if text:
                    return await self._finalize_reply_text(event, text)
        except Exception as exc:
            logger.warning(f"[LLM_like] LLM 响应生成失败，将使用兜底回复: {exc}")

        return await self._finalize_reply_text(event, fallback) if fallback else fallback

    # ─── Autonomous Decision & Generation ──────────────────────────────
    async def _decide_and_generate_reaction(
        self,
        event: AstrMessageEvent,
        target_name: str,
        sender_name: str,
        raw_text: str,
        is_self: bool,
        challenge_ctx: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """让 LLM 结合人设与上下文自主考虑给不给点赞，并生成反应台词（返回 should_like 与原始模型台词）。"""
        target_desc = f"用户本人「{sender_name}」" if is_self else f"用户「{sender_name}」指定的群友「{target_name}」"
        clean_user_text = self._normalize_message_text(raw_text) or raw_text

        history_note = ""
        if challenge_ctx:
            prev_user = challenge_ctx.get("user_text", "")
            prev_bot = challenge_ctx.get("bot_reply", "")
            history_note = (
                f"【前情提要】：\n"
                f"刚才对方曾向你求赞说：『{prev_user}』，当时你暂时拒绝或提出了条件：『{prev_bot}』。\n"
                f"现在对方继续回应了你：『{clean_user_text}』。\n"
            )

        prompt = (
            f"【对话求赞事件】：\n"
            f"{history_note}"
            f"{target_desc}刚刚对你发送了：『{clean_user_text}』，对方正在向你要 QQ 名片赞/主页赞（或回应你刚才提出的点赞条件）。\n"
            f"请完全基于你当前的人设性格、心情、与对方的互动氛围，以及对方说话的态度（例如是礼貌诚恳、撒娇讨好、兑现了条件，还是敷衍命令），自主考虑决定你这次【给不给TA点赞】：\n"
            f"【决策与输出格式要求】：\n"
            f"1. 如果你考虑后决定【给TA点赞】（例如对方态度好、讨你欢心、满足了你的要求，或者你心情不错/嘴硬心软勉为其难答应），请在第一行最开头输出 [ACTION:LIKE]，紧接着输出你的口语回复台词（如爽快答应、傲娇表示勉为其难、要求对方也回赞你等）；\n"
            f"2. 如果你考虑后决定【暂时不给/拒绝点赞】（例如觉得对方太敷衍、想傲娇逗逗对方、或要求对方先夸你/叫你好听的才给赞），请在第一行最开头输出 [ACTION:REJECT]，紧接着输出你的口语回复台词（如傲娇拒绝、调侃戏弄、提出交换条件等）；\n"
            f"特别提示：[ACTION:LIKE] 与 [ACTION:REJECT] 仅为系统动作标记，随后的台词必须直接为自然的角色对话，严禁包含任何思考过程或多余解释。"
        )

        guide = str(self._cfg("custom_prompt_guide", "") or "").strip()
        if guide:
            prompt += f"\n额外决策与人设要求：{guide}"

        umo = getattr(event, "unified_msg_origin", None)
        provider = self.context.get_using_provider(umo)
        fallback_agree = "哼，看在你这么诚恳的份上，就给你点赞吧~"

        if provider is None:
            return True, fallback_agree

        try:
            persona_prompt = await self._get_persona_prompt(event)
            is_voice_active = self._is_genie_active(event)
            genie = self._find_genie() if is_voice_active else None

            if is_voice_active and genie is not None:
                try:
                    hint = genie.build_prompt_injection_hint() or ""
                    if hint:
                        persona_prompt = f"{persona_prompt or ''}{hint}"
                    req_prompt = (
                        f"{prompt}\n"
                        "【特别输出规范】：动作标签后的台词部分，必须同时严格输出 <zh>中文回复</zh> 与 <ja>日本語の返信</ja> 两种语言标签格式，"
                        "<ja> 标签内必须为地道日语台词（严禁任何英文或动作括号，涉及中文人名/昵称必须转为日文假名读法，如若曦->アガサシ）。"
                    )
                except Exception as exc:
                    logger.warning(f"[LLM_like] 读取 genie 双语注入提示失败: {exc}")
                    req_prompt = prompt
            else:
                req_prompt = (
                    f"{prompt}\n"
                    "【语言输出要求】：动作标签后的回复台词，请直接且仅使用自然流畅的纯中文单语言对白，"
                    "严禁输出任何日文翻译、日文假名或多语言结构，直接输出纯中文内容以节省token。"
                )

            resp = await provider.text_chat(
                prompt=req_prompt,
                session_id=None,
                contexts=[],
                image_urls=[],
                system_prompt=persona_prompt or None,
            )
            text = (resp.completion_text or "").strip()
            if not text:
                return True, fallback_agree

            should_like = True
            match = self.RE_ACTION_TAG.search(text)
            if match:
                action_word = match.group(1).upper()
                should_like = action_word == "LIKE"
                text = self.RE_ACTION_TAG.sub("", text).strip()

            return should_like, text
        except Exception as exc:
            logger.warning(f"[LLM_like] 决策判断 LLM 调用失败: {exc}")
            return True, fallback_agree

    @staticmethod
    def _claim_event(event: AstrMessageEvent) -> bool:
        """Prevent duplicate execution across command and plain message handlers."""
        if getattr(event, "_qq_like_claimed", False):
            return True
        event._qq_like_claimed = True
        return False

    # ─── Unified Execution Pipeline (Direct Unconditional Flow) ────────
    async def _handle_like_flow(self, event: AstrMessageEvent):
        """Unconditional like execution with cooldown and message yielding."""
        sender_id_str = str(event.get_sender_id() or "").strip()
        if not sender_id_str:
            yield event.plain_result("无法识别当前发送者 QQ。")
            return

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

            group_id = event.get_group_id()
            at_sender = bool(self._cfg("at_sender", False))

            if group_id and at_sender and At is not None and Plain is not None:
                chain = [At(qq=int(sender_id_str)), Plain(f" {reply_text}")]
                yield event.chain_result(chain)
            else:
                yield event.plain_result(reply_text)
        except Exception as exc:
            logger.exception(f"[LLM_like] 处理点赞流程发生异常: {exc}")
            yield event.plain_result(f"点赞时出现异常：{exc}")
        finally:
            self._in_flight_users.discard(sender_id_str)

    # ─── Autonomous Decision Flow (LLM Considers Whether to Like) ──────
    async def _handle_autonomous_flow(
        self,
        event: AstrMessageEvent,
        challenge_ctx: dict[str, Any] | None = None,
        notify_cooldown: bool = False,
    ):
        """Autonomous decision flow: Let LLM consider whether to give a like or reject based on persona."""
        sender_id_str = str(event.get_sender_id() or "").strip()
        if not sender_id_str:
            return

        now = time.time()
        cd = max(1, int(self._cfg("cooldown_seconds", 10)))
        last_time = self._user_last_triggered.get(sender_id_str, 0.0)

        if sender_id_str in self._in_flight_users:
            if notify_cooldown:
                yield event.plain_result("正在为你处理点赞请求，请稍候~")
            return

        if now - last_time < cd:
            if notify_cooldown:
                remain = max(1, int(cd - (now - last_time)))
                yield event.plain_result(f"点赞过于频繁啦，请稍候 {remain} 秒后再试哦。")
            return

        self._in_flight_users.add(sender_id_str)
        self._user_last_triggered[sender_id_str] = now
        umo = str(getattr(event, "unified_msg_origin", None) or "default")
        challenge_key = (umo, sender_id_str)

        try:
            if challenge_ctx and challenge_ctx.get("target_id"):
                target_id = int(challenge_ctx["target_id"])
                target_name = str(challenge_ctx.get("target_name") or "你")
                is_self = bool(challenge_ctx.get("is_self", True))
            else:
                target_id, target_name, is_self = self._resolve_target(event)

            raw_text = str(event.get_message_str() or "").strip()
            sender_name = str(event.get_sender_name() or "").strip() or "你"

            should_like, raw_reply_text = await self._decide_and_generate_reaction(
                event,
                target_name=target_name,
                sender_name=sender_name,
                raw_text=raw_text,
                is_self=is_self,
                challenge_ctx=challenge_ctx,
            )

            if should_like and target_id:
                self._pending_like_challenges.pop(challenge_key, None)
                actual_likes, max_likes, status, err_detail = await self._execute_send_like(
                    event,
                    target_id,
                )
                logger.info(
                    f"[LLM_like] LLM考虑后同意为 {target_name}({target_id}) 点赞: "
                    f"actual={actual_likes}, max={max_likes}, status={status}"
                )
                if status in ("already_maxed", "failed"):
                    reply_text = await self._generate_reaction(
                        event,
                        target_name=target_name,
                        is_self=is_self,
                        actual_likes=actual_likes,
                        max_likes=max_likes,
                        status=status,
                        error_detail=err_detail,
                    )
                else:
                    reply_text = await self._finalize_reply_text(event, raw_reply_text)
            else:
                logger.info(f"[LLM_like] LLM考虑后决定暂不为 {target_name}({target_id}) 点赞 (REJECT)")
                clean_bot_reply = self._clean_display_text(raw_reply_text)
                self._pending_like_challenges[challenge_key] = {
                    "time": time.time(),
                    "user_text": self._normalize_message_text(raw_text) or raw_text,
                    "bot_reply": clean_bot_reply,
                    "target_id": target_id,
                    "target_name": target_name,
                    "is_self": is_self,
                }
                self._user_last_triggered[sender_id_str] = max(0.0, now - cd + 2.0)
                reply_text = await self._finalize_reply_text(event, raw_reply_text)

            group_id = event.get_group_id()
            at_sender = bool(self._cfg("at_sender", False))

            if group_id and at_sender and At is not None and Plain is not None:
                chain = [At(qq=int(sender_id_str)), Plain(f" {reply_text}")]
                yield event.chain_result(chain)
            else:
                yield event.plain_result(reply_text)
        except Exception as exc:
            logger.exception(f"[LLM_like] 自主点赞流程发生异常: {exc}")
        finally:
            self._in_flight_users.discard(sender_id_str)

    # ─── Commands (with wake prefix / @) ───────────────────────────────
    @filter.command("赞我", alias={"like", "点赞", "赞", "点名片赞", "名片赞", "给我点赞", "求赞", "主页赞", "主页点赞"})
    async def like_cmd(self, event: AstrMessageEvent):
        """【指令触发】向 Bot 请求名片赞，支持由 LLM 考虑给不给点赞或直接点满。"""
        if self._claim_event(event):
            return
        if self._should_llm_decide():
            async for res in self._handle_autonomous_flow(event, notify_cooldown=True):
                yield res
        else:
            async for res in self._handle_like_flow(event):
                yield res
        event.stop_event()

    # ─── Plain Message Trigger (no prefix required) ────────────────────
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_plain_message(self, event: AstrMessageEvent):
        """【普通消息触发】监听一切形式的点赞指令、自然语言求赞问询及条件追赞。"""
        if not self._cfg("enable_plain_trigger", True):
            return
        if getattr(event, "_qq_like_claimed", False):
            return

        raw_text = str(event.get_message_str() or "").strip()
        if not raw_text:
            return

        norm_text = self._normalize_message_text(raw_text)
        if not norm_text or self.RE_META_EXCLUDE.search(norm_text):
            return

        if self._is_directed_at_other_human(event, raw_text):
            return

        is_cmd = bool(self.RE_LIKE_COMMAND.search(norm_text))
        is_natural = bool(self.RE_LIKE_NATURAL.search(norm_text))

        umo = str(getattr(event, "unified_msg_origin", None) or "default")
        sender_id_str = str(event.get_sender_id() or "").strip()
        challenge_key = (umo, sender_id_str)
        challenge_ctx = self._pending_like_challenges.get(challenge_key)
        if challenge_ctx and (time.time() - float(challenge_ctx.get("time", 0.0)) > 180.0):
            self._pending_like_challenges.pop(challenge_key, None)
            challenge_ctx = None

        if is_cmd:
            if self._claim_event(event):
                return
            if self._should_llm_decide():
                async for res in self._handle_autonomous_flow(
                    event,
                    challenge_ctx=challenge_ctx,
                    notify_cooldown=True,
                ):
                    yield res
            else:
                async for res in self._handle_like_flow(event):
                    yield res
            event.stop_event()
            return

        if is_natural and self._cfg("enable_autonomous_decision", True):
            if self._claim_event(event):
                return
            async for res in self._handle_autonomous_flow(
                event,
                challenge_ctx=challenge_ctx,
                notify_cooldown=False,
            ):
                yield res
            event.stop_event()
            return

    # ─── Global LLM Hooks (Catch Any Creative/Indirect Like Request) ───
    @filter.on_llm_request(priority=50)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        """在常规 LLM 对话（含私聊、@唤醒与 group_chat_plus）中注入点赞决策标记说明，让用户以任何方式要赞都能触发。"""
        if not self._cfg("enable_llm_tool", True) and not self._should_llm_decide():
            return
        if getattr(event, "_qq_like_claimed", False):
            return

        umo = str(getattr(event, "unified_msg_origin", None) or "default")
        sender_id_str = str(event.get_sender_id() or "").strip()
        challenge_key = (umo, sender_id_str)
        challenge_ctx = self._pending_like_challenges.get(challenge_key)
        if challenge_ctx and (time.time() - float(challenge_ctx.get("time", 0.0)) > 180.0):
            self._pending_like_challenges.pop(challenge_key, None)
            challenge_ctx = None

        challenge_line = ""
        if challenge_ctx:
            prev_user = challenge_ctx.get("user_text", "")
            prev_bot = challenge_ctx.get("bot_reply", "")
            challenge_line = (
                f"【待决求赞互动】：该用户刚才曾向你求赞（『{prev_user}』），你当时暂时拒绝或提出条件（『{prev_bot}』）。"
                "若对方本次发言满足了你的条件或打动了你，请包含 `[ACTION:LIKE]` 给TA点赞。\n"
            )

        guide = str(self._cfg("custom_prompt_guide", "") or "").strip()
        guide_line = f"额外点赞偏好要求：{guide}\n" if guide else ""
        like_instruction = (
            "\n【QQ名片赞/主页赞自主决策能力】\n"
            "你拥有为当前对话用户点 QQ 资料卡名片赞（主页赞）的能力。当用户在对话中以任何方式向你要赞、求赞、暗示想让你去TA主页/资料卡点赞、或者兑现了你之前提出的点赞条件时：\n"
            "请结合你的人设性格、心情与对方的态度，自主考虑给不给TA点赞：\n"
            "- 如果你决定【给TA点赞】，请在回复内容中包含 `[ACTION:LIKE]` 标记（系统会自动识别该标记并为对方送上满额名片赞，且会在展示前自动剥除该标记）；\n"
            "- 如果你决定【不给/暂时拒绝点赞】，请直接按人设拒绝或提出傲娇条件（可带 `[ACTION:REJECT]` 或不带标记）；\n"
            "- 如果当前对话与要赞/点赞无关，请正常回复，不要输出上述标记。\n"
            f"{challenge_line}{guide_line}"
        )
        current_sys = getattr(req, "system_prompt", "") or ""
        if "[ACTION:LIKE]" not in current_sys:
            req.system_prompt = f"{current_sys}{like_instruction}"

    @filter.on_llm_response(priority=99990)
    async def on_llm_response(self, event: AstrMessageEvent, response: Any) -> None:
        """解析常规对话 LLM 响应中的 [ACTION:LIKE] / [ACTION:REJECT] 标记，自动执行点赞并剥除标记。"""
        if not response:
            return
        text = getattr(response, "completion_text", None)
        if not text or not isinstance(text, str):
            return

        matches = list(self.RE_ACTION_TAG.finditer(text))
        if not matches:
            return

        should_like = any(m.group(1).upper() == "LIKE" for m in matches)
        cleaned_text = self.RE_ACTION_TAG.sub("", text).strip()
        response.completion_text = cleaned_text

        umo = str(getattr(event, "unified_msg_origin", None) or "default")
        sender_id_str = str(event.get_sender_id() or "").strip()
        challenge_key = (umo, sender_id_str)

        if should_like and not getattr(event, "_qq_like_executed", False):
            self._pending_like_challenges.pop(challenge_key, None)
            target_id, target_name, _ = self._resolve_target(event)
            if target_id:
                try:
                    actual_likes, max_likes, status, err_detail = await self._execute_send_like(
                        event,
                        target_id,
                    )
                    logger.info(
                        f"[LLM_like] 常规对话钩子触发点赞 {target_name}({target_id}): "
                        f"actual={actual_likes}, max={max_likes}, status={status}, err={err_detail}"
                    )
                except Exception as exc:
                    logger.warning(f"[LLM_like] 常规对话钩子执行点赞失败: {exc}")

    # ─── LLM Tool (Function Calling) ───────────────────────────────────
    @filter.llm_tool(name="send_qq_like")
    async def send_qq_like(
        self,
        event: AstrMessageEvent,
        target_qq: str = "",
    ) -> str:
        """给指定用户或当前发送者点 QQ 资料卡名片赞（主页赞，送上当日最大上限点赞）。当用户以任何方式求赞、要赞、希望得到名片赞/主页赞，或者询问你是否可以/愿不愿意给TA点赞，且你在人设上考虑后决定同意给TA点赞时调用此工具。如果你根据人设或当前情绪考虑后不想给TA点赞，则不要调用此工具。

        Args:
            target_qq(string): 要点赞的目标用户的QQ账号（纯数字字符串）。如果未指定或为空，默认给当前对话者点赞。
        """
        if not self._cfg("enable_llm_tool", True):
            return "点赞工具当前已在插件配置中禁用。"

        sender_id_str = str(event.get_sender_id() or "").strip()
        target_id_str = str(target_qq or "").strip()
        if not target_id_str or not target_id_str.isdigit():
            target_id_str = sender_id_str

        if not target_id_str or not target_id_str.isdigit():
            return "未找到有效的点赞目标 QQ 号。"

        target_id = int(target_id_str)
        is_self = target_id_str == sender_id_str
        target_desc = "对方" if is_self else f"用户(QQ:{target_id})"

        if getattr(event, "_qq_like_executed", False):
            return f"点赞成功：已为{target_desc}送出名片赞。请按照你的人设性格回复对方。"

        actual_likes, max_likes, status, err_detail = await self._execute_send_like(
            event,
            target_id,
        )

        if status == "full_success":
            return f"点赞成功：已为{target_desc}送出 {actual_likes} 次名片赞（已达本次上限 {max_likes} 次）。请按照你的人设性格回复对方。"
        elif status == "partial_success":
            return f"点赞成功：已为{target_desc}送出 {actual_likes} 次名片赞（因达今日限制停止）。请按照你的人设性格回复对方。"
        elif status == "already_maxed":
            return f"点赞提示：今日为{target_desc}的点赞已达上限（送出 0 次）。请按照你的人设性格告知对方今天已经点满啦、明天再来。"
        else:
            return f"点赞失败：{err_detail or '接口错误'}。请按照你的人设性格告知对方。"
