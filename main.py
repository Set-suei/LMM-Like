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
    "astrbot_plugin_LLM_like",
    "Codex",
    "LLM赞我：为用户送上每日上限名片赞，支持LLM自主决策、函数调用(Tool Calling)、当前人设与语音插件(genie)个性化反应",
    "1.2.0",
)
class QQLikePlugin(Star):
    """QQ profile like plugin with autonomous LLM decision, Tool Calling, persona reaction and genie voice adaptation."""

    # Explicit command triggers (e.g. 赞我 / /赞我 / 点赞)
    RE_LIKE_COMMAND = re.compile(
        r"^[/~#.]?\s*(赞我|求赞|点赞|点个赞|帮我点赞|赞我一下|给我点赞|名片赞|like)(?:[！!~呀啊吧哦呢~\s]*)$",
        re.IGNORECASE,
    )

    # Natural conversation like queries (e.g. 小千岁会给我点赞吗 / 能帮我点赞吗 / 愿不愿意给我点赞)
    RE_LIKE_NATURAL = re.compile(
        r"(?:会|能|可以|想|要|愿不愿意|愿意|敢不敢|能不能|帮我).*?(?:给我|给俺|帮我).*?(?:点赞|点个赞|赞我|名片赞|赞一下).*?(?:吗|嘛|吧|呗|不|？|\?|！|!)|"
        r"(?:会给我点赞吗|能给我点赞吗|可以给我点赞吗|要不要给我点赞|给我点赞好不好|求点赞|给我点个赞嘛|给我点赞吧|给俺点赞|点赞我)",
        re.IGNORECASE,
    )

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

    @staticmethod
    def _clean_display_text(text: str) -> str:
        """未开启语音插件时，清洗提取纯中文单语言，彻底剥除日文标签与假名行。"""
        if not text:
            return ""
        # 1. 优先提取 <zh>...</zh>
        zh_matches = re.findall(r"<zh>(.*?)</zh>", text, re.DOTALL | re.IGNORECASE)
        if zh_matches:
            return "\n".join(m.strip() for m in zh_matches if m.strip())
        # 2. 剥除 <ja>...</ja> 及其内容
        cleaned = re.sub(r"<ja>[\s\S]*?</ja>", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<thought>[\s\S]*?</thought>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?[A-Za-z_][^<>]*>", "", cleaned).strip()
        # 3. 逐行过滤掉纯假名日文行
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

    # ─── LLM Reaction ────────────────────────────────────────────────────────────────────────────────────────
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
            fallback = f"今天已经为{target_name}点过赞，名片赞已达上限，不需要再继续点啦！"
        else:
            fallback = f"点赞出了点小状况：{error_detail or '未知错误'}"

        if not self._cfg("enable_llm_reaction", True):
            return fallback

        if status == "full_success":
            status_desc = f"你刚刚成功为{target_desc}送出了 {actual_likes} 个名片赞，已达到本次点赞上限。"
        elif status == "partial_success":
            status_desc = f"你为{target_desc}送出了 {actual_likes} 个赞，因达到今日点赞上限而停止。"
        elif status == "already_maxed":
            status_desc = f"你尝试为{target_desc}点赞，但系统提示对方赞数已经达到今日上限（或已被点满），本次送出 0 个赞。"
        else:
            status_desc = f"你尝试为{target_desc}点赞时操作失败，原因：{error_detail}。"

        guide = str(self._cfg("custom_prompt_guide", "") or "").strip()
        guide_note = f"\n额外指令要求：{guide}" if guide else ""

        sender_name = str(event.get_sender_name() or "").strip() or "用户"
        prompt = (
            f"【系统事件】：用户「{sender_name}」触发了点赞功能。\n"
            f"事件结果：{status_desc}\n"
            f"请根据你当前的人设、性格与说话风格，直接对用户做出一句自然的口语反应（可包含撒娇、感谢、调侃、炫耀、吐槽、暖心关怀等）。\n"
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
                    # 未开启语音功能时，显式注入单语言中文提示词，严禁日文以节省token
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
                    if text.startswith(('"', "“")) and text.endswith(('"', "”")):
                        text = text[1:-1].strip()

                    if is_voice_active and genie is not None:
                        try:
                            voice_plugin_reply = getattr(genie, "voice_plugin_reply", None)
                            if callable(voice_plugin_reply):
                                display = await voice_plugin_reply(event, text)
                                if display:
                                    return display
                        except Exception as exc:
                            logger.warning(f"[LLM_like] genie 语音处理失败，退回单语言展示: {exc}")

                    return self._clean_display_text(text)
        except Exception as exc:
            logger.warning(f"[LLM_like] LLM 响应生成失败，将使用兜底回复: {exc}")

        if fallback:
            if self._is_genie_active(event):
                genie = self._find_genie()
                if genie is not None:
                    try:
                        voice_plugin_reply = getattr(genie, "voice_plugin_reply", None)
                        if callable(voice_plugin_reply):
                            disp = await voice_plugin_reply(event, fallback)
                            if disp:
                                return disp
                    except Exception as e_fb:
                        logger.warning(f"[LLM_like] fallback 语音接入失败: {e_fb}")
            return self._clean_display_text(fallback)

        return fallback

    # ─── Autonomous Decision & Generation ─────────────────────────────────────────────────────────
    async def _decide_and_generate_reaction(
        self,
        event: AstrMessageEvent,
        target_name: str,
        sender_name: str,
        raw_text: str,
        is_self: bool,
    ) -> tuple[bool, str]:
        """自主决定是否点赞并生成反应台词。未开启语音插件时严格单语言中文。"""
        target_desc = f"用户本人「{sender_name}」" if is_self else f"用户「{sender_name}」为你指定的群友「{target_name}」"

        prompt = (
            f"【群聊对话事件】：\n"
            f"{target_desc}刚刚对你发送了：『{raw_text}』，对方表达了想让你为其点赞、询问你愿不愿意/是否给TA点赞。\n"
            f"请完全基于你当前的人设性格喜好、与群友的熟络程度以及当前语境，自主判断你【愿意】还是【不愿/拒绝】为TA点赞：\n"
            f"【输出格式要求】：\n"
            f"1. 如果你判定【同意/愿意】点赞，请在第一行开头输出 [ACTION:LIKE]，紧接着说出你的口语回复台词（如爽快答应、傲娇表示勉为其难、索要回赞、要求叫你主人等）；\n"
            f"2. 如果你判定【拒绝/不愿】点赞，请在第一行开头输出 [ACTION:REJECT]，紧接着说出你的口语回复台词（如傲娇拒绝、戏弄嘲弄、提出无理要求、吐槽等）；\n"
            f"特别提示：[ACTION:LIKE] 与 [ACTION:REJECT] 仅为系统动作标识，随后的台词必须直接为角色对话，不要包含任何旁白、思考过程或多余说明。"
        )

        guide = str(self._cfg("custom_prompt_guide", "") or "").strip()
        if guide:
            prompt += f"\n额外人设要求：{guide}"

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
                # 未开启语音功能时，显式注入单语言中文提示词，严禁日文以节省token
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
                return True, self._clean_display_text(fallback_agree)

            should_like = True
            if "[ACTION:REJECT]" in text:
                should_like = False
                text = text.replace("[ACTION:REJECT]", "").strip()
            elif "[ACTION:LIKE]" in text:
                should_like = True
                text = text.replace("[ACTION:LIKE]", "").strip()

            if text.startswith(('"', "“")) and text.endswith(('"', "”")):
                text = text[1:-1].strip()

            if is_voice_active and genie is not None:
                try:
                    voice_plugin_reply = getattr(genie, "voice_plugin_reply", None)
                    if callable(voice_plugin_reply):
                        display = await voice_plugin_reply(event, text)
                        if display:
                            return should_like, display
                except Exception as exc:
                    logger.warning(f"[LLM_like] genie 语音处理失败，退回单语言展示: {exc}")

            return should_like, self._clean_display_text(text)
        except Exception as exc:
            logger.warning(f"[LLM_like] 决策判断 LLM 调用失败: {exc}")
            if fallback_agree:
                if self._is_genie_active(event):
                    genie = self._find_genie()
                    if genie is not None:
                        try:
                            voice_plugin_reply = getattr(genie, "voice_plugin_reply", None)
                            if callable(voice_plugin_reply):
                                disp = await voice_plugin_reply(event, fallback_agree)
                                if disp:
                                    return True, disp
                        except Exception:
                            pass
                return True, self._clean_display_text(fallback_agree)
            return True, fallback_agree

    @staticmethod
    def _claim_event(event: AstrMessageEvent) -> bool:
        """Prevent duplicate execution across command and plain message handlers."""
        if getattr(event, "_qq_like_claimed", False):
            return True
        event._qq_like_claimed = True
        return False

    # ─── Unified Execution Pipeline (Direct Command Flow) ─────────────
    async def _handle_like_flow(self, event: AstrMessageEvent):
        """Unified like execution with cooldown and message yielding for explicit commands."""
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

    # ─── Autonomous Decision Flow ──────────────────────────────────────
    async def _handle_autonomous_flow(self, event: AstrMessageEvent):
        """Autonomous decision flow: Let LLM decide whether to like or reject based on persona."""
        sender_id_str = str(event.get_sender_id() or "").strip()
        if not sender_id_str:
            return

        now = time.time()
        cd = max(1, int(self._cfg("cooldown_seconds", 10)))
        last_time = self._user_last_triggered.get(sender_id_str, 0.0)

        if sender_id_str in self._in_flight_users:
            return

        if now - last_time < cd:
            return

        self._in_flight_users.add(sender_id_str)
        self._user_last_triggered[sender_id_str] = now

        try:
            target_id, target_name, is_self = self._resolve_target(event)
            raw_text = str(event.get_message_str() or "").strip()
            sender_name = str(event.get_sender_name() or "").strip() or "你"

            should_like, reply_text = await self._decide_and_generate_reaction(
                event,
                target_name=target_name,
                sender_name=sender_name,
                raw_text=raw_text,
                is_self=is_self,
            )

            if should_like and target_id:
                actual_likes, max_likes, status, err_detail = await self._execute_send_like(
                    event,
                    target_id,
                )
                logger.info(
                    f"[LLM_like] 自主判断同意为 {target_name}({target_id}) 点赞: "
                    f"actual={actual_likes}, max={max_likes}, status={status}"
                )

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
        """【普通消息触发】监听点赞指令及自然语言求赞问询，并阻断后续未处理拦截。"""
        if not self._cfg("enable_plain_trigger", True):
            return
        if self._claim_event(event):
            return

        text = str(event.get_message_str() or "").strip()
        is_cmd = bool(self.RE_LIKE_COMMAND.search(text))
        is_natural = bool(self.RE_LIKE_NATURAL.search(text))

        if is_cmd:
            async for res in self._handle_like_flow(event):
                yield res
            event.stop_event()
        elif is_natural and self._cfg("enable_autonomous_decision", True):
            async for res in self._handle_autonomous_flow(event):
                yield res
            event.stop_event()

    # ─── LLM Tool (Function Calling) ───────────────────────────────────
    @filter.llm_tool(name="send_qq_like")
    async def send_qq_like(
        self,
        event: AstrMessageEvent,
        target_qq: str = "",
    ) -> str:
        """给指定用户或当前发送者点 QQ 资料卡名片赞（送上当日最大上限点赞）。当用户求赞、希望得到点赞、或者询问你是否可以/愿不愿意给TA点赞，且你在人设上决定同意点赞时调用此工具。如果你根据人设或当前情绪不想给TA点赞，则不要调用此工具。

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
        is_self = (target_id_str == sender_id_str)
        target_desc = "对方" if is_self else f"用户(QQ:{target_id})"

        actual_likes, max_likes, status, err_detail = await self._execute_send_like(
            event,
            target_id,
        )

        if status == "full_success":
            return f"点赞成功：已为{target_desc}送出 {actual_likes} 次名片赞（已达本次上限 {max_likes} 次）。请按照你的人设性格回复对方。"
        elif status == "partial_success":
            return f"点赞成功：已为{target_desc}送出 {actual_likes} 次名片赞（因达今日限制停止）。请按照你的人设性格回复对方。"
        elif status == "already_maxed":
            return f"点赞提示：今日为{target_desc}的点赞已达上限（送出 0 次）。请按照你的人设性格告知对方明天再来。"
        else:
            return f"点赞失败：{err_detail or '接口错误'}。请按照你的人设性格告知对方。"
