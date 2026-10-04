# LLM点赞 (astrbot_plugin_qq_like)

LLM点赞插件，可以在赞我的同时调用LLM说话

大概效果
<img width="1416" height="779" alt="T`1HSZ9V`$Y}JM7)1RB4Y`Y" src="https://github.com/user-attachments/assets/379fc296-cacb-4e2c-9f12-00cda5847117" />

## 特性
- 自动点赞：发送 赞我、/赞我、点赞 等，自动为发送者（或 @ 的群友）刷满当日名片赞上限；
- 性格反应：自动读取当前会话的人设提示词（Persona Prompt），由大模型根据点赞结果作出个性化反应；
- 语音适配：深度联动 genie 语音合成插件（TTS），遵循会话双语开关，自动输出日文语音协同播报；
- 冷却防刷：内置用户冷却时间与并发锁。
