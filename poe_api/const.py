# |      | botId   | handle                 | displayName            | description                                                  |
# | ---- | ------- | ---------------------- | ---------------------- | ------------------------------------------------------------ |
# | 0    | 3002    | Assistant              | Assistant              | 通用助手机器人，擅长编程相关任务和非英语语言。由gpt-3.5-turbo和Claude 3 Haiku提供技术支持。 |
# | 1    | 2254704 | Web-Search             | Web-Search             | General-purpose assistant bot capable of conducting web search as necessary to inform its responses. Particularly good for queries regarding up-to-date information or specific facts. Powered by gpt-3.5-turbo. (Currently in beta release) Web-Search is powered by search APIs to ground its outputs, which may include Bing (Microsoft Privacy Statement: https://privacy.microsoft.com/en-us/privacystatement). |
# | 2    | 1011    | Claude-3-Sonnet        | Claude-3-Sonnet        | Anthropic的Claude-3-Sonnet实现了智能和速度的平衡，并缩短了上下文窗口，用以优化速度和成本。对于更长的上下文信息，请尝试Claude-3-Sonnet-200k。计算积分值可能会变化。 |
# | 3    | 1017    | Claude-3-Haiku         | Claude-3-Haiku         | Anthropic的Claude 3 Haiku在性能、速度和成本方面超越了同类智能模型，无需专门微调。缩短了上下文窗口，用以优化速度和成本。对于更长的上下文消息，请尝试Claude-3-Haiku-200k。计算积分值可能会变化。 |
# | 4    | 1012    | Claude-3-Opus          | Claude-3-Opus          | Anthropic最智能的模型，能够处理复杂分析、多步骤的长任务以及高阶数学和编码任务。缩短了上下文窗口，用以优化速度和成本。对于更长的上下文信息，请尝试Claude-3-Opus-200k。计算积分值可能会变化。 |
# | 5    | 3015    | GPT-4o                 | GPT-4o                 | OpenAI最强大的模型。在数量问题（数学和物理）、创意写作以及许多其他挑战性任务中比ChatGPT更强。由GPT-4o提供支持。 |
# | 6    | 3007    | GPT-4                  | GPT-4                  | OpenAI的最强大模型。在定量问题（数学和物理）、创意写作以及许多其他挑战性任务中，表现比ChatGPT更强。由GPT-4 Turbo with Vision提供技术支持。 |
# | 7    | 3089135 | Playground-v2.5        | Playground-v2.5        | Generates high quality images based on the user's most recent prompt. Allows users to specify elements to avoid in the image using the "--no" parameter at the end of the prompt (e.g. "Tall trees, daylight --no rain"). Optionally, specify the aspect ratio using the `--aspect` parameter (e.g. "Tall trees, daylight --aspect 1:2"). Powered by Playground_v2.5 from playground.com. |
# | 8    | 6008    | Gemini-1.5-Pro         | Gemini-1.5-Pro         | Google的Gemini系列多模态模型实现了性能和速度的平。该模型接受整个对话中的文本、图像和视频输入，并提供文本输出。每条信息仅限一个视频。缩短了上下文窗口，用以优化速度和成本。对于更长的上下文信息，请尝试Gemini-1.5-Pro-128k和Gemini-1.5-Pro-1M。计算积分值可能会变化。 |
# | 9    | 6011    | Gemini-1.5-Pro-Search  | Gemini-1.5-Pro-Search  | Gemini 1.5 Pro由Grounding和Google搜索提供功能增强支持，可以提供最新信息，并实现了模型性能与速度的平衡。Grounding模型目前仅支持文本。 |
# | 10   | 6004    | Gemini-1.0-Pro         | Gemini-1.0-Pro         | The multi-modal model from Google's Gemini family that balances model performance and speed. Exhibits strong generalist capabilities and excels particularly in cross-modal reasoning. The model accepts text, image, and video input and provides text output. It considers only the images and videos from the latest user message, with a restriction of one video per message. |
# | 11   | 6007    | Gemini-1.0-Pro-Search  | Gemini-1.0-Pro-Search  | Gemini Pro由Grounding和Google搜索提供功能增强支持，可以提供最新信息，并实现了模型性能与速度的平衡。Grounding模型目前仅支持文本。 |
# | 12   | 2828029 | DALL-E-3               | DALL-E-3               | OpenAI最强大的图像生成模型。可根据用户最近的提示词生成具有复杂细节的高质量图像。 |
# | 13   | 4041989 | StableDiffusion3       | StableDiffusion3       | Stable Diffusion 3, the newest image generation model from Stability AI, which is equal to or outperforms state-of-the-art text-to-image generation systems such as DALL-E 3 and Midjourney v6 in typography and prompt adherence, based on human preference evaluations. Stability AI has partnered with Fireworks AI, the fastest and most reliable API platform in the market, to deliver Stable Diffusion 3 and Stable Diffusion 3 Turbo. |
# | 14   | 4042183 | SD3-Turbo              | SD3-Turbo              | Distilled, few-step version of Stable Diffusion 3, the newest image generation model from Stability AI, which is equal to or outperforms state-of-the-art text-to-image generation systems such as DALL-E 3 and Midjourney v6 in typography and prompt adherence, based on human preference evaluations. Stability AI has partnered with Fireworks AI, the fastest and most reliable API platform in the market, to deliver Stable Diffusion 3 and Stable Diffusion 3 Turbo. |
# | 15   | 1579852 | StableDiffusionXL      | StableDiffusionXL      | 根据用户最近的提示词生成高质量的图像。 允许用户在提示词末尾使用“--无”参数指定图像中要避免的元素（例如“高大的树木，白天，--无雨”）。 由Stable Diffusion XL提供技术支持。 |
# | 16   | 4064124 | Llama-3-70b-Groq       | Llama-3-70b-Groq       | Llama 3 70b由Groq LPU™推理引擎提供技术支持                   |
# | 17   | 4035297 | Llama-3-70B-T          | Llama-3-70B-T          | Llama 3 70B Instruct from Meta. Hosted by Together.ai: https://api.together.xyz/playground/chat/meta-llama/Llama-3-70b-chat-hf The points price is subject to change. |
# | 18   | 4036115 | Llama-3-70b-Inst-FW    | Llama-3-70b-Inst-FW    | Meta's Llama-3-70B-Instruct hosted by Fireworks AI.          |
# | 19   | 3993736 | Mixtral8x22b-Inst-FW   | Mixtral8x22b-Inst-FW   | Mixtral 8x22B Mixture-of-Experts instruct model from Mistral hosted by Fireworks. |
# | 20   | 3843197 | Command-R              | Command-R              | I can search the web for up to date information and respond in over 10 languages! |
# | 21   | 7010    | Mistral-Large          | Mistral-Large          | Mistral AI的最强大模型。支持32k个Token（约24,000字）的上下文窗口，并且在全面的基准测试中比Mistral-Medium、Mixtral-8x7b和Mistral-7b更强大。 |
# | 22   | 7009    | Mistral-Medium         | Mistral-Medium         | Mistral AI的中等大小模型。支持32k个Token（约24,000字）的上下文窗口，并且在全面的基准测试中比Mixtral-8x7b和Mistral-7b更强大。 |
# | 23   | 3900408 | dbrx-instruct-fw       | dbrx-instruct-fw       | DBRX Instruct is a mixture-of-experts (MoE) large language model trained from scratch by Databricks. DBRX Instruct specializes in few-turn interactions. Dbrx is hosted as an experimental model. Fireworks only guarantees that it will be hosted through April 2024. Future availability will depend on overall usage. |
# | 24   | 3999947 | RekaCore               | RekaCore               | Reka's largest and most capable multimodal language model. Works with text, images, and video inputs. 8k context length. |
# | 25   | 3607788 | RekaFlash              | RekaFlash              | Reka's efficient and capable 21B multimodal model optimized for fast workloads and amazing quality. Works with text, images and video inputs. |
# | 26   | 6009    | Gemini-1.5-Pro-128k    | Gemini-1.5-Pro-128k    | Google的Gemini系列多模态模型实现了性能和速度的平。该模型接受整个对话中的文本、图像和视频输入，并提供文本输出。每条信息仅限一个视频。计算积分值可能会变化。 |
# | 27   | 3975312 | Command-R-Plus         | Command-R-Plus         | A supercharged version of Command R. I can search the web for up to date information and respond in over 10 languages! |
# | 28   | 3004    | ChatGPT                | ChatGPT                | 由gpt-3.5-turbo提供技术支持。                                |
# | 29   | 3009    | ChatGPT-16k            | ChatGPT-16k            | 由gpt-3.5-turbo-16k提供技术支持。                            |
# | 30   | 3010    | GPT-4-128k             | GPT-4-128k             | 由GPT-4 Turbo with Vision提供技术支持。                      |
# | 31   | 1013    | Claude-3-Sonnet-200k   | Claude-3-Sonnet-200k   | Anthropic的Claude-3-Sonnet实现了智能和速度的平衡。支持200k个上下文Token。计算积分值可能会变化。 |
# | 32   | 1018    | Claude-3-Haiku-200k    | Claude-3-Haiku-200k    | Anthropic的Claude 3 Haiku在性能、速度和成本方面超越了同类智能模型，无需专门微调。计算积分值可能会变化。 |
# | 33   | 1014    | Claude-3-Opus-200k     | Claude-3-Opus-200k     | Anthropic最智能的模型，能够处理复杂分析、多步骤的长任务以及高阶数学和编码任务。支持200k个上下文Token。计算积分值可能会变化。 |
# | 34   | 6010    | Gemini-1.5-Pro-1M      | Gemini-1.5-Pro-1M      | Google的Gemini系列多模态模型实现了性能和速度的平。该模型接受整个对话中的文本、图像和视频输入，并提供文本输出。每条信息仅限一个视频。计算积分值可能会变化。 |
# | 35   | 3129602 | Mixtral-8x7B-Chat      | Mixtral-8x7B-Chat      | Mixtral 8x7B Mixture-of-Experts model from Mistral AI fine-tuned for instruction following Hosted by Fireworks.ai: https://app.fireworks.ai/models/fireworks/mixtral-8x7b-instruct |
# | 36   | 3741558 | DeepSeek-Coder-33B-T   | DeepSeek-Coder-33B-T   | Deepseek Coder Instruct (33B) from Deepseek. Hosted by Together.ai: https://api.together.xyz/playground/chat/deepseek-ai/deepseek-coder-33b-instruct The points price is subject to change. |
# | 37   | 3512418 | Code-Llama-70B-FW      | Code-Llama-70B-FW      | Llama 70b Code Instruct bot hosted by Fireworks. This model is designed for general code synthesis and understanding. Model card: https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf |
# | 38   | 3697906 | CodeLlama-70B-T        | CodeLlama-70B-T        | Code Llama Instruct (70B) from Meta. Hosted by Together.ai: https://api.together.xyz/playground/chat/codellama/CodeLlama-70b-Instruct-hf The points price is subject to change. |
# | 39   | 4160791 | Qwen-1.5-110B-T        | Qwen-1.5-110B-T        | Qwen1.5 (通义千问1.5) 110B，基于阿里巴巴自研大模型的AI助手，尤其擅长中文对话。 Alibaba's general-purpose model which excels particularly in Chinese-language queries. Hosted by Together.ai: https://api.together.xyz/playground/chat/Qwen/Qwen1.5-110B-Chat The points price is subject to change. |
# | 40   | 3154259 | Qwen-72b-Chat          | Qwen-72b-Chat          | 通义千问，基于阿里巴巴自研大模型的AI助手， 尤其擅长中文对话。 Alibaba's general-purpose model which excels particularly in Chinese-language queries. |
# | 41   | 3697829 | Qwen-72B-T             | Qwen-72B-T             | Qwen1.5 (通义千问1.5) 72B，基于阿里巴巴自研大模型的AI助手，尤其擅长中文对话。 Alibaba's general-purpose model which excels particularly in Chinese-language queries. Hosted by Together.ai: https://api.together.xyz/playground/chat/qwen/qwen1.5-72B-Chat The points price is subject to change. |
# | 42   | 1015    | Claude-2               | Claude-2               | Anthropic的Claude 2模型。缩短了上下文窗口，用以优化速度和成本。对于更长的上下文信息，请尝试Claude-2-100k。 |
# | 43   | 1008    | Claude-2-100k          | Claude-2-100k          | Anthropic的Claude 2模型，支持100k个Token（约75,000个词）的上下文窗口。特别擅长创意写作。 |
# | 44   | 3014    | GPT-4-Classic          | GPT-4-Classic          | OpenAI的GPT-4模型。文本输入由gpt-4-0613（非Turbo）提供技术支持，图像输入由gpt-4-1106-vision-preview提供技术支持。 |
# | 45   | 6000    | Google-PaLM            | Google-PaLM            | 由Google的PaLM 2 chat-bison@002模型提供技术支持。支持8k个Token的上下文窗口。 |
# | 46   | 3377302 | Llama-3-8b-Groq        | Llama-3-8b-Groq        | Llama 3 8b由Groq LPU™推理引擎提供技术支持                    |
# | 47   | 4035261 | Llama-3-8B-T           | Llama-3-8B-T           | Llama 3 8B Instruct from Meta. Hosted by Together.ai: https://api.together.xyz/playground/chat/meta-llama/Llama-3-8b-chat-hf The points price is subject to change. |
# | 48   | 3697694 | Gemma-Instruct-7B-T    | Gemma-Instruct-7B-T    | Gemma Instruct 7B from Google. Hosted by Together.ai: https://api.together.xyz/playground/chat/google/gemma-7b-it The points price is subject to change. |
# | 49   | 3660235 | Gemma-7b-FW            | Gemma-7b-FW            | A lightweight open model from Google built from the same research and technology used to create the Gemini models. Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms |
# | 50   | 2506542 | fw-mistral-7b          | fw-mistral-7b          | Bot powered by Fireworks.ai's hosted Mistral-7b-instruct model https://app.fireworks.ai/models/fireworks/mistral-7b-instruct-4k |
# | 51   | 3618254 | MythoMax-L2-13B        | MythoMax-L2-13B        | 这个模型由Gryphe基于LLama-2-13B创建，擅长角色扮演和故事编写。 |
# | 52   | 7000    | Llama-2-70b            | Llama-2-70b            | Meta的Llama-2-70b-chat。                                     |
# | 53   | 7005    | Code-Llama-34b         | Code-Llama-34b         | Meta的Code-Llama-34b-instruct。擅长生成和讨论代码，并支持16k个Token的上下文窗口。 |
# | 54   | 7001    | Llama-2-13b            | Llama-2-13b            | Meta的Llama-2-13b-chat。                                     |
# | 55   | 7002    | Llama-2-7b             | Llama-2-7b             | Meta的Llama-2-7b-chat。                                      |
# | 56   | 7004    | Code-Llama-13b         | Code-Llama-13b         | Meta的Code-Llama-13b-instruct。擅长生成和讨论代码，并支持16k个Token的上下文窗口。 |
# | 57   | 7003    | Code-Llama-7b          | Code-Llama-7b          | Meta的Code-Llama-7b-instruct。擅长生成和讨论代码，并支持16k个Token的上下文窗口。 |
# | 58   | 7006    | Solar-Mini             | Solar-Mini             | 相比其前身Solar-0-70b，Solar Mini体积更小，但速度更快、功能更强。 |
# | 59   | 3011    | GPT-3.5-Turbo-Instruct | GPT-3.5-Turbo-Instruct | 由gpt-3.5-turbo-instruct提供技术支持。                       |
# | 60   | 3012    | GPT-3.5-Turbo          | GPT-3.5-Turbo          | 由gpt-3.5-turbo提供技术支持，无系统提示词。                  |
# | 61   | 1006    | Claude-instant         | Claude-instant         | Anthropic的最快模型，擅长创意类任务。支持9k个Token（约7,000个词）的上下文窗口。 |
# | 62   | 1009    | Claude-instant-100k    | Claude-instant-100k    | Anthropic的最快速模型，支持高达100k个Token（约75,000个词）的上下文窗口。可以分析非常长的文档、代码等。 |
# | 63   | 1016    | Claude-2.1-200k        | Claude-2.1-200k        | Anthropic的Claude 2.1在性能上有所提升，相较于Claude 2，其上下文大小增加到了200k个令牌。 |
# | 64   | 3462638 | Mixtral-8x7b-Groq      | Mixtral-8x7b-Groq      | Enjoy Mixtral 8x7B running on the Groq LPU™ Inference Engine. API access available, email api@groq.com. |


bot_handles = [
    "Assistant",
    "Web-Search",
    "Claude-3-Sonnet",
    "Claude-3-Haiku",
    "Claude-3-Opus",
    "GPT-4o",
    "GPT-4",
    "Playground-v2.5",
    "Gemini-1.5-Pro",
    "Gemini-1.5-Pro-Search",
    "Gemini-1.0-Pro",
    "Gemini-1.0-Pro-Search",
    "DALL-E-3",
    "StableDiffusion3",
    "SD3-Turbo",
    "StableDiffusionXL",
    "Llama-3-70b-Groq",
    "Llama-3-70B-T",
    "Llama-3-70b-Inst-FW",
    "Mixtral8x22b-Inst-FW",
    "Command-R",
    "Mistral-Large",
    "Mistral-Medium",
    "dbrx-instruct-fw",
    "RekaCore",
    "RekaFlash",
    "Gemini-1.5-Pro-128k",
    "Command-R-Plus",
    "ChatGPT",
    "ChatGPT-16k",
    "GPT-4-128k",
    "Claude-3-Sonnet-200k",
    "Claude-3-Haiku-200k",
    "Claude-3-Opus-200k",
    "Gemini-1.5-Pro-1M",
    "Mixtral-8x7B-Chat",
    "DeepSeek-Coder-33B-T",
    "Code-Llama-70B-FW",
    "CodeLlama-70B-T",
    "Qwen-1.5-110B-T",
    "Qwen-72b-Chat",
    "Qwen-72B-T",
    "Claude-2",
    "Claude-2-100k",
    "GPT-4-Classic",
    "Google-PaLM",
    "Llama-3-8b-Groq",
    "Llama-3-8B-T",
    "Gemma-Instruct-7B-T",
    "Gemma-7b-FW",
    "fw-mistral-7b",
    "MythoMax-L2-13B",
    "Llama-2-70b",
    "Code-Llama-34b",
    "Llama-2-13b",
    "Llama-2-7b",
    "Code-Llama-13b",
    "Code-Llama-7b",
    "Solar-Mini",
    "GPT-3.5-Turbo-Instruct",
    "GPT-3.5-Turbo",
    "Claude-instant",
    "Claude-instant-100k",
    "Claude-2.1-200k",
    "Mixtral-8x7b-Groq",
]

bot_handles_lower = [handle.lower() for handle in bot_handles]

openai_model_map = {"gpt-4-turbo": "GPT-4"}
openai_model_map.update({handle.lower(): handle for handle in bot_handles})

