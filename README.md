This is the repository for the LLM worker in Ran Tester UE.

The worker reads `configs/llm_worker/test.yaml` from the ran-tester-ue checkout. `provider` selects the backend. If `provider` is omitted, it runs the local model.

Local GPU model:

```yaml
provider: local
model: deepseek-ai/DeepSeek-Coder-6.7B-Instruct
```

`model` is a Hugging Face id. This path needs a GPU. Do not set `OPENROUTER_API_KEY`.

OpenRouter. Use DeepSeek V4 Flash:

```yaml
provider: openrouter
model: deepseek/deepseek-v4-flash
```

That id is DeepSeek V4 Flash (284B total, 13B active). Put the key only in the ran-tester-ue `.env`, which the controller already loads:

```
OPENROUTER_API_KEY=your-key
```

Do not commit that file, and do not put the key in this repo or in `test.yaml`.

Keep the model in `test.yaml`, next to the prompts for that run. The key is the only setting that belongs in `.env`.
