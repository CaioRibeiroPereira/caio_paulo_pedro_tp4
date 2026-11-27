# Agente de atendimento — OpenAI Agents SDK

O agente usa `Agent`, `Runner.run` e uma ferramenta `@function_tool` para enviar cada
solicitação do cliente ao endpoint JWT-protegido `POST /classify`. Uma
`SQLiteSession` em memória mantém o contexto entre os três turnos da demonstração.

## Configuração

Na raiz do projeto:

```bash
cp .env.example .env
```

Edite `.env` e configure pelo menos:

```dotenv
GEMINI_API_KEY="chave-do-google-ai-studio"
GEMINI_MODEL="identificador-do-modelo"
AGENT_API_URL="http://127.0.0.1:8000"
AGENT_API_USERNAME="alice"
AGENT_API_PASSWORD="alice123"
```

Como alternativa ao usuário e senha, defina `AGENT_API_TOKEN` com um JWT válido.

O agente usa `AsyncOpenAI` com o endpoint compatível do Google
(`https://generativelanguage.googleapis.com/v1beta/openai/`) e entrega esse cliente
ao `OpenAIChatCompletionsModel`. O tracing da plataforma OpenAI é desabilitado, pois
esta configuração não utiliza `OPENAI_API_KEY`.

## Execução

No primeiro terminal, inicie a API:

```bash
source .venv/bin/activate
cd fastapi
uvicorn main:app --reload
```

No segundo terminal, a partir da raiz:

```bash
source .venv/bin/activate
python -m agent.customer_support_agent
```

O script executa três turnos usando a mesma sessão e salva a evidência em
`agent/conversation_demo.json`. Execute novamente antes da entrega para atualizar a
transcrição com uma conversa real.

## Testes sem consumo da OpenAI API

```bash
.venv/bin/python -m pytest agent/test_customer_support_agent.py -v
```

Os testes usam transporte HTTP simulado e não consomem créditos nem exigem chave.
