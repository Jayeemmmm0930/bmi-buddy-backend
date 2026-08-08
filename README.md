# Local School Health AI

This Python service generates Health Buddy replies with a language model that
runs on this computer. It does not use OpenAI or require an API key.

## Run

```powershell
cd C:\Users\tumba\Documents\bmi\backend
py -m pip install -r requirements.txt
py app.py
```

The optimized 0.5B model is already downloaded on this computer. The API starts
immediately and warms the model in the background. Common questions about sleep,
activity, BMI, hydration, and medicine use verified instant answers; other health
questions use the local model. Keep the terminal open while using the Flutter app.
Non-health prompts are rejected by a server-side topic filter before they can reach
the model. Short follow-ups such as `why?` remain available during a health chat.

The defaults can be changed when starting the server if needed:

```powershell
$env:LOCAL_AI_MODEL="Qwen/Qwen2.5-1.5B-Instruct" # slower, larger model
$env:MAX_NEW_TOKENS="72"
py app.py
```

- Local documentation: `http://127.0.0.1:8000/docs`
- Flutter chat endpoint: `POST /api/chat`
- Health check: `GET /health`

The existing Flutter screen can keep calling `AIService.sendMessage(userText)`.
The server remembers the latest six turns by client address, so follow-up
questions work without changing that screen. `AIService` should send JSON like:

```json
{"message": "How can I sleep better?"}
```

and read the `reply` value from the JSON response.

For multiple users behind the same address, send a stable `session_id` in the
request. Send `"reset": true` to begin a new conversation.

Physical phones must use the computer's LAN address and be on the same Wi-Fi.
Android emulators normally use `http://10.0.2.2:8000`.
