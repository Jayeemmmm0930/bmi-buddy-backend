"""Local, generative School Health Buddy API.

Run with ``py app.py``. The language model runs on this computer; no OpenAI
account, API key, or internet connection is used after the model is downloaded.
"""

import asyncio
import os
import re
import socket
import threading
from collections import defaultdict, deque
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ON_RENDER = os.getenv("RENDER", "").lower() == "true"
MODEL_NAME = (
    os.getenv("RENDER_AI_MODEL", "google/flan-t5-small")
    if ON_RENDER
    else os.getenv("LOCAL_AI_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
)
MAX_TURNS = 4
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "72"))

SYSTEM_PROMPT = """You are Health Buddy, a careful school health education assistant.
Answer the student's actual question directly in plain, friendly language. Use the
conversation for follow-up questions. Do not give a generic menu of topics when you
can answer the question. Keep answers concise, normally 2-3 sentences.

When the student greets you, greet them warmly, briefly introduce yourself as BMI
BUDDY's Health Buddy, and ask how you can help with their health or wellbeing.

Safety and accuracy rules:
- Give general education, not a diagnosis. Do not invent facts or claim certainty.
- Never prescribe medicines or doses. For symptoms, suggest a trusted adult, school
  nurse, pharmacist, or clinician when appropriate.
- Never recommend crash diets, fasting, purging, diet pills, or extreme exercise.
- BMI is a screening measure, not a diagnosis. For ages 2-19 it must be interpreted
  using age- and sex-specific BMI-for-age percentiles, not adult BMI cutoffs.
- Ages 6-12 generally need 9-12 hours of sleep; ages 13-18 need 8-10 hours.
- Ages 5-17 should aim for about 60 minutes of moderate-to-vigorous activity daily,
  building up gradually if currently inactive.
- If important details are missing, ask one short follow-up question.
- If asked about something outside health and wellbeing, politely say this assistant
  focuses on health.
"""

URGENT_TERMS = (
    "can't breathe", "cannot breathe", "difficulty breathing", "chest pain",
    "passed out", "unconscious", "seizure", "severe bleeding", "overdose",
    "suicide", "kill myself", "hurt myself", "self harm",
    "nahihirapan huminga", "magpakamatay",
)
URGENT_REPLY = (
    "This may need urgent help. Tell a trusted adult immediately and contact your "
    "local emergency service or go to the nearest emergency department. Do not wait "
    "for this chat or stay alone."
)
OFF_TOPIC_REPLY = (
    "I can only help with health and wellbeing questions. You can ask me about "
    "sleep, food, exercise, hydration, BMI, symptoms, hygiene, mental health, "
    "or other health topics."
)

HEALTH_TOPIC_PATTERN = re.compile(
    r"\b(?:"
    r"health|healthy|wellness|wellbeing|bmi|weight|height|diet|nutrition|food|meal|"
    r"calorie|protein|vitamin|mineral|water|hydrat\w*|sleep|bedtime|tired|fatigue|"
    r"exercise|workout|activity|fitness|sport\w*|walk\w*|run\w*|muscle|bone|body|symptom|"
    r"pain|ache|hurt|sick|illness|disease|fever|cough|cold|flu|headache|stomach|"
    r"breath\w*|heart|blood|skin|wound|injury|allerg\w*|infect\w*|doctor|nurse|"
    r"hospital|clinic|medicine|medication|tablet|dose|mental|emotion\w*|mood|"
    r"stress|anxi\w*|depress\w*|sad|panic|hygiene|wash|tooth|teeth|dental|"
    r"puberty|period|menstru\w*|pregnan\w*|sexual|first aid|emergency|growth|"
    r"eat\w*|diabetes|asthma|cancer|dengue|diarrhea|vomit\w*|nausea|"
    r"kalusugan|sakit|lagnat|ubo|sipon|tulog|pagkain|tubig|ehersisyo|gamot|"
    r"sugat|tiyan|ulo|dibdib|pagod"
    r")\b",
    re.IGNORECASE,
)
NON_HEALTH_TASK_PATTERN = re.compile(
    r"\b(?:code|coding|programming|python|javascript|flutter|essay|poem|song|"
    r"capital of|president|weather|movie|video game)\b",
    re.IGNORECASE,
)
FOLLOW_UP_PATTERN = re.compile(
    r"^(?:why|why not|how so|tell me more|explain|can you explain|"
    r"what should i do|is that normal|what about that|and then)\??$",
    re.IGNORECASE,
)
SOCIAL_PATTERN = re.compile(
    r"^(?:hi|hello|hey|hiya|good\s+(?:morning|afternoon|evening)|"
    r"kumusta|kamusta|thanks|thank\s+you|salamat|bye|goodbye)"
    r"(?:\s+(?:there|health\s+buddy|buddy))?[!.?]*$",
    re.IGNORECASE,
)


def trusted_facts(text: str) -> str:
    """Return factual constraints relevant to this conversation (lightweight RAG)."""
    lowered = text.lower()
    facts: list[str] = []
    if any(word in lowered for word in ("sleep", "bedtime", "tired", "hours")):
        facts.append(
            "Children ages 6-12 need 9-12 hours of sleep per 24 hours. "
            "Teenagers ages 13-18 need 8-10 hours per 24 hours."
        )
    if any(word in lowered for word in ("exercise", "activity", "workout", "sport")):
        facts.append(
            "Children and adolescents ages 5-17 should average at least 60 minutes "
            "of moderate-to-vigorous physical activity daily and include muscle- and "
            "bone-strengthening activity at least 3 days per week."
        )
    if any(word in lowered for word in ("bmi", "weight", "underweight", "overweight")):
        facts.append(
            "For ages 2-19, BMI categories use age- and sex-specific BMI-for-age "
            "percentiles because young people are still growing. Adult BMI cutoffs "
            "must not be used to classify a child or teen."
        )
    if any(word in lowered for word in ("water", "hydration", "dehydrated", "drink")):
        facts.append(
            "Fluid needs vary with age, body size, heat, illness, and activity; there "
            "is no single amount that is correct for every student. Water is usually "
            "the best routine drink, and more may be needed during heat or activity."
        )
    if any(word in lowered for word in ("medicine", "medication", "tablet", "dose")):
        facts.append(
            "Do not recommend a medicine or dose. Direct the student to a parent or "
            "guardian and a clinician or pharmacist who can check age, weight, health "
            "conditions, allergies, and other medicines."
        )
    if not facts:
        return ""
    return (
        "\n\nIMPORTANT VERIFIED FACTS FOR THIS QUESTION:\n- "
        + "\n- ".join(facts)
        + "\nUse these exact facts and do not contradict or replace their numbers."
    )


def quick_answer(question: str) -> str | None:
    """Answer common, well-defined health questions without invoking the LLM."""
    text = " ".join(question.lower().split())
    age_match = re.search(
        r"\b(?:age[sd]?\s*)?(\d{1,2})(?:-year-old|\s*years?\s*old)?\b", text
    )
    age = int(age_match.group(1)) if age_match else None

    if any(word in text for word in ("sleep", "bedtime", "hours of sleep")):
        if age is not None and 6 <= age <= 12:
            return (
                f"At age {age}, aim for 9-12 hours of sleep each day. Keep a "
                "regular bedtime, avoid screens before bed, and tell a trusted "
                "adult if tiredness continues despite enough sleep."
            )
        if age is not None and 13 <= age <= 18:
            return (
                f"At age {age}, aim for 8-10 hours of sleep each day. Keep a "
                "regular bedtime and reduce bright screens and caffeine near bedtime."
            )
        return (
            "Children ages 6-12 generally need 9-12 hours of sleep, while teens "
            "ages 13-18 need 8-10 hours. A regular bedtime and less screen time "
            "before bed can make sleep easier."
        )
    if any(word in text for word in ("exercise", "activity", "workout", "sport")):
        return (
            "Young people ages 5-17 should aim for about 60 minutes of moderate-to-"
            "vigorous activity each day. Start gradually with walking, play, sports, "
            "or another activity you enjoy."
        )
    if any(word in text for word in ("bmi", "underweight", "overweight")):
        return (
            "For ages 2-19, BMI must be interpreted using age- and sex-specific "
            "BMI-for-age percentiles. It is a screening measure, not a diagnosis, "
            "so discuss growth concerns with a qualified health professional."
        )
    if any(word in text for word in ("water", "hydration", "dehydrated")):
        return (
            "Water is usually the best everyday drink. The amount needed varies "
            "with age, body size, weather, illness, and activity, so drink regularly "
            "and take extra water during heat or exercise."
        )
    if any(word in text for word in ("medicine", "medication", "tablet", "dose")):
        return (
            "Ask a parent or guardian and a clinician or pharmacist before taking "
            "medicine. The safe choice and dose depend on age, weight, allergies, "
            "health conditions, and other medicines."
        )
    return None


def is_health_related(question: str, history: list[dict[str, str]]) -> bool:
    """Allow health questions, conversational greetings, and health follow-ups."""
    text = " ".join(question.strip().split())
    if NON_HEALTH_TASK_PATTERN.search(text):
        return False
    # Greetings are sent to the language model so its reply is conversational,
    # not a canned response from the app or API.
    if SOCIAL_PATTERN.fullmatch(text):
        return True
    if HEALTH_TOPIC_PATTERN.search(text):
        return True
    # Preserve a small set of unambiguous follow-ups to an existing health chat.
    return bool(history and FOLLOW_UP_PATTERN.fullmatch(text))


app = FastAPI(title="BMI School Wellness Local AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=100)
    reset: bool = False


class BmiTipRequest(BaseModel):
    bmi: float = Field(gt=5, lt=100)
    age: int | None = Field(default=None, ge=2, le=120)


class ChatResponse(BaseModel):
    reply: str
    kind: Literal["local_ai", "safety", "off_topic"] = "local_ai"


FOOD_CATALOG = {
    "4800016645407": {
        "name": "Sample instant noodles",
        "summary": "Often high in sodium; check the package serving size.",
        "alternative": "Try pancit with vegetables and egg, using less seasoning.",
    },
    "4800361411108": {
        "name": "Sample sweetened drink",
        "summary": "A sweetened drink can add sugar without much fullness.",
        "alternative": "Choose cold water, unsweetened calamansi water, or plain milk.",
    },
}


class LocalHealthModel:
    """Load the local model once and serialize generation for safe CPU/GPU use."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.device = "loading"
        self.loading = False
        self.load_error: str | None = None
        self.is_seq2seq = ON_RENDER or "t5" in MODEL_NAME.lower()
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            self.loading = True
            self.load_error = None
            try:
                # Importing the ML stack is expensive. Keeping it here lets Uvicorn
                # begin serving health checks and quick answers immediately.
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )

                self.torch = torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Loading local AI model: {MODEL_NAME}", flush=True)
                local_only = os.getenv("HF_LOCAL_FILES_ONLY", "0") == "1"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME, local_files_only=local_only
                )
                # Render's free service has 512 MB of RAM. FLAN-T5 Small is kept
                # in bfloat16 there so the model remains generative and fits beside
                # Python, FastAPI, PyTorch, and tokenizer memory.
                dtype = (
                    torch.float16 if self.device == "cuda"
                    else torch.bfloat16 if ON_RENDER
                    else torch.float32
                )
                model_class = (
                    AutoModelForSeq2SeqLM if self.is_seq2seq
                    else AutoModelForCausalLM
                )
                model = model_class.from_pretrained(
                    MODEL_NAME,
                    dtype=dtype,
                    local_files_only=local_only,
                    low_cpu_mem_usage=ON_RENDER,
                )
                model.to(self.device)
                model.eval()
                self.model = model
                print(f"Local AI ready on {self.device}", flush=True)
            except Exception as error:
                self.load_error = str(error)
                raise
            finally:
                self.loading = False

    def warm_up(self) -> None:
        try:
            self._load()
        except Exception as error:
            print(f"Local AI failed to load: {error}", flush=True)

    def answer(self, history: list[dict[str, str]], question: str) -> str:
        self._load()
        conversation_text = " ".join(
            item["content"] for item in history if item["role"] == "user"
        )
        grounding = trusted_facts(f"{conversation_text} {question}")
        with self._generation_lock, self.torch.inference_mode():
            if self.is_seq2seq:
                turns = [SYSTEM_PROMPT + grounding]
                for item in history:
                    speaker = "Student" if item["role"] == "user" else "Health Buddy"
                    turns.append(f'{speaker}: {item["content"]}')
                turns.append(f"Student: {question}\nHealth Buddy:")
                prompt = "\n".join(turns)
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT + grounding},
                    *history,
                    {"role": "user", "content": question},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            reply_tokens = (
                output[0]
                if self.is_seq2seq
                else output[0, inputs["input_ids"].shape[1]:]
            )
            reply = self.tokenizer.decode(
                reply_tokens, skip_special_tokens=True
            ).strip()
        if not reply:
            raise RuntimeError("The local model returned an empty response.")
        # A small local model can occasionally copy an adult/child sleep range to the
        # wrong age. Reject that contradiction instead of showing unsafe information.
        age_match = re.search(r"\b(?:age[sd]?\s*)?(\d{1,2})(?:-year-old|\s*years?\s*old)?\b", question.lower())
        if age_match and any(word in question.lower() for word in ("sleep", "hours")):
            age = int(age_match.group(1))
            expected = "9-12" if 6 <= age <= 12 else "8-10" if 13 <= age <= 18 else None
            if expected and expected not in reply.replace("–", "-").replace("—", "-"):
                raise RuntimeError("The local model contradicted the verified sleep range.")
        return reply


health_model = LocalHealthModel()
conversations: defaultdict[str, deque[dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=MAX_TURNS * 2)
)


@app.on_event("startup")
def start_model_warmup() -> None:
    if os.getenv("DISABLE_MODEL_WARMUP", "0") != "1":
        threading.Thread(target=health_model.warm_up, daemon=True).start()


def is_urgent(message: str) -> bool:
    text = " ".join(message.lower().split())
    return any(term in text for term in URGENT_TERMS)


def conversation_key(payload: ChatRequest, request: Request) -> str:
    if payload.session_id:
        return payload.session_id
    return request.client.host if request.client else "local"


@app.get("/health")
def health() -> dict[str, str | bool]:
    result: dict[str, str | bool] = {
        "status": "ok", "service": "school-health-buddy", "ai": "local",
        "model": MODEL_NAME, "model_loaded": health_model.model is not None,
        "model_loading": health_model.loading,
    }
    if health_model.load_error:
        result["model_error"] = health_model.load_error
    return result


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "School Health Buddy local AI is running",
        "health_check": "/health", "documentation": "/docs",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    key = conversation_key(payload, request)
    if payload.reset:
        conversations.pop(key, None)
    if is_urgent(payload.message):
        return ChatResponse(reply=URGENT_REPLY, kind="safety")
    history = list(conversations[key])
    if not is_health_related(payload.message, history):
        return ChatResponse(reply=OFF_TOPIC_REPLY, kind="off_topic")
    reply = await asyncio.to_thread(health_model.answer, history, payload.message)
    conversations[key].append({"role": "user", "content": payload.message})
    conversations[key].append({"role": "assistant", "content": reply})
    return ChatResponse(reply=reply)


@app.post("/api/bmi-tip")
def bmi_tip(payload: BmiTipRequest) -> dict[str, str]:
    if payload.age is not None and payload.age < 20:
        tip = (
            f"A BMI of {payload.bmi:.1f} for someone age {payload.age} must be checked "
            "on an age- and sex-specific BMI-for-age growth chart. BMI alone cannot "
            "diagnose health; discuss growth patterns with a parent or guardian and a "
            "qualified health professional."
        )
    elif payload.age is None:
        tip = (
            f"Your calculated BMI is {payload.bmi:.1f}. To interpret it accurately, age "
            "is needed because children and teens use BMI-for-age percentiles rather "
            "than adult cutoffs."
        )
    else:
        tip = (
            f"Your calculated BMI is {payload.bmi:.1f}. BMI is only a screening measure "
            "and should be considered with medical history and other health information."
        )
    return {"tip": tip}


@app.get("/api/nutrition/{barcode}")
def nutrition(barcode: str) -> dict[str, str]:
    return FOOD_CATALOG.get(
        barcode,
        {
            "name": f"Product {barcode}",
            "summary": "Not yet in the school catalog. Check serving size, sugar, sodium, protein, and fiber on the label.",
            "alternative": "Try fruit, boiled egg, corn, peanuts, monggo, or water for an affordable option.",
        },
    )


def local_ip_address() -> str:
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect(("8.8.8.8", 80))
        return str(connection.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        connection.close()


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8000")))
    lan_ip = local_ip_address()
    print("\nSchool Health Buddy local AI", flush=True)
    print(f"Local:   http://127.0.0.1:{port}", flush=True)
    print(f"Network: http://{lan_ip}:{port}", flush=True)
    print(f"Docs:    http://{lan_ip}:{port}/docs\n", flush=True)
    uvicorn.run(app, host=host, port=port)
