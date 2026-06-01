# Evaluator

A [Streamlit](https://streamlit.io/) application for comparing and evaluating responses from multiple large language models, and for running structured, multi-agent web research.

The app combines two tools in a single interface:

- **Compare Models** — send the same prompt to up to three models (OpenAI, Anthropic, Gemini) at once, view their responses side by side, and optionally have an impartial LLM judge score each response against a rubric.
- **Research Agent** — a multi-agent research assistant that scopes a request with you, plans the work, and fans out to specialised web-search agents to produce structured research notes.

## Features

### Compare Models
- Select up to **3 models** across OpenAI, Anthropic, and Gemini providers from a single multiselect.
- Models are queried **concurrently** (via `asyncio`) and rendered in side-by-side columns.
- Show/hide individual model outputs after the chat has started.
- Maintains independent conversation history per model across turns.
- Optional **evaluation mode**: an evaluator agent scores every response 1–5 on instruction following, correctness, completeness, reasoning quality, practical usefulness, and clarity, with strengths, weaknesses, and an explanation for each.

### Research Agent
A pipeline of cooperating agents that route work between each other:
- **Research Scope** — clarifies the request, confirms context with you, and decides whether the task is simple or complex.
- **Generic Search** — answers straightforward requests with a single web search.
- **Search Planner** — breaks complex requests into up to 5 workstreams and assigns each to a specialised researcher (profile, organisation, news, market, technical).
- **Researcher** — runs the planned searches in parallel using the OpenAI web-search tool and returns structured, source-grounded notes.
- **Summarise & save** — condense the whole research conversation into a single titled document, make it downloadable as Markdown, **and** push it to a Notion database in one click.

### Other
- **Generated files sidebar** for downloading produced content as Markdown.
- Optional **password gate** via a `PASSWORD` environment variable.
- Optional **Notion integration** (`utils/store.py`) — summarised research documents are saved to a Notion data source (with title and topic) when Notion credentials are configured.

## Project structure

```
evaluator/
├── main.py                       # Entry point: auth, config, session init, page navigation
├── pages_app/
│   ├── compare_models.py         # Multi-model chat + evaluator UI
│   ├── research.py               # Research agent UI
│   └── research_prompts.yaml     # System prompts for the specialised researchers
├── utils/
│   ├── utility.py                # Session state, config loading, auth, helpers
│   ├── openai.py                 # OpenAI chat agent
│   ├── anthropic.py              # Anthropic (Claude) chat agent
│   ├── gemini.py                 # Gemini chat agent
│   ├── evaluator.py              # Rubric-based evaluator agent
│   ├── research_scope.py         # Scoping / routing agent
│   ├── search_planner.py         # Breaks research into workstreams
│   ├── generic_search.py         # Single-query web search agent
│   ├── researcher.py             # Web-search worker agent
│   ├── summarise.py              # Summarises a research chat into a titled document
│   └── store.py                  # Saves documents to a Notion database
├── requirements.txt
└── .env.example
```

## Getting started

### Prerequisites
- Python 3.10+
- API keys for the providers you intend to use (OpenAI, Anthropic, and/or Gemini).

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
| --- | --- |
| `PASSWORD` | Optional. If set, gates the app behind a password. Leave blank to disable. |
| `OPENAI_API_KEY` | OpenAI API key. |
| `ANTHROPIC_API_KEY` | Anthropic API key. |
| `GEMINI_API_KEY` | Gemini API key. |
| `OPENAI_MODELS` | Comma-separated list of OpenAI models to offer. |
| `GEMINI_MODELS` | Comma-separated list of Gemini models to offer. |
| `ANTHROPIC_MODELS` | Comma-separated list of Anthropic models to offer. |
| `NOTION_API_KEY` | Optional. For saving notes to Notion. |
| `NOTION_DATA_SOURCE_ID` | Optional. Target Notion data source. |

The models listed in `*_MODELS` populate the model picker on the Compare Models page.

### Running

```bash
streamlit run main.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Usage

**Compare Models**
1. Pick up to 3 models and optionally toggle evaluation on.
2. Type a prompt — responses stream into side-by-side columns.
3. With evaluation on, expand each response's evaluation panel to see rubric scores and comments.
4. Use *"Start new chat and choose models again"* to reset and reselect models.

**Research Agent**
1. Describe what you want to research.
2. The scope agent will ask clarifying questions and confirm the research statement.
3. Once scoped, it routes to a single search or a planned set of specialised researchers.
4. Results are returned as structured notes in the chat.
5. Click *"Summarise chat into document"* in the sidebar to condense the conversation into a downloadable Markdown file; if Notion is configured, the document is also saved to your Notion database.

## Notes
- Model names in `.env.example` are placeholders — replace them with models available to your API keys.
