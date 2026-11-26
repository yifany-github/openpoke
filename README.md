# OpenPoke 🌴

OpenPoke is a simplified, open-source take on [Interaction Company’s](https://interaction.co/about) [Poke](https://poke.com/) assistant—built to show how a multi-agent orchestration stack can feel genuinely useful. It keeps the handful of things Poke is great at (email triage, reminders, and persistent agents) while staying easy to spin up locally.

- Multi-agent FastAPI backend that mirrors Poke's interaction/execution split, powered by [OpenRouter](https://openrouter.ai/).
- Gmail tooling via [Composio](https://composio.dev/) for drafting/replying/forwarding without leaving chat.
- Trigger scheduler and background watchers for reminders and "important email" alerts.
- Next.js web UI that proxies everything through the shared `.env`, so plugging in API keys is the only setup.

## This fork: fixing agent overload

Execution agents in OpenPoke persist forever but were never managed:
- the roster grew without bound and was injected, as bare names, into every turn;
- each agent's full history was re-sent on every run.

This fork bounds both:
- a metadata registry, with a relevance-ranked top-K roster with descriptions,
  `search_agents`, and duplicate-name detection;
- rolling memory compaction for execution agents.

On an independent 300-agent workload, correct routing goes from **50% to 88%** (Sonnet 5.5,
paired p < 0.001). Execution prompts stay under ~15k tokens however often an agent is
reused.

| Read | For |
|---|---|
| [DESIGN.md](DESIGN.md) | the problem, the fix, the results (5-minute read) |
| [EVALUATION.md](EVALUATION.md) | methodology, ablations, caveats, how the prompt evolved |
| [OTHER_FINDINGS.md](OTHER_FINDINGS.md) | 21 other issues found in OpenPoke, including a retired default model that stopped the app from running |

- **Code:** `server/services/execution/{roster,agent_ranker,agent_memory}.py`, `server/agents/interaction_agent/{agent,tools}.py`
- **Run the tests and evaluators:**
  ```bash
  python3.11 -m venv .venv && .venv/bin/pip install -r server/requirements.txt pytest tiktoken
  .venv/bin/python -m pytest                              # 67 tests, network blocked
  .venv/bin/python -m evals.prompt_growth                 # prompt size vs. #agents and #reuses
  .venv/bin/python -m evals.routing_offline --vocab fresh # routing recall, K sweep, ablations
  .venv/bin/python -m evals.routing_llm --vocab fresh --n 40 --yes --delay 2   # live (needs OPENROUTER_API_KEY)
  ```
- **New settings** (all optional): `OPENPOKE_{INTERACTION,EXECUTION,SEARCH,SUMMARIZER,CLASSIFIER}_MODEL`,
  `OPENPOKE_AGENT_ROSTER_LIMIT` (20), `OPENPOKE_AGENT_HISTORY_WINDOW` (5),
  `OPENPOKE_AGENT_COMPACTION_BATCH` (5), `OPENPOKE_MAX_OUTPUT_TOKENS` (4096),
  `OPENPOKE_LLM_MAX_CONCURRENCY` (8), `OPENPOKE_LLM_TIMEOUT` (60 s), `OPENPOKE_TURN_TIMEOUT` (180 s).

## Requirements
- Python 3.10+
- Node.js 18+
- npm 9+

## Quickstart
1. **Clone and enter the repo.**
   ```bash
   git clone https://github.com/shlokkhemani/OpenPoke
   cd OpenPoke
   ```
2. **Create a shared env file.** Copy the template and open it in your editor:
   ```bash
   cp .env.example .env
   ```
3. **Get your API keys and add them to `.env`:**
   
   **OpenRouter (Required)**
   - Create an account at [openrouter.ai](https://openrouter.ai/)
   - Generate an API key
   - Replace `your_openrouter_api_key_here` with your actual key in `.env`
   
   **Composio (Required for Gmail)**
   - Sign in at [composio.dev](https://composio.dev/)
   - Create an API key
   - Set up Gmail integration and get your auth config ID
   - Replace `your_composio_api_key_here` and `your_gmail_auth_config_id_here` in `.env`
4. **(Required) Create and activate a Python 3.10+ virtualenv:**
   ```bash
   # Ensure you're using Python 3.10+
   python3.10 -m venv .venv
   source .venv/bin/activate
   
   # Verify Python version (should show 3.10+)
   python --version
   ```
   On Windows (PowerShell):
   ```powershell
   # Use Python 3.10+ (adjust path as needed)
   python3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # Verify Python version
   python --version
   ```

5. **Install backend dependencies:**
   ```bash
   pip install -r server/requirements.txt
   ```
6. **Install frontend dependencies:**
   ```bash
   npm install --prefix web
   ```
7. **Start the FastAPI server:**
   ```bash
   python -m server.server --reload
   ```
8. **Start the Next.js app (new terminal):**
   ```bash
   npm run dev --prefix web
   ```
9. **Connect Gmail for email workflows.** With both services running, open [http://localhost:3000](http://localhost:3000), head to *Settings → Gmail*, and complete the Composio OAuth flow. This step is required for email drafting, replies, and the important-email monitor.

The web app proxies API calls to the Python server using the values in `.env`, so keeping both processes running is required for end-to-end flows.

## Project Layout
- `server/` – FastAPI application and agents
- `web/` – Next.js app
- `server/data/` – runtime data (ignored by git)

## License
MIT — see [LICENSE](LICENSE).
