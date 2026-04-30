"""
Phase 3: AgentTestPhase — autonomous page testing via browser-use Agent.

The agent reads the page description, observes the page visually, and
autonomously interacts with all features.  Results include a textual
summary, screenshots, action list, and errors.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext, PhaseResult
from htmleval.core.phase import Phase
from htmleval.phases.agent_test.keyscan import discover_keys
from htmleval.phases.agent_test.strategy import build_interaction_guide

logger = logging.getLogger("htmleval")

_RETRY_MAX = 5
_RETRY_BACKOFF = [5, 10, 15, 20, 30]

_BROWSER_ARGS = [
    "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-ui-for-media-stream",
]


def _is_resource_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        isinstance(exc, BlockingIOError)
        or "resource temporarily unavailable" in msg
        or "errno 11" in msg
        or "pthread_create" in msg
        or "cannot allocate memory" in msg
        or "connection closed while reading from the driver" in msg
        or "target page, context or browser has been closed" in msg
        or "target crashed" in msg
        or ("spawn" in msg and "eagain" in msg)
        or "browser closed" in msg
        or "browser has been closed" in msg
        or ("browserstartevent" in msg and "timed out" in msg)
        or ("cdp connection" in msg and "timed out" in msg)
        or "timed out during opening handshake" in msg
        or "connecttimeout" in msg
        or "connectionerror" in msg
        or "remotedisconnected" in msg
        or "connection refused" in msg
    )


def _build_task_prompt(
    page_url: str,
    query: str,
    input_types: List[str],
    discovered_keys: Optional[List[str]] = None,
    game_vars: Optional[dict] = None,
) -> str:
    """Compose the universal agent task prompt for any HTML page."""
    interaction_guide = build_interaction_guide(
        input_types,
        discovered_keys=discovered_keys,
        game_vars=game_vars,
    )

    return f"""You are an automated HTML page quality tester. Thoroughly test the page and produce a detailed report.

Visit this URL: {page_url}

## What this page should do:
{query}

## Testing procedure:
1. Load the page — observe initial rendering. Any blank areas, broken layout, or visible errors?
2. Activate entry points: "Start", "Submit", "Play", "Enter", "OK" buttons or similar.
3. Test every feature mentioned in the description above:
   - Click all buttons and interactive elements
   - Fill in any forms or input fields and submit them
   - Test navigation (menus, tabs, pagination, back/next)
   - If there is animated or canvas-based content, observe and interact with it
   - If there are multiple modes, states, or views — navigate through them
4. Complete at least one full user workflow from start to finish.
5. Observe and note: unresponsive elements, visual glitches, broken layout, on-page error messages, missing content.

{interaction_guide}

## Your report (write in English):
1. **Rendering** — how does the initial page look? Any blank/broken areas?
2. **Feature status** — for each feature in the description: working / partial / broken / missing
3. **Bug list** — be specific (what happened vs. what should have happened)
4. **Missing features** — described but not present or non-functional
5. **Overall quality**: Excellent / Good / Fair / Poor / Broken"""


async def _safe_agent_run(agent: Any, max_steps: int) -> Any:
    """Run browser-use agent, converting its internal TimeoutError → RuntimeError."""
    try:
        return await agent.run(max_steps=max_steps)
    except (TimeoutError, asyncio.TimeoutError) as e:
        raise RuntimeError(f"browser-use internal timeout: {e}") from e


def _extract_history(history: Any, output_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "agent_summary": "", "screenshots": [],
        "actions": [], "errors": [], "steps_taken": 0,
    }
    if history is None:
        return result

    try:
        result["agent_summary"] = history.final_result() or ""
    except Exception:
        try:
            result["agent_summary"] = str(history)
        except Exception:
            pass

    try:
        all_paths = history.screenshot_paths(return_none_if_not_screenshot=False)
        valid = [p for p in (all_paths or []) if p and Path(p).exists()]
        for i, src in enumerate(valid):
            dest = str(output_dir / f"screenshot_agent_{i}.png")
            shutil.copy2(src, dest)
            result["screenshots"].append(dest)
    except Exception as e:
        logger.warning(f"Failed to extract agent screenshots: {e}")

    try:
        result["actions"] = history.action_names() or []
    except Exception:
        pass

    try:
        errs = history.errors()
        result["errors"] = [str(e) for e in errs] if errs else []
    except Exception:
        pass

    try:
        result["steps_taken"] = history.number_of_steps()
    except Exception:
        try:
            result["steps_taken"] = len(getattr(history, "agent_steps", []))
        except Exception:
            pass

    return result


class AgentTestPhase(Phase):
    """Phase 3 — browser-use Agent autonomously tests the HTML page."""

    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self._agent_llm: Any = None

    @property
    def name(self) -> str:
        return "agent_test"

    def gate(self, ctx: EvalContext) -> bool:
        if ctx.should_skip:
            return False
        render = ctx.get_phase("render_test")
        if render is not None and render.data.get("rendered") is False:
            ctx.skip_reason = "render_test: page did not render"
            return False
        return True

    async def execute(self, ctx: EvalContext) -> PhaseResult:
        from browser_use import Agent, Browser

        cfg = self.config.agent
        input_types = ctx.get_phase("static_analysis")
        input_types = input_types.data.get("input_types", []) if input_types else []

        # ── Pre-scan: discover working keys before running the LLM agent ──
        discovered_keys: List[str] = []
        game_vars: dict = {}
        if "keyboard" in input_types and ctx.game_url_http:
            logger.info("[agent] %s: running key discovery scan…", ctx.game_id)
            try:
                scan = await asyncio.wait_for(
                    discover_keys(ctx.game_url_http, list(_BROWSER_ARGS)),
                    timeout=20,
                )
                discovered_keys = scan.get("working", [])
                game_vars       = scan.get("game_vars", {})
                if scan.get("error"):
                    logger.warning("[agent] key scan error: %s", scan["error"])
                else:
                    logger.info("[agent] %s: discovered keys=%s  game_vars=%s",
                                ctx.game_id, discovered_keys, list(game_vars.keys()))
            except asyncio.TimeoutError:
                logger.warning("[agent] %s: key scan timed out", ctx.game_id)
            except Exception as e:
                logger.warning("[agent] %s: key scan exception: %s", ctx.game_id, e)

        task_prompt = _build_task_prompt(
            ctx.game_url_http, ctx.query, input_types,
            discovered_keys=discovered_keys,
            game_vars=game_vars,
        )
        llm = self._get_or_create_llm(cfg)

        result_data: Dict[str, Any] = {
            "agent_completed": False, "agent_summary": "",
            "steps_taken": 0, "actions": [], "errors": [],
            "discovered_keys": discovered_keys,  # from pre-scan
            "game_vars_initial": game_vars,       # from pre-scan
        }
        error_bucket: List[str] = []
        history: Optional[Any] = None

        def _make_browser():
            return Browser(
                headless=True, disable_security=True,
                enable_default_extensions=False, args=list(_BROWSER_ARGS),
            )

        def _make_agent(br):
            return Agent(
                task=task_prompt, llm=llm, browser=br,
                use_vision=True, max_actions_per_step=5,
                generate_gif=False, llm_timeout=cfg.llm_timeout,
            )

        browser = _make_browser()
        agent   = _make_agent(browser)

        for attempt in range(_RETRY_MAX + 1):
            try:
                agent_task = asyncio.create_task(
                    asyncio.wait_for(_safe_agent_run(agent, cfg.max_steps), timeout=cfg.timeout)
                )
                await asyncio.sleep(3)
                history = await agent_task
                result_data["agent_completed"] = True
                break

            except asyncio.TimeoutError:
                error_bucket.append(f"Agent timed out after {cfg.timeout}s")
                logger.warning(f"Agent timed out after {cfg.timeout}s — extracting partial results")
                if agent and hasattr(agent, "history"):
                    history = agent.history
                break

            except Exception as e:
                err_str = str(e)
                is_resource = _is_resource_error(e)

                if attempt < _RETRY_MAX and is_resource:
                    wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                    logger.warning(f"Agent resource error (attempt {attempt+1}), retry in {wait}s: {e}")
                    browser = _make_browser()
                    agent   = _make_agent(browser)
                    await asyncio.sleep(wait)
                    continue

                if "browser-use internal timeout" in err_str and not is_resource:
                    error_bucket.append(err_str)
                    if agent and hasattr(agent, "history"):
                        history = agent.history
                    break

                error_bucket.append(f"{type(e).__name__}: {err_str}")
                logger.error(f"Agent error: {traceback.format_exc()}")
                if agent and hasattr(agent, "history"):
                    history = agent.history
                break

        extracted = _extract_history(history, ctx.output_dir)
        result_data["agent_summary"] = extracted["agent_summary"]
        result_data["actions"]       = extracted["actions"]
        result_data["errors"]        = extracted["errors"] + error_bucket
        result_data["steps_taken"]   = extracted["steps_taken"]

        if result_data["agent_summary"] and not result_data["agent_completed"]:
            result_data["agent_completed"] = True

        if browser:
            try:
                await asyncio.wait_for(browser.close(), timeout=10)
            except Exception:
                pass

        return PhaseResult(
            phase_name=self.name,
            success=result_data["agent_completed"],
            data=result_data,
            screenshots=extracted["screenshots"],
            errors=error_bucket,
        )

    def _get_or_create_llm(self, cfg: Any) -> Any:
        if self._agent_llm is None:
            from browser_use import ChatOpenAI as BrowserUseChatOpenAI
            self._agent_llm = BrowserUseChatOpenAI(
                base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model,
            )
        return self._agent_llm
