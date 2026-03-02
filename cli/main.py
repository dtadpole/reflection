"""CLI entry point for the reflection system."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from agenix.config import ReflectionConfig, load_config, make_log_path
from agenix.storage.fs_backend import FSBackend

app = typer.Typer(name="reflection", help="Self-evolving multi-agent coding system.")
agent_app = typer.Typer(help="Run individual agents.")
queues_app = typer.Typer(help="Queue management.")
cards_app = typer.Typer(help="Manage knowledge cards.")
experiences_app = typer.Typer(help="Manage agent experiences.")
logs_app = typer.Typer(help="View execution logs.")
services_app = typer.Typer(help="Manage remote services.")
tunnel_app = typer.Typer(help="Manage SSH tunnels for port forwarding.")
app.add_typer(agent_app, name="agent")
app.add_typer(queues_app, name="queues")
app.add_typer(cards_app, name="cards")
app.add_typer(experiences_app, name="experiences")
app.add_typer(logs_app, name="logs")
app.add_typer(services_app, name="services")
services_app.add_typer(tunnel_app, name="tunnel")


def _load_config(
    config_path: Optional[Path] = None,
    env: Optional[str] = None,
) -> ReflectionConfig:
    """Load config, optionally overriding the environment."""
    cfg = load_config(config_path)
    if env:
        cfg.storage.env = env
    return cfg


def _make_run_tag(agent_name: str = "run") -> str:
    """Generate a run tag from the current timestamp.

    Format: <agent_name>_YYYYMMDD_HHMMSS (e.g. solver_20260302_004300).
    """
    return agent_name + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _find_endpoint_by_name(config: ReflectionConfig, name: str):
    """Find a service endpoint by name, or return None."""
    for ep in config.services.endpoints:
        if ep.name == name:
            return ep
    return None


def _bootstrap(config: ReflectionConfig, run_tag: str | None = None):
    """Create all runtime objects for pipeline execution.

    Returns (pipeline, runner, fs_backend).

    Endpoint wiring:
    - _one: kb_eval (verifier)
    - _two: text_embedding (RemoteEmbedder) + reranker (retriever rerank variant)
    """
    from agenix.pipeline import Pipeline
    from agenix.runner import ClaudeRunner
    from agenix.tools.loader import load_tool
    from agenix.tools.registry import ToolRegistry
    from tools.knowledge.baseline.store import KnowledgeStore

    fs = FSBackend(config.storage)
    fs.initialize()

    run_dir = config.storage.run_path(run_tag) if run_tag else None
    experiences_dir = config.storage.experiences_path

    ep_two = _find_endpoint_by_name(config, "_two")
    ep_one = _find_endpoint_by_name(config, "_one")

    # Build KnowledgeStore — use RemoteEmbedder when _two is available
    if ep_two:
        from tools.knowledge.baseline.embedder import RemoteEmbedder
        from tools.knowledge.baseline.index import LanceIndex

        embedder = RemoteEmbedder(config=ep_two.text_embedding, dimension=4096)
        lance = LanceIndex(db_path=config.storage.lance_path, vector_dim=4096)
        store = KnowledgeStore(
            config=config, fs_backend=fs, lance_index=lance, embedder=embedder,
        )
    else:
        store = KnowledgeStore(config=config, fs_backend=fs)
    store.initialize()

    registry = ToolRegistry()

    # Load retriever tool — use rerank variant when reranker is available on _two
    if ep_two:
        from services.reranker.baseline.client import RerankerClient

        rr_client = RerankerClient(ep_two.reranker)
        retriever_def = load_tool("retriever", variant="rerank")
        registry.register(
            retriever_def.create_fn(
                knowledge_store=store, reranker_client=rr_client
            )
        )
    else:
        retriever_def = load_tool("retriever", variant="baseline")
        registry.register(retriever_def.create_fn(knowledge_store=store))

    # Load verifier tool (kb_eval variant) when _one is available
    if ep_one:
        from services.kb_eval.baseline.client import KbEvalClient

        kb_client = KbEvalClient(ep_one.kb_eval)
        verifier_def = load_tool("verifier", variant="kb_eval")
        registry.register(verifier_def.create_fn(kb_eval_client=kb_client))

    runner = ClaudeRunner(
        tool_registry=registry, run_dir=run_dir, experiences_dir=experiences_dir,
    )
    pipeline = Pipeline(
        config=config,
        runner=runner,
        knowledge_store=store,
        fs_backend=fs,
    )
    return pipeline, runner, fs


# --- Top-level commands ---


@app.command()
def orchestrate(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment (prod/int/test_<user>)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Run all agents as managed subprocesses."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)

    from agenix.orchestrator import Orchestrator

    orch = Orchestrator(
        cfg.orchestrator,
        env=cfg.storage.env,
        config_path=str(config) if config else None,
        verbose=verbose,
        logs_dir=cfg.storage.logs_path,
    )
    exit_code = orch.run()
    raise typer.Exit(exit_code)


@app.command()
def run(
    iterations: int = typer.Option(1, "-n", "--iterations", help="Number of iterations to run."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment (prod/int/test_<user>)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Run N iterations of the full reflection pipeline."""
    _setup_logging(verbose)
    log = logging.getLogger("reflection.run")
    cfg = _load_config(config, env)
    pipeline, _, _ = _bootstrap(cfg)

    succeeded = 0
    failed = 0
    for i in range(1, iterations + 1):
        run_tag = _make_run_tag()
        typer.echo(f"--- Iteration {i}/{iterations} [{run_tag}] ---")
        try:
            result = pipeline.run_iteration(run_tag, iteration=i)
            status = "correct" if result.is_correct else "incorrect"
            typer.echo(
                f"  Problem: {result.problem_id} | "
                f"Experience: {result.experience_id} | "
                f"Result: {status} | "
                f"Cards: {len(result.cards_created)}"
            )
            succeeded += 1
        except Exception as exc:
            log.error("Iteration %d failed: %s", i, exc, exc_info=True)
            typer.echo(f"  FAILED: {exc}", err=True)
            failed += 1

    typer.echo(f"Done. {succeeded} succeeded, {failed} failed.")


@app.command()
def solve(
    description: str = typer.Argument(..., help="Problem description to solve."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Solve a single problem using the solver agent."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)
    pipeline, _, fs = _bootstrap(cfg)

    from agenix.storage.models import Problem

    problem = Problem(title="Ad-hoc Problem", description=description, domain="general")
    fs.save_problem(problem)

    run_tag = _make_run_tag()
    experience = pipeline._run_solver(run_tag, problem)

    typer.echo(f"Experience: {experience.experience_id}")
    typer.echo(f"Correct: {experience.is_correct}")
    if experience.code_solution:
        typer.echo(f"Solution:\n{experience.code_solution}")


@app.command()
def status(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Show system status: counts of problems, experiences, and cards."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)
    fs = FSBackend(cfg.storage)

    problems = fs.count_problems()
    experiences = fs.count_experiences()
    cards_knowledge = fs.count_cards("knowledge")
    cards_reflection = fs.count_cards("reflection")
    cards_insight = fs.count_cards("insight")

    typer.echo(f"Environment: {cfg.storage.env}")
    typer.echo(f"Data root:   {cfg.storage.env_path}")
    typer.echo(f"Problems:    {problems}")
    typer.echo(f"Experiences:  {experiences}")
    typer.echo(f"Cards:       {cards_knowledge + cards_reflection + cards_insight}")
    typer.echo(f"  Knowledge: {cards_knowledge}")
    typer.echo(f"  Reflection: {cards_reflection}")
    typer.echo(f"  Insight:   {cards_insight}")


# --- Agent sub-commands ---


@agent_app.command("curator")
def agent_curator(
    n: int = typer.Option(100, "-n", help="Number of problems to sample."),
    levels: Optional[str] = typer.Option(
        None, "--levels", help="Comma-separated levels (level_1,level_2,level_3,level_4)."
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Load problems from KernelBench and enqueue them."""
    cfg = _load_config(config, env)
    _setup_logging(verbose, log_file=make_log_path(cfg.storage.logs_path, "curator"))
    fs = FSBackend(cfg.storage)
    fs.initialize()

    from ulid import ULID

    from agenix.agents.curator_handler import run_curator
    from agenix.queue.fs_queue import FSQueue

    queue = FSQueue("problems", cfg.storage)
    level_list = levels.split(",") if levels else None
    experience_id = str(ULID())
    conv_path = cfg.storage.experiences_path / "curator" / f"{experience_id}.jsonl"
    problems = run_curator(
        fs, queue, n=n, levels=level_list, seed=seed, conversation_path=conv_path,
    )
    typer.echo(f"Created {len(problems)} problems.")


@agent_app.command("solver")
def agent_solver(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Take one problem from the queue, solve it, and exit."""
    cfg = _load_config(config, env)
    _setup_logging(verbose, log_file=make_log_path(cfg.storage.logs_path, "solver"))
    run_tag = _make_run_tag("solver")
    pipeline, runner, fs = _bootstrap(cfg, run_tag=run_tag)

    from agenix.agents.solver_handler import SolverHandler
    from agenix.queue.fs_queue import FSQueue

    problems_q = FSQueue("problems", cfg.storage)
    problems_q.initialize()
    experiences_q = FSQueue("experiences", cfg.storage)

    message = problems_q.dequeue()
    if message is None:
        typer.echo("No problems in queue.")
        raise typer.Exit()

    handler = SolverHandler(
        runner=runner,
        fs_backend=fs,
        knowledge_store=pipeline.store,
        experiences_queue=experiences_q,
        run_tag=run_tag,
    )

    typer.echo(
        f"Solving: {message.payload.get('title', message.message_id)} "
        f"(run_tag={run_tag})"
    )
    try:
        handler.handle(message)
        problems_q.complete(message.message_id)
        typer.echo("Solver finished successfully.")
    except Exception as exc:
        problems_q.fail(message.message_id, error=str(exc))
        typer.echo(f"Solver failed: {exc}", err=True)
        raise typer.Exit(code=1)


@agent_app.command("critic")
def agent_critic(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Run the critic agent (polls experiences queue)."""
    cfg = _load_config(config, env)
    _setup_logging(verbose, log_file=make_log_path(cfg.storage.logs_path, "critic"))
    run_tag = _make_run_tag("critic")
    pipeline, runner, fs = _bootstrap(cfg, run_tag=run_tag)

    from agenix.agent_loop import QueueAgentLoop
    from agenix.agents.critic_handler import CriticHandler
    from agenix.queue.fs_queue import FSQueue

    experiences_q = FSQueue("experiences", cfg.storage)

    handler = CriticHandler(
        runner=runner,
        fs_backend=fs,
        knowledge_store=pipeline.store,
    )
    loop = QueueAgentLoop(experiences_q, handler)
    typer.echo(f"Critic agent started (run_tag={run_tag}).")
    loop.run()


@agent_app.command("organizer")
def agent_organizer(
    interval: int = typer.Option(300, "--interval", help="Interval between runs (seconds)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Run the organizer agent (periodic knowledge synthesis)."""
    cfg = _load_config(config, env)
    _setup_logging(verbose, log_file=make_log_path(cfg.storage.logs_path, "organizer"))
    run_tag = _make_run_tag("organizer")
    pipeline, runner, fs = _bootstrap(cfg, run_tag=run_tag)

    from agenix.agent_loop import ScheduledAgentLoop
    from agenix.agents.organizer_handler import OrganizerHandler

    handler = OrganizerHandler(
        runner=runner,
        fs_backend=fs,
        knowledge_store=pipeline.store,
        run_tag=run_tag,
    )
    loop = ScheduledAgentLoop(handler, interval=float(interval))
    typer.echo(f"Organizer agent started (interval={interval}s, run_tag={run_tag}).")
    loop.run()


@agent_app.command("insight-finder")
def agent_insight_finder(
    interval: int = typer.Option(600, "--interval", help="Interval between runs (seconds)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Run the insight finder agent (periodic meta-pattern detection)."""
    cfg = _load_config(config, env)
    _setup_logging(verbose, log_file=make_log_path(cfg.storage.logs_path, "insight_finder"))
    run_tag = _make_run_tag("insight_finder")
    pipeline, runner, fs = _bootstrap(cfg, run_tag=run_tag)

    from agenix.agent_loop import ScheduledAgentLoop
    from agenix.agents.insight_handler import InsightHandler

    handler = InsightHandler(
        runner=runner,
        fs_backend=fs,
        knowledge_store=pipeline.store,
        run_tag=run_tag,
    )
    loop = ScheduledAgentLoop(handler, interval=float(interval))
    typer.echo(f"Insight finder agent started (interval={interval}s, run_tag={run_tag}).")
    loop.run()


# --- Queue sub-commands ---


@queues_app.command("status")
def queues_status(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Show queue status (pending/processing/done/failed counts)."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)

    from agenix.queue.fs_queue import FSQueue
    from agenix.queue.models import MessageState

    queue_names = ["problems", "experiences"]
    for name in queue_names:
        q = FSQueue(name, cfg.storage)
        q.initialize()
        pending = q.count(MessageState.PENDING)
        processing = q.count(MessageState.PROCESSING)
        done = q.count(MessageState.DONE)
        failed = q.count(MessageState.FAILED)
        typer.echo(
            f"{name:15s}  pending={pending}  processing={processing}  "
            f"done={done}  failed={failed}"
        )


# --- Logs sub-commands ---


@logs_app.command("show")
def logs_show(
    run: str = typer.Option(..., "--run", help="Run tag to show logs for."),
    agent_filter: Optional[str] = typer.Option(
        None, "--agent", help="Filter by agent name."
    ),
    event_type: Optional[str] = typer.Option(
        None, "--type", help="Filter by event type."
    ),
    limit: int = typer.Option(100, "-n", "--limit", help="Max events to show."),
    config: Optional[Path] = typer.Option(None, "--config"),
    env: Optional[str] = typer.Option(None, "--env"),
) -> None:
    """Show execution log events for a run."""
    cfg = _load_config(config, env)
    log_path = cfg.storage.execution_log_path(run)

    if not log_path.exists():
        typer.echo(f"No execution log found at {log_path}")
        raise typer.Exit(1)

    import json

    count = 0
    with open(log_path) as f:
        for line in f:
            if count >= limit:
                break
            event = json.loads(line)
            if agent_filter and event.get("agent") != agent_filter:
                continue
            if event_type and event.get("event_type") != event_type:
                continue

            ts = event["timestamp"][:19]
            et = event["event_type"]
            ag = event.get("agent", "")
            dur = event.get("duration_ms")
            dur_str = f"  {dur}ms" if dur else ""
            err = event.get("error", "")
            err_str = f"  ERR: {err[:60]}" if err else ""
            data_summary = ""
            data = event.get("data", {})
            if data:
                parts = [f"{k}={v}" for k, v in list(data.items())[:3]]
                data_summary = "  " + ", ".join(parts)

            typer.echo(f"{ts}  {et:20s}  {ag:12s}{dur_str}{data_summary}{err_str}")
            count += 1

    if count == 0:
        typer.echo("No matching events.")


@logs_app.command("summary")
def logs_summary(
    run: str = typer.Option(..., "--run", help="Run tag to summarize."),
    config: Optional[Path] = typer.Option(None, "--config"),
    env: Optional[str] = typer.Option(None, "--env"),
) -> None:
    """Show summary statistics for a run's execution log."""
    cfg = _load_config(config, env)
    log_path = cfg.storage.execution_log_path(run)

    if not log_path.exists():
        typer.echo(f"No execution log found at {log_path}")
        raise typer.Exit(1)

    import json
    from collections import Counter

    events = []
    with open(log_path) as f:
        for line in f:
            events.append(json.loads(line))

    typer.echo(f"Run: {run}")
    typer.echo(f"Events: {len(events)}")

    type_counts = Counter(e["event_type"] for e in events)
    typer.echo("\nEvent counts:")
    for et, count in type_counts.most_common():
        typer.echo(f"  {et:25s}  {count}")

    # Agent completion stats
    completions = [e for e in events if e["event_type"] == "agent_completed"]
    if completions:
        typer.echo("\nAgent completions:")
        for c in completions:
            d = c.get("data", {})
            typer.echo(
                f"  {d.get('agent_name', '?'):15s}  "
                f"{c.get('duration_ms', 0):>6d}ms  "
                f"turns={d.get('num_turns', 0)}  "
                f"cost=${d.get('cost_usd', 0):.4f}"
            )

    # Errors
    errors = [e for e in events if e.get("error")]
    if errors:
        typer.echo(f"\nErrors: {len(errors)}")
        for e in errors:
            typer.echo(f"  [{e['event_type']}] {e['error'][:80]}")


# --- Cards sub-commands ---


@cards_app.command("list")
def cards_list(
    card_type: Optional[str] = typer.Option(
        None, "--type", help="Filter by type (knowledge/reflection/insight).",
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """List knowledge cards."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)
    fs = FSBackend(cfg.storage)

    cards = fs.list_cards(card_type=card_type or None)
    if not cards:
        typer.echo("No cards found.")
        return

    for card in cards:
        typer.echo(f"[{card.card_type:10s}] {card.card_id}  {card.title}")


@cards_app.command("search")
def cards_search(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(5, "-k", "--top-k", help="Number of results."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Semantic search over knowledge cards."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)

    from tools.knowledge.baseline.store import KnowledgeStore

    fs = FSBackend(cfg.storage)
    store = KnowledgeStore(config=cfg, fs_backend=fs)

    results = store.search(query=query, limit=top_k)
    if not results:
        typer.echo("No results found.")
        return

    for r in results:
        score = round(1.0 - r.get("_distance", 0.0), 4)
        typer.echo(f"[{r['card_type']:10s}] {r['card_id']}  {r['title']}  (score={score})")


# --- Experiences sub-commands ---


@experiences_app.command("list")
def experiences_list(
    correct: Optional[bool] = typer.Option(None, "--correct", help="Filter by correctness."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """List agent experiences."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)
    fs = FSBackend(cfg.storage)

    experiences = fs.list_experiences(is_correct=correct)
    if not experiences:
        typer.echo("No experiences found.")
        return

    for e in experiences:
        status = "correct" if e.is_correct else "incorrect"
        typer.echo(
            f"[{status:9s}] {e.experience_id}  "
            f"problem={e.problem_id}  "
            f"steps={len(e.steps)}"
        )


# --- Services sub-commands ---


@services_app.command("status")
def services_status(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Health-check all configured service endpoints."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    from services.health import HealthChecker

    checker = HealthChecker(cfg.services)
    results = asyncio.run(checker.check_all())

    if not results:
        typer.echo("No service endpoints configured.")
        return

    for h in results:
        typer.echo(
            f"[{h.status.value:7s}] {h.name}  "
            f"endpoint={h.endpoint}  "
            f"devices={','.join(h.devices)}  "
            f"pending={h.pending_requests}"
        )


@services_app.command("deploy")
def services_deploy(
    name: str = typer.Argument(..., help="Endpoint name to deploy to."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Deploy kbEval to a named endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer

    deployer = ServiceDeployer(cfg.services)
    ok = asyncio.run(deployer.deploy_kb_eval(endpoint))
    if ok:
        typer.echo(f"Deployed kbEval to {name}.")
    else:
        typer.echo(f"Deploy to {name} failed.", err=True)
        raise typer.Exit(1)


@services_app.command("stop")
def services_stop(
    name: str = typer.Argument(..., help="Endpoint name to stop."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Stop kbEval on a named endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer

    deployer = ServiceDeployer(cfg.services)
    ok = asyncio.run(deployer.stop_kb_eval(endpoint))
    if ok:
        typer.echo(f"Stopped kbEval on {name}.")
    else:
        typer.echo(f"Stop on {name} failed.", err=True)
        raise typer.Exit(1)


@services_app.command("deploy-embedding")
def services_deploy_embedding(
    name: str = typer.Argument(..., help="Endpoint name to deploy to."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Deploy text embedding service to a named endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer

    deployer = ServiceDeployer(cfg.services)
    ok = asyncio.run(deployer.deploy_text_embedding(endpoint))
    if ok:
        typer.echo(f"Deployed text-embedding to {name}.")
    else:
        typer.echo(f"Deploy text-embedding to {name} failed.", err=True)
        raise typer.Exit(1)


@services_app.command("stop-embedding")
def services_stop_embedding(
    name: str = typer.Argument(..., help="Endpoint name to stop."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Stop text embedding service on a named endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer

    deployer = ServiceDeployer(cfg.services)
    ok = asyncio.run(deployer.stop_text_embedding(endpoint))
    if ok:
        typer.echo(f"Stopped text-embedding on {name}.")
    else:
        typer.echo(f"Stop text-embedding on {name} failed.", err=True)
        raise typer.Exit(1)


@services_app.command("logs-embedding")
def services_logs_embedding(
    name: str = typer.Argument(..., help="Endpoint name."),
    lines: int = typer.Option(50, "-n", "--lines", help="Number of log lines."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """View recent text embedding logs from a remote endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer

    deployer = ServiceDeployer(cfg.services)
    output = asyncio.run(deployer.logs_text_embedding(endpoint, lines=lines))
    typer.echo(output)


@services_app.command("deploy-reranker")
def services_deploy_reranker(
    name: str = typer.Argument(..., help="Endpoint name to deploy to."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Deploy reranker service (vLLM + FastAPI) to a named endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer

    deployer = ServiceDeployer(cfg.services)
    ok = asyncio.run(deployer.deploy_reranker(endpoint))
    if ok:
        typer.echo(f"Deployed reranker to {name}.")
    else:
        typer.echo(f"Deploy reranker to {name} failed.", err=True)
        raise typer.Exit(1)


@services_app.command("stop-reranker")
def services_stop_reranker(
    name: str = typer.Argument(..., help="Endpoint name to stop."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Stop reranker service on a named endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer

    deployer = ServiceDeployer(cfg.services)
    ok = asyncio.run(deployer.stop_reranker(endpoint))
    if ok:
        typer.echo(f"Stopped reranker on {name}.")
    else:
        typer.echo(f"Stop reranker on {name} failed.", err=True)
        raise typer.Exit(1)


@services_app.command("logs-reranker")
def services_logs_reranker(
    name: str = typer.Argument(..., help="Endpoint name."),
    lines: int = typer.Option(50, "-n", "--lines", help="Number of log lines."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """View recent reranker logs from a remote endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer

    deployer = ServiceDeployer(cfg.services)
    output = asyncio.run(deployer.logs_reranker(endpoint, lines=lines))
    typer.echo(output)


@services_app.command("health")
def services_health(
    name: str = typer.Argument(..., help="Endpoint name to check."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Detailed health check of a single endpoint (all services)."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer
    from services.health import HealthChecker

    checker = HealthChecker(cfg.services)

    # SSH check
    ssh_ok = asyncio.run(checker.check_ssh(endpoint))
    typer.echo(f"SSH:      {'ok' if ssh_ok else 'failed'}")

    # kbEval health
    h = asyncio.run(checker.check_endpoint(endpoint))
    typer.echo("\n--- kbEval ---")
    typer.echo(f"Status:   {h.status.value}")
    typer.echo(f"Endpoint: {h.endpoint}")
    typer.echo(f"Devices:  {', '.join(h.devices) or 'none'}")
    typer.echo(f"Pending:  {h.pending_requests}")

    # text_embedding health
    te = asyncio.run(checker.check_text_embedding(endpoint))
    typer.echo("\n--- text-embedding ---")
    typer.echo(f"Status:   {te.status.value}")
    typer.echo(f"Endpoint: {te.endpoint}")
    typer.echo(f"Devices:  {', '.join(te.devices) or 'none'}")

    # reranker health
    rr = asyncio.run(checker.check_reranker(endpoint))
    typer.echo("\n--- reranker ---")
    typer.echo(f"Status:   {rr.status.value}")
    typer.echo(f"Endpoint: {rr.endpoint}")
    typer.echo(f"Devices:  {', '.join(rr.devices) or 'none'}")

    # systemd status
    if ssh_ok:
        deployer = ServiceDeployer(cfg.services)
        sd_status = asyncio.run(deployer.systemd_status_kb_eval(endpoint))
        typer.echo(f"\n--- systemd: kb-eval ---\n{sd_status}")
        sd_status_te = asyncio.run(deployer.systemd_status_text_embedding(endpoint))
        typer.echo(f"\n--- systemd: text-embedding ---\n{sd_status_te}")
        sd_status_rr = asyncio.run(deployer.systemd_status_reranker(endpoint))
        typer.echo(f"\n--- systemd: reranker ---\n{sd_status_rr}")


@services_app.command("logs")
def services_logs(
    name: str = typer.Argument(..., help="Endpoint name."),
    lines: int = typer.Option(50, "-n", "--lines", help="Number of log lines."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """View recent kbEval logs from a remote endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer

    deployer = ServiceDeployer(cfg.services)
    output = asyncio.run(deployer.logs_kb_eval(endpoint, lines=lines))
    typer.echo(output)


@services_app.command("test")
def services_test(
    name: str = typer.Argument(..., help="Endpoint name to test."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Test a service endpoint: SSH, systemd, health, and a basic eval round-trip.

    Automatically opens an SSH tunnel for HTTP access to the remote service.
    """
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    ok = asyncio.run(_run_service_test(cfg, endpoint, verbose))
    if not ok:
        raise typer.Exit(1)


async def _run_service_test(
    cfg: ReflectionConfig,
    endpoint,
    verbose: bool,
) -> bool:
    """Run all service tests for an endpoint. Returns True if all passed.

    Expects SSH tunnels to be running (via `reflection services tunnel start`).
    Uses endpoint.kb_eval.base_url which points to localhost through the tunnel.
    """
    from services.deploy import ServiceDeployer
    from services.health import HealthChecker
    from services.kb_eval.baseline.client import KbEvalClient
    from services.ssh_tunnel.tunnel import check_port

    failed = False

    # 1. SSH connectivity
    checker = HealthChecker(cfg.services)
    ssh_ok = await checker.check_ssh(endpoint)
    typer.echo(f"SSH:          {'ok' if ssh_ok else 'FAILED'}")
    if not ssh_ok:
        typer.echo("Cannot reach host via SSH. Aborting.", err=True)
        return False

    # 2. systemd status
    deployer = ServiceDeployer(cfg.services)
    sd_status = await deployer.systemd_status_kb_eval(endpoint)
    is_active = "active (running)" in sd_status
    typer.echo(f"systemd:      {'active' if is_active else 'FAILED'}")
    if not is_active:
        failed = True

    # 3. Check tunnel is running (base_url should point to localhost)
    from urllib.parse import urlparse

    parsed = urlparse(endpoint.kb_eval.base_url)
    tunnel_port = parsed.port or 80
    tunnel_ok = check_port(tunnel_port, parsed.hostname or "localhost")
    typer.echo(f"Tunnel:       {'ok' if tunnel_ok else 'FAILED'} ({endpoint.kb_eval.base_url})")
    if not tunnel_ok:
        typer.echo(
            "Tunnel not reachable. Start tunnels first: reflection services tunnel start",
            err=True,
        )
        return False

    client = KbEvalClient(endpoint.kb_eval)

    # 4. Health endpoint
    health = await client.health()
    health_ok = health.status.value == "running"
    typer.echo(f"Health:       {'ok' if health_ok else 'FAILED'}")
    if health_ok:
        typer.echo(f"  Devices:    {', '.join(health.devices)}")
        typer.echo(f"  Pending:    {health.pending_requests}")
    else:
        failed = True

    # 5. Basic eval round-trip
    if health_ok:
        typer.echo("Eval:         running...")
        ref_code = (
            "import torch\nimport torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super(Model, self).__init__()\n"
            "    def forward(self, x):\n"
            "        return torch.relu(x)\n\n"
            "batch_size = 16\ndim = 16384\n\n"
            "def get_inputs():\n"
            "    return [torch.rand(batch_size, dim)]\n\n"
            "def get_init_inputs():\n"
            "    return []\n"
        )
        gen_code = (
            "import torch\nimport torch.nn as nn\n\n"
            "class ModelNew(nn.Module):\n"
            "    def __init__(self):\n"
            "        super(ModelNew, self).__init__()\n"
            "    def forward(self, x):\n"
            "        return torch.relu(x)\n"
        )
        try:
            result = await client.eval(
                reference_code=ref_code,
                generated_code=gen_code,
                code_type="pytorch",
            )
            eval_ok = result.compiled and result.correctness
            typer.echo(
                f"Eval:         {'ok' if eval_ok else 'FAILED'}  "
                f"compiled={result.compiled}  "
                f"correct={result.correctness}  "
                f"runtime={result.runtime:.2f}ms"
            )
            if not eval_ok:
                failed = True
        except Exception as e:
            typer.echo(f"Eval:         FAILED  {e}")
            failed = True

    if not failed:
        typer.echo(f"\nAll checks passed for {endpoint.name}.")
    return not failed


# --- Tunnel sub-commands ---


@tunnel_app.command("start")
def tunnel_start(
    name: Optional[str] = typer.Argument(None, help="Tunnel name (all if omitted)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Start SSH tunnel(s) via the OS service manager."""
    _setup_logging(verbose)
    cfg = _load_config(config)

    from services.ssh_tunnel.tunnel import get_manager

    manager = get_manager()
    tunnels = cfg.tunnels.tunnels

    if not tunnels:
        typer.echo("No tunnels configured in config/tunnels.yaml.")
        return

    if name:
        tunnel = _find_tunnel(cfg, name)
        manager.start(tunnel)
        typer.echo(f"Started tunnel {tunnel.name}.")
    else:
        manager.start_all(tunnels)
        typer.echo(f"Started {len(tunnels)} tunnel(s).")


@tunnel_app.command("stop")
def tunnel_stop(
    name: Optional[str] = typer.Argument(None, help="Tunnel name (all if omitted)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Stop SSH tunnel(s)."""
    _setup_logging(verbose)
    cfg = _load_config(config)

    from services.ssh_tunnel.tunnel import get_manager

    manager = get_manager()
    tunnels = cfg.tunnels.tunnels

    if name:
        tunnel = _find_tunnel(cfg, name)
        manager.stop(tunnel)
        typer.echo(f"Stopped tunnel {tunnel.name}.")
    else:
        manager.stop_all(tunnels)
        typer.echo(f"Stopped {len(tunnels)} tunnel(s).")


@tunnel_app.command("restart")
def tunnel_restart(
    name: Optional[str] = typer.Argument(None, help="Tunnel name (all if omitted)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Restart SSH tunnel(s). Use after changing config/tunnels.yaml."""
    _setup_logging(verbose)
    cfg = _load_config(config)

    from services.ssh_tunnel.tunnel import get_manager

    manager = get_manager()
    tunnels = cfg.tunnels.tunnels

    if name:
        tunnel = _find_tunnel(cfg, name)
        manager.restart(tunnel)
        typer.echo(f"Restarted tunnel {tunnel.name}.")
    else:
        manager.restart_all(tunnels)
        typer.echo(f"Restarted {len(tunnels)} tunnel(s).")


@tunnel_app.command("status")
def tunnel_status(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Show status of all SSH tunnels."""
    _setup_logging(verbose)
    cfg = _load_config(config)

    from services.ssh_tunnel.tunnel import get_manager

    manager = get_manager()
    tunnels = cfg.tunnels.tunnels

    if not tunnels:
        typer.echo("No tunnels configured in config/tunnels.yaml.")
        return

    for t in tunnels:
        st = manager.status(t)
        status_str = "running" if st.running else "stopped"
        pid_str = f"  pid={st.pid}" if st.pid else ""
        ports = ", ".join(
            f"{f.local_port}->{f.remote_host}:{f.remote_port}"
            for f in st.forwards
        )
        typer.echo(f"[{status_str:7s}] {st.name}  {ports}{pid_str}")


def _find_tunnel(cfg: ReflectionConfig, name: str):
    """Find tunnel by name or exit with error."""
    for t in cfg.tunnels.tunnels:
        if t.name == name:
            return t
    available = [t.name for t in cfg.tunnels.tunnels]
    typer.echo(f"Tunnel '{name}' not found. Available: {available}", err=True)
    raise typer.Exit(1)


def _find_endpoint(cfg: ReflectionConfig, name: str):
    """Find endpoint by name or exit with error."""
    for ep in cfg.services.endpoints:
        if ep.name == name:
            return ep
    available = [ep.name for ep in cfg.services.endpoints]
    typer.echo(f"Endpoint '{name}' not found. Available: {available}", err=True)
    raise typer.Exit(1)


# --- Helpers ---


def _setup_logging(verbose: bool, log_file: Path | None = None) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setLevel(logging.DEBUG)  # always capture full detail to file
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logging.getLogger().addHandler(fh)
