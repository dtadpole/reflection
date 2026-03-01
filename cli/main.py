"""CLI entry point for the reflection system."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from agenix.config import ReflectionConfig, load_config
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import CardType

app = typer.Typer(name="reflection", help="Self-evolving multi-agent coding system.")
cards_app = typer.Typer(help="Manage knowledge cards.")
trajectories_app = typer.Typer(help="Manage solver trajectories.")
services_app = typer.Typer(help="Manage remote services.")
app.add_typer(cards_app, name="cards")
app.add_typer(trajectories_app, name="trajectories")
app.add_typer(services_app, name="services")


def _load_config(
    config_path: Optional[Path] = None,
    env: Optional[str] = None,
) -> ReflectionConfig:
    """Load config, optionally overriding the environment."""
    cfg = load_config(config_path)
    if env:
        cfg.storage.env = env
    return cfg


def _make_run_tag() -> str:
    """Generate a run tag from the current timestamp."""
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _bootstrap(config: ReflectionConfig):
    """Create all runtime objects for pipeline execution.

    Returns (pipeline, runner, fs_backend).
    """
    from agenix.knowledge.store import KnowledgeStore
    from agenix.pipeline import Pipeline
    from agenix.runner import ClaudeRunner
    from agenix.tools import (
        create_code_executor_tool,
        create_kb_eval_tool,
        create_retriever_tool,
    )
    from agenix.tools.registry import ToolRegistry

    fs = FSBackend(config.storage)
    fs.initialize()

    store = KnowledgeStore(config=config, fs_backend=fs)
    store.initialize()

    registry = ToolRegistry()
    registry.register(create_retriever_tool(store))
    registry.register(create_code_executor_tool(config.code_executor))

    # Register kbEval tool if any endpoints are configured
    if config.services.endpoints:
        from services.kb_eval.client import KbEvalClient

        kb_client = KbEvalClient(config.services.endpoints[0].kb_eval)
        registry.register(create_kb_eval_tool(kb_client))

    runner = ClaudeRunner(tool_registry=registry)
    pipeline = Pipeline(
        config=config,
        runner=runner,
        knowledge_store=store,
        fs_backend=fs,
    )
    return pipeline, runner, fs


# --- Top-level commands ---


@app.command()
def run(
    iterations: int = typer.Option(1, "-n", "--iterations", help="Number of iterations to run."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment (prod/int/test_<user>)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Run N iterations of the full reflection pipeline."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)
    pipeline, _, _ = _bootstrap(cfg)

    for i in range(1, iterations + 1):
        run_tag = _make_run_tag()
        typer.echo(f"--- Iteration {i}/{iterations} [{run_tag}] ---")
        result = pipeline.run_iteration(run_tag, iteration=i)
        status = "correct" if result.is_correct else "incorrect"
        typer.echo(
            f"  Problem: {result.problem_id} | "
            f"Trajectory: {result.trajectory_id} | "
            f"Result: {status} | "
            f"Cards: {len(result.cards_created)}"
        )

    typer.echo("Done.")


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
    trajectory = pipeline._run_solver(run_tag, problem)

    typer.echo(f"Trajectory: {trajectory.trajectory_id}")
    typer.echo(f"Correct: {trajectory.is_correct}")
    if trajectory.code_solution:
        typer.echo(f"Solution:\n{trajectory.code_solution}")


@app.command()
def status(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Show system status: counts of problems, trajectories, and cards."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)
    fs = FSBackend(cfg.storage)

    problems = fs.count_problems()
    trajectories = fs.count_trajectories()
    cards_knowledge = fs.count_cards(CardType.KNOWLEDGE)
    cards_reflection = fs.count_cards(CardType.REFLECTION)
    cards_insight = fs.count_cards(CardType.INSIGHT)

    typer.echo(f"Environment: {cfg.storage.env}")
    typer.echo(f"Data root:   {cfg.storage.env_path}")
    typer.echo(f"Problems:    {problems}")
    typer.echo(f"Trajectories: {trajectories}")
    typer.echo(f"Cards:       {cards_knowledge + cards_reflection + cards_insight}")
    typer.echo(f"  Knowledge: {cards_knowledge}")
    typer.echo(f"  Reflection: {cards_reflection}")
    typer.echo(f"  Insight:   {cards_insight}")


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

    ct = None
    if card_type:
        try:
            ct = CardType(card_type)
        except ValueError:
            valid = [t.value for t in CardType]
            typer.echo(f"Invalid card type '{card_type}'. Valid: {valid}", err=True)
            raise typer.Exit(1)

    cards = fs.list_cards(card_type=ct)
    if not cards:
        typer.echo("No cards found.")
        return

    for card in cards:
        typer.echo(f"[{card.card_type.value:10s}] {card.card_id}  {card.title}")


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

    from agenix.knowledge.store import KnowledgeStore

    fs = FSBackend(cfg.storage)
    store = KnowledgeStore(config=cfg, fs_backend=fs)

    results = store.search(query=query, limit=top_k)
    if not results:
        typer.echo("No results found.")
        return

    for r in results:
        score = round(1.0 - r.get("_distance", 0.0), 4)
        typer.echo(f"[{r['card_type']:10s}] {r['card_id']}  {r['title']}  (score={score})")


# --- Trajectories sub-commands ---


@trajectories_app.command("list")
def trajectories_list(
    run_tag: Optional[str] = typer.Option(None, "--run", help="Filter by run tag."),
    correct: Optional[bool] = typer.Option(None, "--correct", help="Filter by correctness."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """List solver trajectories."""
    _setup_logging(verbose)
    cfg = _load_config(config, env)
    fs = FSBackend(cfg.storage)

    trajectories = fs.list_trajectories(run_tag=run_tag, is_correct=correct)
    if not trajectories:
        typer.echo("No trajectories found.")
        return

    for t in trajectories:
        status = "correct" if t.is_correct else "incorrect"
        typer.echo(
            f"[{status:9s}] {t.trajectory_id}  "
            f"problem={t.problem_id}  "
            f"steps={len(t.steps)}"
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


@services_app.command("health")
def services_health(
    name: str = typer.Argument(..., help="Endpoint name to check."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config TOML."),
    env: Optional[str] = typer.Option(None, "--env", help="Environment."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Detailed health check of a single endpoint."""
    import asyncio

    _setup_logging(verbose)
    cfg = _load_config(config, env)

    endpoint = _find_endpoint(cfg, name)

    from services.deploy import ServiceDeployer
    from services.health import HealthChecker

    checker = HealthChecker(cfg.services)
    h = asyncio.run(checker.check_endpoint(endpoint))

    typer.echo(f"Name:     {h.name}")
    typer.echo(f"Status:   {h.status.value}")
    typer.echo(f"Endpoint: {h.endpoint}")
    typer.echo(f"Devices:  {', '.join(h.devices) or 'none'}")
    typer.echo(f"Pending:  {h.pending_requests}")

    # SSH check
    ssh_ok = asyncio.run(checker.check_ssh(endpoint))
    typer.echo(f"SSH:      {'ok' if ssh_ok else 'failed'}")

    # systemd status
    if ssh_ok:
        deployer = ServiceDeployer(cfg.services)
        sd_status = asyncio.run(deployer.systemd_status_kb_eval(endpoint))
        typer.echo(f"\n--- systemd ---\n{sd_status}")


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


def _find_endpoint(cfg: ReflectionConfig, name: str):
    """Find endpoint by name or exit with error."""
    for ep in cfg.services.endpoints:
        if ep.name == name:
            return ep
    available = [ep.name for ep in cfg.services.endpoints]
    typer.echo(f"Endpoint '{name}' not found. Available: {available}", err=True)
    raise typer.Exit(1)


# --- Helpers ---


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s %(levelname)s %(message)s")
