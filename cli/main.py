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
tunnel_app = typer.Typer(help="Manage SSH tunnels for port forwarding.")
app.add_typer(cards_app, name="cards")
app.add_typer(trajectories_app, name="trajectories")
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


def _make_run_tag() -> str:
    """Generate a run tag from the current timestamp."""
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _bootstrap(config: ReflectionConfig):
    """Create all runtime objects for pipeline execution.

    Returns (pipeline, runner, fs_backend).
    """
    from agenix.pipeline import Pipeline
    from agenix.runner import ClaudeRunner
    from agenix.tools.loader import load_tool
    from agenix.tools.registry import ToolRegistry
    from tools.knowledge.baseline.store import KnowledgeStore

    fs = FSBackend(config.storage)
    fs.initialize()

    store = KnowledgeStore(config=config, fs_backend=fs)
    store.initialize()

    registry = ToolRegistry()

    # Load retriever tool — use rerank variant when reranker endpoint is available
    if config.services.endpoints:
        from services.reranker.baseline.client import RerankerClient

        rr_client = RerankerClient(config.services.endpoints[0].reranker)
        retriever_def = load_tool("retriever", variant="rerank")
        registry.register(
            retriever_def.create_fn(
                knowledge_store=store, reranker_client=rr_client
            )
        )
    else:
        retriever_def = load_tool("retriever", variant="baseline")
        registry.register(retriever_def.create_fn(knowledge_store=store))

    # Load verifier tool (kb_eval variant) if endpoints are configured
    if config.services.endpoints:
        from services.kb_eval.baseline.client import KbEvalClient

        kb_client = KbEvalClient(config.services.endpoints[0].kb_eval)
        verifier_def = load_tool("verifier", variant="kb_eval")
        registry.register(verifier_def.create_fn(kb_eval_client=kb_client))

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


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s %(levelname)s %(message)s")
