import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from autostats.core.agent.events import ErrorEvent, TextDelta, ToolCallStarted, ToolResultReady, TurnComplete
from autostats.core.agent.loop import AutoStatsAgent
from autostats.core.config import get_settings
from autostats.core.session.store import SessionStore

app = typer.Typer(help="AutoStats: an AI agent that automates statistical analyses.")
console = Console()


def _agent() -> AutoStatsAgent:
    settings = get_settings()
    store = SessionStore(settings.data_dir / "sessions.db")
    return AutoStatsAgent(store, settings)


def _run_turn(agent: AutoStatsAgent, session_id: str, message: str) -> None:
    for event in agent.run_turn(session_id, message):
        if isinstance(event, ToolCallStarted):
            console.print(f"[dim]-> calling {event.tool_name}({event.arguments})[/dim]")
        elif isinstance(event, ToolResultReady):
            console.print(f"[dim]<- {event.tool_name} done[/dim]")
        elif isinstance(event, TextDelta):
            console.print(event.text, end="")
        elif isinstance(event, TurnComplete):
            console.print()
        elif isinstance(event, ErrorEvent):
            console.print(f"[bold red]Error:[/bold red] {event.message}")


@app.command()
def chat(session_id: str = typer.Option(None, help="Resume an existing session id.")) -> None:
    """Start an interactive chat session with the AutoStats agent."""
    session_id = session_id or uuid.uuid4().hex[:8]
    agent = _agent()
    console.print(f"[bold]AutoStats session {session_id}[/bold] (Ctrl-D to exit)\n")
    while True:
        try:
            message = console.input("[bold cyan]you>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break
        if not message.strip():
            continue
        _run_turn(agent, session_id, message)


@app.command()
def upload(file_path: Path, session_id: str = typer.Option(None)) -> None:
    """Upload a dataset file into a session, then chat about it."""
    session_id = session_id or uuid.uuid4().hex[:8]
    agent = _agent()
    console.print(f"Session: {session_id}")
    _run_turn(agent, session_id, f"Please load the dataset at {file_path.resolve()}.")


@app.command()
def datasets(session_id: str) -> None:
    """List datasets loaded in a session."""
    agent = _agent()
    manager = agent.data_manager(session_id)
    table = Table("dataset_id", "rows", "cols", "source", "trust")
    for meta in manager.list_datasets():
        table.add_row(meta.dataset_id, str(meta.n_rows), str(meta.n_cols), meta.source, meta.trust_level)
    console.print(table)


@app.command()
def report(session_id: str, output: Path = typer.Option(Path("report.html")), fmt: str = typer.Option("html")) -> None:
    """Export a report from a session's accumulated analysis results."""
    from autostats.core.reporting.builder import ReportBuilder

    settings = get_settings()
    store = SessionStore(settings.data_dir / "sessions.db")
    builder = ReportBuilder(store)
    path = builder.build(session_id, output, fmt=fmt)
    console.print(f"Report written to {path}")


if __name__ == "__main__":
    app()
