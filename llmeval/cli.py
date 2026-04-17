"""Typer CLI entrypoint for llmeval.

Commands are thin wrappers around the core library.
"""

import typer
from rich.console import Console

app = typer.Typer(
    name="llmeval",
    help="LLM evaluation and regression testing framework.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed llmeval version."""
    from llmeval import __version__

    console.print(f"llmeval {__version__}")


@app.command()
def run(
    suite: str = typer.Option(
        ..., "--suite", "-s", help="Path to a YAML/JSON test suite file."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override the model specified in the suite."
    ),
) -> None:
    """Run a test suite against a model and print results.

    Args:
        suite: Path to the YAML/JSON test suite file.
        model: Optional model override (e.g. ``gpt-4o``, ``claude-opus-4-20250514``).
    """
    console.print("[yellow]`llmeval run` is not yet implemented.[/yellow]")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
