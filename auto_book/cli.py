"""Command-line parsing for Auto_Book."""

import argparse

from rich.console import Console
from rich.prompt import Prompt

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="auto-book",
        description="Autonomous Book Writing Agent",
    )
    parser.add_argument(
        "brief",
        nargs="*",
        help="The book topic or brief.",
    )
    parser.add_argument(
        "--config",
        default="configs/full.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--genre",
        default="non-fiction how-to",
        help="Book genre/type used by the planner.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint in the output directory.",
    )
    return parser.parse_args()


def get_user_brief() -> str:
    """Ask interactively for a book brief."""

    console.print("\n[bold cyan]Auto_Book - Book Generator[/]\n")
    return Prompt.ask(
        "[bold]What should the book be about?[/]\n"
        "Describe the topic, target audience, and any specific requirements"
    )
