"""Auto_Book CLI entry point."""

from datetime import datetime, timedelta
from pathlib import Path
import uuid

from rich.console import Console

from auto_book.cli import get_user_brief, parse_args
from auto_book.config import load_settings, set_settings
from auto_book.models.run_state import RunPhase, RunState
from auto_book.orchestrator.checkpointer import (
    load_checkpoint,
    run_state_to_graph_state,
)
from auto_book.orchestrator.graph import build_graph
from auto_book.utils.logger import get_logger, setup_logger
from auto_book.utils.rate_limiter import (
    extract_retry_after_seconds,
    is_rate_limit_error,
)

console = Console()


def main() -> None:
    """Run the Auto_Book CLI."""

    args = parse_args()
    cfg = load_settings(args.config)
    set_settings(cfg)
    output_dir = args.output or cfg.output.directory
    log_file = str(Path(output_dir) / "run.log")

    setup_logger(
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        log_file=log_file,
    )
    logger = get_logger()

    if args.resume:
        run_state = load_checkpoint(output_dir)
        if run_state is None:
            console.print("[red]No checkpoint found to resume from.[/]")
            return
        if run_state.phase == RunPhase.COMPLETED:
            console.print(
                f"[green]Checkpoint already completed: run {run_state.run_id}[/]"
            )
            return
        wait_seconds = _remaining_rate_limit_wait(run_state)
        if run_state.phase == RunPhase.FAILED and is_rate_limit_error(
            run_state.failure_reason
        ):
            console.print(
                "[yellow]Previous failure was a rate limit; retrying the first "
                "unfinished chapter.[/]"
            )
        initial_state = run_state_to_graph_state(run_state)
        console.print(f"[cyan]Resuming run {run_state.run_id}...[/]")
    else:
        brief = " ".join(args.brief).strip() or get_user_brief()
        run_state = RunState(
            run_id=str(uuid.uuid4())[:8],
            user_brief=brief,
            genre=args.genre,
            output_directory=output_dir,
        )
        initial_state = run_state_to_graph_state(run_state)
        console.print(f"[green]Starting run {run_state.run_id}...[/]")

    logger.info("Run id: %s", initial_state["run_id"])
    logger.info("Output directory: %s", initial_state["output_directory"])

    graph = build_graph()
    try:
        result = graph.invoke(initial_state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Run interrupted by user.[/]")
        logger.warning("Run interrupted by user")
        return
    except Exception as exc:
        console.print(f"\n[red]Run failed: {exc}[/]")
        logger.exception("Run failed with exception")
        return

    phase = result.get("phase")
    if isinstance(phase, RunPhase):
        phase_label = phase.value
    else:
        phase_label = str(phase)

    if phase == RunPhase.FAILED or phase_label == RunPhase.FAILED.value:
        console.print(f"[red]Run failed: {result.get('error', 'Unknown error')}[/]")
    elif phase == RunPhase.RATE_LIMITED or phase_label == RunPhase.RATE_LIMITED.value:
        wait_seconds = int(result.get("retry_after_seconds") or 0)
        console.print(
            "[yellow]Run paused by a provider rate limit. "
            f"Resume in about {_format_duration(wait_seconds)} with "
            f"`python -m auto_book --resume --output {output_dir}`.[/]"
        )
    else:
        console.print(f"[green]Run finished with phase: {phase_label}[/]")


def _remaining_rate_limit_wait(run_state: RunState) -> int:
    """Return seconds left before a rate-limited checkpoint should resume."""

    if run_state.phase not in {RunPhase.RATE_LIMITED, RunPhase.FAILED}:
        return 0
    if run_state.phase == RunPhase.FAILED and not is_rate_limit_error(
        run_state.failure_reason
    ):
        return 0

    resume_at = run_state.resume_not_before
    if resume_at is None and run_state.failure_reason:
        retry_after = extract_retry_after_seconds(run_state.failure_reason)
        if retry_after is not None:
            resume_at = run_state.updated_at + timedelta(seconds=retry_after)

    if resume_at is None:
        return 0
    return max(0, int((resume_at - datetime.now()).total_seconds()))


def _format_duration(seconds: int) -> str:
    """Format a short human-readable wait duration."""

    seconds = max(0, seconds)
    minutes, remaining = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {remaining}s"
    return f"{remaining}s"


if __name__ == "__main__":
    main()
