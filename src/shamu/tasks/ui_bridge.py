"""UI bridge for task planning and execution across CLI (Rich) and GUI (PySide6)."""

from __future__ import annotations

import logging
from typing import Any, Optional, Set, Tuple

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from shamu.tasks.models import (
    PlanCompletedEvent,
    PlanStartedEvent,
    StepCompletedEvent,
    StepFailedEvent,
    StepStartedEvent,
    StepWaitingApprovalEvent,
    TaskEvent,
    TaskExecutionReport,
    TaskPlan,
    TaskStatus,
    TaskStep,
)

logger = logging.getLogger(__name__)


class TaskUIBridge:
    """Bridges task events and models to terminal (Rich) and desktop GUI (PySide6)."""

    def __init__(
        self,
        console: Optional[Console] = None,
        gui_bridge: Optional[Any] = None,
    ) -> None:
        self.console = console or Console()
        if isinstance(getattr(self.console, "width", None), int) and self.console.width < 120:
            self.console.width = 120
        self.gui_bridge = gui_bridge
        self._handled_events: Set[Tuple[Any, ...]] = set()

    def _get_event_key(self, event: TaskEvent) -> Tuple[Any, ...]:
        """Generates a unique deduplication key for an event to prevent duplicate output."""
        if isinstance(event, PlanStartedEvent):
            return (event.plan_id, "plan_started")
        elif isinstance(event, StepStartedEvent):
            return (event.plan_id, event.step_id, "step_started")
        elif isinstance(event, StepWaitingApprovalEvent):
            return (event.plan_id, event.step_id, "step_waiting")
        elif isinstance(event, StepCompletedEvent):
            return (event.plan_id, event.step_id, "step_completed")
        elif isinstance(event, StepFailedEvent):
            return (event.plan_id, event.step_id, "step_failed")
        elif isinstance(event, PlanCompletedEvent):
            return (event.plan_id, "plan_completed")
        return (event.plan_id, type(event).__name__, id(event))

    def _emit_gui_signal(self, signal_name: str, args_multi: tuple, arg_single: Any = None) -> None:
        """Safely forwards a payload to a PySide6 signal on gui_bridge, handling both multi-arg and single-arg forms."""
        if self.gui_bridge is None:
            return
        sig = getattr(self.gui_bridge, signal_name, None)
        if sig is None or not hasattr(sig, "emit"):
            return
        try:
            sig.emit(*args_multi)
        except TypeError:
            try:
                if hasattr(sig, "__getitem__"):
                    sig[object].emit(arg_single if arg_single is not None else args_multi[0])
                else:
                    sig.emit(arg_single if arg_single is not None else args_multi[0])
            except Exception as e:
                logger.debug("Failed fallback emit for %s: %s", signal_name, e)
        except Exception as exc:
            logger.debug("Failed to emit %s on gui_bridge: %s", signal_name, exc)

    def render_plan(self, plan: TaskPlan) -> Table:
        """Constructs a Rich Table displaying the structured task execution plan."""
        table = Table(
            title=f"📋 Task Plan: [bold cyan]{escape(plan.objective)}[/bold cyan]",
            caption=f"Plan ID: {escape(plan.id)} | Steps: {len(plan.steps)}",
            border_style="cyan",
            header_style="bold cyan",
        )
        table.add_column("#", justify="center", style="dim")
        table.add_column("Step ID", style="bold yellow")
        table.add_column("Title & Description", style="white")
        table.add_column("Tool Call", style="cyan")
        table.add_column("Prerequisites", style="magenta")
        table.add_column("Verification", style="green")
        table.add_column("Status", justify="center", style="blue")

        for i, step in enumerate(plan.steps, 1):
            tool_str = step.tool_name or "[dim]None[/dim]"
            deps_str = ", ".join(step.dependencies) if step.dependencies else "[dim]None[/dim]"

            ver_str = "[dim]None[/dim]"
            if step.verification:
                ver_str = f"{step.verification.type.value}: {step.verification.target}"
                if step.verification.expected:
                    ver_str += f" (expected: '{step.verification.expected}')"

            status_style = {
                TaskStatus.PENDING: "[dim]PENDING[/dim]",
                TaskStatus.WAITING_APPROVAL: "[bold yellow]APPROVAL[/bold yellow]",
                TaskStatus.IN_PROGRESS: "[bold cyan]RUNNING[/bold cyan]",
                TaskStatus.COMPLETED: "[bold green]COMPLETED[/bold green]",
                TaskStatus.FAILED: "[bold red]FAILED[/bold red]",
                TaskStatus.SKIPPED: "[dim yellow]SKIPPED[/dim yellow]",
            }.get(step.status, str(step.status.value))

            desc_text = f"[bold]{escape(step.title)}[/bold]"
            if step.description:
                desc_text += f"\n[dim]{escape(step.description)}[/dim]"

            table.add_row(
                str(i),
                escape(step.id),
                desc_text,
                tool_str,
                deps_str,
                ver_str,
                status_style,
            )

        self.console.print(table)
        self._emit_gui_signal(
            "task_plan_created",
            (plan.id, plan.objective, len(plan.steps)),
            arg_single=plan,
        )
        return table

    def print_plan(self, plan: TaskPlan) -> None:
        """Prints the structured plan table to the terminal."""
        self.render_plan(plan)

    def display_plan(self, plan: TaskPlan) -> Table:
        """Alias for render_plan."""
        return self.render_plan(plan)

    def on_event(self, event: TaskEvent) -> None:
        """Handles real-time task progress events, printing to terminal and emitting GUI signals."""
        key = self._get_event_key(event)
        if key in self._handled_events:
            return
        self._handled_events.add(key)

        if isinstance(event, PlanStartedEvent):
            self.console.print(Panel(
                f"[bold cyan]Objective:[/bold cyan] {escape(event.objective)}\n"
                f"[bold]Total Steps:[/bold] {event.total_steps}",
                title="[bold cyan]🚀 Starting Execution of Task Plan[/bold cyan]",
                border_style="cyan",
            ))
            self._emit_gui_signal(
                "task_plan_created",
                (event.plan_id, event.objective, event.total_steps),
                arg_single=event,
            )
            self._emit_gui_signal(
                "task_step_progress",
                ("plan", "started", event.objective),
                arg_single=event,
            )
            self._emit_gui_signal("status_updated", (f"Executing plan ({event.total_steps} steps)...",), arg_single=f"Executing plan ({event.total_steps} steps)...")

        elif isinstance(event, StepStartedEvent):
            step_tag = escape(f"[{event.step_id}]")
            tool_info = f" -> Tool: [cyan]{escape(event.tool_name)}[/cyan]" if event.tool_name else ""
            self.console.print(f"  [bold blue]▶ Step {step_tag}:[/bold blue] {escape(event.step_title)}{tool_info}")
            self._emit_gui_signal(
                "task_step_progress",
                (event.step_id, "in_progress", ""),
                arg_single=event,
            )
            self._emit_gui_signal("status_updated", (f"Step {event.step_id}: {event.step_title}",), arg_single=f"Step {event.step_id}: {event.step_title}")

        elif isinstance(event, StepWaitingApprovalEvent):
            step_tag = escape(f"[{event.step_id}]")
            self.console.print(
                f"  [bold yellow]⚠️ Approval Required for {step_tag}:[/bold yellow] Tool [cyan]{escape(event.tool_name)}[/cyan]"
            )
            self._emit_gui_signal(
                "task_step_progress",
                (event.step_id, "waiting_approval", event.tool_name),
                arg_single=event,
            )
            self._emit_gui_signal("status_updated", (f"Waiting approval: {event.tool_name}",), arg_single=f"Waiting approval: {event.tool_name}")

        elif isinstance(event, StepCompletedEvent):
            step_tag = escape(f"[{event.step_id}]")
            ver_status = " [green]✓ Verified[/green]" if event.verified else " [yellow]! Unverified[/yellow]"
            self.console.print(f"  [bold green]✓ Step {step_tag} Completed[/bold green]{ver_status}")
            res_summary = (event.result or "").strip()
            if res_summary:
                lines = res_summary.splitlines()
                first_line = lines[0][:120] + ("..." if len(lines[0]) > 120 or len(lines) > 1 else "")
                self.console.print(f"    [dim]↳ {escape(first_line)}[/dim]")
            self._emit_gui_signal(
                "task_step_progress",
                (event.step_id, "completed", event.result),
                arg_single=event,
            )

        elif isinstance(event, StepFailedEvent):
            step_tag = escape(f"[{event.step_id}]")
            self.console.print(f"  [bold red]✗ Step {step_tag} Failed:[/bold red] {escape(event.error)}")
            self._emit_gui_signal(
                "task_step_progress",
                (event.step_id, "failed", event.error),
                arg_single=event,
            )

        elif isinstance(event, PlanCompletedEvent):
            style = "green" if event.success else "red"
            sym = "✓" if event.success else "✗"
            title = f"[bold {style}]{sym} Task Plan {'Completed Successfully' if event.success else 'Failed'}[/bold {style}]"
            self.console.print(Panel(
                f"[bold]Summary:[/bold] {escape(event.summary)}\n"
                f"[bold]Steps Completed:[/bold] {event.completed_steps}/{event.total_steps}",
                title=title,
                border_style=style,
            ))
            self._emit_gui_signal(
                "task_plan_completed",
                (event.plan_id, event.success, event.summary),
                arg_single=event,
            )
            self._emit_gui_signal("status_updated", ("Plan completed" if event.success else "Plan failed",), arg_single="Plan completed" if event.success else "Plan failed")

    async def handle_event(self, event: TaskEvent) -> None:
        """Async event handler invoked during TaskExecutor execution."""
        self.on_event(event)

    def render_report(self, report: TaskExecutionReport) -> Panel:
        """Renders the final TaskExecutionReport into a Rich Panel with step details."""
        color = "green" if report.status == TaskStatus.COMPLETED else "red"
        title = f"[bold {color}]Task Execution Summary: {report.status.value.upper()}[/bold {color}]"

        # Build steps breakdown table
        table = Table(
            title="Step Execution Breakdown",
            border_style=color,
            header_style=f"bold {color}",
        )
        table.add_column("Step", style="bold white")
        table.add_column("Title", style="white")
        table.add_column("Tool", style="cyan")
        table.add_column("Status")
        table.add_column("Verified")
        table.add_column("Details")

        for step in report.step_summaries:
            status_val = step.get("status", "")
            if status_val == "completed":
                step_status = "[bold green]COMPLETED[/bold green]"
            elif status_val == "failed":
                step_status = "[bold red]FAILED[/bold red]"
            elif status_val == "skipped":
                step_status = "[bold yellow]SKIPPED[/bold yellow]"
            else:
                step_status = f"[dim]{status_val}[/dim]"

            ver_val = step.get("verified")
            if ver_val is True:
                ver_display = "[green]PASSED[/green]"
            elif ver_val is False:
                ver_display = "[red]FAILED[/red]"
            else:
                ver_display = "[dim]N/A[/dim]"

            details = step.get("error") or step.get("result") or ""
            details_line = details.replace("\n", " ").strip()
            if len(details_line) > 80:
                details_line = details_line[:77] + "..."

            table.add_row(
                escape(step.get("id", "")),
                escape(step.get("title", "")),
                escape(step.get("tool") or "[dim]None[/dim]"),
                step_status,
                ver_display,
                escape(details_line),
            )

        summary_content = (
            f"[bold]Objective:[/bold] {escape(report.objective)}\n"
            f"[bold]Plan ID:[/bold] {escape(report.plan_id)}\n"
            f"[bold]Status:[/bold] [{color}]{report.status.value.upper()}[/{color}]\n"
            f"[bold]Steps Completed:[/bold] {report.completed_steps}/{report.total_steps} "
            f"([dim]Failed: {report.failed_steps}[/dim])\n"
            f"[bold]Duration:[/bold] {report.duration_seconds:.2f}s\n"
            f"[bold]Final Output:[/bold] {escape(report.final_output)}\n"
        )

        self.console.print(Panel(summary_content, title=title, border_style=color))
        self.console.print(table)

        return Panel(table, title=title, border_style=color)

    def print_report(self, report: TaskExecutionReport) -> None:
        """Prints the execution report to the terminal."""
        self.render_report(report)

    def display_report(self, report: TaskExecutionReport) -> Panel:
        """Alias for render_report."""
        return self.render_report(report)
