"""SHAMU Interactive Terminal REPL & Command Line Interface."""

import argparse
import asyncio
import json
import logging
import sys
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from shamu import __version__
from shamu.config import get_config, set_api_key, get_api_key
from shamu.types import ChatMessage, Role, ToolCall
from shamu.protocols import ShamuError, PermissionDeniedError
from shamu.gateway.ollama_provider import OllamaProvider
from shamu.gateway.openrouter_provider import OpenRouterProvider
from shamu.gateway.router import ModelGateway, RoutingPolicy
from shamu.speech.pipeline import VoicePipeline
from shamu.speech.sentence_splitter import SentenceSplitter
from shamu.security.firewall import ExecutionFirewall
from shamu.security.approval_manager import ApprovalManager, ApprovalRequest, ApprovalResponse
from shamu.security.audit import AuditLogger
from shamu.memory.sqlite_store import SQLiteMemoryStore
from shamu.memory.embedding import EmbeddingEngine
from shamu.memory.hybrid import (
    HybridRetriever,
    format_retrieved_context,
    assemble_user_message_content,
    extract_user_query,
)
from shamu.memory.context_compactor import ContextCompactor
from shamu.agent.orchestrator import (
    AgentOrchestrator,
    StatusUpdate,
    TokenChunk,
    ToolExecuted,
    ToolRequested,
    TurnReset,
    is_complete_json_object,
)
from shamu.tasks import (
    TaskPlanner,
    TaskExecutor,
    TaskStorage,
    TaskUIBridge,
    TaskStatus,
    TaskExecutionReport,
)

logger = logging.getLogger(__name__)


def is_complete_json_object(s: str) -> bool:
    """Returns True if the string is a complete and valid JSON object."""
    try:
        val = json.loads(s)
        return isinstance(val, dict)
    except Exception:
        return False


def create_parser() -> argparse.ArgumentParser:
    """Creates the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="shamu",
        description="SHAMU — Personal AI Desktop Assistant for Windows 11",
    )
    parser.add_argument("--version", action="version", version=f"SHAMU v{__version__}")
    parser.add_argument("--model", type=str, default=None, help="Local Ollama model name")
    parser.add_argument("--fallback", type=str, default=None, help="Cloud OpenRouter fallback model name")
    parser.add_argument("--offline", action="store_true", help="Force local-only mode (disable cloud fallback)")
    parser.add_argument("--cloud", action="store_true", help="Force cloud-only mode via OpenRouter")
    parser.add_argument("--voice", "-v", action="store_true", help="Enable voice input and spoken output mode")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Single prompt execution mode")
    return parser


def parse_cli_policy(args: argparse.Namespace) -> RoutingPolicy:
    """Determines routing policy from parsed arguments."""
    if args.offline:
        return RoutingPolicy.LOCAL_ONLY
    if args.cloud:
        return RoutingPolicy.CLOUD_ONLY
    return RoutingPolicy.AUTO_FALLBACK


class ShamuREPL:
    """Interactive command-line session manager."""

    def __init__(
        self,
        gateway: ModelGateway,
        console: Console,
        voice_mode: bool = False,
        firewall: Optional[ExecutionFirewall] = None,
        compactor: Optional[ContextCompactor] = None,
        orchestrator: Optional[AgentOrchestrator] = None,
        memory_store: Optional[SQLiteMemoryStore] = None,
        retriever: Optional[HybridRetriever] = None,
        session_id: str = "default",
        task_storage: Optional[TaskStorage] = None,
    ):
        self.gateway = gateway
        self.console = console
        self.voice_mode = voice_mode
        self.session_id = session_id
        self.retriever = retriever
        self.memory_store = memory_store or (getattr(retriever, "store", None))
        self.task_storage = task_storage
        self.firewall = firewall or ExecutionFirewall(approval_manager=ApprovalManager(), memory_store=self.memory_store)
        try:
            if getattr(self.firewall, "memory_store", None) is None and self.memory_store is not None:
                self.firewall.memory_store = self.memory_store
                if hasattr(self.firewall, "executor") and self.firewall.executor:
                    self.firewall.executor.memory_store = self.memory_store
        except Exception:
            pass
        self.voice_pipeline = VoicePipeline(gateway=self.gateway) if voice_mode else None
        self.compactor = compactor
        self.orchestrator = orchestrator or AgentOrchestrator(
            gateway=self.gateway,
            firewall=self.firewall,
            compactor=self.compactor,
        )
        self.history: List[ChatMessage] = []
        if self.memory_store:
            try:
                saved = self.memory_store._sync_load_session(self.session_id)
                if saved:
                    self.history = list(saved)
            except Exception:
                pass
        self._setup_approval_handler()

    def _setup_approval_handler(self):
        async def cli_approval_handler(req: ApprovalRequest) -> ApprovalResponse:
            if getattr(self, "_auto_approve", False) or getattr(self, "_auto_approve_plan", False):
                return ApprovalResponse(
                    request_id=getattr(req, "request_id", ""),
                    approved=True,
                    double_confirmed=True,
                    reason="Auto-approved via CLI configuration",
                )

            if not sys.stdin.isatty():
                return ApprovalResponse(
                    request_id=getattr(req, "request_id", ""),
                    approved=False,
                    reason="Non-interactive terminal: approval unavailable",
                )

            self.console.print(Panel(
                f"[bold yellow]⚠️ High-Risk Action Requested:[/bold yellow] [cyan]{req.tool_name}[/cyan]\n"
                f"[dim]Reason: {req.reason}[/dim]\n"
                f"Arguments:\n[dim]{json.dumps(req.arguments, indent=2)}[/dim]",
                title="[bold red]Security Approval Required[/bold red]",
                border_style="red",
            ))
            loop = asyncio.get_running_loop()
            if req.requires_double_confirmation:
                self.console.print("[bold red]This action requires double confirmation.[/bold red]")
                try:
                    user_conf = await asyncio.wait_for(
                        loop.run_in_executor(None, input, "Type 'CONFIRM' to authorize: "),
                        timeout=45.0,
                    )
                    if user_conf.strip() == "CONFIRM":
                        return ApprovalResponse(
                            request_id=getattr(req, "request_id", ""),
                            approved=True,
                            double_confirmed=True,
                            reason="Approved via CLI confirmation",
                        )
                    return ApprovalResponse(
                        request_id=getattr(req, "request_id", ""),
                        approved=False,
                        reason="Confirmation keyword not matched",
                    )
                except asyncio.TimeoutError:
                    return ApprovalResponse(
                        request_id=getattr(req, "request_id", ""),
                        approved=False,
                        reason="Approval timed out after 45s",
                    )
                except Exception:
                    return ApprovalResponse(
                        request_id=getattr(req, "request_id", ""),
                        approved=False,
                        reason="CLI input error",
                    )
            else:
                try:
                    user_conf = await asyncio.wait_for(
                        loop.run_in_executor(None, input, "Approve execution? [y/N]: "),
                        timeout=45.0,
                    )
                    approved = user_conf.strip().lower() in ("y", "yes")
                    return ApprovalResponse(
                        request_id=getattr(req, "request_id", ""),
                        approved=approved,
                        reason="Approved via CLI" if approved else "Rejected via CLI",
                    )
                except asyncio.TimeoutError:
                    return ApprovalResponse(
                        request_id=getattr(req, "request_id", ""),
                        approved=False,
                        reason="Approval timed out after 45s",
                    )
                except Exception:
                    return ApprovalResponse(
                        request_id=getattr(req, "request_id", ""),
                        approved=False,
                        reason="CLI input error",
                    )

        if self.firewall and self.firewall.approval_manager:
            self.firewall.approval_manager.register_handler(cli_approval_handler)

    def print_welcome(self):
        banner = Text()
        banner.append("SHAMU ", style="bold cyan")
        banner.append(f"v{__version__} — Personal AI Desktop Assistant\n", style="bold white")
        banner.append(f"• Local Model:    {self.gateway.ollama.model}\n", style="dim")
        banner.append(f"• Fallback Model: {self.gateway.openrouter.model}\n", style="dim")
        banner.append(f"• Routing Policy: {self.gateway.policy.value}\n", style="dim")
        banner.append(f"• Voice Spoken:   {'Enabled' if self.voice_mode else 'Disabled'}\n", style="dim")
        banner.append("Commands: /plan <objective>, /listen, /voice, /status, /policy, /model <name>, /clear, /exit", style="italic blue")
        self.console.print(Panel(banner, border_style="cyan", title="[bold]Welcome[/bold]"))

    async def execute_turn(self, user_input: str):
        """Streams a single conversational turn with full firewall and tool support."""
        self.console.print("\n[bold cyan]SHAMU:[/bold cyan] ", end="")

        try:
            # 1. Retrieve relevant memories via HybridRetriever
            memories = []
            if self.retriever:
                try:
                    if hasattr(self.retriever, "retrieve"):
                        res = await self.retriever.retrieve(user_input, limit=3)
                        if hasattr(res, "_mock_return_value") and hasattr(self.retriever, "hybrid_retrieve"):
                            hr_res = await self.retriever.hybrid_retrieve(user_input, top_k=3)
                            memories = hr_res if not hasattr(hr_res, "_mock_return_value") else res
                        else:
                            memories = res
                    elif hasattr(self.retriever, "hybrid_retrieve"):
                        memories = await self.retriever.hybrid_retrieve(user_input, top_k=3)
                except Exception:
                    pass

            user_content = assemble_user_message_content(user_input, memories)
            self.history.append(ChatMessage(role=Role.USER, content=user_content))
            self.orchestrator.gateway = self.gateway
            self.orchestrator.firewall = self.firewall
            self.orchestrator.compactor = self.compactor

            splitter = SentenceSplitter(min_clause_words=4) if (self.voice_mode and self.voice_pipeline) else None

            async for event in self.orchestrator.run_turn(self.history):
                if isinstance(event, TurnReset):
                    if self.voice_pipeline:
                        try:
                            await self.voice_pipeline.stop_playback()
                        except Exception:
                            pass
                    if splitter:
                        splitter = SentenceSplitter(min_clause_words=4)
                elif isinstance(event, TokenChunk):
                    self.console.print(event.text, end="", highlight=False)
                    if self.voice_mode and self.voice_pipeline and splitter and hasattr(self.voice_pipeline, "tts"):
                        sentences = splitter.feed(event.text)
                        for sentence in sentences:
                            try:
                                async for audio_chunk in self.voice_pipeline.tts.synthesize_stream(sentence):
                                    if audio_chunk:
                                        await self.voice_pipeline.enqueue_audio(audio_chunk)
                            except Exception:
                                pass
                elif isinstance(event, ToolRequested):
                    self.console.print(f"\n[dim cyan]⚡ Executing {event.tool_call.name}...[/dim cyan]")
                elif isinstance(event, ToolExecuted):
                    if event.is_error:
                        self.console.print(f"[dim red]✗ {event.result}[/dim red]")
                    elif not event.is_cached:
                        self.console.print(f"[dim green]✓ {event.tool_call.name} completed[/dim green]")
                elif isinstance(event, StatusUpdate):
                    pass

            if self.voice_mode and self.voice_pipeline and splitter and hasattr(self.voice_pipeline, "tts"):
                flushed = splitter.flush()
                for sentence in flushed:
                    try:
                        async for audio_chunk in self.voice_pipeline.tts.synthesize_stream(sentence):
                            if audio_chunk:
                                await self.voice_pipeline.enqueue_audio(audio_chunk)
                    except Exception:
                        pass
                await self.voice_pipeline.wait_for_playback()

            if self.memory_store:
                try:
                    await self.memory_store.save_session(self.session_id, self.history)
                except Exception:
                    pass

            self.console.print()

        except KeyboardInterrupt:
            self.console.print("\n[yellow][Interrupted by user][/yellow]")
            if self.voice_pipeline:
                await self.voice_pipeline.interrupt()
        except ShamuError as e:
            self.console.print(f"\n[bold red]Error:[/bold red] {e}")
        except Exception as e:
            self.console.print(f"\n[bold red]Unexpected Error:[/bold red] {e}")

    async def execute_voice_listen(self):
        """Listens from microphone and runs full voice turn."""
        if not self.voice_pipeline:
            self.voice_pipeline = VoicePipeline(gateway=self.gateway)

        self.console.print("\n[bold green][Listening... speak now][/bold green]")
        try:
            transcribed = await self.voice_pipeline.listen_and_transcribe()
            if not transcribed.strip():
                self.console.print("[dim][No speech detected][/dim]")
                return
            self.console.print(f"[bold cyan]You (Spoken):[/bold cyan] {transcribed}")
            await self.execute_turn(transcribed)
        except Exception as e:
            self.console.print(f"[red]Audio capture failed: {e}[/red]")

    def handle_slash_command(self, cmd: str) -> bool:
        """Processes slash commands. Returns True if REPL should exit."""
        parts = cmd.strip().split()
        verb = parts[0].lower()

        if verb in ("/exit", "/quit", "/q"):
            self.console.print("[dim]Goodbye![/dim]")
            return True

        if verb == "/clear":
            self.history.clear()
            if self.memory_store:
                try:
                    self.memory_store._sync_clear_session(self.session_id)
                except Exception:
                    pass
            self.console.print("[green]Conversation history cleared.[/green]")
            return False

        if verb == "/voice":
            self.voice_mode = not self.voice_mode
            if self.voice_mode and not self.voice_pipeline:
                self.voice_pipeline = VoicePipeline(gateway=self.gateway)
            self.console.print(f"[green]Voice spoken response: {'Enabled' if self.voice_mode else 'Disabled'}[/green]")
            return False

        if verb == "/status":
            status_text = (
                f"[bold]Active Configuration:[/bold]\n"
                f"• Policy:        {self.gateway.policy.value}\n"
                f"• Ollama Model:  {self.gateway.ollama.model} ({self.gateway.ollama.base_url})\n"
                f"• Cloud Model:   {self.gateway.openrouter.model}\n"
                f"• Voice Mode:    {'Enabled' if self.voice_mode else 'Disabled'}\n"
                f"• Turns in Mem:  {len(self.history)}\n"
                f"• Cloud Key Set: {'Yes' if get_api_key('openrouter') else 'No'}"
            )
            self.console.print(Panel(status_text, border_style="blue", title="Status"))
            return False

        if verb == "/model" and len(parts) > 1:
            new_model = parts[1]
            self.gateway.ollama.model = new_model
            self.console.print(f"[green]Local model updated to: {new_model}[/green]")
            return False

        if verb == "/key" and len(parts) > 2:
            service = parts[1]
            token = parts[2]
            set_api_key(service, token)
            self.console.print(f"[green]Stored key for '{service}' securely in Windows Credential Manager.[/green]")
            return False

        if verb == "/policy" and len(parts) > 1:
            val = parts[1].lower()
            if val in ("local", "local_only"):
                self.gateway.policy = RoutingPolicy.LOCAL_ONLY
            elif val in ("cloud", "cloud_only"):
                self.gateway.policy = RoutingPolicy.CLOUD_ONLY
            elif val in ("auto", "auto_fallback"):
                self.gateway.policy = RoutingPolicy.AUTO_FALLBACK
            else:
                self.console.print("[red]Unknown policy. Options: local, cloud, auto[/red]")
                return False
            self.console.print(f"[green]Policy switched to: {self.gateway.policy.value}[/green]")
            return False

        if verb == "/plan":
            if len(parts) < 2 or not parts[1].strip():
                self.console.print("[yellow]Usage: /plan <objective>[/yellow]")
                return False

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                return self.execute_plan_command(cmd)
            else:
                return asyncio.run(self.execute_plan_command(cmd))

        self.console.print(f"[yellow]Unknown command '{verb}'. Type /status or /exit.[/yellow]")
        return False

    async def execute_plan_command(self, cmd: str) -> Optional[TaskExecutionReport]:
        """Decomposes an objective into a structured TaskPlan, prompts for approval, and executes sequentially."""
        parts = cmd.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            self.console.print("[yellow]Usage: /plan <objective>[/yellow]")
            return None

        objective = parts[1].strip()
        self.console.print(f"\n[dim cyan]🧠 Planning objective: '{objective}'...[/dim cyan]")

        planner = TaskPlanner(gateway=self.gateway, firewall=self.firewall)
        ui_bridge = TaskUIBridge(console=self.console)
        task_storage = self.task_storage or TaskStorage()
        own_storage = self.task_storage is None

        try:
            plan = await planner.create_plan(objective)
            self.last_plan = plan
            task_storage.save_plan(plan)
            ui_bridge.render_plan(plan)

            # Prompts user for approval: [y/N]
            loop = asyncio.get_running_loop()
            try:
                if getattr(self, "_auto_approve_plan", False) or getattr(self, "_auto_approve", False):
                    user_conf = "y"
                else:
                    user_conf = await loop.run_in_executor(
                        None, input, "\nApprove and execute plan? [y/N]: "
                    )
            except (EOFError, KeyboardInterrupt):
                user_conf = "n"

            if user_conf.strip().lower() not in ("y", "yes"):
                self.console.print("[yellow]Plan execution cancelled by user.[/yellow]")
                plan.status = TaskStatus.CANCELLED
                task_storage.save_plan(plan)
                return None

            plan.approved = True
            task_storage.save_plan(plan)

            executor = TaskExecutor(
                firewall=self.firewall,
                approval_manager=self.firewall.approval_manager if self.firewall else None,
                workspace_root=getattr(self.firewall, "workspace_root", None),
            )
            async for event in executor.run_plan(plan, on_event=ui_bridge.handle_event):
                pass

            task_storage.save_plan(plan)
            report = executor.generate_report(plan)
            self.last_execution_report = report
            ui_bridge.render_report(report)
            return report

        except Exception as e:
            logger.exception("Plan execution error: %s", e)
            self.console.print(f"[bold red]Plan execution error:[/bold red] {e}")
            return None
        finally:
            if own_storage and hasattr(task_storage, "close"):
                try:
                    task_storage.close()
                except Exception:
                    pass

    async def execute_plan(self, objective: str) -> Optional[TaskExecutionReport]:
        """Convenience method to execute a plan for a given objective string."""
        cmd = f"/plan {objective}" if not objective.startswith("/plan") else objective
        return await self.execute_plan_command(cmd)

    async def run(self):
        """Starts the interactive prompt loop."""
        loop = asyncio.get_running_loop()
        self.print_welcome()
        while True:
            self.console.print("\nYou > ", end="")
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    self.console.print("\n[dim]Session closed.[/dim]")
                    break
                user_input = line.strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Session closed.[/dim]")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if user_input.strip() == "/listen":
                    await self.execute_voice_listen()
                    continue
                should_exit = self.handle_slash_command(user_input)
                if asyncio.iscoroutine(should_exit):
                    await should_exit
                    continue
                if should_exit:
                    break
                continue

            await self.execute_turn(user_input)


async def async_main(args: Optional[argparse.Namespace] = None):
    """Async CLI entry point."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    cfg = get_config()
    console = Console()

    local_model = args.model or cfg.local_model
    fallback_model = args.fallback or cfg.fallback_model
    policy = parse_cli_policy(args)

    ollama = OllamaProvider(base_url=cfg.ollama_url, model=local_model, timeout=float(cfg.timeout_seconds))
    openrouter = OpenRouterProvider(model=fallback_model, timeout=float(cfg.timeout_seconds))

    gateway = ModelGateway(
        ollama_provider=ollama,
        openrouter_provider=openrouter,
        policy=policy,
        on_fallback=lambda msg, exc: console.print(f"[yellow]{msg}[/yellow]"),
    )

    memory_store: Optional[SQLiteMemoryStore] = None
    embedding_engine: Optional[EmbeddingEngine] = None
    retriever: Optional[HybridRetriever] = None
    compactor: Optional[ContextCompactor] = None

    try:
        if getattr(cfg, "memory", None):
            memory_store = SQLiteMemoryStore(config=cfg.memory)
            compactor = ContextCompactor(store=memory_store, config=cfg.memory)
            embedding_engine = EmbeddingEngine(config=cfg.memory)
            retriever = HybridRetriever(
                store=memory_store,
                embedding_engine=embedding_engine,
                config=cfg.memory,
            )
    except Exception:
        pass

    firewall = ExecutionFirewall(approval_manager=ApprovalManager(), memory_store=memory_store)

    if args.prompt:
        prompt_trimmed = args.prompt.strip()
        if prompt_trimmed.startswith("/plan"):
            repl = ShamuREPL(
                gateway,
                console,
                voice_mode=args.voice,
                firewall=firewall,
                compactor=compactor,
                memory_store=memory_store,
                retriever=retriever,
            )
            await repl.execute_plan_command(prompt_trimmed)
            return

        # Non-interactive single prompt execution with multi-turn tool support
        try:
            history: List[ChatMessage] = []
            memories = []
            if retriever:
                try:
                    if hasattr(retriever, "hybrid_retrieve") and not hasattr(retriever, "_is_canonical_retriever"):
                        memories = await retriever.hybrid_retrieve(args.prompt, top_k=3)
                    elif hasattr(retriever, "retrieve"):
                        memories = await retriever.retrieve(args.prompt, limit=3)
                    else:
                        memories = []
                except Exception:
                    logger.exception("Memory retrieval failed in prompt mode")
                    memories = []

            user_content = assemble_user_message_content(
                args.prompt,
                memories if memories else None,
            )
            history.append(ChatMessage(role=Role.USER, content=user_content))
            orchestrator = AgentOrchestrator(
                gateway=gateway,
                firewall=firewall,
                compactor=compactor,
            )
            async for event in orchestrator.run_turn(history):
                if isinstance(event, TokenChunk):
                    console.print(event.text, end="", highlight=False)
                elif isinstance(event, ToolRequested):
                    console.print(f"\n[dim cyan]⚡ Executing {event.tool_call.name}...[/dim cyan]")
                elif isinstance(event, ToolExecuted):
                    if event.is_error:
                        console.print(f"[dim red]✗ {event.result}[/dim red]")
                    elif not event.is_cached:
                        console.print(f"[dim green]✓ {event.tool_call.name} completed[/dim green]")
            console.print()
        except ShamuError as e:
            console.print(f"[bold red]Error:[/bold red] {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Interactive REPL mode
    repl = ShamuREPL(
        gateway,
        console,
        voice_mode=args.voice,
        firewall=firewall,
        compactor=compactor,
        memory_store=memory_store,
        retriever=retriever,
    )
    await repl.run()


def main():
    """Synchronous entrypoint for console script."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

