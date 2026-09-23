from __future__ import annotations

import hashlib
import json
import time
from typing import Any, TypedDict

from app.config import get_settings
from app.context import ContextBuilder, MemoryExtractor
from app.northwind import METRIC_DEFINITIONS
from app.runtime import CheckpointManager, EventBus
from app.model import BaseModelGateway
from app.presenter import build_business_answer
from app.prompts import PromptRegistry
from app.repository import Repository
from app.tools import ToolRegistry

from langgraph.graph import END, START, StateGraph

STOP_MESSAGES = {
    "canceled": "任务已取消，已停止后续调用。",
    "token_budget": "已达到本次Token预算，分析提前结束。",
    "runtime_limit": "已达到本次运行时限，分析提前结束。",
    "loop_limit": "已达到模型调用次数上限，分析提前结束。",
    "tool_limit": "已达到工具调用次数上限，分析提前结束。",
    "repeated_call": "检测到重复工具调用，已停止继续尝试。",
    "sql_repair_limit": "SQL修复次数已用尽，请检查问题或数据结构。",
    "permission_denied": "查询违反数据访问限制，已停止执行。",
    "non_retryable_error": "工具发生不可重试错误，已停止执行。",
}


class AgentState(TypedDict, total=False):
    run_id: str
    thread_id: str
    user_id: str
    question: str
    token_budget: int
    system_prompt: str
    messages: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    last_tool_result: dict[str, Any] | None
    last_query_result: dict[str, Any] | None
    first_sql_success: bool | None
    sql_failure_count: int
    unresolved_errors: dict[str, dict[str, Any]]
    stop_reason: str
    answer_text: str
    answer_payload: dict[str, Any]
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    loop_count: int
    tool_call_count: int
    repeated_calls: dict[str, int]
    verification_passed: bool
    should_replan: bool
    error: str
    started_at_epoch: float


class AnalysisGraph:
    """Bounded LangGraph loop; execution checks are not answer-correctness checks."""

    def __init__(
        self,
        repo: Repository,
        context_builder: ContextBuilder,
        prompts: PromptRegistry,
        model: BaseModelGateway,
        tools: ToolRegistry,
        events: EventBus,
    ) -> None:
        self.repo = repo
        self.context_builder = context_builder
        self.prompts = prompts
        self.model = model
        self.tools = tools
        self.events = events
        self.memory = MemoryExtractor()
        self.settings = get_settings()
        self.checkpoints = CheckpointManager()
        self.compiled = self._compile_langgraph()

    @property
    def runtime_name(self) -> str:
        return "LangGraph + PostgreSQL Checkpoint"

    def _compile_langgraph(self):
        graph = StateGraph(AgentState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("model", self._model)
        graph.add_node("tools", self._tools)
        graph.add_node("verify", self._verify)
        graph.add_node("memory", self._memory)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "model")
        graph.add_conditional_edges("model", self._after_model, {"tools": "tools", "verify": "verify"})
        graph.add_edge("tools", "model")
        graph.add_conditional_edges("verify", self._after_verify, {"model": "model", "memory": "memory"})
        graph.add_edge("memory", END)
        return graph.compile(checkpointer=self.checkpoints.saver)

    def invoke(self, state: AgentState) -> AgentState:
        config = {
            "configurable": {"thread_id": state["run_id"]},
            "recursion_limit": self.settings.max_agent_loops * 4 + 10,
        }
        return dict(self.compiled.invoke(state, config=config))

    def _prepare(self, state: AgentState) -> dict[str, Any]:
        self.events.publish(state["run_id"], "context_building", {"message": "正在理解问题和业务口径"})
        pack = self.context_builder.build(
            run_id=state["run_id"],
            thread_id=state["thread_id"],
            user_id=state["user_id"],
            question=state["question"],
            token_budget=state["token_budget"],
        )
        metrics = "\n".join(f"- {item['name']}：{item['formula']}" for item in METRIC_DEFINITIONS)
        sql_policy = self.prompts.render(
            "nl2sql", question=state["question"], schema=pack.schema, metrics=metrics
        )
        rendered = self.prompts.render(
            "agent",
            context=pack.prompt_text(),
            sql_policy=sql_policy.text,
            token_budget=state["token_budget"],
            max_loops=self.settings.max_agent_loops,
        )
        for prompt in (sql_policy, rendered):
            self.repo.save_prompt_snapshot(
                state["run_id"], prompt.name, prompt.version, prompt.prompt_hash, prompt.text
            )
        self.events.publish(
            state["run_id"], "context_ready",
            {"message": "已准备相关数据结构和业务知识", "estimated_tokens": pack.estimated_tokens},
        )
        return {
            "system_prompt": rendered.text,
            "messages": [{"role": "user", "content": state["question"]}],
            "pending_tool_calls": [],
            "last_tool_result": None,
            "last_query_result": None,
            "first_sql_success": None,
            "sql_failure_count": 0,
            "unresolved_errors": {},
            "stop_reason": "",
            "answer_text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_tokens": getattr(pack, "embedding_tokens", 0),
            "loop_count": 0,
            "tool_call_count": 0,
            "repeated_calls": {},
            "verification_passed": False,
            "should_replan": False,
            "error": "",
            "started_at_epoch": time.time(),
        }

    def _runtime_stop(self, state: AgentState) -> str:
        if state.get("stop_reason"):
            return state["stop_reason"]
        if self.repo.get_run(state["run_id"]).status == "canceled":
            return "canceled"
        total = sum(int(state.get(key, 0)) for key in ("input_tokens", "output_tokens", "embedding_tokens"))
        if total >= state["token_budget"]:
            return "token_budget"
        if time.time() - state["started_at_epoch"] >= self.settings.max_runtime_seconds:
            return "runtime_limit"
        return ""

    def _stop(self, state: AgentState, reason: str) -> dict[str, Any]:
        message = STOP_MESSAGES.get(reason, STOP_MESSAGES["non_retryable_error"])
        if not state.get("stop_reason"):
            self.events.publish(state["run_id"], "stopped", {"reason": reason, "message": message})
        return {"stop_reason": reason, "answer_text": message, "pending_tool_calls": [],
                "verification_passed": False, "should_replan": False}

    def _model(self, state: AgentState) -> dict[str, Any]:
        reason = self._runtime_stop(state)
        if not reason and int(state.get("loop_count", 0)) >= self.settings.max_agent_loops:
            reason = "loop_limit"
        if reason:
            return self._stop(state, reason)
        loops = int(state.get("loop_count", 0)) + 1

        self.events.publish(state["run_id"], "thinking", {"message": "正在选择合适的数据工具", "loop": loops})
        self.events.publish(
            state["run_id"],
            "answer_reset",
            {"message": "正在组织回答", "loop": loops},
            durable=False,
        )
        streamed = False

        def on_token(delta: str) -> None:
            nonlocal streamed
            if not delta:
                return
            if not streamed:
                self.events.publish(
                    state["run_id"],
                    "answer_start",
                    {"message": "正在生成回答"},
                    durable=False,
                )
                streamed = True
            self.events.publish(
                state["run_id"],
                "answer_delta",
                {"delta": delta},
                durable=False,
            )

        decision = self.model.invoke(
            state["system_prompt"],
            state.get("messages", []),
            self.tools.schemas(),
            on_token=on_token,
        )
        if streamed:
            self.events.publish(
                state["run_id"],
                "answer_end",
                {"message": "回答生成完成"},
                durable=False,
            )
        messages = list(state.get("messages", []))
        pending = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in decision.tool_calls
        ]
        messages.append({
            "role": "assistant",
            "content": decision.content,
            "tool_calls": pending,
        })
        return {
            "messages": messages,
            "pending_tool_calls": pending,
            # Text accompanying a tool request is not a final answer.
            "answer_text": "" if pending else (decision.content or ""),
            "input_tokens": int(state.get("input_tokens", 0)) + decision.input_tokens,
            "output_tokens": int(state.get("output_tokens", 0)) + decision.output_tokens,
            "loop_count": loops,
        }

    def _after_model(self, state: AgentState) -> str:
        return "tools" if state.get("pending_tool_calls") else "verify"

    def _tools(self, state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        repeated = dict(state.get("repeated_calls", {}))
        last_result = state.get("last_tool_result")
        last_query = state.get("last_query_result")
        count = int(state.get("tool_call_count", 0))
        first_sql_success = state.get("first_sql_success")
        sql_failures = int(state.get("sql_failure_count", 0))
        errors = dict(state.get("unresolved_errors", {}))
        stop = self._runtime_stop(state)

        for call in state.get("pending_tool_calls", []):
            stop = stop or self._runtime_stop(state)
            if not stop and count >= self.settings.max_tool_calls:
                stop = "tool_limit"
            if not stop:
                fingerprint = hashlib.sha256(
                    f"{call['name']}|{json.dumps(call['arguments'], sort_keys=True, ensure_ascii=False)}".encode()
                ).hexdigest()
                repeated[fingerprint] = repeated.get(fingerprint, 0) + 1
                if repeated[fingerprint] > 2:
                    stop = "repeated_call"
            if stop:
                # Every requested call gets a response, even when it is skipped.
                last_result = {"ok": False, "skipped": True, "error": STOP_MESSAGES[stop],
                               "error_type": stop, "retryable": False}
                self.repo.add_tool_call(state["run_id"], call["name"], call["arguments"], last_result, False, 0)
            else:
                self.events.publish(
                    state["run_id"], "tool_running",
                    {"message": self._business_tool_message(call["name"]), "tool": call["name"]},
                )
                last_result = self.tools.execute(state["run_id"], call["name"], call["arguments"], self.repo)
                count += 1
                if call["name"] == "query_database":
                    # Keep the first attempted SQL outcome across all graph turns.
                    if first_sql_success is None:
                        first_sql_success = bool(last_result.get("ok"))
                        self.repo.update_run(state["run_id"], first_sql_success=first_sql_success)
                    last_query = last_result
                    if not last_result.get("ok"):
                        sql_failures += 1
                if last_result.get("ok"):
                    errors.pop(call["name"], None)
                else:
                    errors[call["name"]] = last_result
                    if last_result.get("retryable") is False:
                        stop = "permission_denied" if last_result.get("error_type") == "permission_denied" else "non_retryable_error"
                    elif sql_failures > self.settings.max_sql_repairs:
                        stop = "sql_repair_limit"
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": last_result,
            })
            self.events.publish(
                state["run_id"], "tool_finished",
                {"message": "数据工具执行完成" if last_result.get("ok") else "数据工具未完成",
                 "ok": last_result.get("ok"), "error_type": last_result.get("error_type"),
                 "skipped": bool(last_result.get("skipped"))},
            )

        updates: dict[str, Any] = {
            "messages": messages,
            "pending_tool_calls": [],
            "last_tool_result": last_result,
            "last_query_result": last_query,
            "first_sql_success": first_sql_success,
            "sql_failure_count": sql_failures,
            "unresolved_errors": errors,
            "tool_call_count": count,
            "repeated_calls": repeated,
        }
        if stop:
            updates.update(self._stop(state, stop))
        return updates

    def _verify(self, state: AgentState) -> dict[str, Any]:
        self.events.publish(state["run_id"], "verifying", {"message": "正在检查执行是否完整"})
        stop = self._runtime_stop(state)
        if stop:
            return self._stop(state, stop)
        result = state.get("last_tool_result")
        text = (state.get("answer_text") or "").strip()
        passed = bool(text)
        reason = "回答已生成"
        if result is not None:
            passed = bool(result.get("ok")) and bool(text)
            reason = "工具执行成功且已形成回答" if passed else result.get("error", "结果不完整")
        if state.get("unresolved_errors"):
            passed = False
            reason = "；".join(
                f"{name}: {item.get('error', '工具失败')}。{item.get('recovery_hint', '')}"
                for name, item in state["unresolved_errors"].items()
            )

        can_retry = (
            not passed
            and int(state.get("loop_count", 0)) < self.settings.max_agent_loops
            and int(state.get("tool_call_count", 0)) < self.settings.max_tool_calls
        )
        messages = list(state.get("messages", []))
        if can_retry:
            messages.append({
                "role": "user",
                "content": f"上一步未通过验证：{reason}。请更换查询方式，不要重复相同调用。",
            })
            self.events.publish(state["run_id"], "replanning", {"message": "结果不完整，正在调整分析方法"})
        else:
            self.events.publish(
                state["run_id"], "verified",
                {"message": "执行完整性检查结束", "passed": passed, "reason": reason},
            )
        updates = {"verification_passed": passed, "should_replan": can_retry, "messages": messages}
        if not passed and not can_retry:
            limit = "loop_limit" if int(state.get("loop_count", 0)) >= self.settings.max_agent_loops else "tool_limit"
            updates.update(self._stop(state, limit))
        return updates

    def _after_verify(self, state: AgentState) -> str:
        return "model" if state.get("should_replan") else "memory"

    def _memory(self, state: AgentState) -> dict[str, Any]:
        candidates = [] if state.get("stop_reason") == "canceled" else self.memory.persist(
            self.repo, state["user_id"], state["question"]
        )
        tokens = self._token_summary(state)
        payload = build_business_answer(
            state.get("answer_text") or "本次分析未能形成有效结论。",
            # Inspecting schema/metrics after SQL must not erase its result table.
            state.get("last_query_result") or state.get("last_tool_result"),
            tokens=tokens,
            verification_passed=bool(state.get("verification_passed")),
        )
        payload["execution"] = {
            "stop_reason": state.get("stop_reason", ""),
            "first_sql_success": state.get("first_sql_success"),
            "sql_failure_count": state.get("sql_failure_count", 0),
            "check_scope": "execution_integrity_only",
        }
        self.events.publish(
            state["run_id"], "completed",
            {"message": "分析完成", "memory_updates": len(candidates)},
        )
        return {"answer_payload": payload}

    def _token_summary(self, state: AgentState) -> dict[str, Any]:
        input_tokens = int(state.get("input_tokens", 0))
        output_tokens = int(state.get("output_tokens", 0))
        embedding_tokens = int(state.get("embedding_tokens", 0))
        total = input_tokens + output_tokens + embedding_tokens
        estimated_cost = (
            input_tokens / 1_000_000 * self.settings.input_cost_per_million
            + output_tokens / 1_000_000 * self.settings.output_cost_per_million
            + embedding_tokens / 1_000_000 * self.settings.embedding_cost_per_million
        )
        return {
            "input": input_tokens,
            "output": output_tokens,
            "embedding": embedding_tokens,
            "total": total,
            "budget": state["token_budget"],
            "usage_ratio": round(total / max(1, state["token_budget"]) * 100, 2),
            "estimated_cost": round(estimated_cost, 6),
            "currency": "CNY配置估算",
        }

    @staticmethod
    def _business_tool_message(name: str) -> str:
        return {
            "inspect_schema": "正在确认数据结构",
            "query_database": "正在查询经营数据",
            "get_metric_definition": "正在核对指标口径",
            "web_search": "正在联网查找外部资料",
        }.get(name, "正在调用分析工具")
