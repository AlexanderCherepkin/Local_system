"""Production Circuit Breaker with State Machine and Metrics."""
import asyncio
import time
from enum import Enum
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitMetrics:
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    failures_in_window: list = field(default_factory=list)
    last_failure_time: Optional[float] = None
    state_change_times: Dict[str, float] = field(default_factory=dict)

    def record_success(self):
        self.success_count += 1
        self.total_requests += 1

    def record_failure(self):
        self.failure_count += 1
        self.total_requests += 1
        self.failures_in_window.append(time.time())
        self.last_failure_time = time.time()

    def get_failure_rate(self, window_seconds: float = 60) -> float:
        cutoff = time.time() - window_seconds
        self.failures_in_window = [t for t in self.failures_in_window if t > cutoff]
        return len(self.failures_in_window) / max(1, self.total_requests)


class CircuitOpenError(Exception):
    def __init__(self, message="Circuit breaker is OPEN", state=CircuitState.OPEN, recovery_time=None):
        super().__init__(message)
        self.state = state
        self.recovery_time = recovery_time


class CircuitBreaker:
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        window_seconds: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.window_seconds = window_seconds
        
        self._state = CircuitState.CLOSED
        self._half_open_call_count = 0
        self._last_state_change_time: float = time.time()
        self.metrics = CircuitMetrics()

    @property
    def state(self) -> CircuitState:
        return self._get_current_state()

    def _get_current_state(self) -> CircuitState:
        if self._state == CircuitState.CLOSED:
            return CircuitState.CLOSED
        elif self._state == CircuitState.OPEN:
            elapsed = time.time() - self.metrics.state_change_times.get("open", 0)
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_call_count = 0
                self.metrics.state_change_times["half_open"] = time.time()
        return self._state

    async def execute(self, operation: Callable[[], Any], record_success: bool = True) -> Any:
        current_time = time.time()
        current_state = self._get_current_state()
        
        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(
                message=f"Circuit {self.name} is OPEN",
                state=current_state,
                recovery_time=self.recovery_timeout - (current_time - self.metrics.state_change_times.get("open", 0))
            )

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_call_count >= self.half_open_max_calls:
                self._state = CircuitState.OPEN
                self.metrics.state_change_times["open"] = time.time()
                raise CircuitOpenError(message="Circuit exceeded max calls in HALF_OPEN", state=CircuitState.OPEN)

        try:
            result = operation()
            
            if record_success:
                self.metrics.record_success()
            
            if current_state == CircuitState.HALF_OPEN:
                self._half_open_call_count += 1
                self._state = CircuitState.CLOSED
                self.metrics.state_change_times["closed"] = time.time()

            return result
        except Exception as e:
            self.metrics.record_failure()
            
            if current_state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self.metrics.state_change_times["open"] = time.time()

            # Check threshold for CLOSED -> OPEN transition
            if current_state == CircuitState.CLOSED and self.metrics.failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self.metrics.state_change_times["open"] = time.time()

        raise e

    def reset(self):
        self._state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()

    def get_health_report(self) -> Dict:
        current_state = self._get_current_state()
        return {
            "name": self.name,
            "state": current_state.value,
            "failure_count": self.metrics.failure_count,
            "success_count": self.metrics.success_count,
            "total_requests": self.metrics.total_requests,
            "failure_rate": self.metrics.get_failure_rate(self.window_seconds),
        }
