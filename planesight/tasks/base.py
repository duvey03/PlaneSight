"""Background-task harness for PlaneSight.

Wraps QgsTask so long-running work (data fetch, derivative computation,
detection, plane fitting) never blocks the QGIS UI. Concrete tasks either pass a
callable or subclass PlaneSightTask and override run_work().
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from qgis.core import Qgis, QgsMessageLog, QgsTask


class PlaneSightTask(QgsTask):
    """Base QgsTask that runs a pure-Python callable off the UI thread.

    The result (or the raised exception) is delivered to ``on_finished`` on the
    main thread, so callers can update layers/UI safely.
    """

    def __init__(
        self,
        description: str,
        work: Optional[Callable[[], Any]] = None,
        on_finished: Optional[Callable[[bool, Any], None]] = None,
    ):
        super().__init__(description, QgsTask.CanCancel)
        self._work = work
        self._on_finished = on_finished
        self._result: Any = None
        self._exception: Optional[BaseException] = None

    def run_work(self) -> Any:
        """Override to perform the work. Default: call the provided callable."""
        if self._work is None:
            raise NotImplementedError(
                "Provide a work callable or override run_work()."
            )
        return self._work()

    def run(self) -> bool:
        """QgsTask entry point (executes on a worker thread)."""
        try:
            self._result = self.run_work()
            return True
        except BaseException as exc:
            self._exception = exc
            QgsMessageLog.logMessage(
                f"PlaneSight task failed: {exc}", "PlaneSight", Qgis.Critical
            )
            return False

    def finished(self, ok: bool) -> None:
        """QgsTask callback (executes on the main thread)."""
        if self._on_finished is not None:
            self._on_finished(ok, self._result if ok else self._exception)
