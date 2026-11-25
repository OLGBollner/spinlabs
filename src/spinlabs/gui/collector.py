"""Garbage collection on the Tk thread only (no Matplotlib or Tk import here, so it is tested headless)."""

from __future__ import annotations

import gc

try:
    from tkinter import TclError as _TclError
except ImportError:  # no Tk: the collector still works with any root that has after()
    class _TclError(Exception):
        pass


class MainThreadCollector:
    """Run Python's cyclic garbage collector on the Tk thread only.

    Tk objects (variables, images, widgets of rebuilt figures) end up in
    reference cycles.  If the automatic collector fires on the background
    worker thread (a large production cell allocates a lot there), their
    ``__del__`` calls Tcl from the wrong thread and Tcl aborts the process
    ("Tcl_AsyncDelete: async handler deleted by the wrong thread").  Automatic
    collection is therefore switched off and replaced by a poll on the Tk event
    loop that collects with the same generation thresholds.
    """

    INTERVAL_MS = 200

    def __init__(self, root) -> None:
        self.root = root
        gc.disable()
        self._job = self.root.after(self.INTERVAL_MS, self._tick)
        if hasattr(root, "bind"):
            root.bind("<Destroy>", self._stop, add="+")

    @staticmethod
    def collect() -> int | None:
        """Collect the generation automatic collection would; ``None`` if below the threshold."""
        t0, t1, t2 = gc.get_threshold()
        c0, c1, c2 = gc.get_count()
        if c0 < t0:
            return None
        generation = 2 if c1 >= t1 and c2 >= t2 else 1 if c1 >= t1 else 0
        gc.collect(generation)
        return generation

    def _tick(self) -> None:
        self.collect()
        try:
            self._job = self.root.after(self.INTERVAL_MS, self._tick)
        except _TclError:  # the window is gone
            self._job = None

    def _stop(self, event=None) -> None:
        if event is not None and event.widget is not self.root:
            return  # a child widget was destroyed
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except _TclError:
                pass
            self._job = None
