import json
import os
from collections import defaultdict

import torch


class LayerProfiler:
    """A profiler for recording layer-wise execution time and model size."""

    def __init__(self):
        self.records = defaultdict(list)
        self.starts = {}
        self.static_info = {}
        self.enabled = False

    # Public APIs ---------------------------------------------------------

    def enable(self):
        """Enable the profiler and reset existing records."""
        self.enabled = True
        self.reset()
        print("[Profiler] Profiler enabled.")

    def disable(self):
        """Disable the profiler (records are kept until reset)."""
        self.enabled = False

    def reset(self):
        """Clear all recorded data."""
        self.records.clear()
        self.starts.clear()
        self.static_info.clear()

    # Recording helpers ---------------------------------------------------

    def start_record(self, name: str):
        if not self.enabled:
            return
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.starts[name] = start_event

    def end_record(self, name: str):
        if not self.enabled or name not in self.starts:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = self.starts[name].elapsed_time(end_event)
        self.records[name].append(latency_ms)
        del self.starts[name]

    def add_static_info(self, name: str, value):
        if self.enabled:
            self.static_info[name] = value

    # Reporting -----------------------------------------------------------

    def report(self, filename: str = "profile_results.json"):
        if not self.records:
            print("[Profiler] No records to report.")
            return {}

        avg_times = {
            name: sum(latencies) / len(latencies)
            for name, latencies in sorted(self.records.items())
            if latencies
        }

        report_data = {
            "static_info": self.static_info,
            "avg_times_ms": avg_times,
            "all_times_ms": self.records,
        }

        try:
            with open(filename, "w") as f:
                json.dump(report_data, f, indent=4)
            print(f"[Profiler] Report saved to {os.path.abspath(filename)}")
        except IOError as e:
            print(f"[Profiler] Error saving report file: {e}")

        # Console summary
        print("\n--- Profiler Report ---")
        if self.static_info:
            print("\n--- Model Size ---")
            for name, value in sorted(self.static_info.items()):
                print(f"- {name}: {value}")
        if avg_times:
            print("\n--- Average Execution Time (ms) ---")
            for name, avg_time in sorted(avg_times.items()):
                print(f"- {name}: {avg_time:.4f} ms")
        print("--- End of Report ---\n")

        return report_data


# A global profiler instance for convenience --------------------------------
layer_profiler = LayerProfiler() 