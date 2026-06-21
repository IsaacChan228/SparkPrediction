#!/usr/bin/env python3
"""Start a small SparkSession, report JVM heap, restart with larger heap, and report again.

Usage:
    python scripts/check_and_increase_spark_heap.py [--new-mem 24g]

Notes:
- Requires PySpark installed and a JDK on PATH to run `jmap`/`jcmd` for detailed heap info.
- If JDK tools are missing, the script will still start Spark and print the Java PID and command line.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time


def find_java_procs() -> list[dict]:
    # Try psutil first if available
    try:
        import psutil

        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            if p.info.get("name") and p.info["name"].lower().startswith("java"):
                procs.append({"pid": p.info["pid"], "cmdline": " ".join(p.info.get("cmdline") or [])})
        return procs
    except Exception:
        pass

    # Fallback: PowerShell query (Windows)
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' } | Select-Object ProcessId,CommandLine | ConvertTo-Json",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        out = p.stdout.strip()
        if not out:
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        procs = []
        for item in data:
            pid = int(item.get("ProcessId"))
            cmdline = item.get("CommandLine") or ""
            procs.append({"pid": pid, "cmdline": cmdline})
        return procs
    except Exception:
        return []


def run_jdk_tools(pid: int) -> None:
    tools = [
        ("jcmd", ["jcmd", str(pid), "VM.flags"]),
        ("jmap", ["jmap", "-heap", str(pid)]),
        ("jstat", ["jstat", "-gc", str(pid), "1", "1"]),
    ]
    for name, cmd in tools:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if p.returncode == 0:
                print(f"--- {name} output for PID {pid} ---")
                print(p.stdout)
            else:
                print(f"{name} returned non-zero (rc={p.returncode}); stderr:\n{p.stderr}")
        except FileNotFoundError:
            print(f"{name} not found on PATH; install a JDK to use {name}.")
        except Exception as ex:
            print(f"Error running {name}: {ex}")


def start_spark_session(extra_conf: dict | None = None):
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName("heap-check").master(os.environ.get("SPARK_MASTER", "local[1]"))
    if extra_conf:
        for k, v in extra_conf.items():
            builder = builder.config(k, v)
    # Try to bind to localhost to avoid Windows hostname issues
    builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    spark = builder.getOrCreate()
    return spark


def inspect_current_java():
    procs = find_java_procs()
    if not procs:
        print("No java.exe processes found.")
        return
    for p in procs:
        print(f"Found java PID={p['pid']}")
        print(f"  cmdline: {p['cmdline'][:200]}")
        run_jdk_tools(p["pid"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-mem", default="24g", help="New driver memory (e.g. 24g)")
    args = parser.parse_args()

    print("Current SPARK_DRIVER_MEMORY:", os.environ.get("SPARK_DRIVER_MEMORY"))

    print("\nStarting initial SparkSession (default memory)...")
    spark = start_spark_session()
    time.sleep(1)
    inspect_current_java()

    print("Stopping SparkSession...")
    try:
        spark.stop()
    except Exception:
        pass
    del spark
    time.sleep(1)

    print(f"\nSetting SPARK_DRIVER_MEMORY={args.new_mem} and restarting SparkSession...")
    os.environ["SPARK_DRIVER_MEMORY"] = args.new_mem
    # also set extraJavaOptions to ensure -Xmx is applied
    extra = {
        "spark.driver.memory": args.new_mem,
        "spark.driver.extraJavaOptions": f"-Xmx{args.new_mem}",
    }
    spark2 = start_spark_session(extra_conf=extra)
    time.sleep(1)
    inspect_current_java()

    print("Stopping SparkSession (after resize)...")
    try:
        spark2.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
