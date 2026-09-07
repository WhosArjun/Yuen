#!/usr/bin/env python3
"""
main.py

CLI entrypoint for the local agent.

    $ python main.py
    $ python main.py --model qwen2.5:7b --workspace ./my_project

No API key is ever requested or required. This talks only to a local
Ollama server.
"""

import argparse
import sys

from config import Config
from agent.llm import OllamaNotInstalledError, OllamaNotRunningError, OllamaModelNotFoundError
from agent.agent import Agent

BANNER = r"""
 _                     _      _                    _
| |    ___   ___ __ _ | |    / \   __ _  ___ _ __ | |_
| |   / _ \ / __/ _` || |   / _ \ / _` |/ _ \ '_ \| __|
| |__| (_) | (_| (_| || |__/ ___ \ (_| |  __/ | | | |_
|_____\___/ \___\__,_||_____/   \_\__, |\___|_| |_|\__|
                                   |___/
"""


def parse_args():
    p = argparse.ArgumentParser(description="Local, free, open-source AI agent running on Ollama.")
    p.add_argument("--model", help="Ollama model tag, e.g. llama3.1:8b")
    p.add_argument("--workspace", help="Workspace directory the agent can read/write")
    p.add_argument("--ollama-host", help="Ollama server URL (default http://localhost:11434)")
    p.add_argument("--max-steps", type=int, help="Max tool-call steps per turn")
    p.add_argument("--keep-alive", help="How long Ollama keeps the model loaded, e.g. '30m', '1h', '-1' (default 30m)")
    p.add_argument("--temperature", type=float, help="Sampling temperature, lower = more deterministic (default 0.2)")
    p.add_argument("--no-confirm", action="store_true", help="Disable confirmation prompts (NOT recommended)")
    p.add_argument("--config", help="Path to a config.json file")
    p.add_argument("--session", default="default", help="Memory session id (separate conversation/history)")
    return p.parse_args()


def check_ollama_or_exit(llm_model: str, host: str):
    try:
        from agent.llm import OllamaLLM
    except Exception as e:
        print(f"Fatal import error: {e}")
        sys.exit(1)

    try:
        llm = OllamaLLM(model=llm_model, host=host)
    except OllamaNotInstalledError as e:
        print("=" * 70)
        print("The 'ollama' Python package is not installed.")
        print(str(e))
        print("=" * 70)
        sys.exit(1)

    try:
        llm.ensure_ready()
    except OllamaNotRunningError as e:
        print("=" * 70)
        print("Could not reach a running Ollama server.")
        print(str(e))
        print("If you don't have Ollama installed yet:")
        print("  1. Download it from https://ollama.com/download")
        print("  2. Install it, then run:  ollama serve")
        print("  3. In another terminal:   ollama pull " + llm_model)
        print("=" * 70)
        sys.exit(1)
    except OllamaModelNotFoundError as e:
        print("=" * 70)
        print(str(e))
        print("=" * 70)
        sys.exit(1)

    return llm


def main():
    args = parse_args()

    overrides = {
        "model": args.model,
        "workspace": args.workspace,
        "ollama_host": args.ollama_host,
        "max_steps": args.max_steps,
        "keep_alive": args.keep_alive,
        "temperature": args.temperature,
        "require_confirmation": False if args.no_confirm else None,
    }
    config = Config.load(config_path=args.config, overrides=overrides)

    print(BANNER)
    print(f"Model:      {config.model}")
    print(f"Workspace:  {config.workspace}")
    print(f"Ollama:     {config.ollama_host}")
    print(f"Max steps:  {config.max_steps}")
    print(f"Confirm dangerous actions: {config.require_confirmation}")
    print()

    check_ollama_or_exit(config.model, config.ollama_host)

    def status_callback(msg: str):
        print(f"\nAgent: {msg}")

    agent = Agent(config, session_id=args.session, status_callback=status_callback)

    print("Type your request below. Type 'exit' or 'quit' to leave, 'plan' to show the current plan.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if user_input.lower() == "plan":
            print(agent.planner.get_plan_text())
            continue

        final = agent.run_turn(user_input)
        print(f"\nAgent: {final}\n")


if __name__ == "__main__":
    main()
