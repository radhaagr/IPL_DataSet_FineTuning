#!/usr/bin/env python
import json

class StatsTool:
    """Example: loads a pre-aggregated JSON and returns simple stats.

    In your solution, compute from the provided IPL dataset (e.g., with pandas).
    """
    def __init__(self, path='ipl_stats_sample.json'):
        self.data = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}

    def last_n_overs_runs(self, team: str, n: int = 5) -> int:
        # Stub: return fixed number or lookup in self.data
        return 45

class RetrieverAgent:
    def __init__(self, tool: StatsTool):
        self.tool = tool
    def act(self, query: str):
        ctx = f"Context: Team A scored {self.tool.last_n_overs_runs('Team A', 5)} runs in last 5 overs."
        return ctx

class AnalystAgent:
    def __init__(self, model):
        self.model = model
    def act(self, query: str, context: str):
        # Stub: call fine-tuned model with query+context (implement in your solution)
        return f"Answer to '{query}' based on {context}"

def main():
    tool = StatsTool()
    retriever = RetrieverAgent(tool)
    analyst = AnalystAgent(model='stub-model')
    q = "What was Team A's runs in the last 5 overs?"
    context = retriever.act(q)
    answer = analyst.act(q, context)
    print(answer)

if __name__ == '__main__':
    main()
