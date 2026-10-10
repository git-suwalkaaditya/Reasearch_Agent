"""Research agent: plan -> search -> read -> verify -> reflect -> synthesize -> cited report."""
__version__ = "1.0.0"


def __getattr__(name):  # lazy import so `import research_agent` stays light
    if name == "ResearchAgent":
        from .agent import ResearchAgent

        return ResearchAgent
    if name == "Settings":
        from .config import Settings

        return Settings
    raise AttributeError(name)
