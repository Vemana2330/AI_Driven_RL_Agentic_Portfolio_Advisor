# tools/tool_base.py

class BaseTool:
    """
    Minimal BaseTool-compatible class for our project.
    This provides the same interface.
    we need: a name, description, and a run/_run method.
    """

    name: str = "base_tool"
    description: str = "Base tool"

    def __init__(self, **kwargs):
        # Accept arbitrary kwargs for future flexibility
        pass

    def _run(self, *args, **kwargs):
        """
        Subclasses override this to implement their logic.
        """
        raise NotImplementedError("_run() must be implemented by the tool subclass.")

    def run(self, *args, **kwargs):
        """
        CrewAI-style entrypoint (if needed later).
        For now we mostly call _run directly.
        """
        return self._run(*args, **kwargs)
