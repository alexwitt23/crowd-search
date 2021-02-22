"""This shared storage objects sits between the explorer and trainer nodes
to act as a hub for collating the relavant information between the different
groups of nodes.

We'd like for the trainer to be able to upload its new policy weights and the
explorer to upload the history of each new game. We use this object as the
intermediate step so the other nodes can continue to execute their jobs."""


class SharedStorage:
    """Shared storage object. Simple interface which basically acts as a
    holding spot for different data in the training pipeline."""

    def __init__(self) -> None:
        """Initialize the obect."""
        self.history = []
        self.policy = None

    def update_policy(self, policy):
        """Recieve a policy object and update local copy."""
        self.policy = policy

    def get_policy(self):
        """Return policy object."""
        return self.policy

    def upload_history(self, history):
        """Add history to internal list."""
        self.history.append(history)

    def get_history(self):
        """Return the internal history list."""
        return self.history

    def clear_history(self):
        """Clear out the internal history. Called after history is
        returned to a trainer node."""
        self.history.clear()
