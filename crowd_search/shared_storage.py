"""This shared storage objects sits between the explorer and trainer nodes
to act as a hub for collating the relavant information between the different
groups of nodes.

We'd like for the trainer to be able to upload its new policy weights and the
explorer to upload the history of each new game. We use this object as the
intermediate step so the other nodes can continue to execute their jobs."""


class SharedStorage:
    def __init__(self) -> None:
        self.history = []
        self.policy = None

    def update_policy(self, policy):
        self.policy = policy

    def get_policy(self):
        return self.policy

    def upload_history(self, history):
        self.history.append(history)

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history.clear()
