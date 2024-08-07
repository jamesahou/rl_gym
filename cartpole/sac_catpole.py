from dataclasses import dataclass
import time
import os 
import tyro

@dataclass
class Args:
    seed: int = 1


class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
    
    def add(self, s, a, r, s_, d):
        if len(self.storage) > self.max_size:
            self.storage.pop(0)
        self.storage.append((s, a, r, s_, d))
    
    def sample(self, batch_size):


if __name__ == "__main__":
