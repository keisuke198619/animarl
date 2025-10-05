import numpy as np
import random
import components.sum_tree as sum_tree


class Experience(object):
    """ The class represents prioritized experience replay buffer.

        The class has functions: store samples, pick samples with
        probability in proportion to sample's priority, update
        each sample's priority, reset alpha.

        see https://arxiv.org/pdf/1511.05952.pdf .

        """

    def __init__(self, memory_size, from_demo, alpha=1):
        self.alpha = alpha
        self.from_demo = from_demo
        if from_demo:
            self.tree = sum_tree.SumTree(int(memory_size//2))
            self.memory_size = int(memory_size//2) # 25000
        else:
            self.tree = sum_tree.SumTree(memory_size)
            self.memory_size = memory_size # 5000
    def add(self, priority):
        index = self.tree.add(priority**self.alpha)
        #if len(np.array(priority).shape)>0:
        if priority != 1:
            import pdb; pdb.set_trace()
        return index

    def select(self, batch_size):

        if self.tree.filled_size() < batch_size:
            return None

        indices = []
        priorities = []
        # if not self.from_demo:
        for _ in range(batch_size):
            r = random.random()
            priority, index = self.tree.find(r)
            priorities.append(priority)
            indices.append(index)
            self.priority_update([index], [0])  # To avoid duplicating
        '''else:
            for _ in range(int(batch_size//2)):
                r_ = random.random()
                priority, index = self.tree.find(r_)
                priorities.append(priority)
                priorities.append(priority)
                indices.append(2*index)
                indices.append(2*index+1)
                self.priority_update([2*index,2*index+1], [0,0])  # To avoid duplicating '''           

        self.priority_update(indices, priorities)  # Revert priorities

        return indices

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

                Parameters
                ----------
                indices :
                        list of sample indices
                """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, np.array(p)**self.alpha) # np.array(p)
