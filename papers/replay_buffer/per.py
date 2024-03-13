import random
import numpy as np


class PER:
    """
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/replay_buffer.py
    """
    def __init__(self, capacity, alpha):
        """
        ### Initialize
        """
        # We use a power of $2$ for capacity because it simplifies the code and debugging
        self.capacity = capacity # capacity 버퍼의 크기, 2 ** capacity 만큼의 버퍼 사용
        # $\alpha$
        self.alpha = alpha # alpha == 0 => uniformly sampling replay buffer

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = [0 for _ in range(2 * self.capacity)] # ** 여야 하는거 아닌가?
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = 1. # 처음 들어오는 transition에 대해서는 maximum(= 1)의 우선 순위를 매김으로써 최소한 한번은 학습에 사용될 수 있도록 함.

        # Arrays for buffer
        self.data = {
            'obs': np.zeros(shape=(capacity, 4, 84, 84), dtype=np.uint8), # 현재 상태 3, 84, 84 의미(obs의 차원)
            'action': np.zeros(shape=capacity, dtype=np.int32), # 현재 상태에서 어떤 행동을 했는지
            'reward': np.zeros(shape=capacity, dtype=np.float32), # 보상
            'next_obs': np.zeros(shape=(capacity, 4, 84, 84), dtype=np.uint8), # 다음 상태 
            'done': np.zeros(shape=capacity, dtype=np.bool) # 에피소드의 종료 여부 
        }
        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0
        
    def add(self, obs, action, reward, next_obs, done):
        """
        ### Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['done'][idx] = done

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = self.max_priority ** self.alpha
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)
        
    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])
            
    
    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
            
    def _sum(self):
        """
        #### $\sum_k p_k^\alpha$
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]
    
    def _min(self):
        """
        #### $\min_k p_k^\alpha$
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]
    
    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity
    
    def sample(self, batch_size, beta):
        """
        ### Sample from buffer
        """

        # Initialize samples
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }  
        
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx
            
        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = self._min() / self._sum()
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.size) ** (-beta)
        
        for i in range(batch_size):
            idx = samples['indexes'][i]
            # $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            # $w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$
            weight = (prob * self.size) ** (-beta)
            # Normalize by $\frac{1}{\max_i w_i}$,
            #  which also cancels off the $\frac{1}{N}$ term
            samples['weights'][i] = weight / max_weight

        # Get samples data
        for k, v in self.data.items():
            samples[k] = v[samples['indexes']] # 인덱스에 맞는 샘플 데이터 저장 indexes: 인덱스들의 리스트임으로 buffer에서 해당 인덱스들에 해당하는 샘플들이 추출 됨.
 
        return samples

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        for idx, priority in zip(indexes, priorities):
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)

            # Calculate $p_i^\alpha$
            priority_alpha = priority ** self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.size


    
    
    
