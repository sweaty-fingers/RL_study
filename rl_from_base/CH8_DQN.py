import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit) # 선입 선출, 정의한 큐의 크기보다 더 많이 데이터가 들어오면 자동으로 가장 오래된 데이터가 빠짐.
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n) # buffer에서 n개의 샘플 추출
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], [] # done_mask: 종료상태의 벨류 값이 0이 되도록 하기위한 masking 값. 종료상태에서는 0, 나머지 상태에서는 1의 값을 갖음.
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)
    
    
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
        
        
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a) # 각 상태에서의 qout에서 특정 액션을 취했을 때의 가치함수 추출
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) # max 값의 output은 (values, indices) -> value값 추출한 후 차원을 맞춰주기 위해 unsqueeze
        
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0 # 종료면 0 아니면 1
            memory.put((s,a,r/100.0,s_prime, done_mask)) # 보상의 값이 너무 커서 100으로 나눠줌으로써 스케일 조정 (경험적 조정)
            s = s_prime

            score += r # 평가 지표
            if done:
                break
            
        if memory.size()>2000: # 2000개의 데이터(에피소드 x)가 쌓이기 전까지는 학습 x, 버퍼에 데이터가 충분히 쌓이지 않은 채로 학습을 하게되면 초기 데이터에 모델이 편향될 수 있음.
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict()) # print_interval 이 지날때마다 타겟 네트워크 갱신, 결과 출력
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()
    
if __name__ == "__main__":
    main()