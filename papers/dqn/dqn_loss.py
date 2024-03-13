from typing import Tuple

import torch
from torch import nn
from papers.replay_buffer.per import PER

class QFuncLoss(nn.Module):
    """
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/__init__.py
    """
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, q: torch.Tensor, action: torch.Tensor, double_q: torch.Tensor,
                target_q: torch.Tensor, done: torch.Tensor, reward: torch.Tensor,
                weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        * `q` - $Q(s;\theta_i)$
        * `action` - $a$
        * `double_q` - $\textcolor{cyan}Q(s';\textcolor{cyan}{\theta_i})$
        * `target_q` - $\textcolor{orange}Q(s';\textcolor{orange}{\theta_i^{-}})$
        * `done` - whether the game ended after taking the action
        * `reward` - $r$
        * `weights` - weights of the samples from prioritized experienced replay
        """
        # $Q(s,a;\theta_i)$
        q_sampled_action = q.gather(-1, action.to(torch.long).unsqueeze(-1)).squeeze(-1)
        
    
        # Gradients shouldn't propagate gradients
        # $$r + \gamma \textcolor{orange}{Q}
        #                 \Big(s',
        #                     \mathop{\operatorname{argmax}}_{a'}
        #                         \textcolor{cyan}{Q}(s', a'; \textcolor{cyan}{\theta_i}); \textcolor{orange}{\theta_i^{-}}
        #                 \Big)$$
        with torch.no_grad():
            # Get the best action at state $s'$
            # $$\mathop{\operatorname{argmax}}_{a'}
            #  \textcolor{cyan}{Q}(s', a'; \textcolor{cyan}{\theta_i})$$
            
            best_next_action = torch.argmax(double_q, -1) # double dqn의 loss
            
            # Get the q value from the target network for the best action at state $s'$
            # $$\textcolor{orange}{Q}
            # \Big(s',\mathop{\operatorname{argmax}}_{a'}
            # \textcolor{cyan}{Q}(s', a'; \textcolor{cyan}{\theta_i}); \textcolor{orange}{\theta_i^{-}}
            # \Big)$$
            
            best_next_q_value = target_q.gather(-1, best_next_action.unsqueeze(-1)).squeeze(-1)
            
            # Calculate the desired Q value.
            # We multiply by `(1 - done)` to zero out
            # the next state Q values if the game ended.
            #
            # $$r + \gamma \textcolor{orange}{Q}
            #                 \Big(s',
            #                     \mathop{\operatorname{argmax}}_{a'}
            #                         \textcolor{cyan}{Q}(s', a'; \textcolor{cyan}{\theta_i}); \textcolor{orange}{\theta_i^{-}}
            #                 \Big)$$
            q_update = reward + self.gamma * best_next_q_value * (1 - done) # 액션이 end 상황이면 done == 1
            
            # Temporal difference error $\delta$ is used to weigh samples in replay buffer
            td_error = q_sampled_action - q_update
            
            
            # We take [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) instead of
            # mean squared error loss because it is less sensitive to outliers
            losses = self.huber_loss(q_sampled_action, q_update) # loss 함수로 huberLoss 사용
            loss = torch.mean(weights * losses)
            
            return td_error, loss



