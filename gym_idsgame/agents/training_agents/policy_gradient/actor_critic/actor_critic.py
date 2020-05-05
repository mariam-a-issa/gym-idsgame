"""
An agent for the IDSGameEnv that implements the REINFORCE with Baseline (Critic) Policy Gradient algorithm.
"""
from typing import Union, List
import numpy as np
import time
import tqdm
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.envs.constants import constants
from gym_idsgame.agents.training_agents.policy_gradient.actor_critic.model import ActorCriticNN
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent import PolicyGradientAgent
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig

class ActorCriticAgent(PolicyGradientAgent):
    """
    An implementation of the REINFORCE with Baseline Policy Gradient algorithm
    """
    def __init__(self, env:IdsGameEnv, config: PolicyGradientAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(ActorCriticAgent, self).__init__(env, config)
        self.attacker_policy_network = None
        self.defender_policy_network = None
        self.critic_loss_fn = None
        self.attacker_optimizer = None
        self.defender_optimizer = None
        self.attacker_lr_decay = None
        self.defender_lr_decay = None
        self.tensorboard_writer = SummaryWriter(self.config.tensorboard_dir)
        self.initialize_models()
        self.tensorboard_writer.add_hparams(self.config.hparams_dict(), {})
        self.machine_eps = np.finfo(np.float32).eps.item()

    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """

        # Initialize models
        self.attacker_policy_network = ActorCriticNN(self.config.input_dim, self.config.output_dim_attacker,
                                                     self.config.hidden_dim,
                                                     num_hidden_layers=self.config.num_hidden_layers,
                                                     hidden_activation=self.config.hidden_activation)
        self.defender_policy_network = ActorCriticNN(self.config.input_dim, self.config.output_dim_defender,
                                                     self.config.hidden_dim,
                                                     num_hidden_layers=self.config.num_hidden_layers,
                                                     hidden_activation=self.config.hidden_activation)

        # Specify device
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:0")
            self.config.logger.info("Running on the GPU")
        else:
            device = torch.device("cpu")
            self.config.logger.info("Running on the CPU")

        self.attacker_policy_network.to(device)
        self.defender_policy_network.to(device)

        # Construct loss function
        if self.config.critic_loss_fn == "MSE":
            self.critic_loss_fn = torch.nn.MSELoss()
        elif self.config.critic_loss_fn == "Huber":
            self.critic_loss_fn = torch.nn.SmoothL1Loss()
        else:
            raise ValueError("Loss function not recognized")

        # Define Optimizer. The call to model.parameters() in the optimizer constructor will contain the learnable
        # parameters of the layers in the model
        if self.config.optimizer == "Adam":
            self.attacker_optimizer = torch.optim.Adam(self.attacker_policy_network.parameters(), lr=self.config.alpha)
            self.defender_optimizer = torch.optim.Adam(self.defender_policy_network.parameters(), lr=self.config.alpha)
        elif self.config.optimizer == "SGD":
            self.attacker_optimizer = torch.optim.SGD(self.attacker_policy_network.parameters(), lr=self.config.alpha)
            self.defender_optimizer = torch.optim.SGD(self.defender_policy_network.parameters(), lr=self.config.alpha)
        else:
            raise ValueError("Optimizer not recognized")

        # LR decay
        if self.config.lr_exp_decay:
            self.attacker_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.attacker_optimizer,
                                                                       gamma=self.config.lr_decay_rate)
            self.defender_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.attacker_optimizer,
                                                                            gamma=self.config.lr_decay_rate)


    def training_step(self, saved_rewards : List[float], saved_log_probs : List[torch.Tensor],
                      saved_state_values : List[torch.Tensor], attacker=True) -> torch.Tensor:
        """
        Performs a training step of the Deep-Q-learning algorithm (implemented in PyTorch)

        :param saved_rewards list of rewards encountered in the latest episode trajectory
        :param saved_log_probs list of log-action probabilities (log p(a|s)) encountered in the latest episode trajectory
        :param saved_state_values list of state values encountered in the latest episode trajectory
        :return: loss
        """

        R = 0
        policy_loss = [] # list to save actor (policy) loss
        value_loss = []  # list to save critic (value) loss
        returns = [] # list to save the true (observed) values

        # Create discounted returns. When episode is finished we can go back and compute the observed cumulative
        # discounted reward by using the observed rewards
        for r in saved_rewards[::-1]:
            R = r + self.config.gamma * R
            returns.insert(0, R)
        num_rewards = len(returns)

        # convert list to torch tensor
        returns = torch.tensor(returns)

        # normalize
        std = returns.std()
        if num_rewards < 2:
            std = 0
        returns = (returns - returns.mean()) / (std + self.machine_eps)

        # Compute PG "loss" which in reality is the expected reward, which we want to maximize with gradient ascent
        for log_prob, state_value, R in zip(saved_log_probs, saved_state_values, returns):
            # Compute the advantage which will be used as a baseline in REINFORCE to reduce the gradient variance
            # Intuitively, the advantage tells us how much better the observed reward was compared to the expected reward
            # If the advantage of an action is high, it means that the current policy should be modified to reinforce
            # that action. That is, the advantage tells us for every action much better that action is than
            # the average action.
            advantage = R - state_value.item()

            # negative log probsince we are doing gradient descent (not ascent)
            policy_loss.append(-log_prob * advantage)

            # calculate critic loss using Huber loss
            value_loss.append(self.critic_loss_fn(state_value, torch.tensor([R])))

        # Compute gradient and update models
        if attacker:
            # reset gradients
            self.attacker_optimizer.zero_grad()
            # sum up all the values of policy losses and value losses
            loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
            # perform backprop
            loss.backward()
            # maybe clip gradient
            if self.config.clip_gradient:
                torch.nn.utils.clip_grad_norm_(self.attacker_policy_network.parameters(), 1)
            # gradient descent step
            self.attacker_optimizer.step()
        else:
            # reset gradients
            self.defender_optimizer.zero_grad()
            # sum up all the values of policy losses and value losses
            loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
            # perform backprop
            loss.backward()
            # maybe clip gradient
            if self.config.clip_gradient:
                torch.nn.utils.clip_grad_norm_(self.defender_policy_network.parameters(), 1)
            # gradient descent step
            self.defender_optimizer.step()

        return loss


    def get_action(self, state: np.ndarray, attacker : bool = True) -> Union[int, torch.Tensor]:
        """
        Samples an action from the policy network

        :param state: the state to sample an action for
        :param attacker: boolean flag whether running in attacker mode (if false assume defender)
        :return: The sampled action id
        """
        state = torch.from_numpy(state.flatten()).float()

        # Move to GPU if using GPU
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:0")
            state = state.to(device)

        # Calculate legal actions
        if attacker:
            actions = list(range(self.env.num_attack_actions))
            legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))
            non_legal_actions = list(filter(lambda action: not self.env.is_attack_legal(action), actions))
        else:
            actions = list(range(self.env.num_defense_actions))
            legal_actions = list(filter(lambda action: self.env.is_defense_legal(action), actions))
            non_legal_actions = list(filter(lambda action: not self.env.is_defense_legal(action), actions))

        # Forward pass using the current policy network to predict P(a|s)
        if attacker:
            action_probs, state_value = self.attacker_policy_network(state)
        else:
            action_probs, state_value = self.defender_policy_network(state)

        # Set probability of non-legal actions to 0
        action_probs_1 = action_probs.clone()
        action_probs_1[non_legal_actions] = 0

        # Use torch.distributions package to create a parameterizable probability distribution of the learned policy
        # PG uses a trick to turn the gradient into a stochastic gradient which we can sample from in order to
        # approximate the true gradient (which we can’t compute directly). It can be seen as an alternative to the
        # reparameterization trick
        policy_dist = Categorical(action_probs_1)

        # Sample an action from the probability distribution
        action = policy_dist.sample()

        # log_prob returns the log of the probability density/mass function evaluated at value.
        # save the log_prob as it will use later on for computing the policy gradient
        # policy gradient theorem says that the stochastic gradient of the expected return of the current policy is
        # the log gradient of the policy times the expected return, therefore we save the log of the policy distribution
        # now and use it later to compute the gradient once the episode has finished.
        log_prob = policy_dist.log_prob(action)

        return action.item(), log_prob, state_value


    def train(self) -> ExperimentResult:
        """
        Runs the REINFORCE with Baseline algorithm

        :return: Experiment result
        """
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs

        attacker_state = self.update_state(attacker_obs, defender_obs, [])
        defender_state = self.update_state(defender_obs, attacker_obs, [])

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []
        episode_avg_attacker_loss = []
        episode_avg_defender_loss = []

        # Logging
        self.outer_train.set_description_str("[Train] epsilon:{:.2f},avg_a_R:{:.2f},avg_d_R:{:.2f},"
                                             "avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                                             "acc_D_R:{:.2f}".format(self.config.epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Training
        for episode in range(self.config.num_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            episode_attacker_loss = 0.0
            episode_defender_loss = 0.0
            saved_attacker_log_probs = []
            saved_attacker_rewards = []
            saved_attacker_state_values = []
            saved_defender_log_probs = []
            saved_defender_rewards = []
            saved_defender_state_values = []
            while not done:
                if self.config.render:
                    self.env.render(mode="human")

                if not self.config.attacker and not self.config.defender:
                    raise AssertionError("Must specify whether training an attacker agent or defender agent")

                # Default initialization
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_action, attacker_log_prob, attacker_state_value = self.get_action(attacker_state, attacker=True)
                    saved_attacker_log_probs.append(attacker_log_prob)
                    saved_attacker_state_values.append(attacker_state_value)
                if self.config.defender:
                    defender_action, defender_log_prob, defender_state_value = self.get_action(defender_state, attacker=False)
                    saved_defender_log_probs.append(defender_log_prob)
                    saved_defender_state_values.append(defender_state_value)

                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _ = self.env.step(action)

                # Update metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                saved_attacker_rewards.append(attacker_reward)
                episode_defender_reward += defender_reward
                saved_defender_rewards.append(defender_reward)
                episode_step += 1

                # Move to the next state
                obs = obs_prime
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender
                attacker_state = self.update_state(attacker_obs, defender_obs, attacker_state)
                defender_state = self.update_state(defender_obs, attacker_obs, defender_state)

            # Render final frame
            if self.config.render:
                self.env.render(mode="human")

            # Perform Policy Gradient updates
            if self.config.attacker:
                loss = self.training_step(saved_attacker_rewards, saved_attacker_log_probs,
                                          saved_attacker_state_values, attacker=True)
                episode_attacker_loss += loss.item()

            if self.config.defender:
                loss = self.training_step(saved_defender_rewards, saved_defender_log_probs,
                                          saved_defender_state_values, attacker=False)
                episode_defender_loss += loss.item()

            # Decay LR after every episode
            lr = self.config.alpha
            if self.config.lr_exp_decay:
                self.attacker_lr_decay.step()
                lr = self.attacker_lr_decay.get_lr()[0]

            # Record episode metrics
            self.num_train_games += 1
            self.num_train_games_total += 1
            if self.env.state.hacked:
                self.num_train_hacks += 1
                self.num_train_hacks_total += 1
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            if episode_step > 0:
                if self.config.attacker:
                    episode_avg_attacker_loss.append(episode_attacker_loss/episode_step)
                if self.config.defender:
                    episode_avg_defender_loss.append(episode_defender_loss / episode_step)
            else:
                if self.config.attacker:
                    episode_avg_attacker_loss.append(episode_attacker_loss)
                if self.config.defender:
                    episode_avg_defender_loss.append(episode_defender_loss)

            episode_steps.append(episode_step)

            # Log average metrics every <self.config.train_log_frequency> episodes
            if episode % self.config.train_log_frequency == 0:
                if self.num_train_games > 0 and self.num_train_games_total > 0:
                    self.train_hack_probability = self.num_train_hacks / self.num_train_games
                    self.train_cumulative_hack_probability = self.num_train_hacks_total / self.num_train_games_total
                else:
                    self.train_hack_probability = 0.0
                    self.train_cumulative_hack_probability = 0.0
                self.log_metrics(episode, self.train_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 episode_avg_attacker_loss, episode_avg_defender_loss, lr=lr)

                # Log values and gradients of the parameters (histogram summary) to tensorboard

                if self.config.attacker:
                    for tag, value in self.attacker_policy_network.named_parameters():
                        tag = tag.replace('.', '/')
                        self.tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), episode)
                        self.tensorboard_writer.add_histogram(tag + '_attacker/grad', value.grad.data.cpu().numpy(),
                                                              episode)

                if self.config.defender:
                    for tag, value in self.defender_policy_network.named_parameters():
                        tag = tag.replace('.', '/')
                        self.tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), episode)
                        self.tensorboard_writer.add_histogram(tag + '_defender/grad', value.grad.data.cpu().numpy(),
                                                              episode)

                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_steps = []
                self.num_train_games = 0
                self.num_train_hacks = 0

            # Run evaluation every <self.config.eval_frequency> episodes
            if episode % self.config.eval_frequency == 0:
                self.eval(episode)

            # Save models every <self.config.checkpoint_frequency> episodes
            if episode % self.config.checkpoint_freq == 0:
                self.save_model()

            # Reset environment for the next episode and update game stats
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=True)
            attacker_state = self.update_state(attacker_obs, defender_obs, [])
            defender_state = self.update_state(defender_obs, attacker_obs, [])
            self.outer_train.update(1)

            # Anneal epsilon linearly
            self.anneal_epsilon()

        self.config.logger.info("Training Complete")

        # Final evaluation (for saving Gifs etc)
        self.eval(self.config.num_episodes-1, log=False)

        # Save networks
        self.save_model()

        return self.train_result

    def update_state(self, observation_1: np.ndarray, observation_2: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Update approximative Markov state

        :param observation_1: latest observation
        :param state: current state
        :return: new state
        """
        if self.env.fully_observed():
            if self.config.state_length == 1:
                return np.array(observation_1 + observation_2)
            if len(state) == 0:
                temp = np.append(observation_1, observation_2)
                s = np.array([temp] * self.config.state_length)
                return s
            temp = np.append(observation_1, observation_2)
            state = np.append(state[1:], np.array([temp]), axis=0)
            return state
        else:
            if self.config.state_length == 1:
                return np.array(observation_1)
            if len(state) == 0:
                return np.array([observation_1] * self.config.state_length)
            state = np.append(state[1:], np.array([observation_1]), axis=0)
            return state

    def eval(self, train_episode, log=True) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :param train_episode: the train episode to keep track of logging
        :param log: whether to log the result
        :return: None
        """
        self.config.logger.info("Starting Evaluation")
        time_str = str(time.time())

        self.num_eval_games = 0
        self.num_eval_hacks = 0

        if len(self.eval_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting eval with non-empty result object")
        if self.config.eval_episodes < 1:
            return
        done = False

        # Video config
        if self.config.video:
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
                                      video_frequency=self.config.video_frequency)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_eval = tqdm.tqdm(total=self.config.eval_episodes, desc='Eval Episode', position=1)
        self.outer_eval.set_description_str(
            "[Eval] avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
            "acc_D_R:{:.2f}".format(0.0, 0,0, 0.0, 0.0, 0.0, 0.0))

        # Eval
        attacker_obs, defender_obs = self.env.reset(update_stats=False)
        attacker_state = self.update_state(attacker_obs, defender_obs, [])
        defender_state = self.update_state(defender_obs, attacker_obs, [])

        for episode in range(self.config.eval_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)

                # Default initialization
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_action, _, _ = self.get_action(attacker_state, attacker=True)
                if self.config.defender:
                    defender_action, _, _ = self.get_action(defender_state, attacker=False)
                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _ = self.env.step(action)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender
                attacker_state = self.update_state(attacker_obs, defender_obs, attacker_state)
                defender_state = self.update_state(defender_obs, attacker_obs, defender_state)

            # Render final frame when game completed
            if self.config.eval_render:
                self.env.render()
                time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, episode_step))

            # Record episode metrics
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Update eval stats
            self.num_eval_games +=1
            self.num_eval_games_total += 1
            if self.env.state.detected:
                self.eval_attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
            if self.env.state.hacked:
                self.eval_attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.num_eval_hacks += 1
                self.num_eval_hacks_total +=1

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0 and log:
                if self.num_eval_hacks > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                if self.num_eval_games_total > 0:
                    self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                        self.num_eval_games_total)
                self.log_metrics(episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 eval = True, update_stats=False)

            # Save gifs
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(train_episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

                # Add frames to tensorboard
                for idx, frame in enumerate(self.env.episode_frames):
                    self.tensorboard_writer.add_image(str(train_episode) + "_eval_frames/" + str(idx),
                                                       frame, global_step=train_episode,
                                                      dataformats = "HWC")


            # Reset for new eval episode
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=False)
            attacker_state = self.update_state(attacker_obs, defender_obs, attacker_state)
            defender_state = self.update_state(defender_obs, attacker_obs, defender_state)
            self.outer_eval.update(1)

        # Log average eval statistics
        if log:
            if self.num_eval_hacks > 0:
                self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
            if self.num_eval_games_total > 0:
                self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                    self.num_eval_games_total)

            self.log_metrics(train_episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                             episode_steps, eval=True, update_stats=True)

        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def save_model(self) -> None:
        """
        Saves the PyTorch Model Weights

        :return: None
        """
        time_str = str(time.time())
        if self.config.save_dir is not None:
            if self.config.attacker:
                path = self.config.save_dir + "/" + time_str + "_attacker_policy_network.pt"
                self.config.logger.info("Saving policy-network to: {}".format(path))
                torch.save(self.attacker_policy_network.state_dict(), path)
            if self.config.defender:
                path = self.config.save_dir + "/" + time_str + "_defender_policy_network.pt"
                self.config.logger.info("Saving policy-network to: {}".format(path))
                torch.save(self.defender_policy_network.state_dict(), path)
        else:
            self.config.logger.warning("Save path not defined, not saving policy-networks to disk")
