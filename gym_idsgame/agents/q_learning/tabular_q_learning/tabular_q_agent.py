"""
An agent for the IDSGameEnv that implements the tabular Q-learning algorithm.
"""
from typing import Union
import numpy as np
import time
import tqdm
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.envs.constants import constants
from gym_idsgame.agents.q_learning.q_agent import QAgent

class TabularQAgent(QAgent):
    """
    A simple implementation of the Q(0)-learning algorithm (Sutton & Barto).
    Q-learning is a one-step off-policy algorithm
    """
    def __init__(self, env:IdsGameEnv, config: QAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(TabularQAgent, self).__init__(env, config)
        self.Q_attacker = np.random.rand(self.env.num_states, self.env.num_attack_actions)
        self.Q_defender = np.random.rand(1, self.env.num_attack_actions + 1)

    def get_action(self, s, eval=False, attacker=True) -> int:
        """
        Sample an action using an epsilon-greedy policy with respect to the current Q-values

        :param s: the state to sample an action for
        :param eval: whether sampling an action in eval mode (greedy without exploration)
        :param attacker: if true, sample action from attacker, else use defender
        :return: a sampled action
        """
        if attacker:
            actions = list(range(self.env.num_attack_actions))
            legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))
        else:
            actions = list(range(self.env.num_defense_actions))
            legal_actions = list(filter(lambda action: self.env.is_defense_legal(action), actions))
        if np.random.rand() < self.config.epsilon and not eval:
            return np.random.choice(legal_actions)
        max_legal_action_value = float("-inf")
        max_legal_action = float("-inf")
        if attacker:
            for i in range(len(self.Q_attacker[s])):
                if i in legal_actions and self.Q_attacker[s][i] > max_legal_action_value:
                    max_legal_action_value = self.Q_attacker[s][i]
                    max_legal_action = i
        else:
            for i in range(len(self.Q_defender[s])):
                if i in legal_actions and self.Q_defender[s][i] > max_legal_action_value:
                    max_legal_action_value = self.Q_defender[s][i]
                    max_legal_action = i
        if max_legal_action == float("-inf") or max_legal_action_value == float("-inf"):
            raise AssertionError("Error when selecting action greedily according to the Q-function")
        return max_legal_action

    def train(self) -> ExperimentResult:
        """
        Runs the Q(0)-learning algorithm

        :return: Experiment result
        """
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        attacker_obs, defender_obs = self.env.reset(update_stats=False)

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_train.set_description_str("[Train] epsilon:{:.2f},avg_a_R:{:.2f},avg_d_R:{:.2f},"
                                             "avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                                             "acc_D_R:{:.2f}".format(self.config.epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Training
        for episode in range(self.config.num_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.render:
                    self.env.render(mode="human")

                if not self.config.attacker and not self.config.defender:
                    raise AssertionError("Must specify whether training an attacker agent or defender agent")

                # Default initialization
                attacker_state_node_id = 0
                defender_state_node_id = 0
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_state_node_id = self.env.get_attacker_node_from_observation(attacker_obs)
                    attacker_action = self.get_action(attacker_state_node_id, attacker=True)
                if self.config.defender:
                    defender_action = self.get_action(defender_state_node_id, attacker=False)
                action = (attacker_action, defender_action)

                # Take a step in the environment
                reward, obs_prime, done = self.step_and_update(action, attacker_state_node_id, defender_state_node_id)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender

            # Record episode metrics
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Log average metrics every <self.config.train_log_frequency> episodes
            if episode % self.config.train_log_frequency == 0:
                self.log_metrics(episode, self.train_result, episode_attacker_rewards, episode_defender_rewards,
                                 episode_steps)
                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_steps = []

            # Run evaluation every <self.config.eval_frequency> episodes
            if episode % self.config.eval_frequency == 0:
                self.eval()

            # Reset environment for the next episode and update game stats
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=True)
            self.outer_train.update(1)

            # Anneal epsilon linearly
            self.anneal_epsilon()

        self.config.logger.info("Training Complete")

        # Final evaluation (for saving Gifs etc)
        self.eval(log=False)

        # Log and return
        self.log_state_values()

        # Save Q Table
        self.save_q_table()

        return self.train_result

    def step_and_update(self, action, attacker_state_node_id, defender_state_node_id) -> Union[float, np.ndarray, bool]:
        obs_prime, reward, done, info = self.env.step(action)
        attacker_reward, defender_reward = reward
        attacker_obs_prime, defender_obs_prime = obs_prime

        if self.config.attacker:
            state_prime_node_id = self.env.get_attacker_node_from_observation(attacker_obs_prime)
            self.q_learning_update(attacker_state_node_id, action, attacker_reward, state_prime_node_id, attacker=True)

        if self.config.defender:
            state_prime_node_id = 0
            self.q_learning_update(defender_state_node_id, action, defender_reward, state_prime_node_id, attacker=False)

        return reward, obs_prime, done

    def q_learning_update(self, s : int, a : int, r : float, s_prime : int, attacker=True) -> None:
        """
        Performs a q_learning update

        :param s: the state id
        :param a: the action id
        :param r: the reward
        :param s_prime: the result state id
        :param attacker: boolean flag, if True update attacker Q, otherwise update defender Q
        :return: None
        """
        if attacker:
            self.Q_attacker[s, a] = self.Q_attacker[s, a] + self.config.alpha * (r + self.config.gamma * np.max(self.Q_attacker[s_prime])
                                                                                 - self.Q_attacker[s, a])
        else:
            self.Q_defender[s, a] = self.Q_defender[s, a] + self.config.alpha * (
                        r + self.config.gamma * np.max(self.Q_defender[s_prime])
                        - self.Q_defender[s, a])

    def eval(self, log=True) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :param log: whether to log the result
        :return: None
        """
        self.config.logger.info("Starting Evaluation")
        time_str = str(time.time())

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

        for episode in range(self.config.eval_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)

                # Default initialization
                attacker_state_node_id = 0
                defender_state_node_id = 0
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_state_node_id = self.env.get_attacker_node_from_observation(attacker_obs)
                    attacker_action = self.get_action(attacker_state_node_id, eval=True, attacker=True)
                if self.config.defender:
                    defender_action = self.get_action(defender_state_node_id, eval=True, attacker=False)
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
            if self.env.state.detected:
                self.eval_attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
            if self.env.state.hacked:
                self.eval_attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.num_eval_hacks += 1

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0 and log:
                if self.num_eval_hacks > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                self.log_metrics(episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                                 episode_steps, eval = True)
                episode_attacker_rewards = []
                episode_steps = []

            # Save gifs
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

            # Reset for new eval episode
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=False)
            self.outer_eval.update(1)

        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def log_state_values(self) -> None:
        """
        Utility function for printing the state-values according to the learned Q-function
        :return:
        """
        if self.config.attacker:
            self.config.logger.info("--- Attacker State Values ---")
            for i in range(len(self.Q_attacker)):
                state_value = sum(self.Q_attacker[i])
                node_id = i
                self.config.logger.info("s:{},V(s):{}".format(node_id, state_value))
            self.config.logger.info("--------------------")

        if self.config.defender:
            self.config.logger.info("--- Defender State Values ---")
            for i in range(len(self.Q_defender)):
                state_value = sum(self.Q_defender[i])
                node_id = i
                self.config.logger.info("s:{},V(s):{}".format(node_id, state_value))
            self.config.logger.info("--------------------")

    def save_q_table(self) -> None:
        """
        Saves Q table to disk in binary npy format

        :return: None
        """
        if self.config.save_dir is not None:
            if self.config.attacker:
                self.config.logger.info("Saving Q-table to: {}".format(self.config.save_dir + "/attacker_q_table.npy"))
                np.save(self.config.save_dir + "/attacker_q_table.npy", self.Q_attacker)
            if self.config.defender:
                self.config.logger.info("Saving Q-table to: {}".format(self.config.save_dir + "/defender_q_table.npy"))
                np.save(self.config.save_dir + "/defender_q_table.npy", self.Q_defender)
        else:
            self.config.logger.warning("Save path not defined, not saving Q table to disk")