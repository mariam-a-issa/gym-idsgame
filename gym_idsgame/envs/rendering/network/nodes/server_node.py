from typing import Union
import pyglet
from pyglet import clock
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.rendering.network.nodes.resource_node import ResourceNode
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.node_type import NodeType

class ServerNode(ResourceNode):
    """
    Represents a server node in the grid network
    """
    def __init__(self, render_config: RenderConfig, row: int, col: int):
        """
        Initializes the node

        :param render_config: render config, e.g font size, avatar, etc.
        :param row: the row in the grid
        :param col: the column in the grid
        """
        avatar = pyglet.resource.image(render_config.server_filename)
        super(ServerNode, self).__init__(avatar, render_config, render_config.background)
        self.col = col
        self.row = row
        self.scale = render_config.server_scale
        self.reset()

    @property
    def node_type(self) -> NodeType:
        """
        :return: the type of the node (SERVER)
        """
        return NodeType.SERVER

    def init_labels(self) -> None:
        """
        Initializes the labels of the node

        :return: None
        """
        attack_label_x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2
        attack_label_y = self.row * int((self.render_config.rect_size) / 1.5) + self.render_config.rect_size / 4
        defense_label_x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2
        defense_label_y = self.row * int((self.render_config.rect_size) / 1.5) + self.render_config.rect_size / 7
        det_label_x = self.col * self.render_config.rect_size + self.render_config.rect_size / 3
        det_label_y = self.row * int((self.render_config.rect_size) / 1.5) + self.render_config.rect_size / 3
        self.create_labels(attack_label_x=attack_label_x, attack_label_y=attack_label_y,
                           defense_label_x=defense_label_x, defense_label_y=defense_label_y,
                           det_label_x=det_label_x, det_label_y=det_label_y)

    def simulate_attack(self, attack_type:int, edges_list:list=None) -> bool:
        """
        Simulates an attack against the node.

        :param attack_type: the type of the attack
        :param edges_list: edges list for visualization (blinking)
        :return: True if the attack was successful otherwise False
        """
        for i in range(0, self.render_config.num_blinks):
            if i % 2 == 0:
                clock.schedule_once(self.blink_red_attack, self.render_config.blink_interval * i)
            else:
                clock.schedule_once(self.blink_black_attack, self.render_config.blink_interval * i)
        if self.attack_values[attack_type] < self.render_config.game_config.max_value-1:
            self.attack_values[attack_type] += 1
        self.attack_label.text = self.attack_text
        if self.attack_values[attack_type] > self.defense_values[attack_type]:
            return True # successful attack
        else:
            return False

    def blink_red_attack(self, dt, edges_list:list=None) -> None:
        """
        Makes the node and its links blink red to visualize an attack

        :param dt: the time since the last scheduled blink
        :param edges_list: list of edges to blink
        :return: None
        """
        color = constants.RENDERING.RED
        color_list = list(color) + list(color)
        for edges in self.incoming_edges:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.RENDERING.RED_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.RENDERING.RED

    def blink_black_attack(self, dt, edges_list:list=None) -> None:
        """
        Makes the node and its links blink black to visualize an attack

        :param dt: the time since the last scheduled blink
        :param edges_list: list of edges to blink
        :return: None
        """
        color = constants.RENDERING.BLACK
        color_list = list(color) + list(color)
        for edges in self.incoming_edges:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.RENDERING.BLACK_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.RENDERING.WHITE

    def center_avatar(self) -> None:
        """
        Utiltiy function for centering the avatar inside a cell

        :return: The centered coordinates in the grid
        """
        self.x = self.col*self.render_config.rect_size + self.render_config.rect_size/2.3
        self.y = int((self.render_config.rect_size/1.5))*self.row + self.render_config.rect_size/3.5

    def get_link_coords(self, upper:bool=True, lower:bool=False) -> Union[float, float, int, int]:
        """
        Gets the coordinates of link endpoints of the node

        :param upper: if True, returns the upper endpoint
        :param lower: if False, returns the lower endpoint
        :return: (x-coordinate, y-coordinate, grid-column, grid-row)
        """
        if upper:
            x = self.col*self.render_config.rect_size + self.render_config.rect_size/2
            y = (self.row+1)*(self.render_config.rect_size/1.5) - self.render_config.rect_size/6
        elif lower:
            x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2
            y = (self.row + 1) * (self.render_config.rect_size / 1.5) - self.render_config.rect_size / 1.75
        return x, y, self.col, self.row