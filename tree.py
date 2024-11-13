from manim import *
from manim.camera.camera import Camera
import networkx as nx
import numpy as np

class LabeledDot(Dot):
    def __init__(self, label, radius=None, **kwargs) -> None:
        if isinstance(label, str):
            rendered_label = MathTex(label, color=BLACK)
        else:
            rendered_label = label

        if radius is None:
            radius = (
                0.1 + max(rendered_label.get_width(), rendered_label.get_height()) / 2
            )
        Dot.__init__(self, radius=radius, **kwargs)
        rendered_label.move_to(self.get_center())
        self.add(rendered_label)

class Tree(Mobject):
    def __init__(self, l, max_depth=7, **kwargs):
        super().__init__(**kwargs)
        self.l = l
        self.max_depth = max_depth
        self.max_matrix_depth = 3

        self.line_length = 3
        self.label_scale = 0.7
        self.unlabeled_vertex_scale = 0.2
        self.line_width = 10

        self.graph = nx.Graph()
        self.vertex_positions = dict()

        self.construct_level(0, np.array([0,0,0]))


    def matrix_from_type(self, type):
        if type != self.l:
            return np.array([[1, 0], [type, self.l]])
        else:
            return np.array([[self.l, 0], [0, 1]])
    
    def latex_matrix(self, mat):
        return "\\begin{pmatrix} %d & %d \\\\ %d & %d \\end{pmatrix}" % (mat[0,0], mat[0,1], mat[1,0], mat[1,1])

    def inverse_type(self, type):
        if type == self.l:
            return 0
        else:
            return self.l
    
    def relative_angle(self, type):
        return 2*np.pi/(self.l+1) * type

    def matrix_to_bytes(self, matrix: np.array) -> bytes:
        return matrix.tobytes()
    
    def bytes_to_matrix(self, bytes_data: bytes) -> np.array:
        # Reshape to 2x2 since we know the size
        return np.frombuffer(bytes_data, dtype=np.int64).reshape(2, 2)

    def construct_level(self, depth, pos, angle_of_0=0, type=None, matrix=np.array([[1,0],[0,1]])):
        # Create the vertex
        if depth < self.max_matrix_depth:
            node = LabeledDot(
                self.latex_matrix(matrix),
                color=WHITE
            ).move_to(pos).set_z_index(2).scale(self.label_scale)
            
        else:
            node = Dot(
                radius=self.unlabeled_vertex_scale,
                color=BLACK
            ).move_to(pos).set_z_index(2)
        node.scale(1 / (1.5**depth))
        print
        self.add(node)
        hashable_matrix = self.matrix_to_bytes(matrix)
        self.graph.add_node(hashable_matrix)
        self.vertex_positions[hashable_matrix] = pos

        if depth == self.max_depth:
            return
        
        # Create the children
        for i in range(self.l + 1):
            if type is not None and i == self.inverse_type(type):
                continue
            angle = angle_of_0 + self.relative_angle(i)
            rel_pos = np.array([np.cos(angle), np.sin(angle), 0])
            
            new_pos = pos + self.line_length * rel_pos / (2**depth)
            
            edge = Line(pos, new_pos, color=BLACK, stroke_width=self.line_width/(1.5**depth)).set_z_index(0)
            self.add(edge)

            new_angle_of_0 = angle - np.pi - self.relative_angle(self.inverse_type(i))
            new_matrix = matrix @ self.matrix_from_type(i)
            self.construct_level(depth+1, new_pos, new_angle_of_0, i, new_matrix)
    def path(self, directions, edge_start=0, edge_end=0):
        # Input: a list of directions (0, ..., l) which determine a path in the tree from the root
        # Optionally, edge_start and edge_end can be set to determine what percentage of the edge is cut off for the arrow
        # Output: a path in the graph as a list of lines
        lines = []
        v_0 = np.array([[1,0], [0,1]])
        v_1 = None
        for depth, step in enumerate(directions):
            step_matrix = self.matrix_from_type(step)
            v_1 = v_0 @ step_matrix
            # Debug prints
            v_0_pos = self.vertex_positions[self.matrix_to_bytes(v_0)]
            v_1_pos = self.vertex_positions[self.matrix_to_bytes(v_1)]
            arrow_start = (1-edge_start)*v_0_pos + edge_start*v_1_pos
            arrow_end = edge_end*v_0_pos + (1-edge_end)*v_1_pos
            print(arrow_end, arrow_start)

        

            lines.append(LabeledLine(self.latex_matrix(step_matrix),
                                      font_size=15/(1.5**depth),
                                      label_color=BLUE,
                                      frame_fill_color=WHITE,
                                      label_frame = True,
                                      start=arrow_start, 
                                      end=arrow_end, 
                                      color=BLUE, stroke_width=self.line_width/(1.5**depth)).set_z_index(4))
            v_0 = v_1.copy()
        return lines

            
    
class TreeScene(Scene):
    def construct(self):
        l = 2
        tree = Tree(l)
        self.add(tree)


class WalkInTree(MovingCameraScene):
    def construct(self):
        l = 2
        tree = Tree(l,max_depth=9)
        self.add(tree)
        path = tree.path([1,0,0])
        original_width = self.camera.frame.width.copy()
        for arrow in path:
            self.play(self.camera.frame.animate.move_to(arrow).set(width=self.camera.frame.width*0.8))
            self.play(Create(arrow))
            self.wait(1)
        self.play(self.camera.frame.animate.move_to(ORIGIN).set(width=original_width))



# To run this script, run the following command:
# manim -pr PIXELS,PIXELS tree.py TreeScene