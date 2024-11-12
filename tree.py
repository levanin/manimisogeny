from manim import *
from manim.camera.camera import Camera
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
        

        self.construct_level(0, np.array([0,0,0]))

    def matrix_from_type(self, type):
        if type != self.l:
            return np.matrix([[1, 0], [type, self.l]])
        else:
            return np.matrix([[self.l, 0], [0, 1]])
    
    def latex_matrix(self, mat):
        return "\\begin{pmatrix} %d & %d \\\\ %d & %d \\end{pmatrix}" % (mat[0,0], mat[0,1], mat[1,0], mat[1,1])

    def inverse_type(self, type):
        if type == self.l:
            return 0
        else:
            return self.l
    
    def relative_angle(self, type):
        return 2*np.pi/(self.l+1) * type

    def construct_level(self, depth, pos, angle_of_0=0, type=None, matrix=np.matrix([[1,0],[0,1]])):
        # Create the vertex
        if depth < self.max_matrix_depth:
            node = LabeledDot(
                self.latex_matrix(matrix),
                color=WHITE
            ).move_to(pos).set_z_index(1).scale(self.label_scale)
        else:
            node = Dot(
                radius=self.unlabeled_vertex_scale,
                color=BLACK
            ).move_to(pos).set_z_index(1)
        node.scale(1 / (1.5**depth))
        self.add(node)
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
    
class TreeScene(Scene):
    def construct(self):
        l = 4
        tree = Tree(l)
        self.add(tree)


# To run this script, run the following command:
# manim -pr PIXELS,PIXELS tree.py TreeScene