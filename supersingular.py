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

class Graph(VMobject):
    def __init__(self, vertex_color=BLACK, edge_color=BLACK, **kwargs):
        super().__init__(**kwargs)
        
        self.graph = nx.read_adjlist('adjlist.txt', delimiter=',', create_using=nx.MultiGraph)
        self.layout = nx.kamada_kawai_layout(self.graph, dim=2,scale=4)
        self.add_vertices()
        self.add_edges()
    def add_vertices(self):
        for n in self.graph.nodes:
            pos = self.get_layout_position(n)
            self.add(Dot(pos, color=BLACK))

    def get_layout_position(self, vertex):
        return np.array([*self.layout[vertex],0])

    def add_edges(self):
        for (start, end, multiplicity) in self.graph.edges:
            multiplicity += 1
            print(multiplicity)
            pos0 = self.get_layout_position(start)
            pos1 = self.get_layout_position(end)
            if start == end:
                self.add(CubicBezier(pos0, pos0 + [1,1,1], pos1 + [-1,1,-1], pos1, color=BLACK))
            elif multiplicity == 1:
                self.add(Line(pos0, pos1, color=BLACK))
            elif multiplicity > 1:
                self.add(CubicBezier(
                    pos0,
                    pos0 + [1,1,1],
                    pos1 + [-1,1,-1],
                    pos1,color=BLACK))
                self.add(CubicBezier(
                    pos0 + [0,0,1],
                    pos0 + [1,1,2],
                    pos1 + [-1,1,2],
                    pos1 + [0,0,1],color=BLACK))


    # def path(self, directions, edge_start=0, edge_end=0):
    #     # Input: a list of directions (0, ..., l) which determine a path in the tree from the root
    #     # Optionally, edge_start and edge_end can be set to determine what percentage of the edge is cut off for the arrow
    #     # Output: a path in the graph as a list of lines
    #     lines = []
    #     v_0 = np.array([[1,0], [0,1]])
    #     v_1 = None
    #     for depth, step in enumerate(directions):
    #         step_matrix = self.matrix_from_type(step)
    #         v_1 = v_0 @ step_matrix
            
    #         v_0_pos = self.vertex_positions[self.matrix_to_bytes(v_0)]
    #         v_1_pos = self.vertex_positions[self.matrix_to_bytes(v_1)]

    #         arrow_start = (1-edge_start)*v_0_pos + edge_start*v_1_pos
    #         arrow_end = edge_end*v_0_pos + (1-edge_end)*v_1_pos

    #         lines.append(LabeledLine(self.latex_matrix(step_matrix),
    #                                   font_size=15/(1.5**depth),
    #                                   label_color=BLUE,
    #                                   frame_fill_color=WHITE,
    #                                   label_frame = True,
    #                                   label_position=0.6,
    #                                   start=arrow_start, 
    #                                   end=arrow_end,
    #                                   color=BLUE, stroke_width=self.line_width/(1.5**depth)).set_z_index(1))
    #         v_0 = v_1.copy()
    #     return lines

            
    
class GraphFigure(Scene):
    def construct(self):
        l = 2
        graph = Graph(adj_list)
        self.add(graph)




# To run this script, run the following command:
# manim -pr PIXELS,PIXELS tree.py TreeScene

adj_list = [['0', '380'], ['0', '380'], ['0', '380'], ['196', '196'], ['196', '246'], ['196', '246'], ['380', '382'], ['252', '380'], ['172', '246'], ['108', '246'], ['377', '382'], ['340', '382'], ['156', '252'], ['128', '252'], ['172', '211*z2 + 342'], ['172', '172*z2 + 170'], ['108', '231'], ['108', '227'], ['377', '335*z2 + 289'], ['377', '48*z2 + 241'], ['340', '375'], ['340', '340'], ['72', '156'], ['156', '227'], ['128', '215*z2 + 46'], ['128', '168*z2 + 261'], ['375', '211*z2 + 342'], ['118*z2 + 5', '211*z2 + 342'], ['375', '172*z2 + 170'], ['265*z2 + 123', '172*z2 + 170'], ['231', '119*z2 + 47'], ['231', '264*z2 + 166'], ['227', '347'], ['215*z2 + 46', '335*z2 + 289'], ['192*z2 + 25', '335*z2 + 289'], ['48*z2 + 241', '168*z2 + 261'], ['191*z2 + 217', '48*z2 + 241'], ['72', '72'], ['72', '72'], ['215*z2 + 46', '62*z2 + 91'], ['321*z2 + 153', '168*z2 + 261'], ['118*z2 + 5', '265*z2 + 123'], ['118*z2 + 5', '287*z2 + 240'], ['265*z2 + 123', '96*z2 + 144'], ['119*z2 + 47', '264*z2 + 166'], ['119*z2 + 47', '62*z2 + 91'], ['321*z2 + 153', '264*z2 + 166'], ['347', '192*z2 + 25'], ['347', '191*z2 + 217'], ['192*z2 + 25', '191*z2 + 217'], ['62*z2 + 91', '287*z2 + 240'], ['96*z2 + 144', '321*z2 + 153'], ['96*z2 + 144', '287*z2 + 240']]
['0,380,380,380', '196,196,246,246', '380,382,252', '246,172,108', '382,377,340', '252,156,128', '172,211*z2 + 342,172*z2 + 170', '108,231,227', '377,335*z2 + 289,48*z2 + 241', '340,375,340', '156,72,227', '128,215*z2 + 46,168*z2 + 261', '211*z2 + 342,375,118*z2 + 5', '172*z2 + 170,375,265*z2 + 123', '231,119*z2 + 47,264*z2 + 166', '227,347', '335*z2 + 289,215*z2 + 46,192*z2 + 25', '48*z2 + 241,168*z2 + 261,191*z2 + 217', '375', '72,72,72', '215*z2 + 46,62*z2 + 91', '168*z2 + 261,321*z2 + 153', '118*z2 + 5,265*z2 + 123,287*z2 + 240', '265*z2 + 123,96*z2 + 144', '119*z2 + 47,264*z2 + 166,62*z2 + 91', '264*z2 + 166,321*z2 + 153', '347,192*z2 + 25,191*z2 + 217', '192*z2 + 25,191*z2 + 217', '191*z2 + 217', '62*z2 + 91,287*z2 + 240', '321*z2 + 153,96*z2 + 144', '287*z2 + 240,96*z2 + 144', '96*z2 + 144']