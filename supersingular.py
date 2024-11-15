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
    def __init__(self, vertex_color=BLACK, edge_color=BLACK, dimension=2, **kwargs):
        super().__init__(**kwargs)
        
        self.graph = nx.read_adjlist('adjlist.txt', delimiter=',', create_using=nx.MultiGraph)
        self.layout = nx.kamada_kawai_layout(self.graph, dim=dimension,scale=5)
        self.dimension = dimension
        self.draw_vertices()
        self.draw_edges()
    def draw_vertices(self):
        for n in self.graph.nodes:
            pos = self.get_layout_position(n)
            self.add(Dot(pos, color=BLACK, radius=1.2*DEFAULT_DOT_RADIUS))

    def get_layout_position(self, vertex):
        assert self.dimension in [2, 3]
        if self.dimension == 2:
            return np.array([*self.layout[vertex],0])
        else:
            return np.array([*self.layout[vertex]])
    
    def draw_loop(self, vertex, pos0, multiplicity):
        if multiplicity > 2:
            raise NotImplementedError("Drawing more than 2 loops on one vertex is not implemented.")
        angle_of_loop = 0
        neighbour_positions = [self.get_layout_position(neighbour) for neighbour in self.graph.neighbors(vertex) if neighbour != vertex]
        if len(neighbour_positions) > 0:
            mean_angle = np.mean([np.arctan2(pos[1] - pos0[1], pos[0] - pos0[0]) for pos in neighbour_positions])
            angle_of_loop = mean_angle + np.pi
        dvec1 = np.array([np.cos(angle_of_loop + np.pi/4), np.sin(angle_of_loop + np.pi/4), 0])
        dvec2 = np.array([np.cos(angle_of_loop - np.pi/4), np.sin(angle_of_loop - np.pi/4), 0])
        if multiplicity == 1:
            self.add(CubicBezier(pos0, pos0 + 1.5*dvec1, pos0 + 1.5*dvec2, pos0, color=BLACK))
        else:
            dvec3 = np.array([np.cos(angle_of_loop + np.pi/5), np.sin(angle_of_loop + np.pi/5), 0])
            dvec4 = np.array([np.cos(angle_of_loop - np.pi/5), np.sin(angle_of_loop - np.pi/5), 0])
            self.add(CubicBezier(pos0, pos0 + 1*dvec3, pos0 + 1*dvec4, pos0, color=BLACK))
            self.add(CubicBezier(pos0, pos0 + 1.6*dvec1, pos0 + 1.6*dvec2, pos0, color=BLACK))

        
        
    def draw_multiple_edges(self, pos0, pos1, multiplicity):
        assert multiplicity > 1
        if multiplicity > 3:
            raise NotImplementedError("Drawing more than 3 edges is not implemented.")
        midpoint = (pos0 + pos1) / 2
        magnitude = np.linalg.norm(pos1 - pos0)
        angle = np.arctan2(pos1[1] - pos0[1], pos1[0] - pos0[0])
        spline1 = midpoint + magnitude/4 * np.array([np.cos(angle + np.pi/2), np.sin(angle + np.pi/2), 0])
        spline2 = midpoint + magnitude/4 * np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2), 0])
        self.add(CubicBezier(pos0, spline1, spline1, pos1, color=BLACK))
        self.add(CubicBezier(pos0, spline2, spline2, pos1, color=BLACK))
        if multiplicity == 3:
            self.add(Line(pos0,pos1, color=BLACK))  

    def draw_edges(self):
        already_drawn = set()
        for (start, end, multiplicity) in sorted(self.graph.edges, key=lambda x: x[2], reverse=True):
            multiplicity = multiplicity + 1
            if (end, start) in already_drawn or (start, end) in already_drawn:
                continue
            already_drawn.add((start, end))
            # For some reason networkx stores multiplicities 1 less than they are
            pos0 = self.get_layout_position(start)
            pos1 = self.get_layout_position(end)
            if start == end:
                self.draw_loop(start, pos0, multiplicity)
            elif multiplicity == 1:
                self.add(Line(pos0, pos1, color=BLACK))
            elif multiplicity > 1:
                self.draw_multiple_edges(pos0, pos1, multiplicity)

    def draw_cycle(self, length,number=0, color=BLUE, starting_vertex=None):
        cycles = nx.simple_cycles(self.graph)
        cycles = list(filter(lambda x: len(x) == length, cycles))
        cycle = cycles[number % len(list(cycles))]
        drawn_cycle = []
        
        if starting_vertex is not None and starting_vertex in cycle:
            starting_index = cycle.index(starting_vertex)
            cycle = cycle[starting_index:] + cycle[:starting_index]
        print(cycle)
        if length == 7 and number in [1,4]:
            # This is a hack for the presentation
            cycle = cycle[:1] + list(reversed(cycle[1:]))
        for i in range(len(cycle)):
            v_i = cycle[i]
            v_i_plus_1 = cycle[(i+1) % len(cycle)]
            pos0 = self.get_layout_position(v_i)
            pos1 = self.get_layout_position(v_i_plus_1)
            drawn_edge = Arrow(color=color, max_tip_length_to_length_ratio=0.1)
            drawn_edge.put_start_and_end_on(pos0, pos1)
            drawn_cycle.append(drawn_edge)
        return drawn_cycle
    
    def draw_path(self, start_vertex, end_vertex, color=BLUE):
        shortest_path = nx.shortest_path(self.graph, start_vertex, end_vertex)

        drawn_path = []
        print(shortest_path)
        for i in range(len(shortest_path)-1):
            pos0 = self.get_layout_position(shortest_path[i])
            pos1 = self.get_layout_position(shortest_path[i+1])
            drawn_edge = Arrow(color=color, max_tip_length_to_length_ratio=0.1)
            drawn_edge.put_start_and_end_on(pos0, pos1)
            drawn_path.append(drawn_edge)
        return drawn_path



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
        graph = Graph(adj_list)
        self.add(graph)

class WalkInGraph(MovingCameraScene):
    def construct(self):
        graph = Graph(adj_list)
        self.play(Create(graph), run_time=3)
        #Good one below
        cycle1 = graph.draw_cycle(7,4,color=BLUE, starting_vertex='128')
        cycle2 = graph.draw_cycle(7,1, color=RED, starting_vertex='128')
        original_width = self.camera.frame.width.copy()
        for edge in cycle1:
            #self.play(self.camera.frame.animate.move_to(edge).set(width=0.8*original_width))
            self.wait(3)
            self.play(Create(edge), run_time=3)
        self.wait(2)
        for edge in cycle2:
            self.wait(3)
            self.play(Create(edge), run_time=3)
            
        self.wait(5)
        #self.play(self.camera.frame.animate.move_to(ORIGIN).set(width=original_width))

class WalkInGraph2(MovingCameraScene):
    def construct(self):
        graph = Graph(adj_list)
        self.play(Create(graph), run_time=3)
        #Good one below
        cycle1 = graph.draw_path('128', '196', color=GREEN)
        E_label = MathTex('E', color=GREEN).next_to(graph.get_layout_position('128'), UP)
        Ep_label = MathTex('E\'', color=GREEN).next_to(graph.get_layout_position('196'), UP)
        self.add(E_label)
        original_width = self.camera.frame.width.copy()
        for edge in cycle1:
            #self.play(self.camera.frame.animate.move_to(edge).set(width=0.8*original_width))
            self.wait(2)
            self.play(Create(edge), run_time=3)
        self.play(Write(Ep_label))
        self.wait(2)
        #self.play(self.camera.frame.animate.move_to(ORIGIN).set(width=original_width))

class TestingBench(Scene):
    def construct(self):
        graph = Graph(adj_list)
        self.add(graph)
        cycle1 = graph.draw_cycle(7,4,color=BLUE, starting_vertex='128')
        cycle2 = graph.draw_cycle(7,1, color=RED, starting_vertex='128')
        for edge in cycle1:
            self.add(edge)
        for edge in cycle2:
            self.add(edge)

# To run this script, run the following command:
# manim -pr PIXELS,PIXELS tree.py TreeScene

adj_list = [['0', '380'], ['0', '380'], ['0', '380'], ['196', '196'], ['196', '246'], ['196', '246'], ['380', '382'], ['252', '380'], ['172', '246'], ['108', '246'], ['377', '382'], ['340', '382'], ['156', '252'], ['128', '252'], ['172', '211*z2 + 342'], ['172', '172*z2 + 170'], ['108', '231'], ['108', '227'], ['377', '335*z2 + 289'], ['377', '48*z2 + 241'], ['340', '375'], ['340', '340'], ['72', '156'], ['156', '227'], ['128', '215*z2 + 46'], ['128', '168*z2 + 261'], ['375', '211*z2 + 342'], ['118*z2 + 5', '211*z2 + 342'], ['375', '172*z2 + 170'], ['265*z2 + 123', '172*z2 + 170'], ['231', '119*z2 + 47'], ['231', '264*z2 + 166'], ['227', '347'], ['215*z2 + 46', '335*z2 + 289'], ['192*z2 + 25', '335*z2 + 289'], ['48*z2 + 241', '168*z2 + 261'], ['191*z2 + 217', '48*z2 + 241'], ['72', '72'], ['72', '72'], ['215*z2 + 46', '62*z2 + 91'], ['321*z2 + 153', '168*z2 + 261'], ['118*z2 + 5', '265*z2 + 123'], ['118*z2 + 5', '287*z2 + 240'], ['265*z2 + 123', '96*z2 + 144'], ['119*z2 + 47', '264*z2 + 166'], ['119*z2 + 47', '62*z2 + 91'], ['321*z2 + 153', '264*z2 + 166'], ['347', '192*z2 + 25'], ['347', '191*z2 + 217'], ['192*z2 + 25', '191*z2 + 217'], ['62*z2 + 91', '287*z2 + 240'], ['96*z2 + 144', '321*z2 + 153'], ['96*z2 + 144', '287*z2 + 240']]
['0,380,380,380', '196,196,246,246', '380,382,252', '246,172,108', '382,377,340', '252,156,128', '172,211*z2 + 342,172*z2 + 170', '108,231,227', '377,335*z2 + 289,48*z2 + 241', '340,375,340', '156,72,227', '128,215*z2 + 46,168*z2 + 261', '211*z2 + 342,375,118*z2 + 5', '172*z2 + 170,375,265*z2 + 123', '231,119*z2 + 47,264*z2 + 166', '227,347', '335*z2 + 289,215*z2 + 46,192*z2 + 25', '48*z2 + 241,168*z2 + 261,191*z2 + 217', '375', '72,72,72', '215*z2 + 46,62*z2 + 91', '168*z2 + 261,321*z2 + 153', '118*z2 + 5,265*z2 + 123,287*z2 + 240', '265*z2 + 123,96*z2 + 144', '119*z2 + 47,264*z2 + 166,62*z2 + 91', '264*z2 + 166,321*z2 + 153', '347,192*z2 + 25,191*z2 + 217', '192*z2 + 25,191*z2 + 217', '191*z2 + 217', '62*z2 + 91,287*z2 + 240', '321*z2 + 153,96*z2 + 144', '287*z2 + 240,96*z2 + 144', '96*z2 + 144']